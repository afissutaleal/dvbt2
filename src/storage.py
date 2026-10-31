"""
Persistent Storage Engine for Region Datasets & Preprocessed Results
Ensures that once a region's data is preprocessed, it is saved to disk
and automatically reloaded across application sessions without requiring re-upload.
Includes catalog discovery, incremental merge, and transmitter management.
"""

import os
import json
import shutil
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List


def get_region_dir(data_dir: str, region_code: str) -> str:
    path = os.path.join(data_dir, 'processed', region_code)
    os.makedirs(path, exist_ok=True)
    return path


def save_region_processed_data(
    data_dir: str,
    region_code: str,
    df_clean: pd.DataFrame,
    df_tx_sum: pd.DataFrame,
    stats: Dict[str, Any],
    df_split: Optional[pd.DataFrame] = None
) -> None:
    """Saves cleaned dataset, transmitter summary, and preprocessing stats to disk."""
    reg_dir = get_region_dir(data_dir, region_code)
    
    # Save transmitter summary CSV
    tx_path = os.path.join(reg_dir, 'tx_summary.csv')
    df_tx_sum.to_csv(tx_path, index=False)

    # Save cleaned model dataset CSV
    clean_path = os.path.join(reg_dir, 'clean_dataset.csv')
    df_clean.to_csv(clean_path, index=False)

    # Save stats JSON
    stats_path = os.path.join(reg_dir, 'stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    # Save split dataset CSV if present
    if df_split is not None:
        split_path = os.path.join(reg_dir, 'split_dataset.csv')
        df_split.to_csv(split_path, index=False)


def save_region_split_data(
    data_dir: str,
    region_code: str,
    df_split: pd.DataFrame
) -> None:
    """Saves consolidated split dataset to disk."""
    reg_dir = get_region_dir(data_dir, region_code)
    split_path = os.path.join(reg_dir, 'split_dataset.csv')
    df_split.to_csv(split_path, index=False)


def save_region_transmitters(
    data_dir: str,
    region_code: str,
    df_tx: pd.DataFrame
) -> None:
    """Updates only the transmitter summary file and updates stats count."""
    reg_dir = get_region_dir(data_dir, region_code)
    tx_path = os.path.join(reg_dir, 'tx_summary.csv')
    df_tx.to_csv(tx_path, index=False)
    
    stats_path = os.path.join(reg_dir, 'stats.json')
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                s = json.load(f)
            s['n_transmitters'] = len(df_tx)
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(s, f, indent=2)
        except Exception:
            pass


def append_region_drive_test_data(
    data_dir: str,
    region_code: str,
    df_new_clean: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Appends new cleaned drive test records to the existing clean_dataset.csv.
    Performs spatial deduplication and updates stats.json.
    """
    reg_dir = get_region_dir(data_dir, region_code)
    clean_path = os.path.join(reg_dir, 'clean_dataset.csv')
    stats_path = os.path.join(reg_dir, 'stats.json')
    
    if os.path.exists(clean_path):
        df_old = pd.read_csv(clean_path)
        df_merged = pd.concat([df_old, df_new_clean], ignore_index=True)
    else:
        df_merged = df_new_clean.copy()

    # Deduplicate matching frequency and location within 10 meters / 4 decimals
    dedup_cols = [c for c in ['frequency_hz', 'Latitude', 'Longitude'] if c in df_merged.columns]
    if len(dedup_cols) == 3:
        df_merged = df_merged.drop_duplicates(subset=dedup_cols, keep='last').reset_index(drop=True)

    df_merged.to_csv(clean_path, index=False)

    stats = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        except Exception:
            pass
    stats['n_matched_model'] = len(df_merged)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    return df_merged, stats


def load_region_processed_data(
    data_dir: str,
    region_code: str
) -> Optional[Dict[str, Any]]:
    """
    Loads saved preprocessed data from disk for a given region.
    Returns dictionary with 'df_tx_summary', 'df_clean', 'stats', 'df_split' if found.
    """
    reg_dir = os.path.join(data_dir, 'processed', region_code)
    tx_path = os.path.join(reg_dir, 'tx_summary.csv')
    clean_path = os.path.join(reg_dir, 'clean_dataset.csv')
    stats_path = os.path.join(reg_dir, 'stats.json')
    split_path = os.path.join(reg_dir, 'split_dataset.csv')

    if not os.path.exists(tx_path):
        return None

    try:
        df_tx_sum = pd.read_csv(tx_path)
        df_clean = pd.read_csv(clean_path) if os.path.exists(clean_path) else None
        
        stats = {}
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)

        df_split = pd.read_csv(split_path) if os.path.exists(split_path) else None

        return {
            'df_tx_summary': df_tx_sum,
            'df_clean': df_clean,
            'stats': stats,
            'df_split': df_split
        }
    except Exception:
        return None


def delete_region_processed_data(data_dir: str, region_code: str) -> bool:
    """Removes stored processed data for a region."""
    reg_dir = os.path.join(data_dir, 'processed', region_code)
    if os.path.exists(reg_dir):
        try:
            shutil.rmtree(reg_dir)
            return True
        except Exception:
            return False
    return False


def delete_region_completely(data_dir: str, models_dir: str, region_code: str) -> bool:
    """
    Completely removes a region, including:
    1. data/processed/{region_code}/ (transmitters, clean data, split data, stats)
    2. models/{region_code}* (models and metrics)
    Works whether the model has been trained or not.
    """
    success = True
    reg_dir = os.path.join(data_dir, 'processed', region_code)
    if os.path.exists(reg_dir):
        try:
            shutil.rmtree(reg_dir)
        except Exception:
            success = False

    if os.path.exists(models_dir):
        for fname in os.listdir(models_dir):
            if fname.startswith(f"{region_code}_") or fname == f"{region_code}.json":
                try:
                    os.remove(os.path.join(models_dir, fname))
                except Exception:
                    success = False

    return success


def rename_region(data_dir: str, region_code: str, new_name: str) -> bool:
    """
    Renames the display title of a region non-destructively by updating
    'region_name' inside its stats.json file.
    Does not modify transmitter data, clean datasets, splits, or models.
    """
    reg_dir = os.path.join(data_dir, 'processed', region_code)
    os.makedirs(reg_dir, exist_ok=True)
    stats_path = os.path.join(reg_dir, 'stats.json')
    
    stats = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        except Exception:
            pass

    stats['region_name'] = str(new_name).strip()
    try:
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        return True
    except Exception:
        return False


def list_available_regions(data_dir: str, models_dir: str) -> List[Dict[str, Any]]:
    """
    Scans data/processed/ and models/ to discover all available saved regions.
    Returns list of dicts with region metadata sorted by importance.
    """
    proc_dir = os.path.join(data_dir, 'processed')
    codes = set()

    if os.path.exists(proc_dir):
        for name in os.listdir(proc_dir):
            if os.path.isdir(os.path.join(proc_dir, name)):
                codes.add(name)

    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith('_xgb_model.json'):
                code = f.replace('_xgb_model.json', '')
                codes.add(code)

    legacy_aliases = {'kalimantan__timur__1', 'kalimantan_timur__1_', 'kaltim1', 'kalimantan_timur___1'}
    regions = []
    for code in sorted(list(codes)):
        # Skip legacy duplicate aliases
        if code in legacy_aliases:
            continue
        # Skip temporary test prefixes if real region exists
        if code.startswith('test_') and len(codes) > 1:
            continue

        reg_p = os.path.join(proc_dir, code)
        tx_path = os.path.join(reg_p, 'tx_summary.csv')
        stats_path = os.path.join(reg_p, 'stats.json')
        model_path = os.path.join(models_dir, f"{code}_xgb_model.json")
        metrics_path = os.path.join(models_dir, f"{code}_metrics.json")

        # Derive human friendly title
        if 'balikpapan' in code.lower():
            title = "Balikpapan - Kaltim 1"
        elif code == 'kalimantan_timur_1':
            title = "Kalimantan Timur 1 (Samarinda)"
        elif code == 'kalimantan_timur___1':
            title = "Kalimantan Timur 1 (Balikpapan Sample)"
        elif 'jabar' in code.lower():
            title = "Jawa Barat 1 (Bandung)"
        else:
            clean_name = code.replace('___', ' - ').replace('__', ' ').replace('_', ' ')
            title = " ".join(w.capitalize() for w in clean_name.split())

        n_tx = 0
        if os.path.exists(tx_path):
            try:
                n_tx = len(pd.read_csv(tx_path))
            except Exception:
                pass

        n_pts = 0
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    s = json.load(f)
                    n_pts = s.get('n_matched_model', s.get('n_raw_drivetest', 0))
                    custom_name = s.get('region_name', s.get('title'))
                    if custom_name and str(custom_name).strip():
                        title = str(custom_name).strip()
            except Exception:
                pass

        has_model = os.path.exists(model_path)
        r2_val = None
        rmse_val = None
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    m = json.load(f)
                    r2_val = m.get('test_r2')
                    rmse_val = m.get('test_rmse')
            except Exception:
                pass

        if has_model:
            model_info = f"AI Siap (R² {r2_val:.2f})" if r2_val is not None else "AI Siap"
        else:
            model_info = "Model Belum Dilatih"

        label = f"{title} [{code}] — {n_tx} TX • {n_pts:,} Titik • {model_info}"

        regions.append({
            'code': code,
            'title': title,
            'n_tx': n_tx,
            'n_pts': n_pts,
            'has_model': has_model,
            'r2': r2_val,
            'rmse': rmse_val,
            'label': label
        })

    # Sort so regions with transmitters & models appear first
    regions.sort(key=lambda r: (r['has_model'], r['n_tx'], r['n_pts']), reverse=True)
    return regions
