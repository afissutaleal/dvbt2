"""
Multi-Feature Stratified 3-Way Split Module (70% Train : 15% Val : 15% Test)
Implements:
- Continuous variable binning (Distance q=4, RX Altitude q=3, Azimuth Quadrants bins=4, Clearance LoS/NLoS)
- Composite Stratification Key (Frequency + Distance + RX Altitude + Azimuth + Clearance)
- Safeguard 1: Handling rare categories (< 3 samples) in initial split
- Safeguard 2: Iterative handling of rare categories (< 2 samples) in secondary temp split
- Statistical sanity check table with Clearance metrics (Mean, Median, % LoS, % NLoS)
- 8-panel comprehensive diagnostic distribution plots (Frekuensi, Jarak, RX AMSL, Azimuth, Clearance, LoS/NLoS %, Target Kuat Medan, Proporsi Rasio Dataset)
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from src.features import MODEL_FEATURE_COLS, TARGET_COL


def compute_summary_check_table(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes a comprehensive sanity check comparison table across Train, Validation, and Test datasets.
    Includes geodetic distance, RX AMSL altitude, broadcast frequency, and path clearance metrics.
    """
    freq_div = 1e6 if 'frequency_hz' in X_train.columns and X_train['frequency_hz'].mean() > 1e4 else 1.0

    cols = ['Train (70%)', 'Validation (15%)', 'Test (15%)']
    rows = {}

    # Distance
    if 'distance_km' in X_train.columns:
        rows['Jarak Geodetik: Rata-rata (km)'] = [
            X_train['distance_km'].mean(),
            X_val['distance_km'].mean(),
            X_test['distance_km'].mean()
        ]

    # Azimuth Metrics (Degrees & Trigonometric components)
    def _calc_az_deg(df):
        if 'azimuth_deg' in df.columns:
            return df['azimuth_deg']
        elif 'azimuth_sin' in df.columns and 'azimuth_cos' in df.columns:
            return (np.degrees(np.arctan2(df['azimuth_sin'], df['azimuth_cos'])) + 360.0) % 360.0
        return None

    az_tr = _calc_az_deg(X_train)
    az_va = _calc_az_deg(X_val)
    az_te = _calc_az_deg(X_test)

    if az_tr is not None and az_va is not None and az_te is not None:
        rows['Sudut Azimuth: Rata-rata (°)'] = [
            az_tr.mean(),
            az_va.mean(),
            az_te.mean()
        ]

    if 'azimuth_sin' in X_train.columns:
        rows['Azimuth Sin: Rata-rata'] = [
            X_train['azimuth_sin'].mean(),
            X_val['azimuth_sin'].mean(),
            X_test['azimuth_sin'].mean()
        ]

    if 'azimuth_cos' in X_train.columns:
        rows['Azimuth Cos: Rata-rata'] = [
            X_train['azimuth_cos'].mean(),
            X_val['azimuth_cos'].mean(),
            X_test['azimuth_cos'].mean()
        ]

    # RX Altitude
    if 'rx_antenna_amsl' in X_train.columns:
        rows['Tinggi Antena RX AMSL: Rata-rata (m)'] = [
            X_train['rx_antenna_amsl'].mean(),
            X_val['rx_antenna_amsl'].mean(),
            X_test['rx_antenna_amsl'].mean()
        ]

    # TX Altitude
    if 'tx_antenna_amsl' in X_train.columns:
        rows['Tinggi Antena TX AMSL: Rata-rata (m)'] = [
            X_train['tx_antenna_amsl'].mean(),
            X_val['tx_antenna_amsl'].mean(),
            X_test['tx_antenna_amsl'].mean()
        ]

    # EIRP
    if 'eirp_dbm' in X_train.columns:
        rows['EIRP: Rata-rata (dBm)'] = [
            X_train['eirp_dbm'].mean(),
            X_val['eirp_dbm'].mean(),
            X_test['eirp_dbm'].mean()
        ]

    # Frequency
    freq_col = 'frequency_hz' if 'frequency_hz' in X_train.columns else ('frequency_mhz' if 'frequency_mhz' in X_train.columns else None)
    if freq_col:
        rows['Frekuensi Siaran: Rata-rata (MHz)'] = [
            (X_train[freq_col] / freq_div).mean(),
            (X_val[freq_col] / freq_div).mean(),
            (X_test[freq_col] / freq_div).mean()
        ]

    # Clearance Metrics
    if 'clearance_m' in X_train.columns:
        rows['Clearance Medan: Rata-rata (m)'] = [
            X_train['clearance_m'].mean(),
            X_val['clearance_m'].mean(),
            X_test['clearance_m'].mean()
        ]
        rows['Clearance Medan: Median (m)'] = [
            X_train['clearance_m'].median(),
            X_val['clearance_m'].median(),
            X_test['clearance_m'].median()
        ]
        rows['Proporsi Bebas Halangan (LoS >= 0m) (%)'] = [
            (X_train['clearance_m'] >= 0).mean() * 100.0,
            (X_val['clearance_m'] >= 0).mean() * 100.0,
            (X_test['clearance_m'] >= 0).mean() * 100.0
        ]
        rows['Proporsi Terhalang Kontur (NLoS < 0m) (%)'] = [
            (X_train['clearance_m'] < 0).mean() * 100.0,
            (X_val['clearance_m'] < 0).mean() * 100.0,
            (X_test['clearance_m'] < 0).mean() * 100.0
        ]

    df_summary = pd.DataFrame(rows, index=cols).T
    return df_summary.round(3)


def generate_split_distribution_figure(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> plt.Figure:
    """
    Generates an 8-panel diagnostic figure (2 rows x 4 columns) verifying that features,
    target, and clearance proportions are evenly distributed across Train, Val, and Test sets.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), dpi=100)
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['Train (70%)', 'Validation (15%)', 'Test (15%)']
    palette_colors = {'Train (70%)': '#1f77b4', 'Validation (15%)': '#ff7f0e', 'Test (15%)': '#2ca02c'}

    # Prepare datasets for plotting
    train_plot = X_train.copy()
    train_plot['Dataset'] = 'Train (70%)'
    freq_div = 1e6 if 'frequency_hz' in train_plot.columns and train_plot['frequency_hz'].mean() > 1e4 else 1.0
    freq_src = 'frequency_hz' if 'frequency_hz' in train_plot.columns else 'frequency_mhz'
    train_plot['frequency_mhz_plot'] = train_plot[freq_src] / freq_div
    train_plot['target_dbuv'] = y_train.values

    val_plot = X_val.copy()
    val_plot['Dataset'] = 'Validation (15%)'
    val_plot['frequency_mhz_plot'] = val_plot[freq_src] / freq_div
    val_plot['target_dbuv'] = y_val.values

    test_plot = X_test.copy()
    test_plot['Dataset'] = 'Test (15%)'
    test_plot['frequency_mhz_plot'] = test_plot[freq_src] / freq_div
    test_plot['target_dbuv'] = y_test.values

    for df_p in [train_plot, val_plot, test_plot]:
        if 'azimuth_deg' not in df_p.columns:
            if 'azimuth_sin' in df_p.columns and 'azimuth_cos' in df_p.columns:
                df_p['azimuth_deg'] = (
                    (np.degrees(np.arctan2(df_p['azimuth_sin'], df_p['azimuth_cos'])) + 360.0) % 360.0
                )
            else:
                df_p['azimuth_deg'] = 0.0

    combined_plot = pd.concat([train_plot, val_plot, test_plot], axis=0)

    # -------------------------------------------------------------
    # Panel 1: Distribusi Frekuensi (MHz)
    # -------------------------------------------------------------
    f_tr = train_plot['frequency_mhz_plot']
    f_va = val_plot['frequency_mhz_plot']
    f_te = test_plot['frequency_mhz_plot']
    axes[0, 0].hist([f_tr, f_va, f_te], bins=10, label=labels, color=color_list, alpha=0.85)
    axes[0, 0].set_title('1. Distribusi Frekuensi (MHz)', fontsize=10, fontweight='bold')
    axes[0, 0].set_xlabel('Frekuensi (MHz)', fontsize=9)
    axes[0, 0].legend(fontsize=8)

    # -------------------------------------------------------------
    # Panel 2: Kepadatan Jarak Geodetik (km)
    # -------------------------------------------------------------
    d_tr = train_plot['distance_km']
    d_va = val_plot['distance_km']
    d_te = test_plot['distance_km']
    axes[0, 1].hist(d_tr, bins=25, density=True, alpha=0.4, color=color_list[0], label='Train (70%)')
    axes[0, 1].hist(d_va, bins=25, density=True, alpha=0.4, color=color_list[1], label='Validation (15%)')
    axes[0, 1].hist(d_te, bins=25, density=True, alpha=0.4, color=color_list[2], label='Test (15%)')
    axes[0, 1].set_title('2. Kepadatan Jarak Geodetik (km)', fontsize=10, fontweight='bold')
    axes[0, 1].set_xlabel('Jarak Geodetik (km)', fontsize=9)
    axes[0, 1].legend(fontsize=8)

    # -------------------------------------------------------------
    # Panel 3: Sebaran Ketinggian Penerima (RX AMSL)
    # -------------------------------------------------------------
    axes[0, 2].boxplot(
        [train_plot['rx_antenna_amsl'], val_plot['rx_antenna_amsl'], test_plot['rx_antenna_amsl']],
        tick_labels=labels
    )
    axes[0, 2].set_title('3. Sebaran Ketinggian RX (m AMSL)', fontsize=10, fontweight='bold')
    axes[0, 2].set_ylabel('Altitude / AMSL (meter)', fontsize=9)

    # -------------------------------------------------------------
    # Panel 4: Arah Azimuth Antena (0°-360°)
    # -------------------------------------------------------------
    az_tr = train_plot['azimuth_deg']
    az_va = val_plot['azimuth_deg']
    az_te = test_plot['azimuth_deg']
    axes[0, 3].hist(az_tr, bins=20, density=True, alpha=0.4, color=color_list[0], label='Train (70%)')
    axes[0, 3].hist(az_va, bins=20, density=True, alpha=0.4, color=color_list[1], label='Validation (15%)')
    axes[0, 3].hist(az_te, bins=20, density=True, alpha=0.4, color=color_list[2], label='Test (15%)')
    axes[0, 3].set_title('4. Arah Azimuth Antena (0°-360°)', fontsize=10, fontweight='bold')
    axes[0, 3].set_xlabel('Sudut Arah Bearing (Derajat)', fontsize=9)
    axes[0, 3].legend(fontsize=8)

    # -------------------------------------------------------------
    # Panel 5: Kepadatan Clearance Lintasan Medan (meter)
    # -------------------------------------------------------------
    if 'clearance_m' in train_plot.columns:
        c_tr = train_plot['clearance_m']
        c_va = val_plot['clearance_m']
        c_te = test_plot['clearance_m']
        axes[1, 0].hist(c_tr, bins=25, density=True, alpha=0.4, color=color_list[0], label='Train (70%)')
        axes[1, 0].hist(c_va, bins=25, density=True, alpha=0.4, color=color_list[1], label='Validation (15%)')
        axes[1, 0].hist(c_te, bins=25, density=True, alpha=0.4, color=color_list[2], label='Test (15%)')
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Batas NLoS (0 m)')
        axes[1, 0].set_title('5. Kepadatan Clearance Medan (m)', fontsize=10, fontweight='bold')
        axes[1, 0].set_xlabel('Clearance (meter)', fontsize=9)
        axes[1, 0].legend(fontsize=8)
    else:
        axes[1, 0].text(0.5, 0.5, 'Fitur Clearance Tidak Tersedia', ha='center', va='center', fontsize=11)
        axes[1, 0].set_title('5. Kepadatan Clearance Medan (m)', fontsize=10, fontweight='bold')

    # -------------------------------------------------------------
    # Panel 6: Proporsi Status Propagasi LoS vs NLoS (%)
    # -------------------------------------------------------------
    if 'clearance_m' in train_plot.columns:
        los_nlos_data = []
        for name, df_sub in [('Train (70%)', train_plot), ('Validation (15%)', val_plot), ('Test (15%)', test_plot)]:
            n_los = (df_sub['clearance_m'] >= 0).sum()
            n_nlos = (df_sub['clearance_m'] < 0).sum()
            total = len(df_sub)
            los_nlos_data.append({
                'Dataset': name,
                'LoS (%)': (n_los / total) * 100.0 if total > 0 else 0.0,
                'NLoS (%)': (n_nlos / total) * 100.0 if total > 0 else 0.0
            })
        df_prop = pd.DataFrame(los_nlos_data)

        x_pos = np.arange(len(labels))
        width = 0.35
        rects1 = axes[1, 1].bar(x_pos - width/2, df_prop['LoS (%)'], width, label='LoS (>= 0m)', color='#2ca02c', alpha=0.85)
        rects2 = axes[1, 1].bar(x_pos + width/2, df_prop['NLoS (%)'], width, label='NLoS (< 0m)', color='#d62728', alpha=0.85)
        axes[1, 1].set_title('6. Proporsi LoS vs Terhalang NLoS (%)', fontsize=10, fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(labels, fontsize=8)
        axes[1, 1].set_ylabel('Persentase (%)', fontsize=9)
        axes[1, 1].set_ylim(0, 115)
        axes[1, 1].legend(loc='upper right', fontsize=8)

        for rect in rects1:
            h = rect.get_height()
            axes[1, 1].annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                                xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
        for rect in rects2:
            h = rect.get_height()
            axes[1, 1].annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                                xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Fitur Clearance Tidak Tersedia', ha='center', va='center', fontsize=11)
        axes[1, 1].set_title('6. Proporsi LoS vs Terhalang NLoS (%)', fontsize=10, fontweight='bold')

    # -------------------------------------------------------------
    # Panel 7: Distribusi Kuat Medan Target (dBµV/m)
    # -------------------------------------------------------------
    y_tr = train_plot['target_dbuv']
    y_va = val_plot['target_dbuv']
    y_te = test_plot['target_dbuv']
    axes[1, 2].hist(y_tr, bins=20, density=True, alpha=0.4, color=color_list[0], label='Train (70%)')
    axes[1, 2].hist(y_va, bins=20, density=True, alpha=0.4, color=color_list[1], label='Validation (15%)')
    axes[1, 2].hist(y_te, bins=20, density=True, alpha=0.4, color=color_list[2], label='Test (15%)')
    axes[1, 2].set_title('7. Distribusi Target Kuat Medan (dBµV/m)', fontsize=10, fontweight='bold')
    axes[1, 2].set_xlabel('Kuat Medan (dBµV/m)', fontsize=9)
    axes[1, 2].legend(fontsize=8)

    # -------------------------------------------------------------
    # Panel 8: Rasio Ukuran Dataset Bar Chart
    # -------------------------------------------------------------
    counts = [len(train_plot), len(val_plot), len(test_plot)]
    bars = axes[1, 3].bar(labels, counts, color=color_list, alpha=0.85)
    for b, c in zip(bars, counts):
        pct = (c / sum(counts)) * 100.0 if sum(counts) > 0 else 0.0
        axes[1, 3].text(b.get_x() + b.get_width()/2., b.get_height() + 15, f"{c:,}\n({pct:.1f}%)", 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[1, 3].set_title('8. Rasio Ukuran Dataset (70 : 15 : 15)', fontsize=10, fontweight='bold')
    axes[1, 3].set_ylabel('Jumlah Sampel (Baris)', fontsize=9)
    axes[1, 3].set_ylim(0, max(counts) * 1.25)

    plt.tight_layout()
    return fig


def perform_multi_feature_stratified_split(
    df_model: pd.DataFrame,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, plt.Figure]:
    """
    Splits dataset into 70% Train, 15% Validation, and 15% Test with proportional
    representation across Frequency, Distance, RX Altitude, Azimuth, and Clearance.
    """
    df_work = df_model.copy()
    
    # 1. Continuous Binning for Distance, RX Antenna AMSL, Azimuth, and Clearance
    df_work['dist_bin'] = pd.qcut(df_work['distance_km'], q=4, labels=False)
    df_work['rx_bin'] = pd.qcut(df_work['rx_antenna_amsl'], q=3, duplicates='drop', labels=False)
    
    # Convert azimuth sin/cos back to degrees (0-360) and divide into 4 quadrants
    if 'azimuth_sin' in df_work.columns and 'azimuth_cos' in df_work.columns:
        azimuth_deg = np.degrees(np.arctan2(df_work['azimuth_sin'], df_work['azimuth_cos'])) % 360.0
    else:
        azimuth_deg = np.zeros(len(df_work))
    df_work['azimuth_bin'] = pd.cut(azimuth_deg, bins=4, labels=False)

    # Clearance Binning (LoS >= 0 vs NLoS < 0)
    if 'clearance_m' in df_work.columns:
        df_work['clearance_bin'] = np.where(df_work['clearance_m'] >= 0, 'LoS', 'NLoS')
    else:
        df_work['clearance_bin'] = 'LoS'
    
    # 2. Composite Stratification Key
    freq_col = 'frequency_mhz' if 'frequency_mhz' in df_work.columns else 'frequency_hz'
    df_work['strat_key'] = (
        df_work[freq_col].astype(str) + '_' +
        df_work['dist_bin'].astype(str) + '_' +
        df_work['rx_bin'].astype(str) + '_' +
        df_work['azimuth_bin'].astype(str) + '_' +
        df_work['clearance_bin'].astype(str)
    )
    
    # Safeguard 1: Group rare keys (< 3 samples) into 'rare_group' to prevent train_test_split error
    key_counts = df_work['strat_key'].value_counts()
    rare_keys = key_counts[key_counts < 3].index
    df_work.loc[df_work['strat_key'].isin(rare_keys), 'strat_key'] = 'rare_group'
    
    # 3. Separate Feature Matrix X and Target y
    cols_to_use = [c for c in MODEL_FEATURE_COLS if c in df_work.columns]
    X = df_work[cols_to_use].copy()
    y = df_work[TARGET_COL].copy()
    strat_labels = df_work['strat_key'].copy()
    
    # 4. Step 1: Split Train (70%) and Temp (30%)
    X_train, X_temp, y_train, y_temp, strat_train, strat_temp = train_test_split(
        X, y, strat_labels,
        test_size=0.30,
        stratify=strat_labels,
        random_state=random_state
    )
    
    # Safeguard 2: Iterative check on strat_temp for categories with < 2 samples
    strat_temp = strat_temp.copy()
    while True:
        temp_counts = strat_temp.value_counts()
        rare_temp_keys = temp_counts[temp_counts < 2].index
        if len(rare_temp_keys) == 0:
            break
        strat_temp.loc[strat_temp.isin(rare_temp_keys)] = 'rare_group'
        
    # 5. Step 2: Split Temp (30%) into Validation (15%) and Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=strat_temp,
        random_state=random_state
    )
    
    # 6. Comprehensive Summary Sanity Check Table
    summary_check = compute_summary_check_table(X_train, X_val, X_test)
    
    # 7. Comprehensive 8-Panel Diagnostic Distribution Figure
    fig = generate_split_distribution_figure(X_train, y_train, X_val, y_val, X_test, y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, summary_check, fig
