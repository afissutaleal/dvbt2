"""
Feature Engineering & Data Preprocessing Module for DVB-T2 Propagation Prediction
Handles:
- Multi-file Drive Test ingestion & concatenation
- Frequency matching (MHz to Hz)
- Row deduplication & stationary point aggregation (local mean)
- Geodetic distance (Haversine, d != 0)
- Compass bearing & trigonometric transformation (sin, cos)
- Empirical EIRP (dBm) calculation per transmitter:
  EIRP (dBm) = E_max + 20 * log10(d_km) - 44.76
- Separation of 7 Model Features from Spatial Metadata
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

# 8 Official Features fed into the XGBoost Regressor (Including Terrain Clearance)
MODEL_FEATURE_COLS = [
    'frequency_hz',
    'eirp_dbm',
    'distance_km',
    'azimuth_sin',
    'azimuth_cos',
    'tx_antenna_amsl',
    'rx_antenna_amsl',
    'clearance_m'
]

TARGET_COL = 'field_strength_dbuv_m'

SPATIAL_METADATA_COLS = [
    'STN_NAME',
    'SID_LAT',
    'SID_LONG',
    'Latitude',
    'Longitude'
]

# Standard DVB-T2 UHF Band IV/V Reception Thresholds (dBµV/m)
DVBT2_THRESHOLDS = {
    'SANGAT_KUAT': 65.0,   # Merah (#ef4444, >= 65 dBµV/m)
    'KUAT': 54.0,          # Oranye (#f97316, 54 - 65 dBµV/m)
    'SEDANG': 45.2,        # Biru (#3b82f6, 45.2 - 54 dBµV/m)
    'BLANK_SPOT': 0.0      # Transparan (< 45.2 dBµV/m)
}


def load_dataframe_safely(file_obj) -> pd.DataFrame:
    """
    Safely reads CSV or Excel files with automatic encoding detection.
    Specifically handles byte 0xb5 ('µ' symbol in 'Level (dBµV/m)') by testing
    'latin1', 'cp1252', 'utf-8', 'iso-8859-1', and auto-detecting delimiters (, or ;).
    """
    fname = getattr(file_obj, 'name', '')
    if isinstance(fname, str) and fname.lower().endswith(('.xlsx', '.xls')):
        try:
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            return pd.read_excel(file_obj)
        except Exception:
            pass

    encodings = ['latin1', 'cp1252', 'utf-8', 'utf-8-sig', 'iso-8859-1']

    for enc in encodings:
        for sep in [',', ';', None]:
            try:
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                if sep is None:
                    df = pd.read_csv(file_obj, encoding=enc, sep=None, engine='python')
                else:
                    df = pd.read_csv(file_obj, encoding=enc, sep=sep)

                if df is not None and len(df.columns) > 1 and len(df) > 0:
                    return df
            except Exception:
                continue

    # Final fallback
    if hasattr(file_obj, 'seek'):
        file_obj.seek(0)
    return pd.read_csv(file_obj, encoding='latin1', on_bad_lines='skip')


def calculate_haversine_distance(
    lat1: np.ndarray, 
    lon1: np.ndarray, 
    lat2: np.ndarray, 
    lon2: np.ndarray
) -> np.ndarray:
    """
    Calculate geodetic distance in kilometers between pairs of coordinates
    using the Haversine formula (WGS-84 spherical approximation).
    Enforces d > 0 (replaces exact zero with 1e-4 km / 0.1m) to prevent log10(0) errors.
    """
    R = 6371.0088  # Earth's mean radius in km
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    
    d_km = R * c
    # Guarantee d != 0 strictly
    d_km = np.where(d_km <= 0.0, 0.0001, d_km)
    return d_km


def calculate_azimuth_bearing(
    lat1: np.ndarray, 
    lon1: np.ndarray, 
    lat2: np.ndarray, 
    lon2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate compass bearing (azimuth in degrees, 0 to 360) from TX (lat1, lon1) to RX (lat2, lon2),
    along with its trigonometric projections: sin(azimuth) and cos(azimuth).
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)
    
    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    
    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360.0) % 360.0
    
    azimuth_sin = np.sin(bearing_rad)
    azimuth_cos = np.cos(bearing_rad)
    
    return bearing_deg, azimuth_sin, azimuth_cos


def calculate_eirp_dbm(e_max_dbuv_m: float, distance_km: float) -> float:
    """
    Calculates Empirical EIRP in dBm from the maximum field strength (dBuV/m) and its distance (km):
    EIRP (dBm) = E_max (dBuV/m) + 20 * log10(d_km) - 44.76
    """
    d_safe = max(distance_km, 0.0001)
    eirp = e_max_dbuv_m + 20.0 * np.log10(d_safe) - 44.76
    return round(float(eirp), 2)


def calculate_radio_horizon(h_tx_amsl: float, h_rx_amsl: float = 0.0, k_factor: float = 4.0 / 3.0) -> float:
    """
    Calculates line-of-sight Radio Horizon distance in kilometers
    under standard atmospheric refraction (k=4/3, Re = 8495 km):
    d_h = 4.122 * (sqrt(max(h_tx, 0)) + sqrt(max(h_rx, 0))) km
    """
    h_tx_safe = max(float(h_tx_amsl), 0.0)
    h_rx_safe = max(float(h_rx_amsl), 0.0)
    c = np.sqrt(2.0 * (k_factor * 6371.0) / 1000.0) # ~4.122
    d_horizon = c * (np.sqrt(h_tx_safe) + np.sqrt(h_rx_safe))
    return round(float(d_horizon), 2)


def _to_1d_flat(val, dtype=None) -> np.ndarray:
    """
    Safely forces any Series, DataFrame (with duplicate column names),
    or numpy array into a strict 1-dimensional numpy array.
    Prevents ValueError: Buffer has wrong number of dimensions (expected 1, got 2).
    """
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    if isinstance(val, (pd.Series, pd.Index)):
        val = val.to_numpy()
    arr = np.asarray(val)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def process_and_align_datasets(
    df_transmitter: pd.DataFrame,
    df_drivetest_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    End-to-End Preprocessing Pipeline:
    1. Standardize column names without generating duplicate columns
    2. Convert TX FREQ (MHz) to Hz
    3. Drop exact duplicate rows in drive test
    4. Aggregate stationary/idle vehicle points (local mean by Frequency, Lat, Long)
    5. Filter drive test to matching transmitter frequencies
    6. Calculate Haversine distance and Azimuth (sin, cos)
    7. Compute Empirical EIRP (dBm) per transmitter from E_max and d_km
    8. Separate 7 Model Features from Spatial Metadata ensuring strict 1D vectors
    """
    df_tx = df_transmitter.copy()
    df_dt = df_drivetest_raw.copy()

    # Strip column whitespaces and remove duplicate column headers
    df_tx.columns = [str(c).strip() for c in df_tx.columns]
    df_dt.columns = [str(c).strip() for c in df_dt.columns]
    df_tx = df_tx.loc[:, ~df_tx.columns.duplicated(keep='first')].copy()
    df_dt = df_dt.loc[:, ~df_dt.columns.duplicated(keep='first')].copy()

    # 1. Clean & validate transmitter dataframe
    tx_rules = [
        ('STN_NAME', ['STN_NAME', 'STATION_NAME', 'STN', 'NAME']),
        ('FREQ_MHZ', ['FREQ', 'FREQUENCY', 'MHZ']),
        ('SID_LAT', ['SID_LAT', 'TX_LAT', 'LATITUDE', 'LAT']),
        ('SID_LONG', ['SID_LONG', 'SID_LON', 'TX_LON', 'LONGITUDE', 'LONG', 'LON']),
        ('H_ANT', ['H_ANT', 'HANT', 'MASL', 'AMSL', 'HEIGHT', 'TINGGI'])
    ]

    tx_col_map = {}
    for target_name, keywords in tx_rules:
        for col in df_tx.columns:
            if col in tx_col_map:
                continue
            col_upper = str(col).strip().upper()
            if any(k in col_upper for k in keywords):
                tx_col_map[col] = target_name
                break

    df_tx = df_tx.rename(columns=tx_col_map)
    df_tx = df_tx.loc[:, ~df_tx.columns.duplicated(keep='first')].copy()
    
    # Ensure numeric types
    for c in ['FREQ_MHZ', 'SID_LAT', 'SID_LONG', 'H_ANT']:
        if c in df_tx.columns:
            df_tx[c] = pd.to_numeric(df_tx[c], errors='coerce')
    df_tx = df_tx.dropna(subset=['FREQ_MHZ', 'SID_LAT', 'SID_LONG', 'H_ANT']).copy()
    
    # Convert TX MHz to Hz (round to nearest Hz)
    df_tx['FREQ_HZ'] = (df_tx['FREQ_MHZ'] * 1e6).round().astype(np.int64)

    # 2. Clean & validate drive test dataframe
    dt_rules = [
        ('Frequency_Hz', ['FREQ', 'HZ', 'FREQUENCY']),
        ('Level_dBuV_m', ['LEVEL', 'DBUV', 'FIELD', 'DB']),
        ('Longitude', ['LONGITUDE', 'LONG', 'LON']),
        ('Latitude', ['LATITUDE', 'LAT']),
        ('Altitude', ['ALTITUDE', 'ALT', 'ELEVATION', 'ELEV', 'AMSL', 'MASL'])
    ]

    dt_col_map = {}
    for target_name, keywords in dt_rules:
        for col in df_dt.columns:
            if col in dt_col_map:
                continue
            col_upper = str(col).strip().upper()
            if any(k in col_upper for k in keywords):
                dt_col_map[col] = target_name
                break

    df_dt = df_dt.rename(columns=dt_col_map)
    df_dt = df_dt.loc[:, ~df_dt.columns.duplicated(keep='first')].copy()

    for c in ['Frequency_Hz', 'Level_dBuV_m', 'Longitude', 'Latitude', 'Altitude']:
        if c in df_dt.columns:
            df_dt[c] = pd.to_numeric(df_dt[c], errors='coerce')

    df_dt = df_dt.dropna(subset=['Frequency_Hz', 'Level_dBuV_m', 'Longitude', 'Latitude']).copy()
    df_dt['Frequency_Hz'] = df_dt['Frequency_Hz'].round().astype(np.int64)
    if 'Altitude' not in df_dt.columns or df_dt['Altitude'].isna().all():
        df_dt['Altitude'] = 1.5

    n_raw_dt = len(df_dt)

    # 3. Deduplication: Drop exact duplicate rows
    df_dt = df_dt.drop_duplicates().copy()
    n_after_exact_dedup = len(df_dt)

    # 4. Stationary Points Aggregation: Local mean when vehicle is stopped at traffic light
    df_dt_subset = df_dt[['Frequency_Hz', 'Latitude', 'Longitude', 'Level_dBuV_m', 'Altitude']].copy()
    df_dt_agg = df_dt_subset.groupby(
        ['Frequency_Hz', 'Latitude', 'Longitude'], 
        as_index=False
    ).agg({
        'Level_dBuV_m': 'mean',
        'Altitude': 'mean'
    })
    n_after_stationary_agg = len(df_dt_agg)

    # 5. Frequency Matching: Filter drive test to keep only frequencies present in transmitter table
    tx_freqs = set(df_tx['FREQ_HZ'].unique())
    df_matched = df_dt_agg[df_dt_agg['Frequency_Hz'].isin(tx_freqs)].copy()
    n_matched = len(df_matched)

    if n_matched == 0:
        raise ValueError(
            f"Tidak ditemukan kecocokan frekuensi antara data pemancar ({len(tx_freqs)} frekuensi) "
            f"dan data drive test. Pastikan satuan frekuensi pemancar dalam MHz dan drive test dalam Hz."
        )

    # Merge matched drive test with transmitter specs by Frequency
    tx_merge_cols = ['STN_NAME', 'FREQ_MHZ', 'FREQ_HZ', 'SID_LAT', 'SID_LONG', 'H_ANT']
    df_tx_subset = df_tx[tx_merge_cols].loc[:, ~df_tx[tx_merge_cols].columns.duplicated(keep='first')].copy()

    df_merged = pd.merge(
        df_matched,
        df_tx_subset,
        left_on='Frequency_Hz',
        right_on='FREQ_HZ',
        how='inner'
    ).reset_index(drop=True)
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated(keep='first')].copy()

    # 6. Calculate Distance and Azimuth
    tx_lat = _to_1d_flat(df_merged['SID_LAT'], dtype=float)
    tx_lon = _to_1d_flat(df_merged['SID_LONG'], dtype=float)
    rx_lat = _to_1d_flat(df_merged['Latitude'], dtype=float)
    rx_lon = _to_1d_flat(df_merged['Longitude'], dtype=float)

    dists = calculate_haversine_distance(tx_lat, tx_lon, rx_lat, rx_lon)
    bearing_deg, az_sin, az_cos = calculate_azimuth_bearing(tx_lat, tx_lon, rx_lat, rx_lon)

    df_merged['distance_km'] = dists
    df_merged['azimuth_deg'] = bearing_deg
    df_merged['azimuth_sin'] = az_sin
    df_merged['azimuth_cos'] = az_cos

    # 7. Calculate Empirical EIRP (dBm) per Transmitter
    eirp_dict = {}
    tx_summary = []

    for freq_hz, grp in df_merged.groupby('Frequency_Hz'):
        idx_max = grp['Level_dBuV_m'].idxmax()
        row_max = grp.loc[idx_max]
        if isinstance(row_max, pd.DataFrame):
            row_max = row_max.iloc[0]

        e_max = float(row_max['Level_dBuV_m'])
        d_max = float(row_max['distance_km'])
        stn = str(row_max['STN_NAME'])
        f_mhz = float(row_max['FREQ_MHZ'])
        h_ant = float(row_max['H_ANT'])
        lat_tx = float(row_max['SID_LAT'])
        lon_tx = float(row_max['SID_LONG'])

        eirp_val = calculate_eirp_dbm(e_max, d_max)
        eirp_dict[freq_hz] = eirp_val

        tx_summary.append({
            'STN_NAME': stn,
            'FREQ_MHZ': f_mhz,
            'FREQ_HZ': freq_hz,
            'SID_LAT': lat_tx,
            'SID_LONG': lon_tx,
            'H_ANT_masl': h_ant,
            'Horizon_TX_km': calculate_radio_horizon(h_ant, 0.0),
            'E_max_dBuV_m': round(e_max, 2),
            'd_at_Emax_km': round(d_max, 3),
            'EIRP_dBm': eirp_val,
            'n_measurements': len(grp)
        })

    df_tx_summary = pd.DataFrame(tx_summary)

    # Assign EIRP (dBm) to each row
    df_merged['eirp_dbm'] = df_merged['Frequency_Hz'].map(eirp_dict)

    # 8. Calculate Terrain Clearance (clearance_m = h_los - h_max)
    from src.terrain_engine import compute_dataset_clearance
    df_merged['clearance_m'] = compute_dataset_clearance(
        df=df_merged,
        tx_lat_col='SID_LAT',
        tx_lon_col='SID_LONG',
        tx_amsl_col='H_ANT',
        rx_lat_col='Latitude',
        rx_lon_col='Longitude',
        rx_amsl_col='Altitude'
    )

    # 9. Standardize final model dataset using direct column assignment
    n_rows = len(df_merged)
    df_final = pd.DataFrame(index=np.arange(n_rows))
    df_final['frequency_hz'] = _to_1d_flat(df_merged['Frequency_Hz'], dtype=np.int64)
    df_final['frequency_mhz'] = _to_1d_flat(df_merged['FREQ_MHZ'], dtype=float)
    df_final['eirp_dbm'] = _to_1d_flat(df_merged['eirp_dbm'], dtype=float)
    df_final['distance_km'] = _to_1d_flat(df_merged['distance_km'], dtype=float)
    df_final['azimuth_sin'] = _to_1d_flat(df_merged['azimuth_sin'], dtype=float)
    df_final['azimuth_cos'] = _to_1d_flat(df_merged['azimuth_cos'], dtype=float)
    df_final['tx_antenna_amsl'] = _to_1d_flat(df_merged['H_ANT'], dtype=float)
    df_final['rx_antenna_amsl'] = _to_1d_flat(df_merged['Altitude'], dtype=float)
    df_final['clearance_m'] = _to_1d_flat(df_merged['clearance_m'], dtype=float)
    df_final['field_strength_dbuv_m'] = _to_1d_flat(df_merged['Level_dBuV_m'], dtype=float)
    
    # Metadata
    df_final['STN_NAME'] = _to_1d_flat(df_merged['STN_NAME'], dtype=object)
    df_final['SID_LAT'] = _to_1d_flat(df_merged['SID_LAT'], dtype=float)
    df_final['SID_LONG'] = _to_1d_flat(df_merged['SID_LONG'], dtype=float)
    df_final['Latitude'] = _to_1d_flat(df_merged['Latitude'], dtype=float)
    df_final['Longitude'] = _to_1d_flat(df_merged['Longitude'], dtype=float)
    df_final['azimuth_deg'] = _to_1d_flat(df_merged['azimuth_deg'], dtype=float)

    stats = {
        'n_raw_drivetest': n_raw_dt,
        'n_after_exact_dedup': n_after_exact_dedup,
        'n_after_stationary_agg': n_after_stationary_agg,
        'n_matched_model': len(df_final),
        'n_transmitters': len(df_tx_summary)
    }

    return df_final, df_tx_summary, stats

    stats = {
        'n_raw_drivetest': n_raw_dt,
        'n_after_exact_dedup': n_after_exact_dedup,
        'n_after_stationary_agg': n_after_stationary_agg,
        'n_matched_model': len(df_final),
        'n_transmitters': len(df_tx_summary)
    }

    return df_final, df_tx_summary, stats
