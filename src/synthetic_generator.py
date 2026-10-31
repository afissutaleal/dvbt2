"""
Synthetic Realistic DVB-T2 Drive Test & Site Database Generator
Generates physically grounded drive test datasets for multiple Indonesian service areas
(Kalimantan Timur 1 and Jawa Barat 1) with terrain variation, shadow fading,
and realistic antenna radiation patterns.
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, List, Dict
from src.features import (
    calculate_haversine_distance,
    calculate_azimuth_bearing,
    compute_derived_features
)
from src.propagation_models import okumura_hata_field_strength

# Seed for reproducibility
np.random.seed(42)

# Predefined Broadcast Service Areas in Indonesia
PREDEFINED_REGIONS = {
    'Kalimantan Timur 1': {
        'code': 'kaltim1',
        'name': 'Wilayah Layanan Kalimantan Timur 1 (Samarinda - Balikpapan - IKN)',
        'center_lat': -0.85,
        'center_lon': 117.00,
        'bbox': {
            'min_lat': -1.40,
            'max_lat': -0.35,
            'min_lon': 116.70,
            'max_lon': 117.35
        },
        'sites': [
            {
                'site_id': 'TX_K1_01',
                'site_name': 'TVRI Samarinda (Gunung Selili)',
                'tx_latitude': -0.5089,
                'tx_longitude': 117.1593,
                'tx_antenna_amsl': 165.0,  # AMSL in meters
                'frequency_mhz': 546.0,    # UHF Ch 30
                'erp': 20.0,               # kW
                'azimuth_main_lobe': 225.0 # Direction towards Balikpapan/Kukar
            },
            {
                'site_id': 'TX_K1_02',
                'site_name': 'TVRI Balikpapan (Gunung Polisi)',
                'tx_latitude': -1.2534,
                'tx_longitude': 116.8372,
                'tx_antenna_amsl': 142.0,
                'frequency_mhz': 546.0,    # SFN / Multi-MUX
                'erp': 20.0,
                'azimuth_main_lobe': 45.0
            },
            {
                'site_id': 'TX_K1_03',
                'site_name': 'MNC Media Samarinda (Sempaja)',
                'tx_latitude': -0.4650,
                'tx_longitude': 117.1420,
                'tx_antenna_amsl': 130.0,
                'frequency_mhz': 586.0,    # UHF Ch 35
                'erp': 15.0,
                'azimuth_main_lobe': 180.0
            }
        ]
    },
    'Jawa Barat 1': {
        'code': 'jabar1',
        'name': 'Wilayah Layanan Jawa Barat 1 (Bandung Raya - Cimahi)',
        'center_lat': -6.9147,
        'center_lon': 107.6098,
        'bbox': {
            'min_lat': -7.10,
            'max_lat': -6.75,
            'min_lon': 107.45,
            'max_lon': 107.80
        },
        'sites': [
            {
                'site_id': 'TX_JB_01',
                'site_name': 'TVRI Gandasoli (Bukit Panyandaan)',
                'tx_latitude': -6.8672,
                'tx_longitude': 107.6741,
                'tx_antenna_amsl': 1150.0, # High elevation mast
                'frequency_mhz': 586.0,
                'erp': 25.0,
                'azimuth_main_lobe': 230.0
            },
            {
                'site_id': 'TX_JB_02',
                'site_name': 'Trans Media Bandung (Cimenyan)',
                'tx_latitude': -6.8720,
                'tx_longitude': 107.6550,
                'tx_antenna_amsl': 1080.0,
                'frequency_mhz': 634.0,    # UHF Ch 41
                'erp': 20.0,
                'azimuth_main_lobe': 210.0
            }
        ]
    }
}


def _simulate_terrain_elevation(lat: np.ndarray, lon: np.ndarray, region_key: str) -> np.ndarray:
    """
    Simulates realistic terrain altitude (AMSL in meters) using synthetic undulating surfaces.
    """
    if 'kaltim' in region_key.lower():
        # Rolling hills between 15m and 180m AMSL
        elev = 35.0 + 40.0 * np.sin(lat * 35.0) * np.cos(lon * 25.0) + \
               25.0 * np.sin((lat + 0.8) * 15.0) + 15.0 * np.random.uniform(-1, 1, size=len(lat))
        return np.clip(elev, 10.0, 220.0)
    else:
        # Bandung basin surrounded by mountains (650m to 1400m)
        center_lat = -6.9147
        center_lon = 107.6098
        dist_from_basin = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2) * 111.0
        elev = 670.0 + 35.0 * (dist_from_basin ** 1.3) + 20.0 * np.sin(lat * 50.0)
        return np.clip(elev, 650.0, 1450.0)


def generate_region_dataset(
    region_key: str,
    n_drive_samples: int = 3500
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic Drive Test measurement data and Site Database for a selected region.
    Returns: (df_drivetest, df_sites)
    """
    region_info = PREDEFINED_REGIONS.get(region_key, PREDEFINED_REGIONS['Kalimantan Timur 1'])
    sites_list = region_info['sites']
    df_sites = pd.DataFrame(sites_list)
    
    # Generate drive test route points across highways & arterial roads
    bbox = region_info['bbox']
    lat_min, lat_max = bbox['min_lat'], bbox['max_lat']
    lon_min, lon_max = bbox['min_lon'], bbox['max_lon']
    
    # Simulate realistic road networks (clustered trajectories)
    points_per_cluster = 350
    n_clusters = max(1, n_drive_samples // points_per_cluster)
    
    rx_lat_list = []
    rx_lon_list = []
    
    for _ in range(n_clusters):
        start_lat = np.random.uniform(lat_min + 0.05, lat_max - 0.05)
        start_lon = np.random.uniform(lon_min + 0.05, lon_max - 0.05)
        
        # Random walk trajectory representing drive test vehicle
        step_lat = np.random.normal(0, 0.0015, points_per_cluster).cumsum()
        step_lon = np.random.normal(0, 0.0018, points_per_cluster).cumsum()
        
        c_lat = np.clip(start_lat + step_lat, lat_min, lat_max)
        c_lon = np.clip(start_lon + step_lon, lon_min, lon_max)
        
        rx_lat_list.extend(c_lat)
        rx_lon_list.extend(c_lon)
        
    # Trim to exact requested count
    rx_lats = np.array(rx_lat_list[:n_drive_samples])
    rx_lons = np.array(rx_lon_list[:n_drive_samples])
    
    # Ground receiver height: typically 1.5m to 2.5m vehicle roof mast + terrain elevation
    rx_terrain_amsl = _simulate_terrain_elevation(rx_lats, rx_lons, region_key)
    rx_antenna_amsl = rx_terrain_amsl + 1.5
    
    # Assign measurements to nearest/dominant serving transmitter
    records = []
    for i in range(len(rx_lats)):
        r_lat = rx_lats[i]
        r_lon = rx_lons[i]
        r_amsl = rx_antenna_amsl[i]
        
        # Find strongest candidate site at this point
        best_fs = -999.0
        best_site = sites_list[0]
        
        for site in sites_list:
            t_lat = site['tx_latitude']
            t_lon = site['tx_longitude']
            t_amsl = site['tx_antenna_amsl']
            dist_km = calculate_haversine_distance(np.array([t_lat]), np.array([t_lon]),
                                                   np.array([r_lat]), np.array([r_lon]))[0]
            
            # Okumura-Hata baseline
            base_fs = okumura_hata_field_strength(
                frequency_mhz=site['frequency_mhz'],
                tx_height_m=max(30.0, t_amsl - r_amsl + 50.0),
                rx_height_m=1.5,
                distance_km=dist_km,
                erp_kw=site['erp'],
                environment='suburban'
            )
            
            # Antenna directional pattern effect
            az_deg = calculate_azimuth_bearing(np.array([t_lat]), np.array([t_lon]),
                                               np.array([r_lat]), np.array([r_lon]))[0]
            az_diff = np.abs(az_deg - site.get('azimuth_main_lobe', 0.0))
            az_diff = min(az_diff, 360.0 - az_diff)
            pattern_loss = 6.0 * (az_diff / 180.0)**2  # Max ~6dB front-to-back ratio
            
            cand_fs = base_fs - pattern_loss
            if cand_fs > best_fs:
                best_fs = cand_fs
                best_site = site
                
        # Shadow fading (Log-normal shadowing, standard deviation ~ 3.5 dB)
        shadow_fading = np.random.normal(0, 3.2)
        measured_fs = best_fs + shadow_fading
        
        # Sensor measurement limits (e.g., Rohde & Schwarz / Promax DVB-T2 meter)
        measured_fs = np.clip(measured_fs, 15.0, 115.0)
        
        records.append({
            'site_id': best_site['site_id'],
            'site_name': best_site['site_name'],
            'tx_latitude': best_site['tx_latitude'],
            'tx_longitude': best_site['tx_longitude'],
            'tx_antenna_amsl': best_site['tx_antenna_amsl'],
            'frequency_mhz': best_site['frequency_mhz'],
            'erp': best_site['erp'],
            'rx_latitude': round(float(r_lat), 6),
            'rx_longitude': round(float(r_lon), 6),
            'rx_antenna_amsl': round(float(r_amsl), 2),
            'field_strength_dbuv_m': round(float(measured_fs), 2)
        })
        
    df_drivetest = pd.DataFrame(records)
    # Compute derived physical features (distance_km, azimuth_sin, azimuth_cos)
    df_drivetest = compute_derived_features(df_drivetest)
    
    return df_drivetest, df_sites


def populate_default_datasets(base_dir: str, overwrite: bool = False):
    """
    Generates and saves default CSV files for all predefined regions.
    """
    sites_dir = os.path.join(base_dir, 'data', 'sites')
    dt_dir = os.path.join(base_dir, 'data', 'drivetest')
    os.makedirs(sites_dir, exist_ok=True)
    os.makedirs(dt_dir, exist_ok=True)
    
    for r_name, r_info in PREDEFINED_REGIONS.items():
        code = r_info['code']
        sites_path = os.path.join(sites_dir, f"{code}_sites.csv")
        dt_path = os.path.join(dt_dir, f"{code}_drivetest.csv")
        
        if overwrite or not os.path.exists(sites_path) or not os.path.exists(dt_path):
            print(f"Generating realistic dataset for {r_name} ({code})...")
            df_dt, df_sites = generate_region_dataset(r_name, n_drive_samples=3200)
            df_sites.to_csv(sites_path, index=False)
            df_dt.to_csv(dt_path, index=False)
            print(f"Saved: {sites_path} and {dt_path}")
            
    print("Dataset initialization complete.")
