"""
Generate Sample Datasets mirroring user's exact screenshot formats:
1. data/sample_data_transmitter.csv
2. data/sample_DriveT_BPP1.csv
3. data/sample_DriveT_BPP2.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# 1. Transmitter Data (Exact from user screenshot)
tx_data = [
    {"STN_NAME": "TRANS TV BALIKPAPAN", "FREQ (MHz)": 495.25, "SID_LONG": 116.8388889, "SID_LAT": -1.24722222, "H_ANT (masl)": 120},
    {"STN_NAME": "BALIKPAPAN TELEVISI", "FREQ (MHz)": 511.25, "SID_LONG": 116.8516667, "SID_LAT": -1.21861111, "H_ANT (masl)": 98},
    {"STN_NAME": "4687/9529871/5/1/PT.IVM BALIKPAPAN", "FREQ (MHz)": 527.25, "SID_LONG": 116.8480556, "SID_LAT": -1.25000000, "H_ANT (masl)": 125},
    {"STN_NAME": "RCTI NETWORK", "FREQ (MHz)": 543.25, "SID_LONG": 116.8361861, "SID_LAT": -1.26977222, "H_ANT (masl)": 120},
    {"STN_NAME": "SCTV BALIKPAPAN", "FREQ (MHz)": 559.25, "SID_LONG": 116.8386111, "SID_LAT": -1.24805556, "H_ANT (masl)": 124},
    {"STN_NAME": "CIPTA TELEVISI BORNEO NUSANTARA", "FREQ (MHz)": 575.25, "SID_LONG": 116.8379167, "SID_LAT": -1.24872222, "H_ANT (masl)": 103},
    {"STN_NAME": "ANTV BALIKPAPAN", "FREQ (MHz)": 671.25, "SID_LONG": 116.8378889, "SID_LAT": -1.24801944, "H_ANT (masl)": 139},
    {"STN_NAME": "GLOBAL TV", "FREQ (MHz)": 703.25, "SID_LONG": 116.8361861, "SID_LAT": -1.26977222, "H_ANT (masl)": 120},
    {"STN_NAME": "KOMPAS TV BALIKPAPAN", "FREQ (MHz)": 719.25, "SID_LONG": 116.8459222, "SID_LAT": -1.24179444, "H_ANT (masl)": 130},
    {"STN_NAME": "METRO TV BALIKPAPAN", "FREQ (MHz)": 735.25, "SID_LONG": 116.8392806, "SID_LAT": -1.24823889, "H_ANT (masl)": 120},
    {"STN_NAME": "TRANS7 BALIKPAPAN", "FREQ (MHz)": 751.25, "SID_LONG": 116.8388889, "SID_LAT": -1.24722222, "H_ANT (masl)": 120}
]

df_tx = pd.DataFrame(tx_data)
tx_path = os.path.join(data_dir, "sample_data_transmitter.csv")
df_tx.to_csv(tx_path, index=False)
print(f"Saved: {tx_path}")

# 2. Generate 2 Drive Test Files around Balikpapan (Simulating Scanner Log Format)
target_freqs_hz = (df_tx['FREQ (MHz)'] * 1e6).round().astype(np.int64).tolist()
# Add non-broadcast scanner scan frequencies (e.g. FM radio 87.7 - 98.4 MHz like in user's image)
fm_freqs_hz = [87700000, 88100000, 88500000, 89300000, 91700000, 94400000, 95200000, 96800000]
all_scanned_freqs = fm_freqs_hz + target_freqs_hz

def generate_drive_route(n_route_points: int, center_lat: float, center_lon: float):
    lats = [center_lat]
    lons = [center_lon]
    for _ in range(n_route_points - 1):
        lats.append(lats[-1] + np.random.normal(0, 0.002))
        lons.append(lons[-1] + np.random.normal(0, 0.0025))
    return np.array(lats), np.array(lons)

def generate_dt_file(filename: str, route_lats: np.ndarray, route_lons: np.ndarray):
    rows = []
    for i in range(len(route_lats)):
        r_lat = route_lats[i]
        r_lon = route_lons[i]
        # Simulate realistic terrain altitude in Balikpapan (15m to 140m)
        elev = max(10.0, 30.0 + 35.0 * np.sin(r_lat * 40.0) * np.cos(r_lon * 30.0) + np.random.normal(0, 2))

        # Include all scanned frequencies at this physical coordinate
        for f_hz in all_scanned_freqs:
            if f_hz in target_freqs_hz:
                # Find matching transmitter
                matching_tx = df_tx[df_tx['FREQ (MHz)'] == round(f_hz / 1e6, 2)].iloc[0]
                t_lat, t_lon = matching_tx['SID_LAT'], matching_tx['SID_LONG']
                # Geodetic distance approx
                d_km = np.sqrt(((t_lat - r_lat) * 111.139)**2 + ((t_lon - r_lon) * 111.139 * np.cos(np.radians(t_lat)))**2)
                d_km = max(d_km, 0.1)

                # Broadcast UHF field strength model: E = E0 - 28*log10(d) + shadowing
                # High near tower (75-85 dBuV/m), decay with distance, crossing 45.2 at ~35-45km
                e_val = 88.0 - 28.5 * np.log10(d_km) + np.random.normal(0, 3.5)
                e_val = np.clip(e_val, 25.0, 92.0)
            else:
                # FM frequencies
                e_val = float(np.random.uniform(22.0, 75.0))

            rows.append({
                "Time": "50:44.3",
                "Frequency (Hz)": f_hz,
                "Level (dBµV/m)": round(e_val, 1),
                "Longitude (Long)": round(r_lon, 6),
                "Latitude (Latitude)": round(r_lat, 6),
                "Altitude (m)": round(elev, 1),
                "File Info": "Meas. Unit: ArgusSamarinda"
            })

    df = pd.DataFrame(rows)
    # Add a few stationary duplicate coordinates (traffic light scenario)
    stationary_sample = df.sample(n=min(len(df), 150), random_state=42)
    df = pd.concat([df, stationary_sample], ignore_index=True)
    out_path = os.path.join(data_dir, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")

# Generate 2 files: Route 1 (Balikpapan City & Highway) and Route 2 (East Coastal Route)
r1_lats, r1_lons = generate_drive_route(160, -1.247, 116.845)
generate_dt_file("sample_DriveT_BPP1.csv", r1_lats, r1_lons)

r2_lats, r2_lons = generate_drive_route(180, -1.215, 116.865)
generate_dt_file("sample_DriveT_BPP2.csv", r2_lats, r2_lons)

print("Sample generation complete.")
