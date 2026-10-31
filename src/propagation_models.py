"""
Theoretical and Empirical Radio Propagation Baseline Models
Includes Free Space Path Loss, Okumura-Hata (UHF), and DVB-T2 standard thresholds
based on ITU-R P.1546 / BT.1368 and Permenkominfo guidelines.
"""

import numpy as np
from typing import Union, Dict

# Standard DVB-T2 UHF Field Strength Classification (dBuV/m)
DVBT2_THRESHOLDS = {
    'STRONG': 48.0,      # >= 48 dBuV/m: Strong signal, robust reception (ITU-R BT.1368 fixed outdoor / portable)
    'MEDIUM': 38.0,      # 38 - 48 dBuV/m: Moderate / marginal signal, may require directional high-gain antenna
    'BLANK_SPOT': 0.0    # < 38 dBuV/m: Blank spot / below DVB-T2 QEF (Quasi Error Free) demodulation limit
}

COLOR_MAP = {
    'STRONG': '#1b9e77',      # Green
    'MEDIUM': '#d95f02',      # Amber / Orange-Yellow
    'BLANK_SPOT': '#e7298a'   # Red / Magenta
}


def free_space_field_strength(
    erp_kw: Union[float, np.ndarray],
    distance_km: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Theoretical Free Space Field Strength in dBuV/m:
    E = 106.92 + 10*log10(ERP_kW) - 20*log10(d_km)
    """
    d_clamped = np.maximum(distance_km, 0.05)
    return 106.92 + 10.0 * np.log10(np.maximum(erp_kw, 0.001)) - 20.0 * np.log10(d_clamped)


def okumura_hata_field_strength(
    frequency_mhz: Union[float, np.ndarray],
    tx_height_m: Union[float, np.ndarray],
    rx_height_m: Union[float, np.ndarray],
    distance_km: Union[float, np.ndarray],
    erp_kw: Union[float, np.ndarray],
    environment: str = 'suburban'
) -> Union[float, np.ndarray]:
    """
    Okumura-Hata empirical model adapted for UHF broadcast frequencies (470-694 MHz).
    Calculates Path Loss (dB) and converts to Field Strength (dBuV/m).
    """
    f = np.clip(frequency_mhz, 150.0, 1500.0)
    ht = np.clip(tx_height_m, 20.0, 500.0)
    hr = np.clip(rx_height_m, 1.0, 30.0)
    d = np.clip(distance_km, 0.1, 100.0)
    
    # Receiver antenna correction factor a(hr)
    log_f = np.log10(f)
    a_hr = (1.1 * log_f - 0.7) * hr - (1.56 * log_f - 0.8)
    
    # Basic urban path loss (dB)
    loss = (
        69.55 + 26.16 * log_f - 13.82 * np.log10(ht) - a_hr +
        (44.9 - 6.55 * np.log10(ht)) * np.log10(d)
    )
    
    # Environment correction
    if environment == 'suburban':
        loss = loss - 2.0 * (np.log10(f / 28.0))**2 - 5.4
    elif environment == 'open' or environment == 'rural':
        loss = loss - 4.78 * (log_f)**2 + 18.33 * log_f - 40.94
        
    # Convert Path Loss (dB) to Field Strength (dBuV/m):
    # Free Space / Dipole reference:
    # E (dBuV/m) = 139.36 + 10*log10(ERP_kW) + 20*log10(f_MHz) - Path_Loss
    field_strength = 139.36 + 10.0 * np.log10(np.maximum(erp_kw, 0.001)) + 20.0 * np.log10(f) - loss
    return field_strength


def classify_field_strength(field_strength: Union[float, np.ndarray]) -> Union[str, np.ndarray]:
    """
    Classify field strength into DVB-T2 reception bands:
    - Strong: >= 48 dBuV/m (Green)
    - Medium: 38 to 48 dBuV/m (Yellow/Amber)
    - Blank Spot: < 38 dBuV/m (Red)
    """
    if isinstance(field_strength, (int, float, np.floating)):
        if field_strength >= DVBT2_THRESHOLDS['STRONG']:
            return 'Strong (>= 48 dBuV/m)'
        elif field_strength >= DVBT2_THRESHOLDS['MEDIUM']:
            return 'Medium (38 - 48 dBuV/m)'
        else:
            return 'Blank Spot (< 38 dBuV/m)'
            
    conditions = [
        field_strength >= DVBT2_THRESHOLDS['STRONG'],
        (field_strength >= DVBT2_THRESHOLDS['MEDIUM']) & (field_strength < DVBT2_THRESHOLDS['STRONG']),
        field_strength < DVBT2_THRESHOLDS['MEDIUM']
    ]
    choices = [
        'Strong (>= 48 dBuV/m)',
        'Medium (38 - 48 dBuV/m)',
        'Blank Spot (< 38 dBuV/m)'
    ]
    return np.select(conditions, choices, default='Blank Spot (< 38 dBuV/m)')


def get_dvbt2_color(val: float) -> str:
    """Return hex color for a given field strength value."""
    if val >= DVBT2_THRESHOLDS['STRONG']:
        return '#2ecc71'  # Green
    elif val >= DVBT2_THRESHOLDS['MEDIUM']:
        return '#f39c12'  # Yellow/Orange
    else:
        return '#e74c3c'  # Red
