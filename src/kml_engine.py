"""
KML / KMZ Masterplan Engine for DVB-T2 Indonesia
Provides extraction, indexing, and spatial geometry querying for:
1. Official DVB-T2 Service Area Boundaries (225 Wilayah Layanan Polygons across Indonesia)
2. Official DVB-T2 Test Points (1,113 Compliance Verification Points)
3. Custom User KML upload and parsing support
"""

import os
import re
import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def calculate_polygon_area_km2(coords: List[List[float]]) -> float:
    """
    Computes the true geodesic area (km²) of a polygon defined by [[lat, lon], ...]
    using spherical excess on WGS84 earth sphere (mean radius R = 6371.0 km).
    """
    if not coords or len(coords) < 3:
        return 0.0
    arr = np.array(coords, dtype=float)
    lats = np.radians(arr[:, 0])
    lons = np.radians(arr[:, 1])
    # Ensure closed ring
    if not (arr[0, 0] == arr[-1, 0] and arr[0, 1] == arr[-1, 1]):
        lats = np.append(lats, lats[0])
        lons = np.append(lons, lons[0])
    R = 6371.0 # Earth radius in kilometers
    # Spherical polygon area via Girard / Green theorem
    area = 0.5 * np.abs(np.sum((lons[1:] - lons[:-1]) * (np.sin(lats[1:]) + np.sin(lats[:-1])))) * (R ** 2)
    return float(area)



class KMLMasterplanEngine:
    """
    Parser and Query Engine for the Official Indonesian DVB-T2 Broadcast Masterplan
    (Permenkominfo No. 6/2019 & ASO Allocations).
    """
    def __init__(self, kml_path: Optional[str] = None):
        self.kml_path = kml_path
        self.service_areas: Dict[str, Dict[str, Any]] = {}
        self.test_points: List[Dict[str, Any]] = []
        self._is_loaded = False

        if kml_path and os.path.exists(kml_path):
            self.load_masterplan(kml_path)

    def load_masterplan(self, kml_path: str) -> bool:
        """Parses the Masterplan KML file into memory."""
        try:
            self.kml_path = kml_path
            tree = ET.parse(kml_path)
            root = tree.getroot()

            # Find all folders
            folders = root.findall('.//{http://www.opengis.net/kml/2.2}Folder')
            if not folders:
                folders = root.findall('.//Folder')

            # Extract Wilayah Layanan (Folder 0 or by name)
            wl_folder = None
            tp_folder = None
            for f in folders:
                n_el = f.find('{http://www.opengis.net/kml/2.2}name')
                if n_el is None:
                    n_el = f.find('name')
                if n_el is not None and n_el.text:
                    t_lower = n_el.text.strip().lower()
                    if 'wilayah layanan' in t_lower:
                        wl_folder = f
                    elif 'test point' in t_lower:
                        tp_folder = f

            # Fallback by index if names not matched
            if wl_folder is None and len(folders) > 0:
                wl_folder = folders[0]
            if tp_folder is None and len(folders) > 2:
                tp_folder = folders[2]

            # 1. Parse Wilayah Layanan Polygons
            self.service_areas.clear()
            if wl_folder is not None:
                pms = wl_folder.findall('.//{http://www.opengis.net/kml/2.2}Placemark')
                if not pms:
                    pms = wl_folder.findall('.//Placemark')

                for pm in pms:
                    n_el = pm.find('{http://www.opengis.net/kml/2.2}name')
                    if n_el is None:
                        n_el = pm.find('name')
                    if n_el is None or not n_el.text:
                        continue

                    raw_kml_name = n_el.text.strip()
                    # Strip all parenthesized date/phase information (e.g., "(30 April 2022)", "(Non-ASO)")
                    clean_name = re.sub(r'\s*\([^)]*\)', '', raw_kml_name).strip()
                    raw_name = clean_name
                    desc_el = pm.find('{http://www.opengis.net/kml/2.2}description')
                    if desc_el is None:
                        desc_el = pm.find('description')
                    desc_text = desc_el.text.strip() if desc_el is not None and desc_el.text else ''

                    coords_el = pm.find('.//{http://www.opengis.net/kml/2.2}coordinates')
                    if coords_el is None:
                        coords_el = pm.find('.//coordinates')

                    if coords_el is not None and coords_el.text:
                        coords: List[List[float]] = []
                        for token in coords_el.text.strip().split():
                            parts = token.split(',')
                            if len(parts) >= 2:
                                try:
                                    lon = float(parts[0])
                                    lat = float(parts[1])
                                    coords.append([lat, lon])
                                except ValueError:
                                    continue

                        if coords:
                            # Parse metadata from description
                            coverage_area_match = re.search(r'wilayah tercakup:\s*(.*?)(?:<br>kanal:|$)', desc_text, re.IGNORECASE | re.DOTALL)
                            channels_match = re.search(r'kanal:\s*([0-9,\s]+)', desc_text, re.IGNORECASE)

                            covered_str = coverage_area_match.group(1).replace('<br>', ', ').replace('\t', ' ').strip() if coverage_area_match else ''
                            covered_str = re.sub(r'\b\d{2}\.\d{2}\b\s*', '', covered_str).strip(', ')
                            channels_str = channels_match.group(1).strip() if channels_match else ''

                            # Standardized key for fuzzy matching
                            norm_key = self._normalize_key(raw_name)

                            # Calculate true geodesic area of service area polygon
                            poly_area = calculate_polygon_area_km2(coords)

                            self.service_areas[norm_key] = {
                                'raw_name': raw_name,
                                'norm_key': norm_key,
                                'description': desc_text,
                                'covered_areas': covered_str,
                                'channels': channels_str,
                                'polygon': coords,
                                'area_km2': poly_area,
                                'center': [
                                    sum(c[0] for c in coords) / len(coords),
                                    sum(c[1] for c in coords) / len(coords)
                                ]
                            }

            # 2. Parse Test Points
            self.test_points.clear()
            if tp_folder is not None:
                pms = tp_folder.findall('.//{http://www.opengis.net/kml/2.2}Placemark')
                if not pms:
                    pms = tp_folder.findall('.//Placemark')

                for pm in pms:
                    n_el = pm.find('{http://www.opengis.net/kml/2.2}name')
                    if n_el is None:
                        n_el = pm.find('name')
                    if n_el is None or not n_el.text:
                        continue

                    tp_name = n_el.text.strip()
                    desc_el = pm.find('{http://www.opengis.net/kml/2.2}description')
                    if desc_el is None:
                        desc_el = pm.find('description')
                    desc_text = desc_el.text.strip() if desc_el is not None and desc_el.text else ''

                    coords_el = pm.find('.//{http://www.opengis.net/kml/2.2}coordinates')
                    if coords_el is None:
                        coords_el = pm.find('.//coordinates')

                    if coords_el is not None and coords_el.text:
                        tokens = coords_el.text.strip().split()
                        if tokens:
                            parts = tokens[0].split(',')
                            if len(parts) >= 2:
                                try:
                                    lon = float(parts[0])
                                    lat = float(parts[1])
                                    # Extract target fieldstrength if available
                                    fs_match = re.search(r'Reference Fieldstrength:\s*([0-9\.]+)', desc_text)
                                    ref_fs = float(fs_match.group(1)) if fs_match else 54.0

                                    self.test_points.append({
                                        'name': tp_name,
                                        'lat': lat,
                                        'lon': lon,
                                        'ref_fs_dbuv': ref_fs,
                                        'description': desc_text,
                                        'service_area_key': self._normalize_key(tp_name)
                                    })
                                except ValueError:
                                    continue

            self._is_loaded = True
            return True

        except Exception as e:
            print(f"Error loading Masterplan KML: {e}")
            return False

    @staticmethod
    def _normalize_key(name: str) -> str:
        """Normalizes a name string for resilient fuzzy matching."""
        s = name.lower()
        # Remove date suffixes like (30 april 2022) or (non-aso)
        s = re.sub(r'\(.*?\)', '', s)
        s = re.sub(r'[^a-z0-9]', '', s)
        return s

    def find_service_area(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Finds a matching Wilayah Layanan polygon using fuzzy token matching.
        Example queries: 'kaltim_1', 'Kalimantan Timur 1', 'Balikpapan - Kaltim 1'
        """
        if not self.service_areas:
            return None

        q_norm = self._normalize_key(query)

        # 1. Exact match on normalized key
        if q_norm in self.service_areas:
            return self.service_areas[q_norm]

        # 2. Substring match
        for k, v in self.service_areas.items():
            if q_norm in k or k in q_norm:
                return v

        # 3. Province + Number match (e.g. 'kaltim1' -> 'kalimantantimur1')
        kaltim_aliases = {
            'kaltim1': 'kalimantantimur1',
            'kaltim2': 'kalimantantimur2',
            'kaltim3': 'kalimantantimur3',
            'kaltim4': 'kalimantantimur4',
            'kaltim5': 'kalimantantimur5',
            'kaltim6': 'kalimantantimur6',
            'kaltim7': 'kalimantantimur7',
            'balikpapan': 'kalimantantimur1',
            'samarinda': 'kalimantantimur1',
            'bontang': 'kalimantantimur1'
        }
        for alias, target in kaltim_aliases.items():
            if alias in q_norm and target in self.service_areas:
                return self.service_areas[target]

        return None

    def get_test_points_for_area(self, area_norm_key: str) -> List[Dict[str, Any]]:
        """Returns test points belonging to a specific service area."""
        results = []
        for tp in self.test_points:
            # Check if test point name references the area
            if area_norm_key in tp['service_area_key'] or tp['service_area_key'] in area_norm_key:
                results.append(tp)
            elif 'kaltim1' in area_norm_key and ('kaltim1' in tp['service_area_key'] or 'kalimantantimur1' in tp['service_area_key']):
                results.append(tp)
        return results

    def list_all_service_areas(self) -> List[Dict[str, str]]:
        """Returns a list of all 225 service areas sorted alphabetically."""
        return [
            {
                'name': v['raw_name'],
                'norm_key': v['norm_key'],
                'covered_areas': v['covered_areas'],
                'channels': v['channels']
            }
            for v in sorted(self.service_areas.values(), key=lambda x: x['raw_name'])
        ]


# Singleton default instance
_masterplan_instance: Optional[KMLMasterplanEngine] = None

def get_masterplan_engine(kml_path: Optional[str] = None) -> KMLMasterplanEngine:
    global _masterplan_instance
    if _masterplan_instance is None or not _masterplan_instance._is_loaded:
        if kml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            kml_path = os.path.join(base_dir, 'data', 'kml', 'masterplan_tv_digital.kml')
        _masterplan_instance = KMLMasterplanEngine(kml_path)
    return _masterplan_instance


def evaluate_test_points_compliance(
    model_manager: Any,
    tx_params: Dict[str, Any],
    test_points: List[Dict[str, Any]],
    tolerance_db: float = 3.0,
    rx_antenna_amsl: float = 1.5,
    strict_nlos_cutoff: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Evaluates field strength predictions at official DVB-T2 Test Points
    and benchmarks them against regulatory coordination limits (Permenkominfo No. 6/2019).
    
    REGULATORY LOGIC (Batas Maksimum Kuat Medan Yang Diperbolehkan):
    Test points are coordination points along the borders of the service area to prevent
    co-channel/adjacent interference into neighboring service areas.
    - If E_pred <= E_ref: 🟢 AMAN / PATUH (TIDAK MEMICU INTERFERENSI)
    - If E_ref < E_pred <= E_ref + tol: 🟡 DALAM TOLERANSI (+3 dB)
    - If E_pred > E_ref + tol: 🔴 MELANGGAR (POTENSI INTERFERENSI KE WILAYAH SEBELAH)
    
    Returns:
    - df_eval: DataFrame containing diagnostic metrics per test point
    - summary: Summary scorecard dictionary (safe_count, tolerable_count, violation_count, compliance_rate)
    """
    import pandas as pd
    if not test_points or model_manager is None or model_manager.active_model is None:
        return pd.DataFrame(), {}

    results = []
    safe_cnt = 0
    tolerable_cnt = 0
    violation_cnt = 0

    for tp in test_points:
        t_lat = float(tp['lat'])
        t_lon = float(tp['lon'])
        t_name = str(tp['name'])
        ref_fs = float(tp.get('ref_fs_dbuv', 54.0))

        pred_res = model_manager.predict_single_point(
            tx_row=tx_params,
            rx_latitude=t_lat,
            rx_longitude=t_lon,
            rx_antenna_amsl=rx_antenna_amsl,
            strict_nlos_cutoff=strict_nlos_cutoff
        )

        fs_pred = float(pred_res['predicted_field_strength_dbuv_m'])
        d_km = float(pred_res['distance_km'])
        clr = float(pred_res.get('clearance_m', 0.0))
        is_los = bool(pred_res.get('is_los', True))
        is_cut_nlos = bool(pred_res.get('is_cut_by_nlos', False))
        is_cut_horizon = bool(pred_res.get('is_cut_by_horizon', False))

        is_estimated = bool(pred_res.get('is_estimated', False) or is_cut_nlos or is_cut_horizon or fs_pred < 45.2)
        pred_label = f"{fs_pred:.1f} dBµV/m (Est.)" if is_estimated else f"{fs_pred:.1f} dBµV/m"

        margin = fs_pred - ref_fs
        upper_bound = ref_fs + tolerance_db

        if fs_pred <= ref_fs:
            if is_cut_nlos:
                status = '🟢 AMAN (TERHALANG BUKIT / NLOS CUTOFF)'
            elif is_cut_horizon:
                status = '🟢 AMAN (DI LUAR RADIO HORIZON)'
            elif is_estimated:
                status = '🟢 AMAN (BLANK SPOT / SINYAL SANGAT LEMAH)'
            else:
                status = '🟢 AMAN / PATUH (TIDAK MEMICU INTERFERENSI)'
            status_code = 'PASS'
            color = '#10b981' # Green (Safe)
            safe_cnt += 1
        elif fs_pred <= upper_bound:
            status = f'🟡 DALAM TOLERANSI (+{tolerance_db:.0f} dB)'
            status_code = 'TOLERABLE'
            color = '#f59e0b' # Amber/Orange (Tolerable)
            tolerable_cnt += 1
        else:
            status = '🔴 MELANGGAR (POTENSI INTERFERENSI WILAYAH SEBELAH)'
            status_code = 'VIOLATION'
            color = '#ef4444' # Red (Regulatory Violation)
            violation_cnt += 1

        results.append({
            'Nama Titik Uji': t_name,
            'Latitude': t_lat,
            'Longitude': t_lon,
            'Jarak ke TX (km)': round(d_km, 2),
            'Clearance Medan (m)': round(clr, 1),
            'Batas Maksimum Acuan (dBµV/m)': ref_fs,
            'Ambang Batas Maks (+3 dB)': round(upper_bound, 1),
            'Prediksi Kuat Medan AI (dBµV/m)': round(fs_pred, 1),
            'Prediksi Display': pred_label,
            'Margin thd Batas Maks (dB)': round(margin, 1),
            'Status Koordinasi': status,
            'status_code': status_code,
            'color': color,
            'is_los': is_los,
            'is_cut_by_nlos': is_cut_nlos,
            'is_cut_by_horizon': is_cut_horizon,
            'is_estimated': is_estimated
        })

    df_eval = pd.DataFrame(results)
    total_pts = len(results)
    compliant_pts = safe_cnt + tolerable_cnt
    compliance_rate = (compliant_pts / total_pts * 100.0) if total_pts > 0 else 0.0
    avg_margin = df_eval['Margin thd Batas Maks (dB)'].mean() if len(df_eval) > 0 else 0.0

    summary = {
        'total_points': total_pts,
        'safe_count': safe_cnt,
        'pass_count': safe_cnt, # Backwards compatibility alias
        'tolerable_count': tolerable_cnt,
        'violation_count': violation_cnt,
        'fail_count': violation_cnt, # Backwards compatibility alias
        'compliant_count': compliant_pts,
        'compliance_rate_pct': round(compliance_rate, 1),
        'avg_margin_db': round(float(avg_margin), 2),
        'tolerance_db': tolerance_db
    }

    return df_eval, summary
