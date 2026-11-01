"""
Aplikasi Prediksi Propagasi Siaran DVB-T2 Berbasis AI & GIS (Multi-Region)
Tema: CloudRF / SOOTHSAYER High-Performance RF Telemetry & GIS Suite
Arsitektur:
1. Peta GIS (Web GIS Map) SELALU tampil di halaman utama (atas).
2. Legenda vertikal gradient bar dBµV/m CloudRF style langsung di peta.
3. Sidebar memuat pemilih wilayah, navigasi data (Pemancar, Hasil Ukur, Split, Model, Coverage, Titik Tunggal), dan kontrol kontekstual.
4. Layar ter-split: Peta interaktif di atas, panel detail interaktif di bawah dengan tombol fleksibel tampil/sembunyi.
5. Menu tambah data baru hanya muncul saat membuat wilayah baru secara terurut (Wizard).
"""

import os
import sys
import json
import importlib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Dynamic Module Reloading
import src.features
import src.terrain_engine
import src.split_utils
import src.model_engine
import src.grid_engine
import src.gis_viz
import src.storage
import src.kml_engine
importlib.reload(src.features)
importlib.reload(src.terrain_engine)
importlib.reload(src.split_utils)
importlib.reload(src.model_engine)
importlib.reload(src.grid_engine)
importlib.reload(src.gis_viz)
importlib.reload(src.storage)
importlib.reload(src.kml_engine)

# Internal Modules
from src.features import (
    MODEL_FEATURE_COLS,
    TARGET_COL,
    SPATIAL_METADATA_COLS,
    DVBT2_THRESHOLDS,
    process_and_align_datasets,
    calculate_haversine_distance,
    calculate_azimuth_bearing,
    calculate_radio_horizon,
    load_dataframe_safely
)
from src.split_utils import (
    perform_multi_feature_stratified_split,
    compute_summary_check_table,
    generate_split_distribution_figure
)
from src.model_engine import DVBT2ModelManager
from src.storage import (
    save_region_processed_data,
    save_region_split_data,
    load_region_processed_data,
    delete_region_processed_data,
    delete_region_completely,
    rename_region,
    save_region_transmitters,
    append_region_drive_test_data,
    list_available_regions
)
from src.grid_engine import (
    generate_radius_grid_boxes,
    predict_coverage_grid_with_subsampling,
    generate_continuous_coverage_raster
)
from src.gis_viz import (
    create_cloudrf_gis_map,
    create_input_data_map,
    create_polygon_coverage_map,
    plot_parity_scatter,
    plot_feature_importance
)

# Page Setup
favicon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "favicon.png")
st.set_page_config(
    page_title="RealFieldAI • Real Measurements, Intelligent Predictions",
    page_icon=favicon_path if os.path.exists(favicon_path) else "📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CloudRF / SOOTHSAYER Modern Executive Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Layout geometry: elevated main view with minimal gap below Streamlit fixed header */
    .block-container {
        padding-top: 2.65rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 99% !important;
    }

    /* Collapse empty element wrappers holding <style> */
    div[data-testid="element-container"]:has(style:only-child) {
        display: none !important;
    }

    /* Elevate sidebar content all the way to the top */
    [data-testid="stSidebar"] {
        padding-top: 0rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
        padding-top: 0.2rem !important;
        padding-bottom: 0rem !important;
        min-height: 2rem !important;
        height: auto !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        margin-top: -1.2rem !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    header[data-testid="stHeader"] {
        background: #ffffff !important;
        height: 2.5rem !important;
        z-index: 9999 !important;
        border-bottom: 1px solid #f1f5f9 !important;
    }

    /* Executive Toolbar Header */
    .cloudrf-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #0B132B 100%);
        color: #F8FAFC;
        padding: 8px 18px;
        border-radius: 8px;
        margin-top: 0px !important;
        margin-bottom: 8px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.12);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .header-left {
        display: flex;
        flex-direction: column;
        gap: 3px;
        min-width: 220px;
    }
    .header-brand {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #38BDF8;
        display: flex;
        align-items: center;
        gap: 8px;
        line-height: 1.2;
    }
    .header-tagline {
        font-size: 0.80rem;
        color: #94A3B8;
        font-weight: 500;
        line-height: 1.3;
        display: flex;
        align-items: center;
        gap: 6px;
        max-width: 55vw;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .header-pills-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        justify-content: flex-end;
    }
    .cloudrf-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 6px;
        letter-spacing: 0.02em;
        white-space: nowrap;
    }
    .pill-channels {
        background: rgba(234, 179, 8, 0.2);
        color: #FDE047;
        border: 1px solid rgba(234, 179, 8, 0.45);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }
    .pill-model-ok {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.45);
    }
    .pill-model-no {
        background: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.45);
    }
    .pill-data {
        background: rgba(99, 102, 241, 0.2);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.45);
    }
    .pill-nlos {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
    }

    /* Split Screen Bar */
    .split-toggle-bar {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px 14px;
        margin-top: 10px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
    }

    /* Primary Action Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
    }
    button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Sidebar Section Divider */
    .sidebar-heading {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748B;
        margin-top: 16px;
        margin-bottom: 6px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Clear any legacy cached objects to ensure latest methods & features are loaded
st.cache_resource.clear()

# Instantiate fresh model manager directly on each script run
manager = src.model_engine.DVBT2ModelManager(models_dir=MODELS_DIR)


# ==============================================================================
# SIDEBAR: KONTROL NAVIGASI & OPERASI CLOUDRF
# ==============================================================================
with st.sidebar:
    # Logo & Nama Aplikasi Resmi RealFieldAI
    logo_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "realfield_ai_logo.png")
    if os.path.exists(logo_file):
        st.image(logo_file, use_container_width=True)
    else:
        st.markdown("""
        <div style="font-family:'Plus Jakarta Sans', sans-serif; font-weight:800; font-size:1.3rem; color:#0B2240; margin-bottom:2px;">
            RealField<span style="color:#1C8B82;">AI</span>
        </div>
        <div style="font-size:0.75rem; color:#475569; font-weight:500; margin-bottom:12px;">
            Real Measurements, Intelligent Predictions
        </div>
        """, unsafe_allow_html=True)

    # 1. Discover all available saved regions from disk & models
    available_regions = list_available_regions(DATA_DIR, MODELS_DIR)
    NEW_REGION_OPT = "➕ [Buat / Tambah Wilayah Baru...]"
    region_options = [r['label'] for r in available_regions] + [NEW_REGION_OPT]

    # Determine default selection
    saved_active = st.session_state.get('active_stored_region')
    def_idx = 0
    if saved_active:
        for idx, r in enumerate(available_regions):
            if r['code'] == saved_active:
                def_idx = idx
                break
    else:
        for idx, r in enumerate(available_regions):
            if 'balikpapan' in r['code'].lower():
                def_idx = idx
                break

    st.markdown('<div class="sidebar-heading">🌐 WILAYAH LAYANAN SIARAN</div>', unsafe_allow_html=True)
    selected_reg_label = st.selectbox(
        "Pilih Wilayah Siaran Tersimpan:",
        options=region_options,
        index=def_idx if len(region_options) > 0 else 0,
        help="Pilih wilayah untuk memuat seluruh paket pemancar, data ukur, dan model AI secara instan."
    )

    is_new_region_mode = (selected_reg_label == NEW_REGION_OPT)

    if is_new_region_mode:
        region_name = "Wilayah Siaran Baru"
        region_code = "wilayah_siaran_baru"
        has_trained_model = False
    else:
        matched_reg = next(r for r in available_regions if r['label'] == selected_reg_label)
        region_code = matched_reg['code']
        region_name = matched_reg['title']
        has_trained_model = matched_reg.get('has_model', False)

        # Single Unified Management Expander: Edit Nama & Hapus Wilayah
        with st.expander("⚙️ Kelola Nama & Hapus Wilayah", expanded=False):
            st.markdown(f"**ID Wilayah:** `{region_code}`")
            status_txt = "🟢 Model AI Terlatih & Siap" if has_trained_model else "🟡 Model Belum Dilatih"
            st.caption(f"Status Model: **{status_txt}**")

            # Bagian 1: Ubah Nama Tampilan Wilayah
            st.markdown("##### ✏️ Ubah Nama Wilayah")
            new_title_in = st.text_input(
                "Nama Tampilan Wilayah:",
                value=region_name,
                key=f"rename_input_{region_code}",
                help="Mengubah nama wilayah siaran tanpa mengubah data pemancar, data ukur, maupun model AI yang tersimpan."
            )
            if st.button("Simpan Perubahan Nama", key=f"btn_rename_{region_code}", use_container_width=True):
                if new_title_in.strip() and new_title_in.strip() != region_name:
                    rename_region(DATA_DIR, region_code, new_title_in.strip())
                    st.session_state['active_stored_region'] = region_code
                    st.success(f"✅ Nama wilayah diperbarui menjadi: **{new_title_in.strip()}**")
                    st.rerun()

            st.markdown("---")
            # Bagian 2: Hapus Wilayah Layanan (Bisa kapan saja, meski model belum dilatih)
            st.markdown("##### 🗑️ Hapus Wilayah Layanan")
            if has_trained_model:
                st.caption("Menghapus data pemancar, titik ukur, dan model AI wilayah ini secara permanen dari disk.")
            else:
                st.caption("Wilayah ini belum memiliki model AI terlatih. Anda tetap dapat menghapus seluruh data wilayah ini secara permanen.")

            confirm_del = st.checkbox("Konfirmasi ingin menghapus wilayah ini", key=f"confirm_del_{region_code}")
            if st.button("Hapus Wilayah Ini Secara Permanen", key=f"btn_del_{region_code}", disabled=not confirm_del, type="secondary", use_container_width=True):
                delete_region_completely(DATA_DIR, MODELS_DIR, region_code)
                for k in ['df_model', 'df_tx_summary', 'stats', 'df_split_full', 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'df_raw_tx', 'df_raw_dt_list', 'active_stored_region', 'dt_files_processed_key', 'tx_file_processed_key', 'active_coverage_grid', 'single_pred_result', 'single_pred_ctx']:
                    st.session_state.pop(k, None)
                st.success(f"🗑️ Wilayah `{region_name}` berhasil dihapus!")
                st.rerun()

    # Auto-load persistent region data from disk
    if not is_new_region_mode:
        active_stored_reg = st.session_state.get('active_stored_region')
        if active_stored_reg != region_code:
            for k in ['df_model', 'df_tx_summary', 'stats', 'df_split_full', 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'summary_check', 'fig_dist', 'df_raw_tx', 'df_raw_dt_list', 'active_coverage_grid', 'single_pred_result', 'single_pred_ctx']:
                st.session_state.pop(k, None)
            st.session_state['active_stored_region'] = region_code

        saved_region_data = load_region_processed_data(DATA_DIR, region_code)
        if saved_region_data is not None:
            if 'df_tx_summary' not in st.session_state and saved_region_data['df_tx_summary'] is not None:
                st.session_state['df_tx_summary'] = saved_region_data['df_tx_summary']
            if 'df_model' not in st.session_state and saved_region_data['df_clean'] is not None:
                st.session_state['df_model'] = saved_region_data['df_clean']
            if 'stats' not in st.session_state and saved_region_data['stats']:
                st.session_state['stats'] = saved_region_data['stats']
            if 'df_split_full' not in st.session_state and saved_region_data['df_split'] is not None:
                df_sp = saved_region_data['df_split']
                st.session_state['df_split_full'] = df_sp
                tr_m = df_sp['split_dataset'] == 'Train (70%)'
                va_m = df_sp['split_dataset'] == 'Validation (15%)'
                te_m = df_sp['split_dataset'] == 'Test (15%)'
                f_cols = [c for c in MODEL_FEATURE_COLS if c in df_sp.columns]
                st.session_state['X_train'] = df_sp.loc[tr_m, f_cols]
                st.session_state['y_train'] = df_sp.loc[tr_m, TARGET_COL]
                st.session_state['X_val'] = df_sp.loc[va_m, f_cols]
                st.session_state['y_val'] = df_sp.loc[va_m, TARGET_COL]
                st.session_state['X_test'] = df_sp.loc[te_m, f_cols]
                st.session_state['y_test'] = df_sp.loc[te_m, TARGET_COL]

        model_loaded = manager.load_region_model(region_code)
    else:
        saved_region_data = None
        model_loaded = False

    # Discover matching Masterplan DVB-T2 Service Area & Test Points
    kml_engine = src.kml_engine.get_masterplan_engine()
    current_sa = kml_engine.find_service_area(region_code) or kml_engine.find_service_area(region_name)
    current_tps = kml_engine.get_test_points_for_area(current_sa['norm_key']) if current_sa else []

    # 2. View / Mode Switcher in Sidebar (When an existing region is selected)
    if not is_new_region_mode:
        st.markdown('<div class="sidebar-heading">📋 MODE TAMPILAN & OPERASI</div>', unsafe_allow_html=True)
        active_view = st.radio(
            "Pilih Mode Analisis:",
            options=[
                "📡 Data Pemancar",
                "🚗 Data Hasil Ukur & Preprocessing",
                "📊 Analisis Split & Stratifikasi",
                "🧠 Model AI & Retraining",
                "🗺️ Simulasi Cakupan (Coverage Grid)",
                "🎯 Prediksi Titik Tunggal"
            ],
            index=0,
            label_visibility="collapsed"
        )

        if active_view == "🗺️ Simulasi Cakupan (Coverage Grid)":
            st.markdown('<div class="sidebar-heading">⚙️ PARAMETER COVERAGE</div>', unsafe_allow_html=True)
        elif active_view == "🎯 Prediksi Titik Tunggal":
            st.markdown('<div class="sidebar-heading">⚙️ PARAMETER TITIK TUNGGAL</div>', unsafe_allow_html=True)
        elif active_view == "🧠 Model AI & Retraining":
            st.markdown('<div class="sidebar-heading">⚙️ PARAMETER PELATIHAN AI</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sidebar-heading">⚙️ PARAMETER OPERASI</div>', unsafe_allow_html=True)

        # Contextual controls based on active view
        df_tx_curr = st.session_state.get('df_tx_summary')
        tx_names_list = list(df_tx_curr['STN_NAME'].unique()) if df_tx_curr is not None and len(df_tx_curr) > 0 else []

        if active_view == "📡 Data Pemancar":
            map_filter_stn = st.selectbox(
                "Sorot Pemancar di Peta:",
                options=["Semua Pemancar"] + tx_names_list,
                index=0
            )

        elif active_view == "🚗 Data Hasil Ukur & Preprocessing":
            map_filter_dt_stn = st.selectbox(
                "Filter Titik Ukur per Pemancar:",
                options=["Semua Pemancar"] + tx_names_list,
                index=0
            )

        elif active_view == "🧠 Model AI & Retraining":
            optuna_trials_input = st.slider("Jumlah Iterasi Optuna (Trials):", min_value=10, max_value=60, value=25, step=5)
            btn_retrain_ai = st.button("🚀 Mulai Pelatihan Ulang Model AI", type="primary", use_container_width=True)

        elif active_view == "🗺️ Simulasi Cakupan (Coverage Grid)":
            cov_tx_selected = st.selectbox("Pilih Pemancar:", tx_names_list, index=0 if len(tx_names_list) > 0 else 0)

            sa_name_display = current_sa.get('raw_name', 'Wilayah Layanan KML') if current_sa else 'Wilayah Layanan KML'
            if current_sa:
                st.caption(f"🏛️ **Wilayah Layanan Masterplan:** `{sa_name_display}`")

            cov_resolution = st.selectbox(
                "Resolusi Grid Analisis:",
                options=[100, 50, 10],
                index=1,
                format_func=lambda x: f"{x}m x {x}m (Maksimum)" if x == 100 else f"{x}m x {x}m",
                help="Ukuran resolusi perhitungan medan siaran: 10m x 10m (mikro), 50m x 50m (semi-urban), dan 100m x 100m (makro)."
            )

            # Jangkauan Komputasi Grid ditentukan otomatis berdasarkan Radio Horizon Maksimal
            from src.grid_engine import calculate_max_radio_horizon_km
            sel_tx_prev = df_tx_curr[df_tx_curr['STN_NAME'] == cov_tx_selected].iloc[0].to_dict() if (df_tx_curr is not None and len(df_tx_curr) > 0 and cov_tx_selected) else {}
            cov_radius = calculate_max_radio_horizon_km(
                tx_params=sel_tx_prev,
                test_points=current_tps,
                df_region=st.session_state.get('df_model')
            ) if sel_tx_prev else 50.0

            st.caption(f"🔭 **Jangkauan Komputasi Otomatis:** `{cov_radius:.1f} km` *(Berdasarkan Radio Horizon Maksimal)*")

            tp_tolerance = st.slider(
                "Toleransi Batas Maksimum Titik Uji (+dB):",
                min_value=1.0,
                max_value=6.0,
                value=3.0,
                step=0.5,
                help="Toleransi batas maksimum kuat medan di titik test point perbatasan untuk mencegah interferensi ke wilayah layanan tetangga (standar koordinasi Permenkominfo No. 6/2019: +3.0 dB)."
            )
            cov_strict_nlos = st.toggle("Strict NLoS Cutoff (Area Terhalang Bukit)", value=False, help="Potong mutlak area terhalang kontur perbukitan (clearance < 0m) menjadi Blank Spot transparan.")
            btn_run_coverage = st.button("🎨 Bangkitkan Peta Coverage", type="primary", use_container_width=True)

        elif active_view == "🎯 Prediksi Titik Tunggal":
            st.info("💡 **Tips:** Klik langsung pada peta untuk memilih titik penerima (Rx), gunakan pintasan titik uji resmi di bawah, atau ketik koordinat manual.")
            sp_tx_selected = st.selectbox("Pilih Pemancar:", tx_names_list, index=0 if len(tx_names_list) > 0 else 0, key="sb_sp_tx")

            if current_tps:
                tp_options = ["-- Pilih Titik Uji Acuan KML --"] + [f"{tp['name']} (Target: {tp['ref_fs_dbuv']} dBµV/m)" for tp in current_tps]
                sel_tp_quick = st.selectbox("📍 Pintasan Titik Uji (Test Point KML):", options=tp_options, index=0, help="Pilih salah satu titik uji resmi untuk mengisi koordinat secara otomatis.")
                if sel_tp_quick != "-- Pilih Titik Uji Acuan KML --":
                    matched_tp_obj = next(t for t in current_tps if t['name'] in sel_tp_quick)
                    if st.session_state.get('last_selected_tp') != matched_tp_obj['name']:
                        st.session_state['last_selected_tp'] = matched_tp_obj['name']
                        st.session_state['sp_rx_lat_val'] = matched_tp_obj['lat']
                        st.session_state['sp_rx_lon_val'] = matched_tp_obj['lon']
                        st.session_state['rx_coord_version'] = st.session_state.get('rx_coord_version', 0) + 1
                        st.rerun()
            
            p_row_init = df_tx_curr[df_tx_curr['STN_NAME'] == sp_tx_selected].iloc[0] if len(tx_names_list) > 0 else None
            default_lat = float(p_row_init['SID_LAT']) + 0.08 if p_row_init is not None else -1.20
            default_lon = float(p_row_init['SID_LONG']) + 0.06 if p_row_init is not None else 116.88

            if 'sp_rx_lat_val' not in st.session_state:
                st.session_state['sp_rx_lat_val'] = default_lat
            if 'sp_rx_lon_val' not in st.session_state:
                st.session_state['sp_rx_lon_val'] = default_lon

            rx_ver = st.session_state.get('rx_coord_version', 0)
            cur_lat = float(st.session_state['sp_rx_lat_val'])
            cur_lon = float(st.session_state['sp_rx_lon_val'])

            sp_rx_lat = st.number_input("Latitude Penerima (Rx):", value=cur_lat, format="%.6f", key=f"input_sp_rx_lat_{rx_ver}")
            sp_rx_lon = st.number_input("Longitude Penerima (Rx):", value=cur_lon, format="%.6f", key=f"input_sp_rx_lon_{rx_ver}")
            st.session_state['sp_rx_lat_val'] = sp_rx_lat
            st.session_state['sp_rx_lon_val'] = sp_rx_lon

            sp_rx_amsl = st.number_input("Tinggi Antena Rx (m AMSL):", value=25.0, step=5.0, key="sp_rx_amsl")
            sp_strict_nlos = st.toggle("Strict NLoS Cutoff", value=False, key="sp_toggle_nlos")
            btn_run_sp = st.button("🎯 Hitung Kuat Medan", type="primary", use_container_width=True)

        chosen_basemap = "satellite"

    else:
        active_view = "➕ Buat Wilayah Baru"
        chosen_basemap = "satellite"


# ==============================================================================
# MAIN VIEW CANVAS: CLOUDRF EXECUTIVE TOOLBAR & MAP DISPLAY
# ==============================================================================

# Dynamic Live Badges
if model_loaded:
    m_r2_val = manager.active_metrics.get('test_r2', '0.00')
    badge_model = f'<span class="cloudrf-pill pill-model-ok">🟢 AI Ready (R² {m_r2_val})</span>'
else:
    badge_model = '<span class="cloudrf-pill pill-model-no">🟡 Model Belum Dilatih</span>'

n_tx_count = len(st.session_state.get('df_tx_summary', []))
n_dt_count = len(st.session_state.get('df_model', []))
badge_data = f'<span class="cloudrf-pill pill-data">💾 {n_tx_count} TX • {n_dt_count:,} Titik</span>' if n_tx_count > 0 else '<span class="cloudrf-pill pill-data">📂 Data Belum Dimuat</span>'

# Service Area Info & Header Construction
if current_sa:
    sa_clean_name = current_sa.get('raw_name', region_name)
    sa_cov_text = current_sa.get('covered_areas', '')
    sa_chan_text = current_sa.get('channels', '')
else:
    sa_clean_name = region_name
    sa_cov_text = ''
    sa_chan_text = ''

# Left Header: Nama Wilayah Layanan sebagai Identitas Utama (Font Besar, tanpa prefix 'Wilayah Layanan:')
# Ditambah info Cakupan Administratif di baris kedua
if sa_cov_text:
    tagline_html = f'<div class="header-tagline" title="Cakupan Administratif: {sa_cov_text}"><span style="color:#C084FC; font-weight:700;">🏛️ Cakupan:</span> <span style="color:#CBD5E1;">{sa_cov_text}</span></div>'
else:
    tagline_html = '<div class="header-tagline"><span style="color:#94A3B8;">RealFieldAI • Real Measurements, Intelligent Predictions</span></div>'

# Right Header: Telemetry Badges (Kanal Frekuensi DVB-T2, Status Model AI R², dan Data TX/Titik)
badge_chan = f'<span class="cloudrf-pill pill-channels" title="Alokasi Kanal Frekuensi DVB-T2">📺 Kanal: CH {sa_chan_text}</span>' if sa_chan_text else ''

# Executive Toolbar Header (Tersebar merata: Identitas & Cakupan di Kiri, Telemetri di Kanan)
st.markdown(f"""
<div class="cloudrf-header">
    <div class="header-left">
        <div class="header-brand">📍 {sa_clean_name}</div>
        {tagline_html}
    </div>
    <div class="header-pills-row">
        {badge_chan}
        {badge_model}
        {badge_data}
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. PRIMARY MAP CANVAS (ALWAYS ON TOP)
# ==============================================================================
# Handle action triggers for Coverage & Single Point prior to rendering map
if not is_new_region_mode:
    # Trigger: Run Coverage Grid
    if active_view == "🗺️ Simulasi Cakupan (Coverage Grid)" and 'btn_run_coverage' in locals() and btn_run_coverage:
        if not model_loaded or manager.active_model is None:
            st.error("⚠️ Model AI belum dilatih untuk wilayah ini. Silakan latih model pada mode 'Model AI & Retraining'.")
        elif df_tx_curr is None or len(df_tx_curr) == 0:
            st.error("⚠️ Data pemancar belum dimuat.")
        else:
            sel_tx_row = df_tx_curr[df_tx_curr['STN_NAME'] == cov_tx_selected].iloc[0].to_dict()
            sa_poly = current_sa['polygon'] if current_sa else None

            if 'cov_radius' not in locals() or cov_radius is None:
                from src.grid_engine import calculate_max_radio_horizon_km
                cov_radius = calculate_max_radio_horizon_km(
                    tx_params=sel_tx_row,
                    test_points=current_tps,
                    df_region=st.session_state.get('df_model')
                )

            with st.spinner(f"Membangkitkan peta coverage continuous raster {cov_resolution}m x {cov_resolution}m (Jangkauan Radio Horizon {cov_radius:.1f} km)..."):
                raster_overlay, df_grid_res, kpis = generate_continuous_coverage_raster(
                    model=manager.active_model,
                    tx_params=sel_tx_row,
                    radius_km=cov_radius,
                    resolution_m=float(cov_resolution),
                    rx_antenna_amsl=1.5,
                    thresholds={'sangat_kuat': 65.0, 'kuat': 54.0, 'sedang': 45.2},
                    strict_nlos_cutoff=cov_strict_nlos,
                    service_area_polygon=sa_poly
                )

                # Evaluasi Kepatuhan Batas Maksimum Titik Uji (Test Points Permenkominfo KML)
                tol_val = tp_tolerance if 'tp_tolerance' in locals() else 3.0
                df_tp_eval, tp_summary = src.kml_engine.evaluate_test_points_compliance(
                    model_manager=manager,
                    tx_params=sel_tx_row,
                    test_points=current_tps,
                    tolerance_db=tol_val,
                    rx_antenna_amsl=1.5,
                    strict_nlos_cutoff=cov_strict_nlos
                )

                st.session_state['active_coverage_grid'] = {
                    'raster_overlay_data': raster_overlay,
                    'df_grid_res': df_grid_res,
                    'kpis': kpis,
                    'stn_name': cov_tx_selected,
                    'radius_km': cov_radius,
                    'resolution_m': cov_resolution,
                    'effective_res_m': cov_resolution,
                    'strict_nlos': cov_strict_nlos,
                    'mode': 'raster',
                    'df_tp_eval': df_tp_eval,
                    'tp_summary': tp_summary,
                    'sa_name': sa_name_display if 'sa_name_display' in locals() else 'Wilayah Layanan KML'
                }
                st.success(f"✅ Peta cakupan raster berhasil dibangkitkan (Jangkauan radio horizon {cov_radius:.1f} km, resolusi {cov_resolution}m x {cov_resolution}m)!")

    # Trigger: Run Single Point Prediction
    if active_view == "🎯 Prediksi Titik Tunggal" and 'btn_run_sp' in locals() and btn_run_sp:
        if not model_loaded or manager.active_model is None:
            st.error("⚠️ Model AI belum dilatih untuk wilayah ini.")
        elif df_tx_curr is None or len(df_tx_curr) == 0:
            st.error("⚠️ Data pemancar belum dimuat.")
        else:
            sel_sp_tx_row = df_tx_curr[df_tx_curr['STN_NAME'] == sp_tx_selected].iloc[0].to_dict()
            single_res = manager.predict_single_point(
                tx_row=sel_sp_tx_row,
                rx_latitude=sp_rx_lat,
                rx_longitude=sp_rx_lon,
                rx_antenna_amsl=sp_rx_amsl,
                strict_nlos_cutoff=sp_strict_nlos
            )
            st.session_state['single_pred_result'] = single_res
            st.session_state['single_pred_ctx'] = {
                'stn_name': str(sel_sp_tx_row['STN_NAME']),
                'tx_lat': float(sel_sp_tx_row['SID_LAT']),
                'tx_lon': float(sel_sp_tx_row['SID_LONG']),
                'rx_lat': float(sp_rx_lat),
                'rx_lon': float(sp_rx_lon),
                'rx_amsl': float(sp_rx_amsl),
                'strict_nlos_cutoff': sp_strict_nlos
            }
            st.success("✅ Prediksi kuat medan titik tunggal berhasil dihitung!")


# Prepare layers for master map
df_tx_for_map = st.session_state.get('df_tx_summary')
df_dt_for_map = st.session_state.get('df_model')
cov_data_stored = st.session_state.get('active_coverage_grid')
df_grid_for_map = cov_data_stored['df_grid_res'] if cov_data_stored is not None else None
raster_for_map = cov_data_stored.get('raster_overlay_data') if cov_data_stored is not None else None

show_tx_on_map = True
show_dt_on_map = (active_view == "🚗 Data Hasil Ukur & Preprocessing" or active_view == "📊 Analisis Split & Stratifikasi")
show_cov_on_map = (active_view == "🗺️ Simulasi Cakupan (Coverage Grid)" and (df_grid_for_map is not None or raster_for_map is not None))

sp_ctx_map = st.session_state.get('single_pred_ctx') if active_view == "🎯 Prediksi Titik Tunggal" else None
sp_res_map = st.session_state.get('single_pred_result') if active_view == "🎯 Prediksi Titik Tunggal" else None

# If user is in Single Point mode and hasn't computed yet, create a lightweight ctx so target marker is visible on the map
if active_view == "🎯 Prediksi Titik Tunggal" and sp_ctx_map is None and 'sp_rx_lat' in locals():
    sel_sp_tx = df_tx_curr[df_tx_curr['STN_NAME'] == sp_tx_selected].iloc[0].to_dict() if len(tx_names_list) > 0 else {}
    sp_ctx_map = {
        'stn_name': str(sel_sp_tx.get('STN_NAME', 'TX')),
        'tx_lat': float(sel_sp_tx.get('SID_LAT', sp_rx_lat)),
        'tx_lon': float(sel_sp_tx.get('SID_LONG', sp_rx_lon)),
        'rx_lat': float(sp_rx_lat),
        'rx_lon': float(sp_rx_lon),
        'rx_amsl': float(sp_rx_amsl),
        'strict_nlos_cutoff': sp_strict_nlos
    }

highlight_stn = None
if active_view == "📡 Data Pemancar" and 'map_filter_stn' in locals():
    highlight_stn = map_filter_stn
elif active_view == "🚗 Data Hasil Ukur & Preprocessing" and 'map_filter_dt_stn' in locals():
    highlight_stn = map_filter_dt_stn
elif active_view == "🗺️ Simulasi Cakupan (Coverage Grid)" and cov_data_stored is not None:
    highlight_stn = cov_data_stored.get('stn_name')
elif active_view == "🎯 Prediksi Titik Tunggal" and sp_ctx_map is not None:
    highlight_stn = sp_ctx_map.get('stn_name')

# Fetch official DVB-T2 Service Area Boundary & Test Points (Masterplan KML)
kml_engine = src.kml_engine.get_masterplan_engine()
current_sa = kml_engine.find_service_area(region_code) or kml_engine.find_service_area(region_name)
current_tps = kml_engine.get_test_points_for_area(current_sa['norm_key']) if current_sa else []

# Render Folium Map
map_obj = create_cloudrf_gis_map(
    df_tx_summary=df_tx_for_map,
    df_matched_dt=df_dt_for_map,
    df_grid_result=df_grid_for_map if (cov_data_stored is None or cov_data_stored.get('mode') != 'raster') else None,
    raster_overlay_data=raster_for_map,
    selected_tx_name=highlight_stn,
    show_transmitters=show_tx_on_map,
    show_drive_test=show_dt_on_map,
    show_coverage_grid=show_cov_on_map,
    single_point_ctx=sp_ctx_map,
    single_point_res=sp_res_map,
    service_area_polygon=current_sa,
    service_area_testpoints=current_tps,
    test_point_evaluations=cov_data_stored.get('df_tp_eval') if cov_data_stored is not None else None,
    basemap_choice=chosen_basemap
)

# Render map directly in the primary top section
map_key = f"cloudrf_map_{region_code}_{active_view}_{chosen_basemap}_{highlight_stn}"
map_output = st_folium(
    map_obj,
    width="100%",
    height=580,
    returned_objects=["last_clicked"],
    key=map_key
)

# Handle interactive click-to-select point in Single Point mode
if active_view == "🎯 Prediksi Titik Tunggal" and map_output and map_output.get("last_clicked"):
    clicked = map_output["last_clicked"]
    c_lat = round(float(clicked["lat"]), 6)
    c_lon = round(float(clicked["lng"]), 6)
    if st.session_state.get('last_handled_map_click') != (c_lat, c_lon):
        st.session_state['last_handled_map_click'] = (c_lat, c_lon)
        st.session_state['sp_rx_lat_val'] = c_lat
        st.session_state['sp_rx_lon_val'] = c_lon
        st.session_state['rx_coord_version'] = st.session_state.get('rx_coord_version', 0) + 1

        if model_loaded and manager.active_model is not None and df_tx_curr is not None and len(df_tx_curr) > 0:
            sel_sp_tx_row = df_tx_curr[df_tx_curr['STN_NAME'] == sp_tx_selected].iloc[0].to_dict()
            single_res = manager.predict_single_point(
                tx_row=sel_sp_tx_row,
                rx_latitude=c_lat,
                rx_longitude=c_lon,
                rx_antenna_amsl=sp_rx_amsl,
                strict_nlos_cutoff=sp_strict_nlos
            )
            st.session_state['single_pred_result'] = single_res
            st.session_state['single_pred_ctx'] = {
                'stn_name': str(sel_sp_tx_row['STN_NAME']),
                'tx_lat': float(sel_sp_tx_row['SID_LAT']),
                'tx_lon': float(sel_sp_tx_row['SID_LONG']),
                'rx_lat': float(c_lat),
                'rx_lon': float(c_lon),
                'rx_amsl': float(sp_rx_amsl),
                'strict_nlos_cutoff': sp_strict_nlos
            }
        else:
            sel_sp_tx_row = df_tx_curr[df_tx_curr['STN_NAME'] == sp_tx_selected].iloc[0].to_dict() if df_tx_curr is not None and len(df_tx_curr) > 0 else {}
            st.session_state['single_pred_ctx'] = {
                'stn_name': str(sel_sp_tx_row.get('STN_NAME', 'TX')),
                'tx_lat': float(sel_sp_tx_row.get('SID_LAT', c_lat)),
                'tx_lon': float(sel_sp_tx_row.get('SID_LONG', c_lon)),
                'rx_lat': float(c_lat),
                'rx_lon': float(c_lon),
                'rx_amsl': float(sp_rx_amsl),
                'strict_nlos_cutoff': sp_strict_nlos
            }
        st.toast(f"📍 Titik Penerima (Rx) Terpilih: {c_lat:.5f}, {c_lon:.5f}")
        st.rerun()


# ==============================================================================
# 2. FLEXIBLE SPLIT SCREEN PANEL (BELOW MAP)
# ==============================================================================
col_toggle, col_info = st.columns([1.2, 2.8])
with col_toggle:
    show_detail_panel = st.toggle("📐 Tampilkan Panel Detail di Bawah Peta", value=True)
with col_info:
    st.caption(f"Mode Aktif: **{active_view}** • Wilayah: **{region_name}** • Peta interaktif selalu aktif di atas.")

st.markdown("---")

if show_detail_panel:
    # --------------------------------------------------------------------------
    # A. WIZARD: BUAT WILAYAH BARU (Sequential Guided Workflow)
    # --------------------------------------------------------------------------
    if is_new_region_mode:
        st.subheader("➕ Penambahan Wilayah Layanan Siaran Baru (Wizard Terurut)")
        st.markdown("""
        Ikuti tahapan di bawah ini secara berurutan untuk mendaftarkan wilayah layanan baru:
        1. Masukkan nama wilayah siaran.
        2. Unggah data spesifikasi pemancar (`.csv`/`.xlsx`).
        3. Unggah satu atau beberapa berkas hasil pengukuran *drive test*.
        4. Sistem akan menyelaraskan data, menghitung EIRP, membagi partisi split, dan melatih model AI.
        """)

        w_col1, w_col2 = st.columns([1.2, 1.8])
        with w_col1:
            st.markdown("##### Langkah 1: Identitas Wilayah")
            new_reg_name_input = st.text_input("Nama Wilayah Layanan Baru:", value="Jawa Barat 1", help="Contoh: Jawa Barat 1, Surabaya - Jatim 1, Medan - Sumut 1")
            clean_new_code = "".join(c if c.isalnum() else "_" for c in new_reg_name_input.strip()).lower()
            st.caption(f"ID Kode Sistem: `{clean_new_code}`")

            new_upload_tx = st.file_uploader("Langkah 2: Berkas Pemancar (CSV/XLSX):", type=['csv', 'xlsx'], key="new_tx_up")
            new_upload_dt = st.file_uploader("Langkah 3: Berkas Drive Test (Bisa Multi-Berkas):", type=['csv', 'xlsx'], accept_multiple_files=True, key="new_dt_up")

            btn_create_region = st.button("🚀 Proses & Simpan Basis Data Wilayah Baru", type="primary", use_container_width=True)

        with w_col2:
            st.markdown("##### 💡 Petunjuk Format Berkas")
            st.info("""
            * **Berkas Pemancar:** Harus memuat kolom Nama Pemancar (`STN_NAME`), Frekuensi MHz (`FREQ_MHZ`), Koordinat (`SID_LAT`, `SID_LONG`), dan Tinggi Antena mdpl (`H_ANT`).
            * **Berkas Drive Test:** Harus memuat kolom Frekuensi Hz (`Frequency_Hz`), Level Sinyal (`Level_dBuV_m`), dan Koordinat (`Latitude`, `Longitude`).
            """)

            if btn_create_region:
                if not new_upload_tx or not new_upload_dt:
                    st.error("❌ Harap unggah berkas pemancar dan minimal satu berkas drive test terlebih dahulu!")
                else:
                    with st.spinner("Memproses, membersihkan duplikasi, dan menyelaraskan wilayah baru..."):
                        try:
                            df_raw_tx_new = load_dataframe_safely(new_upload_tx)
                            df_raw_dt_new = pd.concat([load_dataframe_safely(f) for f in new_upload_dt], ignore_index=True)

                            df_clean_new, df_tx_sum_new, stats_new = process_and_align_datasets(df_raw_tx_new, df_raw_dt_new)
                            stats_new['region_name'] = new_reg_name_input.strip()

                            save_region_processed_data(
                                data_dir=DATA_DIR,
                                region_code=clean_new_code,
                                df_clean=df_clean_new,
                                df_tx_sum=df_tx_sum_new,
                                stats=stats_new
                            )
                            st.session_state['active_stored_region'] = clean_new_code
                            st.success(f"🎉 Basis data untuk wilayah **{new_reg_name_input.strip()}** berhasil dibuat dengan {len(df_tx_sum_new)} pemancar dan {len(df_clean_new):,} titik ukur!")
                            st.rerun()
                        except Exception as err:
                            st.error(f"❌ Gagal memproses berkas: {err}")

    # --------------------------------------------------------------------------
    # B. MODE 1: DATA PEMANCAR (TRANSMITTER)
    # --------------------------------------------------------------------------
    elif active_view == "📡 Data Pemancar":
        st.subheader("📡 Manajemen & Parameter Pemancar Wilayah")
        st.markdown("Tabel ringkasan spesifikasi teknis pemancar DVB-T2 beserta nilai **$\text{EIRP (dBm)}$** empiris terhitung.")

        if df_tx_curr is None or len(df_tx_curr) == 0:
            st.warning("⚠️ Data pemancar belum tersedia untuk wilayah ini.")
        else:
            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Pemancar", f"{len(df_tx_curr)} Menara")
            f_min = df_tx_curr['FREQ_MHZ'].min()
            f_max = df_tx_curr['FREQ_MHZ'].max()
            m2.metric("Rentang Frekuensi", f"{f_min:.1f} – {f_max:.1f} MHz")
            m3.metric("EIRP Tertinggi", f"{df_tx_curr['EIRP_dBm'].max():.2f} dBm")
            m4.metric("Rata-rata Tinggi Antena", f"{df_tx_curr['H_ANT_masl'].mean():.1f} mdpl")

            st.markdown("#### 🛠️ Editor Parameter Pemancar (Live Data Editor)")
            st.caption("Anda dapat mengedit koordinat, tinggi antena, frekuensi, atau EIRP secara langsung pada tabel. Anda juga dapat menambah baris baru di bawah.")

            edited_tx_table = st.data_editor(
                df_tx_curr,
                num_rows="dynamic",
                use_container_width=True,
                key=f"editor_tx_{region_code}",
                column_config={
                    "STN_NAME": st.column_config.TextColumn("Nama Pemancar / Stasiun", required=True),
                    "FREQ_MHZ": st.column_config.NumberColumn("Frekuensi (MHz)", format="%.1f", required=True),
                    "SID_LAT": st.column_config.NumberColumn("Latitude", format="%.6f", required=True),
                    "SID_LONG": st.column_config.NumberColumn("Longitude", format="%.6f", required=True),
                    "H_ANT_masl": st.column_config.NumberColumn("Tinggi Antena (m AMSL)", format="%.1f", required=True),
                    "EIRP_dBm": st.column_config.NumberColumn("EIRP (dBm)", format="%.2f", required=True),
                    "Horizon_TX_km": st.column_config.NumberColumn("Radio Horizon (km)", format="%.1f", disabled=True),
                    "n_measurements": st.column_config.NumberColumn("Jumlah Titik", disabled=True)
                }
            )

            col_btn_tx1, col_btn_tx2, col_btn_tx3 = st.columns([1.5, 1.5, 2])
            with col_btn_tx1:
                if st.button("💾 Simpan Perubahan Pemancar", type="primary", use_container_width=True):
                    try:
                        edited_tx_table['FREQ_MHZ'] = pd.to_numeric(edited_tx_table['FREQ_MHZ'], errors='coerce').fillna(500.0)
                        edited_tx_table['FREQ_HZ'] = (edited_tx_table['FREQ_MHZ'] * 1e6).round().astype(np.int64)
                        edited_tx_table['SID_LAT'] = pd.to_numeric(edited_tx_table['SID_LAT'], errors='coerce')
                        edited_tx_table['SID_LONG'] = pd.to_numeric(edited_tx_table['SID_LONG'], errors='coerce')
                        edited_tx_table['H_ANT_masl'] = pd.to_numeric(edited_tx_table['H_ANT_masl'], errors='coerce').fillna(50.0)
                        edited_tx_table['EIRP_dBm'] = pd.to_numeric(edited_tx_table['EIRP_dBm'], errors='coerce').fillna(70.0)
                        edited_tx_table['Horizon_TX_km'] = [calculate_radio_horizon(float(h), 0.0) for h in edited_tx_table['H_ANT_masl']]

                        save_region_transmitters(DATA_DIR, region_code, edited_tx_table)
                        st.session_state['df_tx_summary'] = edited_tx_table
                        st.success("✅ Perubahan parameter pemancar berhasil disimpan! Peta dan simulasi langsung diperbarui.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Gagal menyimpan pemancar: {e}")

            with col_btn_tx2:
                csv_tx_exp = df_tx_curr.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Unduh Ringkasan Pemancar (CSV)",
                    data=csv_tx_exp,
                    file_name=f"dvbt2_transmitters_{region_code}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # Option to upload additional transmitter CSV file
            with st.expander("📁 Unggah Berkas Pemancar Tambahan (CSV/XLSX)"):
                up_more_tx = st.file_uploader("Pilih Berkas Pemancar Tambahan:", type=['csv', 'xlsx'], key="up_more_tx_file")
                if up_more_tx and st.button("➕ Gabungkan Berkas Pemancar Ini"):
                    try:
                        df_extra_tx = load_dataframe_safely(up_more_tx)
                        # Normalize columns
                        for std, keys in [('STN_NAME', ['NAME', 'STN']), ('FREQ_MHZ', ['FREQ', 'MHZ']), ('SID_LAT', ['LAT']), ('SID_LONG', ['LON']), ('H_ANT_masl', ['ANT', 'HEIGHT', 'MASL'])]:
                            for c in df_extra_tx.columns:
                                if any(k in str(c).upper() for k in keys):
                                    df_extra_tx.rename(columns={c: std}, inplace=True)
                                    break
                        df_merged_tx = pd.concat([df_tx_curr, df_extra_tx], ignore_index=True).drop_duplicates(subset=['STN_NAME', 'FREQ_MHZ'])
                        df_merged_tx['FREQ_HZ'] = (df_merged_tx['FREQ_MHZ'] * 1e6).round().astype(np.int64)
                        df_merged_tx['Horizon_TX_km'] = [calculate_radio_horizon(float(h), 0.0) for h in df_merged_tx['H_ANT_masl']]
                        if 'EIRP_dBm' not in df_merged_tx.columns:
                            df_merged_tx['EIRP_dBm'] = 65.0
                        save_region_transmitters(DATA_DIR, region_code, df_merged_tx)
                        st.session_state['df_tx_summary'] = df_merged_tx
                        st.success("✅ Pemancar berhasil ditambahkan!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"❌ Gagal menggabungkan: {err}")

    # --------------------------------------------------------------------------
    # C. MODE 2: DATA HASIL UKUR & PREPROCESSING
    # --------------------------------------------------------------------------
    elif active_view == "🚗 Data Hasil Ukur & Preprocessing":
        st.subheader("🚗 Data Hasil Pengukuran Lapangan (Drive Test)")
        df_clean = st.session_state.get('df_model')
        stats = st.session_state.get('stats', {})

        if df_clean is None or len(df_clean) == 0:
            st.warning("⚠️ Data hasil ukur belum tersedia untuk wilayah ini.")
        else:
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Mentah Drive Test", f"{stats.get('n_raw_drivetest', len(df_clean)):,} baris")
            m2.metric("Eliminasi Duplikasi Persis", f"{stats.get('n_after_exact_dedup', len(df_clean)):,} baris")
            m3.metric("Agregasi Titik Macet (Local Mean)", f"{stats.get('n_after_stationary_agg', len(df_clean)):,} baris")
            m4.metric("Data Siap Training (Matched)", f"{len(df_clean):,} titik", delta=f"{len(df_tx_curr)} Pemancar")

            # Ingestion expander for new drive test files
            with st.expander("📥 Tambah Berkas Pengukuran Lapangan Baru (Drive Test Multi-File)", expanded=False):
                st.caption("Unggah berkas hasil ukur terbaru untuk menambahkan titik baru atau memperbarui dataset wilayah ini.")
                up_dt_new = st.file_uploader("Unggah Berkas Drive Test (.csv / .xlsx):", type=['csv', 'xlsx'], accept_multiple_files=True, key="up_dt_in_split")

                if up_dt_new:
                    c_m1, c_m2 = st.columns(2)
                    with c_m1:
                        btn_inc_merge = st.button("➕ Gabungkan dengan Data Eksisting (Incremental Merge)", type="primary", use_container_width=True)
                    with c_m2:
                        btn_ovr_replace = st.button("🔄 Ganti Seluruh Data Ukur (Replace/Overwrite)", type="secondary", use_container_width=True)

                    if btn_inc_merge:
                        with st.spinner("Menyelaraskan data baru dan menggabungkan..."):
                            df_concat_new = pd.concat([load_dataframe_safely(f) for f in up_dt_new], ignore_index=True)
                            df_tx_align = df_tx_curr.copy()
                            if 'H_ANT_masl' in df_tx_align.columns and 'H_ANT' not in df_tx_align.columns:
                                df_tx_align['H_ANT'] = df_tx_align['H_ANT_masl']
                            df_new_c, _, stats_n = process_and_align_datasets(df_tx_align, df_concat_new)
                            df_merged, stats_merged = append_region_drive_test_data(DATA_DIR, region_code, df_new_c)
                            stats_merged['region_name'] = region_name
                            st.session_state['df_model'] = df_merged
                            st.session_state['stats'] = stats_merged
                            for k in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'summary_check', 'fig_dist', 'df_split_full']:
                                st.session_state.pop(k, None)
                            st.success(f"🎉 Berhasil digabungkan! Total titik ukur saat ini: **{len(df_merged):,}** titik. Buka menu 'Model AI & Retraining' untuk melatih ulang model.")
                            st.rerun()

                    if btn_ovr_replace:
                        with st.spinner("Mengganti dataset ukur lama dengan berkas baru..."):
                            df_concat_new = pd.concat([load_dataframe_safely(f) for f in up_dt_new], ignore_index=True)
                            df_tx_align = df_tx_curr.copy()
                            if 'H_ANT_masl' in df_tx_align.columns and 'H_ANT' not in df_tx_align.columns:
                                df_tx_align['H_ANT'] = df_tx_align['H_ANT_masl']
                            df_new_c, df_tx_n, stats_n = process_and_align_datasets(df_tx_align, df_concat_new)
                            stats_n['region_name'] = region_name
                            save_region_processed_data(DATA_DIR, region_code, df_new_c, df_tx_n, stats_n)
                            st.session_state['df_model'] = df_new_c
                            st.session_state['df_tx_summary'] = df_tx_n
                            st.session_state['stats'] = stats_n
                            for k in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'summary_check', 'fig_dist', 'df_split_full']:
                                st.session_state.pop(k, None)
                            st.success("🔄 Dataset ukur wilayah ini berhasil diganti sepenuhnya!")
                            st.rerun()

            # Data table
            c_f1, c_f2 = st.columns([1.5, 2.5])
            with c_f1:
                filter_stn_table = st.selectbox("Saring Tampilan Berdasarkan Stasiun:", ["Semua Stasiun"] + list(df_tx_curr['STN_NAME'].unique()), key="filter_dt_stn_table")
            with c_f2:
                csv_clean_exp = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh Dataset Preprocessed Siap AI (CSV)", data=csv_clean_exp, file_name=f"dvbt2_preprocessed_{region_code}.csv", mime="text/csv")

            df_tbl_show = df_clean.copy() if filter_stn_table == "Semua Stasiun" else df_clean[df_clean['STN_NAME'] == filter_stn_table].copy()

            # Ensure azimuth_deg, azimuth_sin, and azimuth_cos exist
            if 'azimuth_deg' not in df_tbl_show.columns and 'azimuth_sin' in df_tbl_show.columns and 'azimuth_cos' in df_tbl_show.columns:
                df_tbl_show['azimuth_deg'] = (np.degrees(np.arctan2(df_tbl_show['azimuth_sin'], df_tbl_show['azimuth_cos'])) + 360) % 360
            elif 'azimuth_deg' not in df_tbl_show.columns and all(c in df_tbl_show.columns for c in ['SID_LAT', 'SID_LONG', 'Latitude', 'Longitude']):
                deg, s, c = calculate_azimuth_bearing(df_tbl_show['SID_LAT'].values, df_tbl_show['SID_LONG'].values, df_tbl_show['Latitude'].values, df_tbl_show['Longitude'].values)
                df_tbl_show['azimuth_deg'] = deg
                df_tbl_show['azimuth_sin'] = s
                df_tbl_show['azimuth_cos'] = c

            show_cols = [
                'STN_NAME', 'frequency_mhz', 'distance_km', 'azimuth_deg',
                'azimuth_sin', 'azimuth_cos', 'clearance_m', 'eirp_dbm',
                'field_strength_dbuv_m', 'Latitude', 'Longitude',
                'tx_antenna_amsl', 'rx_antenna_amsl', 'SID_LAT', 'SID_LONG', 'frequency_hz'
            ]
            valid_show_cols = [c for c in show_cols if c in df_tbl_show.columns]

            st.dataframe(
                df_tbl_show[valid_show_cols],
                use_container_width=True,
                height=350,
                column_config={
                    "STN_NAME": st.column_config.TextColumn("Pemancar"),
                    "frequency_mhz": st.column_config.NumberColumn("Frekuensi (MHz)", format="%.2f"),
                    "distance_km": st.column_config.NumberColumn("Jarak (km)", format="%.3f"),
                    "azimuth_deg": st.column_config.NumberColumn("Azimuth (°)", format="%.2f"),
                    "azimuth_sin": st.column_config.NumberColumn("Azimuth Sin", format="%.4f"),
                    "azimuth_cos": st.column_config.NumberColumn("Azimuth Cos", format="%.4f"),
                    "clearance_m": st.column_config.NumberColumn("Clearance (m)", format="%.2f"),
                    "eirp_dbm": st.column_config.NumberColumn("EIRP (dBm)", format="%.2f"),
                    "field_strength_dbuv_m": st.column_config.NumberColumn("Level (dBµV/m)", format="%.2f"),
                    "Latitude": st.column_config.NumberColumn("Latitude Rx", format="%.6f"),
                    "Longitude": st.column_config.NumberColumn("Longitude Rx", format="%.6f"),
                    "tx_antenna_amsl": st.column_config.NumberColumn("Tx AMSL (m)", format="%.1f"),
                    "rx_antenna_amsl": st.column_config.NumberColumn("Rx AMSL (m)", format="%.1f")
                }
            )
            st.caption(f"Menampilkan {len(df_tbl_show):,} baris data pengukuran lapangan lengkap dengan fitur spasial (Jarak, Azimuth sudut & proyeksi trigonometris, Terrain Clearance medan). Target pelatihan AI: `field_strength_dbuv_m`.")

    # --------------------------------------------------------------------------
    # D. MODE 3: ANALISIS STRATIFIED SPLIT
    # --------------------------------------------------------------------------
    elif active_view == "📊 Analisis Split & Stratifikasi":
        st.subheader("📊 Analisis Multi-Feature Stratified Split (70% Train, 15% Val, 15% Test)")
        df_clean = st.session_state.get('df_model')

        if df_clean is None or len(df_clean) == 0:
            st.warning("⚠️ Data belum diproses. Silakan muat data pada mode 'Data Hasil Ukur'.")
        else:
            btn_run_split = st.button("📊 Jalankan Multi-Feature Stratified Split (70:15:15)", type="primary")

            if btn_run_split or 'X_train' not in st.session_state:
                with st.spinner("Menjalankan stratifikasi multi-fitur & kuantisasi kuantil..."):
                    X_tr, y_tr, X_va, y_va, X_te, y_te, sum_check, fig_dist = perform_multi_feature_stratified_split(df_clean)
                    st.session_state['X_train'] = X_tr
                    st.session_state['y_train'] = y_tr
                    st.session_state['X_val'] = X_va
                    st.session_state['y_val'] = y_va
                    st.session_state['X_test'] = X_te
                    st.session_state['y_test'] = y_te
                    st.session_state['summary_check'] = sum_check
                    st.session_state['fig_dist'] = fig_dist

                    # Build split full dataset and persist
                    df_sp_full = df_clean.copy()
                    df_sp_full['split_dataset'] = 'Unassigned'
                    df_sp_full.loc[X_tr.index, 'split_dataset'] = 'Train (70%)'
                    df_sp_full.loc[X_va.index, 'split_dataset'] = 'Validation (15%)'
                    df_sp_full.loc[X_te.index, 'split_dataset'] = 'Test (15%)'
                    st.session_state['df_split_full'] = df_sp_full
                    save_region_split_data(DATA_DIR, region_code, df_sp_full)

            X_tr = st.session_state['X_train']
            X_va = st.session_state['X_val']
            X_te = st.session_state['X_test']
            sum_check = st.session_state.get('summary_check')
            fig_dist = st.session_state.get('fig_dist')

            if sum_check is None:
                sum_check = compute_summary_check_table(X_tr, X_va, X_te)
                st.session_state['summary_check'] = sum_check

            if fig_dist is None and all(k in st.session_state for k in ['y_train', 'y_val', 'y_test']):
                fig_dist = generate_split_distribution_figure(
                    X_tr, st.session_state['y_train'],
                    X_va, st.session_state['y_val'],
                    X_te, st.session_state['y_test']
                )
                st.session_state['fig_dist'] = fig_dist

            # Metric Cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Train Set (70%)", f"{len(X_tr):,} baris")
            c2.metric("Validation Set (15%)", f"{len(X_va):,} baris")
            c3.metric("Test Set (15%)", f"{len(X_te):,} baris")
            c4.metric("Total Terstratifikasi", f"{len(X_tr) + len(X_va) + len(X_te):,} baris")

            # Sanity Check Table
            st.markdown("#### 📋 Tabel Verifikasi Statistik (Sanity Check)")
            st.dataframe(sum_check, use_container_width=True)

            # 8-Panel Diagnostic Plot
            if fig_dist is not None:
                st.markdown("#### 📈 Visualisasi Distribusi Multi-Fitur Antar Partisi")
                st.pyplot(fig_dist, use_container_width=True)

            if 'df_split_full' in st.session_state:
                csv_sp = st.session_state['df_split_full'].to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh Dataset Partisi Lengkap (CSV)", data=csv_sp, file_name=f"dvbt2_stratified_split_{region_code}.csv", mime="text/csv")

    # --------------------------------------------------------------------------
    # E. MODE 4: MODEL AI & RETRAINING
    # --------------------------------------------------------------------------
    elif active_view == "🧠 Model AI & Retraining":
        st.subheader("🧠 Model Propagasi AI (XGBoost Regressor + Optuna HPO)")
        
        # Check trigger from sidebar
        if 'btn_retrain_ai' in locals() and btn_retrain_ai:
            if 'X_train' not in st.session_state:
                st.error("⚠️ Partisi split belum tersedia. Silakan jalankan Split pada mode 'Analisis Split & Stratifikasi' terlebih dahulu.")
            else:
                progress_bar = st.progress(0.0)
                status_txt = st.empty()

                def update_optuna_ui(trial_num, val_rmse, params):
                    pct = (trial_num + 1) / optuna_trials_input
                    progress_bar.progress(min(pct, 1.0))
                    status_txt.text(f"Iterasi Optuna {trial_num + 1}/{optuna_trials_input} | Best Val RMSE: {val_rmse:.4f} dBµV/m")

                with st.spinner(f"Menjalankan optimasi Bayesian Optuna ({optuna_trials_input} trials) & melatih model final..."):
                    best_params, study = manager.run_optuna_hpo(
                        X_train=st.session_state['X_train'],
                        y_train=st.session_state['y_train'],
                        X_val=st.session_state['X_val'],
                        y_val=st.session_state['y_val'],
                        n_trials=optuna_trials_input,
                        trial_callback=update_optuna_ui
                    )
                    final_xgb, metrics = manager.train_final_model(
                        X_train=st.session_state['X_train'],
                        y_train=st.session_state['y_train'],
                        X_val=st.session_state['X_val'],
                        y_val=st.session_state['y_val'],
                        X_test=st.session_state['X_test'],
                        y_test=st.session_state['y_test'],
                        best_params=best_params,
                        region_code=region_code
                    )

                progress_bar.progress(1.0)
                status_txt.text("Pelatihan Selesai!")
                st.success(f"🎉 Model AI untuk wilayah **{region_name}** berhasil dilatih dan disimpan ke disk lokal!")
                st.rerun()

        # Display Evaluation Performance
        if manager.active_model is not None and manager.active_metrics:
            metrics = manager.active_metrics
            st.markdown("#### 📊 Tabel Evaluasi Metrik Kinerja Model")
            df_metrics = pd.DataFrame([{
                "Model Arsitektur": "XGBoost Regressor",
                "Train RMSE": f"{metrics['train_rmse']:.4f}",
                "Val RMSE": f"{metrics['val_rmse']:.4f}",
                "Test RMSE (dBµV/m)": f"{metrics['test_rmse']:.4f}",
                "Test MAE (dBµV/m)": f"{metrics['test_mae']:.4f}",
                "Test R² Score": f"{metrics['test_r2']:.4f}"
            }])
            st.dataframe(df_metrics, use_container_width=True)

            p_c1, p_c2 = st.columns(2)
            with p_c1:
                if 'X_test' in st.session_state:
                    X_te = st.session_state['X_test']
                    booster = manager.active_model.get_booster()
                    expected_features = booster.feature_names if booster and booster.feature_names else MODEL_FEATURE_COLS
                    cols_to_use = [col for col in expected_features if col in X_te.columns]
                    y_test_pred = manager.active_model.predict(X_te[cols_to_use])
                    fig_par = plot_parity_scatter(
                        st.session_state['y_test'].values,
                        y_test_pred,
                        r2=metrics.get('test_r2', 0.0),
                        rmse=metrics.get('test_rmse', 0.0)
                    )
                    st.plotly_chart(fig_par, use_container_width=True)
                else:
                    st.info("Muat data split untuk menampilkan Parity Plot.")

            with p_c2:
                fig_imp = plot_feature_importance(metrics.get('feature_importances', {}))
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning(f"⚠️ Model untuk wilayah **{region_name}** belum dilatih. Gunakan tombol 'Mulai Pelatihan Ulang Model AI' di sidebar untuk melatih model.")

    # --------------------------------------------------------------------------
    # F. MODE 5: SIMULASI CAKUPAN (COVERAGE GRID)
    # --------------------------------------------------------------------------
    elif active_view == "🗺️ Simulasi Cakupan (Coverage Grid)":
        res_display = f"{cov_data_stored.get('resolution_m', 50)}m x {cov_data_stored.get('resolution_m', 50)}m" if cov_data_stored is not None else "100m x 100m / 50m x 50m / 10m x 10m"
        st.subheader(f"🗺️ Analisis & Simulasi Cakupan Sinyal ({res_display} - Peta Raster Telekomunikasi)")
        
        if cov_data_stored is None:
            st.info("💡 Tekan tombol **'🎨 Bangkitkan Peta Coverage'** di panel kiri (sidebar) untuk menghitung simulasi cakupan.")
        else:
            kpis = cov_data_stored['kpis']
            df_grid_res = cov_data_stored['df_grid_res']
            stn_sim = cov_data_stored['stn_name']
            scope_lbl = cov_data_stored.get('scope_label', 'Wilayah Layanan Resmi')
            sa_name = cov_data_stored.get('sa_name', 'Wilayah Layanan KML')

            st.caption(f"📍 **Evaluasi Wilayah Layanan:** `{sa_name}` • Pemancar: **{stn_sim}** • Tampilan Heatmap: *Pancaran Alami Gelombang (Dibatasi Radio Horizon, Sensitivitas ≥45.2 dBµV/m, & Kontur Topografi)*")

            # Section 1: Official Service Area Coverage Performance
            if kpis.get('has_service_area') and kpis.get('wl_total_area_km2', 0) > 0:
                st.markdown("#### 🏛️ Kinerja Cakupan Wilayah Layanan Resmi (Masterplan DVB-T2 KML)")
                st.caption(f"Evaluasi cakupan terhadap total luas wilayah layanan administratif **{sa_name}** ({kpis['wl_total_area_km2']:,.1f} km²). Hanya sel penerimaan di dalam batas poligon resmi yang dihitung.")

                # Row 1: Overarching Service Area KPIs
                w1, w2, w3, w4 = st.columns(4)
                w1.metric("Luas Poligon Wilayah Layanan", f"{kpis['wl_total_area_km2']:,.1f} km²")
                w2.metric("Luas Tercover di Wilayah", f"{kpis['wl_covered_area_km2']:,.1f} km²", delta="≥ 45.2 dBµV/m")
                w3.metric("Persentase Wilayah Tercover", f"{kpis['wl_pct_covered']:.2f}%", delta=f"{kpis['wl_covered_area_km2']:.1f} / {kpis['wl_total_area_km2']:.0f} km²")
                w4.metric("Blank Spot Wilayah Layanan", f"{kpis['wl_blank_area_km2']:,.1f} km²", delta=f"{kpis['wl_pct_blank']:.1f}%", delta_color="inverse")

                # Row 2: Per-Category Breakdown Strictly Inside Wilayah Layanan
                st.markdown("##### 📶 Distribusi Kategori Kuat Sinyal di dalam Wilayah Layanan")
                wc1, wc2, wc3, wc4 = st.columns(4)
                wc1.metric(
                    "🔴 Sangat Kuat (≥65)",
                    f"{kpis.get('wl_sangat_kuat_km2', 0):.1f} km²",
                    delta=f"{kpis.get('wl_pct_sk', 0):.2f}% dari Wilayah"
                )
                wc2.metric(
                    "🟠 Kuat (54-65)",
                    f"{kpis.get('wl_kuat_km2', 0):.1f} km²",
                    delta=f"{kpis.get('wl_pct_k', 0):.2f}% dari Wilayah"
                )
                wc3.metric(
                    "🔵 Sedang (45.2-54)",
                    f"{kpis.get('wl_sedang_km2', 0):.1f} km²",
                    delta=f"{kpis.get('wl_pct_s', 0):.2f}% dari Wilayah"
                )
                wc4.metric(
                    "⚪ Blank Spot (<45.2)",
                    f"{kpis.get('wl_blank_area_km2', 0):.1f} km²",
                    delta=f"{kpis.get('wl_pct_blank', 0):.2f}% (Belum Tercover)",
                    delta_color="inverse"
                )

                # Diagnostic Summary Table for Wilayah Layanan Categories
                df_wl_cats = pd.DataFrame([
                    {
                        "Kategori Sinyal": "🔴 Sangat Kuat",
                        "Ambang Kuat Medan": "≥ 65.0 dBµV/m",
                        "Luas di Wilayah Layanan (km²)": f"{kpis.get('wl_sangat_kuat_km2', 0):.1f}",
                        "% thd Total Wilayah Layanan": f"{kpis.get('wl_pct_sk', 0):.2f}%",
                        "% Komposisi Area Tercover": f"{kpis.get('wl_comp_pct_sk', 0):.1f}%",
                        "Karakteristik & Target Penerimaan": "Antena Portabel / Indoor Tanpa Bantuan"
                    },
                    {
                        "Kategori Sinyal": "🟠 Kuat",
                        "Ambang Kuat Medan": "54.0 - 64.9 dBµV/m",
                        "Luas di Wilayah Layanan (km²)": f"{kpis.get('wl_kuat_km2', 0):.1f}",
                        "% thd Total Wilayah Layanan": f"{kpis.get('wl_pct_k', 0):.2f}%",
                        "% Komposisi Area Tercover": f"{kpis.get('wl_comp_pct_k', 0):.1f}%",
                        "Karakteristik & Target Penerimaan": "Antena Outdoor Standar Perkotaan"
                    },
                    {
                        "Kategori Sinyal": "🔵 Sedang",
                        "Ambang Kuat Medan": "45.2 - 53.9 dBµV/m",
                        "Luas di Wilayah Layanan (km²)": f"{kpis.get('wl_sedang_km2', 0):.1f}",
                        "% thd Total Wilayah Layanan": f"{kpis.get('wl_pct_s', 0):.2f}%",
                        "% Komposisi Area Tercover": f"{kpis.get('wl_comp_pct_s', 0):.1f}%",
                        "Karakteristik & Target Penerimaan": "Batas Minimum DVB-T2 (Antena Outdoor Bertingkat)"
                    },
                    {
                        "Kategori Sinyal": "⚪ Blank Spot",
                        "Ambang Kuat Medan": "< 45.2 dBµV/m",
                        "Luas di Wilayah Layanan (km²)": f"{kpis.get('wl_blank_area_km2', 0):.1f}",
                        "% thd Total Wilayah Layanan": f"{kpis.get('wl_pct_blank', 0):.2f}%",
                        "% Komposisi Area Tercover": "-",
                        "Karakteristik & Target Penerimaan": "Perlu Pemancar Pengisi Sinyal (Gap Filler / SFN Repeater)"
                    }
                ])
                st.dataframe(df_wl_cats, use_container_width=True, hide_index=True)

                st.info(f"💡 **Catatan Rekayasa Siaran:** Wilayah Layanan **{sa_name}** memiliki luas total **{kpis['wl_total_area_km2']:,.1f} km²** yang membentang lintas kabupaten/kota. Satu menara pemancar tunggal (**{stn_sim}**) melayani **{kpis['wl_covered_area_km2']:.1f} km²** (**{kpis['wl_pct_covered']:.2f}%** dari total provinsi). Sisa wilayah layanan (**{kpis['wl_blank_area_km2']:,.1f} km²**) secara regulasi dialokasikan untuk dilayani oleh menara pemancar transmisi lain (*transposer / SFN repeater*) di kabupaten/kota lainnya dalam wilayah layanan yang sama.")

            # Section 2: Physical Transmitter Emission Footprint
            st.markdown("#### 📡 Jangkauan Efektif Pancaran Menara Pemancar (Transmitter Footprint)")
            st.caption("Total area permukaan bumi yang terjangkau pancaran gelombang elektromagnetik dari menara pemancar ini (termasuk sinyal yang melintasi perbatasan wilayah secara alami).")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Area Tercover Pancaran", f"{kpis.get('tx_total_footprint_km2', kpis['covered_area_km2']):.1f} km²", delta="Pancaran Alami")
            k2.metric("Sangat Kuat (≥65 dBµV/m)", f"{kpis.get('tx_sangat_kuat_km2', 0):.1f} km²", delta=f"{kpis.get('tx_pct_sk', 0):.1f}%")
            k3.metric("Kuat (54-65 dBµV/m)", f"{kpis.get('tx_kuat_km2', 0):.1f} km²", delta=f"{kpis.get('tx_pct_k', 0):.1f}%")
            k4.metric("Sedang (45.2-54 dBµV/m)", f"{kpis.get('tx_sedang_km2', 0):.1f} km²", delta=f"{kpis.get('tx_pct_s', 0):.1f}%")

            if kpis.get('n_cut_by_nlos', 0) > 0:
                st.warning(f"✂️ **Strict NLoS Cutoff:** Sebanyak **{kpis['n_cut_by_nlos']:,} area** terhalang kontur perbukitan dipotong mutlak menjadi Blank Spot (Transparan).")

            # Section 3: Official Test Points Regulatory Compliance (Maximum Permissible Limit)
            df_tp_eval = cov_data_stored.get('df_tp_eval')
            tp_summary = cov_data_stored.get('tp_summary')

            if df_tp_eval is not None and len(df_tp_eval) > 0 and tp_summary:
                st.markdown("---")
                st.markdown("#### 🛡️ Evaluasi Kepatuhan Koordinasi Titik Uji (Test Points Masterplan KML)")
                st.caption(f"Evaluasi **Batas Maksimum Kuat Medan yang Diperbolehkan** di titik-titik koordinasi perbatasan sesuai Permenkominfo No. 6/2019 dengan toleransi koordinasi **+{tp_summary['tolerance_db']:.1f} dB**. Jika sinyal prediksi melebihi batas acuan, dinilai melanggar karena dapat mengakibatkan interferensi di wilayah layanan sebelahnya.")

                safe_c = tp_summary.get('safe_count', tp_summary.get('pass_count', 0))
                tol_c = tp_summary.get('tolerable_count', 0)
                viol_c = tp_summary.get('violation_count', tp_summary.get('fail_count', 0))

                tp_c1, tp_c2, tp_c3, tp_c4, tp_c5 = st.columns(5)
                tp_c1.metric("Total Titik Uji", f"{tp_summary['total_points']} Titik")
                tp_c2.metric("🟢 Aman / Patuh (≤ Maks)", f"{safe_c} Titik", delta="Bebas Interferensi")
                tp_c3.metric(f"🟡 Dalam Toleransi (+{tp_summary['tolerance_db']:.0f} dB)", f"{tol_c} Titik", delta="Marginal Aman")
                tp_c4.metric("🔴 Melanggar (> Maks)", f"{viol_c} Titik", delta="Potensi Interferensi", delta_color="inverse")
                tp_c5.metric("Tingkat Kepatuhan", f"{tp_summary['compliance_rate_pct']:.1f}%", delta=f"Margin Rata-rata: {tp_summary['avg_margin_db']:+.1f} dB")

                cols_tp_view = ['Nama Titik Uji', 'Jarak ke TX (km)', 'Clearance Medan (m)', 'Batas Maksimum Acuan (dBµV/m)', 'Ambang Batas Maks (+3 dB)', 'Prediksi Kuat Medan AI (dBµV/m)', 'Margin thd Batas Maks (dB)', 'Status Koordinasi']
                cols_to_show = [c for c in cols_tp_view if c in df_tp_eval.columns]

                # Format Display with (Est.) suffix for Blank Spot / Obstructed / Horizon points
                df_tp_display = df_tp_eval.copy()
                if 'Prediksi Kuat Medan AI (dBµV/m)' in df_tp_display.columns:
                    df_tp_display['Prediksi Kuat Medan AI (dBµV/m)'] = df_tp_display.apply(
                        lambda r: f"{float(r['Prediksi Kuat Medan AI (dBµV/m)']):.1f} (Est.)" if r.get('is_estimated') else f"{float(r['Prediksi Kuat Medan AI (dBµV/m)']):.1f}",
                        axis=1
                    )
                st.dataframe(df_tp_display[cols_to_show], use_container_width=True)

                if viol_c > 0:
                    st.error(f"⚠️ **Peringatan Regulasi Koordinasi:** Terdapat **{viol_c} titik uji perbatasan** yang melanggar batas maksimum kekuatan sinyal yang diperbolehkan. Kekuatan sinyal yang bocor ke wilayah tetangga berpotensi menimbulkan interferensi ko-kanal/kanal-berdampingan. Operator disarankan untuk menurunkan EIRP atau mengatur *down-tilt* antena ke arah tersebut.")
                else:
                    st.success("✅ **Regulasi Koordinasi Terpenuhi:** Seluruh titik uji perbatasan berada di bawah batas maksimum yang diperbolehkan (atau dalam batas toleransi koordinasi). Pancaran pemancar aman dan tidak menimbulkan gangguan interferensi ke wilayah layanan sebelahnya.")

            st.markdown("---")
            st.markdown("#### 📊 Sampel Data Titik Grid Cakupan")
            # Table of Grid Results
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                csv_grid = df_grid_res.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh Data Lengkap Grid Cakupan (CSV)", data=csv_grid, file_name=f"coverage_{stn_sim}.csv", mime="text/csv", use_container_width=True)
            with c_d2:
                df_blank = df_grid_res[~df_grid_res['is_covered']]
                csv_blank = df_blank.to_csv(index=False).encode('utf-8')
                st.download_button(f"📥 Unduh Daftar Blank Spot ({len(df_blank):,} Titik) (CSV)", data=csv_blank, file_name=f"blank_spots_{stn_sim}.csv", mime="text/csv", use_container_width=True)

            cols_show = ['center_lat', 'center_lon', 'distance_km', 'clearance_m', 'median_field_strength_dbuv_m', 'category', 'is_covered']
            if 'in_service_area' in df_grid_res.columns:
                cols_show.append('in_service_area')
            st.dataframe(df_grid_res[cols_show].head(500), use_container_width=True, height=260)
            st.caption(f"Menampilkan 500 sampel titik data pertama dari total {len(df_grid_res):,} titik (Peta Raster Telekomunikasi, Resolusi: **{cov_data_stored.get('resolution_m', 50)}m x {cov_data_stored.get('resolution_m', 50)}m**, Lingkup Evaluasi: **{scope_lbl}**).")

    # --------------------------------------------------------------------------
    # G. MODE 6: PREDIKSI TITIK TUNGGAL (SINGLE POINT)
    # --------------------------------------------------------------------------
    elif active_view == "🎯 Prediksi Titik Tunggal":
        st.subheader("🎯 Kalkulator Prediksi Kuat Medan Titik Tunggal (Single Point)")
        single_res = st.session_state.get('single_pred_result')
        sp_ctx = st.session_state.get('single_pred_ctx')

        if single_res is None or sp_ctx is None:
            st.info("💡 Atur koordinat dan klik **'🎯 Hitung Kuat Medan'** di panel samping (sidebar) untuk melakukan kalkulasi titik tunggal.")
        else:
            fs_pred = single_res['predicted_field_strength_dbuv_m']
            cat = single_res['category']
            clr = single_res['clearance_m']
            is_los = single_res['is_los']
            d_km = single_res['distance_km']
            is_est = single_res.get('is_estimated', False)
            fs_display = f"{fs_pred:.2f} dBµV/m (Est.)" if is_est else f"{fs_pred:.2f} dBµV/m"

            sp_c1, sp_c2, sp_c3, sp_c4 = st.columns(4)
            sp_c1.metric("Prediksi Kuat Medan AI", fs_display, delta=cat)
            sp_c2.metric("Jarak Geodesik Haversine", f"{d_km:.2f} km")
            sp_c3.metric("Terrain Clearance", f"{clr:+.2f} m", delta="Line of Sight (LoS)" if clr >= 0 else "Obstructed (NLoS)")
            sp_c4.metric("Margin Cakrawala Radio", f"{single_res['horizon_margin_km']:.2f} km", delta="Di Dalam Horizon" if is_los else "Di Luar Horizon")

            # Diagnostic parameters
            st.markdown("#### 🔬 Rincian Diagnostik Propagasi")
            df_diag = pd.DataFrame([{
                "Parameter": "Nama Pemancar (Tx)", "Nilai": sp_ctx['stn_name']
            }, {
                "Parameter": "Frekuensi Pemancar", "Nilai": f"{single_res['frequency_mhz']:.2f} MHz"
            }, {
                "Parameter": "Daya Pancar EIRP", "Nilai": f"{single_res['eirp_dbm']:.2f} dBm"
            }, {
                "Parameter": "Elevasi Puncak Rintangan Terburuk (h_max)", "Nilai": f"{single_res['h_max_amsl']:.2f} mdpl"
            }, {
                "Parameter": "Tinggi Garis Pandang LoS (h_los)", "Nilai": f"{single_res['h_los_amsl']:.2f} mdpl"
            }, {
                "Parameter": "Tonjolan Lengkungan Bumi (h_er)", "Nilai": f"{single_res['h_er_m']:.2f} m"
            }, {
                "Parameter": "Jarak Horizon Radio Maksimum", "Nilai": f"{single_res['d_horizon_km']:.2f} km"
            }, {
                "Parameter": "Status Pemotongan Strict NLoS", "Nilai": "DIPOTONG BLANK SPOT" if single_res.get('is_cut_by_nlos') else "NORMAL (AI PROPAGATION)"
            }])
            st.dataframe(df_diag, use_container_width=True)
