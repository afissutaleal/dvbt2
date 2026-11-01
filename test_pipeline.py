"""
Comprehensive Automated Test Script for DVB-T2 Propagation AI & GIS Pipeline
Validates:
1. Ingestion of sample transmitter and multi-file drive test
2. Deduplication, stationary aggregation, frequency matching, and EIRP calculation
3. Multi-Feature Stratified 3-Way Split (70:15:15)
4. Optuna HPO + XGBoost Training & Evaluation
5. 100m x 100m Polygon Grid generation with 33 sub-points & median aggregation
6. Single Point Predictor
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features import process_and_align_datasets
from src.split_utils import perform_multi_feature_stratified_split
from src.model_engine import DVBT2ModelManager
from src.grid_engine import (
    generate_radius_grid_boxes,
    predict_coverage_grid_with_subsampling,
    generate_continuous_coverage_raster
)

def run_test():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')

    print("=== 1. Testing Data Ingestion & Preprocessing ===")
    tx_path = os.path.join(data_dir, "sample_data_transmitter.csv")
    dt1_path = os.path.join(data_dir, "sample_DriveT_BPP1.csv")
    dt2_path = os.path.join(data_dir, "sample_DriveT_BPP2.csv")

    df_tx = pd.read_csv(tx_path)
    df_dt1 = pd.read_csv(dt1_path)
    df_dt2 = pd.read_csv(dt2_path)
    df_dt_concat = pd.concat([df_dt1, df_dt2], ignore_index=True)
    print(f"Loaded {len(df_tx)} transmitters, {len(df_dt_concat)} total raw drive test rows from 2 files.")

    df_clean, df_tx_sum, stats = process_and_align_datasets(df_tx, df_dt_concat)
    print(f"Preprocessing stats: {stats}")
    print(f"Transmitter EIRP Summary:\n{df_tx_sum[['STN_NAME', 'FREQ_MHZ', 'E_max_dBuV_m', 'd_at_Emax_km', 'EIRP_dBm']].head(4)}")

    assert len(df_clean) > 0, "Cleaned dataset is empty!"
    assert 'eirp_dbm' in df_clean.columns, "EIRP column missing!"
    assert 'clearance_m' in df_clean.columns, "Clearance column missing!"
    print(f"Sample Clearance values: {df_clean['clearance_m'].head(5).tolist()}")
    print("Preprocessed & Aligned successfully with Terrain Clearance!")

    print("\n=== 2. Testing Multi-Feature Stratified 3-Way Split (70:15:15) ===")
    X_tr, y_tr, X_va, y_va, X_te, y_te, sum_check, fig = perform_multi_feature_stratified_split(df_clean)
    print(f"Split sizes: Train={len(X_tr)} (70%), Val={len(X_va)} (15%), Test={len(X_te)} (15%)")
    print(f"Sanity Check Table:\n{sum_check}")
    assert len(X_tr) > len(X_va) and len(X_va) == len(X_te), "Split proportions invalid!"
    print("Stratified Split completed successfully!")

    print("\n=== 3. Testing Optuna HPO & XGBoost Regressor Training ===")
    manager = DVBT2ModelManager(models_dir=models_dir)
    best_params, study = manager.run_optuna_hpo(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        n_trials=8 # Short test
    )
    print(f"Best Optuna Val RMSE: {study.best_value:.4f}")

    final_model, metrics = manager.train_final_model(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te,
        best_params=best_params,
        region_code='test_balikpapan'
    )
    print(f"Final Model Metrics:\n"
          f"Train RMSE: {metrics['train_rmse']} | Val RMSE: {metrics['val_rmse']} | "
          f"Test RMSE: {metrics['test_rmse']} dBµV/m | Test MAE: {metrics['test_mae']} dBµV/m | "
          f"Test R²: {metrics['test_r2']}")
    assert metrics['test_r2'] > 0.70, f"R² too low: {metrics['test_r2']}"
    print("XGBoost Model trained and evaluated successfully!")

    print("\n=== 4. Testing Grid Generation (100m / 50m / 10m) with 33 Sub-points & Median ===")
    first_tx = df_tx_sum.iloc[0].to_dict()
    grid_boxes, eff_res = generate_radius_grid_boxes(
        tx_lat=float(first_tx['SID_LAT']),
        tx_lon=float(first_tx['SID_LONG']),
        radius_km=5.0, # 5 km test radius
        resolution_m=100.0 # 100m grid test
    )
    print(f"Generated {len(grid_boxes)} boxes at ~{eff_res}m resolution within 15 km radius.")

    df_grid_res, kpis = predict_coverage_grid_with_subsampling(
        model=final_model,
        tx_params=first_tx,
        grid_boxes=grid_boxes,
        n_subpoints_per_box=33,
        thresholds={'sangat_kuat': 65.0, 'kuat': 54.0, 'sedang': 45.2},
        strict_nlos_cutoff=False
    )
    print(f"Grid Prediction KPIs (AI Mode):\n"
          f"Total Boxes: {kpis['total_boxes']} | Covered Area: {kpis['covered_area_km2']} km² ({kpis['pct_covered']}%) | "
          f"Sangat Kuat: {kpis['pct_sangat_kuat']}% | Kuat: {kpis['pct_kuat']}% | Sedang: {kpis['pct_sedang']}% | "
          f"Blank Spot: {kpis['pct_blank_spot']}%")
    assert len(df_grid_res) == len(grid_boxes), "Grid results count mismatch!"

    # Test Strict NLoS Cutoff Mode
    df_grid_nlos, kpis_nlos = predict_coverage_grid_with_subsampling(
        model=final_model,
        tx_params=first_tx,
        grid_boxes=grid_boxes,
        n_subpoints_per_box=10,
        thresholds={'sangat_kuat': 65.0, 'kuat': 54.0, 'sedang': 45.2},
        strict_nlos_cutoff=True
    )
    print(f"Grid Prediction KPIs (Strict NLoS Cutoff Mode):\n"
          f"Cut by NLoS: {kpis_nlos.get('n_cut_by_nlos', 0)} boxes | "
          f"Covered Area: {kpis_nlos['covered_area_km2']} km² ({kpis_nlos['pct_covered']}%) | "
          f"Blank Spot: {kpis_nlos['pct_blank_spot']}%")
    assert kpis_nlos['pct_covered'] <= kpis['pct_covered'], "Strict NLoS covered area should be <= AI mode!"
    print("Grid sub-sampling & Strict NLoS cutoff verified!")

    print("\n=== 4b. Testing Continuous Telecom Raster Heatmap (CloudRF Style) ===")
    raster_overlay, df_sample_cov, kpis_raster = generate_continuous_coverage_raster(
        model=final_model,
        tx_params=first_tx,
        radius_km=10.0,
        resolution_m=100.0,
        strict_nlos_cutoff=False
    )
    print(f"Raster coverage generated: H={raster_overlay['H']}, W={raster_overlay['W']}, resolution={raster_overlay['resolution_m']}m")
    assert 'image_b64' in raster_overlay, "image_b64 missing from raster overlay!"
    assert 'fs_grid_b64' in raster_overlay, "fs_grid_b64 missing from raster overlay!"
    assert len(df_sample_cov) > 0, "Representative sample points dataframe empty!"
    
    # Check that colors strictly match Red, Orange, Blue, Transparent
    allowed_colors = {'#ef4444', '#f97316', '#3b82f6', 'transparent'}
    sample_colors = set(df_sample_cov['color'].unique())
    print(f"Colors present in sample: {sample_colors}")
    assert sample_colors.issubset(allowed_colors), f"Unexpected colors found: {sample_colors - allowed_colors}"
    print("Continuous Telecom Raster Heatmap & Color Hierarchy verified!")

    print("\n=== 5. Testing Single Point Predictor ===")
    # 5a. AI mode (strict_nlos_cutoff=False)
    single_res = manager.predict_single_point(
        tx_row=first_tx,
        rx_latitude=float(first_tx['SID_LAT']) + 0.05,
        rx_longitude=float(first_tx['SID_LONG']) + 0.04,
        rx_antenna_amsl=25.0,
        strict_nlos_cutoff=False
    )
    print(f"Single Point Result (AI Mode): {single_res}")
    assert 'predicted_field_strength_dbuv_m' in single_res
    assert 'd_horizon_km' in single_res and 'is_los' in single_res, "Radio Horizon keys missing from single point result!"
    assert 'clearance_m' in single_res and 'h_max_amsl' in single_res, "Clearance keys missing from single point result!"
    print(f"Clearance verification: clearance_m={single_res['clearance_m']} m, h_max={single_res['h_max_amsl']} m, h_los={single_res['h_los_amsl']} m")
    print(f"Radio Horizon verification: d_horizon={single_res['d_horizon_km']} km, is_los={single_res['is_los']}, margin={single_res['horizon_margin_km']} km")

    # 5b. Strict NLoS Cutoff Mode (strict_nlos_cutoff=True)
    single_res_nlos = manager.predict_single_point(
        tx_row=first_tx,
        rx_latitude=float(first_tx['SID_LAT']) + 0.05,
        rx_longitude=float(first_tx['SID_LONG']) + 0.04,
        rx_antenna_amsl=25.0,
        strict_nlos_cutoff=True
    )
    print(f"Single Point Result (Strict NLoS Mode): cat={single_res_nlos['category']}, fs={single_res_nlos['predicted_field_strength_dbuv_m']} dBµV/m, is_cut_by_nlos={single_res_nlos['is_cut_by_nlos']}")
    if single_res['clearance_m'] < 0:
        assert single_res_nlos['category'] == 'Blank Spot', "Obstructed point should become Blank Spot under Strict NLoS cutoff!"
        assert single_res_nlos['is_cut_by_nlos'] is True, "is_cut_by_nlos flag must be True!"
    print("Strict NLoS single point cutoff verified!")

    # Test strict horizon cutoff for distant point (> 80 km)
    single_res_far = manager.predict_single_point(
        tx_row=first_tx,
        rx_latitude=float(first_tx['SID_LAT']) + 0.8,
        rx_longitude=float(first_tx['SID_LONG']) + 0.8,
        rx_antenna_amsl=10.0
    )
    print(f"Far Point (Beyond Horizon): d={single_res_far['distance_km']} km, d_horizon={single_res_far['d_horizon_km']} km, category={single_res_far['category']}, fs={single_res_far['predicted_field_strength_dbuv_m']} dBµV/m, is_los={single_res_far['is_los']}")
    assert single_res_far['category'] == 'Blank Spot', "Far point should be strictly Blank Spot!"
    assert single_res_far['predicted_field_strength_dbuv_m'] < 45.2, "Far point field strength should be below 45.2!"

    print("Single Point Predictor verified!")

    print("\n=== 6. Testing DVB-T2 Masterplan KML & Regulatory Test Point Compliance ===")
    from src.kml_engine import get_masterplan_engine, evaluate_test_points_compliance, calculate_polygon_area_km2
    kml_eng = get_masterplan_engine()
    sa_kaltim = kml_eng.find_service_area('kaltim1')
    assert sa_kaltim is not None, "Kalimantan Timur - 1 service area not found in KML!"
    
    sa_area = sa_kaltim.get('area_km2', calculate_polygon_area_km2(sa_kaltim['polygon']))
    print(f"Service Area: {sa_kaltim['raw_name']}, Calculated Geodesic Area: {sa_area:.1f} km²")
    assert sa_area > 20000.0, f"Area should be > 20,000 km²! Got: {sa_area}"

    tps = kml_eng.get_test_points_for_area(sa_kaltim['norm_key'])
    print(f"Test points found for area: {len(tps)}")
    assert len(tps) > 0, "No test points found for Kalimantan Timur - 1!"

    df_tp_res, tp_sum = evaluate_test_points_compliance(
        model_manager=manager,
        tx_params=first_tx,
        test_points=tps,
        tolerance_db=3.0,
        rx_antenna_amsl=1.5,
        strict_nlos_cutoff=False
    )
    print(f"Test Points Evaluation Summary:\n"
          f"Total: {tp_sum['total_points']} | Safe (<= Max Limit): {tp_sum['safe_count']} | "
          f"Tolerable (+3 dB): {tp_sum['tolerable_count']} | Violation (> Max Limit): {tp_sum['violation_count']} | "
          f"Compliance Rate: {tp_sum['compliance_rate_pct']}% | Avg Margin: {tp_sum['avg_margin_db']:+.1f} dB")
    
    # Print diagnostic table (strip non-ascii for cp1252 Windows console compatibility)
    df_clean_print = df_tp_res[['Nama Titik Uji', 'Jarak ke TX (km)', 'Batas Maksimum Acuan (dBµV/m)', 'Prediksi Kuat Medan AI (dBµV/m)', 'Margin thd Batas Maks (dB)', 'Status Koordinasi']].copy()
    df_clean_print['Status Koordinasi'] = df_clean_print['Status Koordinasi'].astype(str).str.encode('ascii', 'ignore').str.decode('ascii')
    print(df_clean_print.to_string())

    assert 'safe_count' in tp_sum and 'violation_count' in tp_sum, "Compliance metrics missing!"
    print("DVB-T2 Masterplan & Test Point Regulatory Evaluation verified!")

    print("\n ALL 6 PIPELINE COMPONENTS TESTED & VALIDATED SUCCESSFULLY!")

if __name__ == '__main__':
    run_test()
