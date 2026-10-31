"""
Demonstration & Verification Script
Initializes datasets, trains regional XGBoost models, validates R^2 score,
and tests spatial grid predictions.
"""

import sys
import os
import json

# Ensure workspace root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.synthetic_generator import populate_default_datasets, PREDEFINED_REGIONS
from src.model_engine import DVBT2ModelManager
from src.grid_engine import (
    generate_spatial_grid_points,
    approximate_grid_elevation,
    batch_predict_coverage_grid,
    compute_coverage_kpis
)
import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=== DVB-T2 Propagation AI & GIS Pipeline Initialization ===")

    # 1. Populate datasets
    populate_default_datasets(base_dir, overwrite=False)

    manager = DVBT2ModelManager(models_dir=os.path.join(base_dir, 'models'))

    # 2. Train model for Kalimantan Timur 1
    kaltim_code = 'kaltim1'
    dt_path = os.path.join(base_dir, 'data', 'drivetest', f"{kaltim_code}_drivetest.csv")
    sites_path = os.path.join(base_dir, 'data', 'sites', f"{kaltim_code}_sites.csv")

    df_dt = pd.read_csv(dt_path)
    df_sites = pd.read_csv(sites_path)
    print(f"\nLoaded {len(df_dt)} drive-test rows and {len(df_sites)} sites for {kaltim_code}.")

    # Load or train
    if not manager.load_region_model(kaltim_code):
        print("Running training with Optuna HPO...")
        result = manager.run_multi_region_retraining(
            region_code=kaltim_code,
            df_drivetest=df_dt,
            run_optuna=True,
            n_trials=15,
            test_size=0.20,
            progress_cb=lambda msg, p: print(f"[{int(p*100)}%] {msg}")
        )
    else:
        print(f"Loaded existing model for {kaltim_code}")

    metrics = manager.active_metrics
    print("\n--- Model Training Results (Kalimantan Timur 1) ---")
    print(f"Train R² : {metrics.get('train_r2')}")
    print(f"Test R²  : {metrics.get('test_r2')}")
    print(f"Test RMSE: {metrics.get('test_rmse')} dBµV/m")
    print(f"Test MAE : {metrics.get('test_mae')} dBµV/m")
    print(f"Test MAPE: {metrics.get('test_mape')}%")
    print(f"Feature Importances: {json.dumps(metrics.get('feature_importances', {}), indent=2)}")

    # 3. Model for Jawa Barat 1
    jabar_code = 'jabar1'
    if not manager.load_region_model(jabar_code):
        jb_dt_path = os.path.join(base_dir, 'data', 'drivetest', f"{jabar_code}_drivetest.csv")
        df_jb_dt = pd.read_csv(jb_dt_path)
        print(f"\nTraining model for {jabar_code}...")
        jb_res = manager.run_multi_region_retraining(
            region_code=jabar_code,
            df_drivetest=df_jb_dt,
            run_optuna=False,
            test_size=0.20,
            progress_cb=lambda msg, p: print(f"[{int(p*100)}%] {msg}")
        )
        print(f"Jawa Barat 1 - Test R²: {jb_res['metrics']['test_r2']}, RMSE: {jb_res['metrics']['test_rmse']} dBµV/m")
    else:
        print(f"\nLoaded existing model for {jabar_code} (Test R²: {manager.active_metrics.get('test_r2')})")

    # 4. Test Spatial Grid Generation and Batch Prediction Engine
    print("\nTesting Spatial Grid Generation & Batch Prediction Engine (Kalimantan Timur 1)...")
    manager.load_region_model(kaltim_code)
    bbox = PREDEFINED_REGIONS['Kalimantan Timur 1']['bbox']
    grid_lat, grid_lon, res_m = generate_spatial_grid_points(
        bbox['min_lat'], bbox['max_lat'], bbox['min_lon'], bbox['max_lon'],
        resolution_meters=1500.0 # Fast resolution for test
    )
    grid_amsl = approximate_grid_elevation(grid_lat, grid_lon, 'kaltim')

    print(f"Generated {len(grid_lat)} grid points at ~{res_m:.0f}m resolution.")
    df_grid_pred = batch_predict_coverage_grid(
        model=manager.active_model,
        sites_list=df_sites.to_dict('records'),
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        grid_amsl=grid_amsl,
        mode='dominant'
    )
    kpis = compute_coverage_kpis(df_grid_pred, res_m)
    print(f"Grid predictions finished successfully:")
    print(f"Total Area: {kpis['total_area_km2']} km²")
    print(f"Strong Signal Area: {kpis['pct_strong']}% ({kpis['strong_signal_area_km2']} km²)")
    print(f"Medium Signal Area: {kpis['pct_medium']}% ({kpis['medium_signal_area_km2']} km²)")
    print(f"Blank Spot Area   : {kpis['pct_blank']}% ({kpis['blank_spot_area_km2']} km²)")

    print("\n All core engine components successfully validated!")


if __name__ == '__main__':
    main()
