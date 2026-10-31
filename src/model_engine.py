"""
Machine Learning Pipeline & Optuna Hyperparameter Optimization Module
Implements:
- Pure XGBoost Regressor with user-specified hyperparameter search spaces
- Optuna Bayesian TPE Sampler with early stopping on Validation Set
- Full Train, Validation, and Test Evaluation Metrics (RMSE, MAE, R²)
- Model persistence to local JSON files (models/{region}_xgb_model.json)
- Single Point Inference Engine for instantaneous field strength calculation
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from typing import Dict, Any, Tuple, Optional, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.features import (
    MODEL_FEATURE_COLS,
    TARGET_COL,
    calculate_haversine_distance,
    calculate_azimuth_bearing,
    calculate_radio_horizon
)

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class DVBT2ModelManager:
    """
    Manages regional XGBoost models: Optuna HPO, training with early stopping,
    evaluation metrics, disk persistence, and single-point/batch inference.
    """

    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.active_model: Optional[xgb.XGBRegressor] = None
        self.active_region: Optional[str] = None
        self.active_metrics: Dict[str, Any] = {}
        self.active_study_summary: Dict[str, Any] = {}

    def get_model_path(self, region_code: str) -> str:
        safe_code = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in region_code).lower()
        return os.path.join(self.models_dir, f"{safe_code}_xgb_model.json")

    def get_metrics_path(self, region_code: str) -> str:
        safe_code = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in region_code).lower()
        return os.path.join(self.models_dir, f"{safe_code}_metrics.json")

    def run_optuna_hpo(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 30,
        trial_callback: Optional[Callable[[int, float, Dict[str, Any]], None]] = None
    ) -> Tuple[Dict[str, Any], optuna.Study]:
        """
        Runs Optuna Bayesian Optimization (TPE Sampler) targeting minimum Validation RMSE.
        Search space strictly conforms to broadcast propagation requirements:
        - n_estimators: 300 - 1000
        - learning_rate: 0.01 - 0.1 (log scale)
        - max_depth: 4 - 8
        - subsample: 0.6 - 1.0
        - colsample_bytree: 0.6 - 1.0
        - min_child_weight: 2 - 8
        - gamma: 0.0 - 0.5
        """
        def objective_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist'
            }

            model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))

            if trial_callback:
                trial_callback(trial.number, val_rmse, trial.params)

            return val_rmse

        study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study_xgb.optimize(objective_xgb, n_trials=n_trials)

        best_params = study_xgb.best_params
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        best_params['tree_method'] = 'hist'

        return best_params, study_xgb

    def train_final_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        best_params: Dict[str, Any],
        region_code: str = 'default_region'
    ) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
        """
        Trains final XGBoost model with optimal parameters and evaluates on
        Train, Validation, and unseen Test sets.
        """
        model_params = best_params.copy()
        model_params['early_stopping_rounds'] = 50
        model_params['random_state'] = 42
        model_params['n_jobs'] = -1
        model_params['tree_method'] = 'hist'

        final_xgb = xgb.XGBRegressor(**model_params)
        final_xgb.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        # Multi-dataset Predictions
        tr_pred = final_xgb.predict(X_train)
        va_pred = final_xgb.predict(X_val)
        te_pred = final_xgb.predict(X_test)

        train_rmse = float(np.sqrt(mean_squared_error(y_train, tr_pred)))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, va_pred)))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, te_pred)))
        test_mae = float(mean_absolute_error(y_test, te_pred))
        test_r2 = float(r2_score(y_test, te_pred))

        feature_importances = dict(zip(
            MODEL_FEATURE_COLS, 
            [round(float(v), 5) for v in final_xgb.feature_importances_]
        ))

        metrics = {
            'train_rmse': round(train_rmse, 4),
            'val_rmse': round(val_rmse, 4),
            'test_rmse': round(test_rmse, 4),
            'test_mae': round(test_mae, 4),
            'test_r2': round(test_r2, 4),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'best_params': best_params,
            'feature_importances': feature_importances,
            'region_code': region_code
        }

        # Save model and metrics locally
        model_path = self.get_model_path(region_code)
        metrics_path = self.get_metrics_path(region_code)
        final_xgb.save_model(model_path)

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        self.active_model = final_xgb
        self.active_region = region_code
        self.active_metrics = metrics

        return final_xgb, metrics

    def load_region_model(self, region_code: str) -> bool:
        """
        Loads saved model from local disk without retraining.
        """
        model_path = self.get_model_path(region_code)
        metrics_path = self.get_metrics_path(region_code)

        if not os.path.exists(model_path):
            return False

        model = xgb.XGBRegressor()
        model.load_model(model_path)
        self.active_model = model
        self.active_region = region_code

        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                self.active_metrics = json.load(f)

        return True

    def predict_single_point(
        self,
        tx_row: Dict[str, Any],
        rx_latitude: float,
        rx_longitude: float,
        rx_antenna_amsl: float = 1.5,
        strict_nlos_cutoff: bool = False
    ) -> Dict[str, Any]:
        """
        Instantaneous prediction of field strength (dBuV/m) at a single coordinate.
        """
        if self.active_model is None:
            raise RuntimeError("Model belum dilatih atau dimuat. Silakan latih model terlebih dahulu.")

        tx_lat = float(tx_row['SID_LAT'])
        tx_lon = float(tx_row['SID_LONG'])
        freq_hz = int(tx_row.get('FREQ_HZ', round(float(tx_row['FREQ_MHZ']) * 1e6)))
        eirp_dbm = float(tx_row['EIRP_dBm'])
        tx_amsl = float(tx_row['H_ANT_masl'])

        # Distance & Azimuth
        dist_km = calculate_haversine_distance(
            np.array([tx_lat]), np.array([tx_lon]),
            np.array([rx_latitude]), np.array([rx_longitude])
        )[0]

        az_deg, az_sin, az_cos = calculate_azimuth_bearing(
            np.array([tx_lat]), np.array([tx_lon]),
            np.array([rx_latitude]), np.array([rx_longitude])
        )

        # Terrain Profile & Clearance Calculation
        from src.terrain_engine import calculate_single_path_clearance
        clearance_info = calculate_single_path_clearance(
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            tx_antenna_amsl=tx_amsl,
            rx_lat=float(rx_latitude),
            rx_lon=float(rx_longitude),
            rx_antenna_amsl=float(rx_antenna_amsl)
        )
        clearance_val = float(clearance_info['clearance_m'])

        X_single = pd.DataFrame([{
            'frequency_hz': freq_hz,
            'eirp_dbm': eirp_dbm,
            'distance_km': dist_km,
            'azimuth_sin': float(az_sin[0]),
            'azimuth_cos': float(az_cos[0]),
            'tx_antenna_amsl': tx_amsl,
            'rx_antenna_amsl': float(rx_antenna_amsl),
            'clearance_m': clearance_val
        }])

        # Adaptive feature matching: supports both 8-feature (with clearance) and legacy 7-feature models
        booster = self.active_model.get_booster()
        expected_features = booster.feature_names if (booster and booster.feature_names) else MODEL_FEATURE_COLS
        cols_to_use = [col for col in expected_features if col in X_single.columns]
        pred_fs = float(self.active_model.predict(X_single[cols_to_use])[0])

        # Radio Horizon Boundary Guardrail
        d_horizon = calculate_radio_horizon(tx_amsl, rx_antenna_amsl)
        horizon_margin = round(float(d_horizon - dist_km), 2)
        is_los = bool(dist_km <= d_horizon)
        is_terrain_obstructed = bool(clearance_val < 0.0)

        # Cutoff rules:
        # 1. Beyond Radio Horizon (always strict cutoff by Earth curvature)
        # 2. Strict NLoS Cutoff (if user toggle enabled and terrain clearance < 0 m)
        is_cut_by_horizon = not is_los
        is_cut_by_nlos = bool(strict_nlos_cutoff and is_terrain_obstructed)
        is_blank = is_cut_by_horizon or is_cut_by_nlos

        if is_blank:
            category = 'Blank Spot'
            color = 'transparent'
            if is_cut_by_horizon:
                d_excess = dist_km - d_horizon
                pred_fs = min(max(pred_fs - (d_excess * 2.0), 10.0), 30.0) # Strictly in Blank Spot range
            else:
                pred_fs = min(pred_fs, 30.0) # Forced into Blank Spot by Strict NLoS cutoff
        else:
            # Classification
            if pred_fs >= 65.0:
                category = 'Sangat Kuat'
                color = '#ef4444' # Merah
            elif pred_fs >= 54.0:
                category = 'Kuat'
                color = '#f97316' # Oranye
            elif pred_fs >= 45.2:
                category = 'Sedang'
                color = '#3b82f6' # Biru
            else:
                category = 'Blank Spot'
                color = 'transparent'

        return {
            'predicted_field_strength_dbuv_m': round(pred_fs, 2),
            'category': category,
            'color': color,
            'distance_km': round(dist_km, 3),
            'azimuth_deg': round(float(az_deg[0]), 2),
            'tx_stn_name': tx_row.get('STN_NAME', 'TX'),
            'frequency_mhz': round(freq_hz / 1e6, 2),
            'eirp_dbm': eirp_dbm,
            'd_horizon_km': d_horizon,
            'horizon_margin_km': horizon_margin,
            'is_los': is_los,
            'clearance_m': clearance_val,
            'h_max_amsl': clearance_info['h_max_amsl'],
            'h_los_amsl': clearance_info['h_los_amsl'],
            'h_er_m': clearance_info['h_er_m'],
            'd1_km': clearance_info['d1_km'],
            'is_terrain_obstructed': is_terrain_obstructed,
            'is_cut_by_horizon': is_cut_by_horizon,
            'is_cut_by_nlos': is_cut_by_nlos,
            'strict_nlos_cutoff': strict_nlos_cutoff,
            'n_samples': clearance_info.get('n_samples', 0),
            'step_m': clearance_info.get('step_m', 500.0)
        }
