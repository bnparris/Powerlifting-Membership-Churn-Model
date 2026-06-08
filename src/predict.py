import joblib
import pandas as pd
import sys
from pathlib import Path
import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
import json  



def generate_predictions():
    """Generate and save predictions for all lifters in the most recent year."""
    panel = pd.read_csv(config.PRODUCTION_PANEL)
    with open(config.METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        features = metadata['features']
        last_complete_year = metadata['last_complete_year']
    model = joblib.load(config.MODEL_PATH)

    panel_rows = panel[panel['Year'] == last_complete_year]
    X = panel_rows[features]

    probs = model.predict_proba(X)[:, 1]


    predictions_df = panel_rows[['Name']].copy()
    predictions_df['ChurnProbability'] = probs
    predictions_df['OfferIntervention'] = (predictions_df['ChurnProbability'] > config.INTERVENTION_THRESHOLD).astype(int)
    predictions_path = config.PREDICTIONS_DIR / f"predictions_for_{last_complete_year +1}.csv"

    above_threshold = predictions_df.loc[predictions_df['ChurnProbability']>config.INTERVENTION_THRESHOLD]

    #uses mean predicted probability as proxy for baseline churn rate (no labels available in production)
    baseline_in_window = above_threshold['ChurnProbability'].mean()

    
    effect = proportion_effectsize(baseline_in_window - config.MDE, baseline_in_window)

    control_size = NormalIndPower().solve_power(effect_size=effect, alpha=config.ALPHA, power=config.POWER)
    control_size = float(control_size[0]) if hasattr(control_size, '__len__') else float(control_size)
    control_size = round(control_size)
    rng = np.random.default_rng(42)
    control_group = rng.choice(above_threshold.index, size=control_size, replace=False)
    predictions_df.loc[control_group, 'OfferIntervention'] = 0
    predictions_df['ControlGroup'] = 0
    predictions_df.loc[control_group, 'ControlGroup'] = 1
    predictions_df.to_csv(predictions_path, index=False)
    
if __name__ == '__main__':
    generate_predictions()


