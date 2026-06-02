import joblib
import pandas as pd
import sys
from pathlib import Path
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
    pred = model.predict(X)
    probs = model.predict_proba(X)[:, 1]


    predictions_df = panel_rows[['Name']].copy()
    predictions_df['ChurnProbability'] = probs
    predictions_df['OfferIntervention'] = (predictions_df['ChurnProbability'] > 0.62).astype(int)
    predictions_path = config.PREDICTIONS_DIR / f"predictions_for_{last_complete_year +1}.csv"
    print(predictions_df.columns)
    predictions_df.to_csv(predictions_path, index=False)
    
if __name__ == '__main__':
    generate_predictions()
    