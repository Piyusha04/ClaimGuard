# ClaimGuard – warranty & return claim fraud detection

1. Install Python 3.10+ and run:  `pip install -r requirements.txt`
2. Start:  `python app.py`   (or run.bat on Windows / ./run.sh on Mac/Linux)
3. Open http://127.0.0.1:5000

First start generates a synthetic dataset (data/claims.csv, ~10% injected fraud) and trains
a Random Forest + Isolation Forest blend (a few seconds). Risk = low (<0.30), medium, high (>=0.60);
edit LOW/HIGH in fraud/model.py. Reasons come from occlusion attribution (SHAP-style);
investigator decisions are saved in data/decisions.json.
To use real data, replace data/claims.csv with the same columns and an is_fraud label column.