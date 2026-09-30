import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from .features import build_features, FEATURES

LOW, HIGH = 0.30, 0.60   # risk score cut-offs for low / medium / high
REASONS = {
 "days_to_claim": lambda v: f"Claim filed {int(v)} days after purchase" + (" (unusually soon)" if v < 7 else ""),
 "no_purchase_record": lambda v: "No matching purchase record for this serial number",
 "serial_mismatch": lambda v: "Serial number differs from the unit that was sold",
 "claims_per_customer": lambda v: f"Customer has {int(v)} claims on file",
 "claim_to_price_ratio": lambda v: f"Claim is {v:.0%} of the purchase price",
 "prior_returns": lambda v: f"{int(v)} earlier returns by this customer",
 "accounts_per_address": lambda v: f"{int(v)} accounts share this address",
 "accounts_per_device": lambda v: f"{int(v)} accounts used this device",
 "customer_claims_90d": lambda v: f"{int(v)} claims by this customer within 90 days",
 "high_value": lambda v: "High-value claim",
}

def tier(s): return "high" if s >= HIGH else "medium" if s >= LOW else "low"

class Detector:
    def fit(self, df):
        X, y = build_features(df), df["is_fraud"]
        self.rf = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced_subsample",
                                         random_state=42, n_jobs=-1)
        # out-of-fold probabilities, so dashboard scores are not inflated by training on the same rows
        oof = cross_val_predict(self.rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                method="predict_proba")[:, 1]
        self.rf.fit(X, y)
        self.iso = IsolationForest(n_estimators=300, contamination=0.08, random_state=42).fit(X)
        raw = -self.iso.score_samples(X)
        self.lo, self.hi = raw.min(), raw.max()
        self.baseline, self.X = X.median(), X
        iso = (raw - self.lo) / (self.hi - self.lo + 1e-9)
        self.scores = np.clip(0.8 * oof + 0.2 * iso, 0, 1)   # supervised + anomaly blend
        self.auc = roc_auc_score(y, self.scores)
        self.importance = {k: float(v) for k, v in zip(FEATURES, self.rf.feature_importances_.round(4))}
        return self

    def explain(self, i, top=4):
        """Occlusion attribution (SHAP-like): risk drop when a feature is set to its typical value."""
        x = self.X.iloc[[i]]
        p0 = self.rf.predict_proba(x)[0, 1]
        rows = []
        for f in FEATURES:
            x2 = x.copy(); x2[f] = self.baseline[f]
            rows.append((f, p0 - self.rf.predict_proba(x2)[0, 1]))
        rows.sort(key=lambda t: -t[1])
        return [{"feature": f, "impact": round(float(c), 3), "text": REASONS[f](x[f].iloc[0])}
                for f, c in rows[:top] if c > 0.005]
