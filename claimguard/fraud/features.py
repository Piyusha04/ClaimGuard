import numpy as np, pandas as pd

FEATURES = ["days_to_claim", "no_purchase_record", "serial_mismatch", "claims_per_customer",
            "claim_to_price_ratio", "prior_returns", "accounts_per_address",
            "accounts_per_device", "customer_claims_90d", "high_value"]

def _claims_90d(g):
    t = g.sort_values().values.astype("datetime64[D]").astype(int)
    n = np.searchsorted(t, t, "right") - np.searchsorted(t, t - 90, "left")
    return pd.Series(n, index=g.sort_values().index)

def build_features(df):
    f = pd.DataFrame(index=df.index)
    f["days_to_claim"] = (df["claim_date"] - df["purchase_date"]).dt.days
    f["no_purchase_record"] = df["registered_serial"].isna().astype(int)
    f["serial_mismatch"] = ((df["serial_number"] != df["registered_serial"])
                            & df["registered_serial"].notna()).astype(int)
    f["claims_per_customer"] = df.groupby("customer_id")["claim_id"].transform("count")
    f["claim_to_price_ratio"] = (df["claim_amount"] / df["price"]).round(3)
    f["prior_returns"] = df["prior_returns"]
    f["accounts_per_address"] = df.groupby("address_id")["customer_id"].transform("nunique")
    f["accounts_per_device"] = df.groupby("device_id")["customer_id"].transform("nunique")
    f["customer_claims_90d"] = (df.groupby("customer_id")["claim_date"].apply(_claims_90d)
                                .reset_index(level=0, drop=True).reindex(df.index))
    f["high_value"] = (df["claim_amount"] > 700).astype(int)
    return f[FEATURES].astype(float)
