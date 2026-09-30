"""Synthetic warranty/return claims with injected fraud patterns."""
import numpy as np, pandas as pd

PRODUCTS = {"Laptop": 900, "Smartphone": 650, "Headphones": 180, "Smartwatch": 320,
            "Vacuum": 400, "Blender": 110, "Camera": 750, "Tablet": 480}
REGIONS = ["North", "South", "East", "West", "Central"]
REASONS = ["Defective on arrival", "Stopped working", "Screen/part damage", "Battery failure",
           "Not as described", "Changed mind"]

def generate(n=6000, seed=42):
    r = np.random.default_rng(seed)
    n_cust = int(n * 0.45)
    prod = r.choice(list(PRODUCTS), n)
    price = np.array([PRODUCTS[p] for p in prod]) * r.uniform(0.9, 1.1, n)
    claim_date = pd.Timestamp("2026-09-30") - pd.to_timedelta(r.integers(0, 540, n), unit="D")
    gap = np.clip(r.gamma(3, 35, n), 5, 700).astype(int)
    df = pd.DataFrame({
        "claim_id": [f"CLM-{100000+i}" for i in range(n)],
        "customer_id": [f"C{c:05d}" for c in r.integers(0, n_cust, n)],
        "product": prod, "price": price.round(2),
        "purchase_date": claim_date - pd.to_timedelta(gap, unit="D"), "claim_date": claim_date,
        "claim_type": r.choice(["Warranty", "Return"], n, p=[.6, .4]),
        "claim_reason": r.choice(REASONS, n, p=[.2, .3, .15, .15, .12, .08]),
        "claim_amount": (price * r.uniform(0.25, 0.95, n)).round(2),
        "prior_returns": r.poisson(0.5, n),
        "region": r.choice(REGIONS, n),
        "address_id": [f"A{a:05d}" for a in r.integers(0, int(n * 0.6), n)],
        "device_id": [f"D{d:05d}" for d in r.integers(0, int(n * 0.7), n)],
    })
    df["registered_serial"] = [f"SN{r.integers(10**7, 10**8)}" for _ in range(n)]
    df["serial_number"] = df["registered_serial"]
    df["is_fraud"] = 0

    idx = r.permutation(n); k = int(n * 0.02)
    a, b, c, d, e = (idx[i*k:(i+1)*k] for i in range(5))
    # 1) product never bought: no purchase record, unknown serial
    df.loc[a, "registered_serial"] = None
    df.loc[a, "serial_number"] = [f"SN{r.integers(10**7, 10**8)}" for _ in a]
    df.loc[a, "claim_type"] = "Warranty"
    # 2) swapped/used item: serial differs from the one sold
    df.loc[b, "serial_number"] = [f"SN{r.integers(10**7, 10**8)}" for _ in b]
    df.loc[b, "claim_reason"] = "Not as described"
    # 3) repeat claimants
    df.loc[c, "customer_id"] = r.choice([f"C{x:05d}" for x in r.integers(0, n_cust, 40)], len(c))
    df.loc[c, "prior_returns"] += r.integers(3, 8, len(c))
    # 4) organized ring: many accounts sharing few addresses/devices
    df.loc[d, "address_id"] = r.choice([f"A9{x:04d}" for x in range(12)], len(d))
    df.loc[d, "device_id"] = r.choice([f"D9{x:04d}" for x in range(10)], len(d))
    df.loc[d, "customer_id"] = [f"C9{x:05d}" for x in r.integers(0, 60, len(d))]
    # 5) inflated amount, claimed right after purchase
    df.loc[e, "claim_amount"] = (df.loc[e, "price"] * r.uniform(1.0, 1.6, len(e))).round(2)
    df.loc[e, "purchase_date"] = df.loc[e, "claim_date"] - pd.to_timedelta(r.integers(0, 4, len(e)), unit="D")
    fraud_idx = np.concatenate([a, b, c, d, e])
    df.loc[fraud_idx, "is_fraud"] = 1
    # hard negatives: honest customers who look a bit suspicious (early failure, big claim, shared address)
    rest = np.setdiff1d(idx, fraud_idx); m = int(n * 0.04)
    h1, h2, h3 = rest[:m], rest[m:2*m], rest[2*m:3*m]
    df.loc[h1, "purchase_date"] = df.loc[h1, "claim_date"] - pd.to_timedelta(r.integers(1, 5, m), unit="D")
    df.loc[h2, "claim_amount"] = (df.loc[h2, "price"] * r.uniform(0.95, 1.2, m)).round(2)
    df.loc[h3, "address_id"] = r.choice([f"A8{x:04d}" for x in range(30)], m)
    # a few real fraud cases that look ordinary
    quiet = r.choice(fraud_idx, int(len(fraud_idx) * 0.12), replace=False)
    df.loc[quiet, "claim_amount"] = (df.loc[quiet, "price"] * 0.5).round(2)
    df.loc[quiet, "prior_returns"] = 0
    return df.sort_values("claim_date").reset_index(drop=True)
