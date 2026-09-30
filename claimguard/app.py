import json, os
import pandas as pd
from flask import Flask, jsonify, render_template, request, abort
from fraud.data import generate
from fraud.model import Detector, tier

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
app = Flask(__name__)
os.makedirs("data", exist_ok=True)
CSV, DEC_FILE = "data/claims.csv", "data/decisions.json"

if os.path.exists(CSV):   # replace data/claims.csv with real claims (same columns) to use your own data
    df = pd.read_csv(CSV, parse_dates=["purchase_date", "claim_date"])
else:
    df = generate(); df.to_csv(CSV, index=False)
det = Detector().fit(df)
df["risk"] = det.scores.round(3)
df["tier"] = [tier(s) for s in df["risk"]]
df["month"] = df["claim_date"].dt.strftime("%Y-%m")
decisions = json.load(open(DEC_FILE)) if os.path.exists(DEC_FILE) else {}

def row(i):
    r = df.iloc[i]
    return {"claim_id": r.claim_id, "customer": r.customer_id, "product": r["product"],
            "region": r.region, "type": r.claim_type, "reason": r.claim_reason,
            "amount": float(r.claim_amount), "date": r.claim_date.strftime("%Y-%m-%d"),
            "risk": float(r.risk), "tier": r.tier, "decision": decisions.get(r.claim_id, "")}

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/summary")
def summary():
    t = df["tier"].value_counts()
    def by(col):
        g = df.groupby(col).agg(claims=("claim_id", "count"), high=("tier", lambda s: int((s == "high").sum())))
        g["rate"] = (g["high"] / g["claims"]).round(3)
        return g.reset_index().rename(columns={col: "name"}).to_dict("records")
    tr = df.groupby(["month", "tier"]).size().unstack(fill_value=0)
    tr = tr.reindex(columns=["low", "medium", "high"], fill_value=0).tail(12)
    return jsonify(total=len(df), tiers={k: int(t.get(k, 0)) for k in ["low", "medium", "high"]},
                   at_risk=float(df.loc[df.tier == "high", "claim_amount"].sum()),
                   auto_approve=float((df.tier == "low").mean()), auc=round(det.auc, 3),
                   product=by("product"), region=by("region"),
                   trend=[{"month": m, **{k: int(v) for k, v in r.items()}} for m, r in tr.iterrows()],
                   importance=det.importance)

@app.route("/api/claims")
def claims():
    t = request.args.get("tier", "high")
    d = df if t == "all" else df[df.tier == t]
    d = d.sort_values("risk", ascending=False).head(int(request.args.get("limit", 150)))
    return jsonify([row(df.index.get_loc(i)) for i in d.index])

@app.route("/api/claims/<cid>")
def claim(cid):
    m = df.index[df.claim_id == cid]
    if not len(m): abort(404)
    i = df.index.get_loc(m[0])
    return jsonify({**row(i), "reasons": det.explain(i)})

@app.route("/api/claims/<cid>/decision", methods=["POST"])
def decide(cid):
    v = (request.json or {}).get("decision")
    if v not in ("approved", "rejected", "escalated"): abort(400)
    decisions[cid] = v
    json.dump(decisions, open(DEC_FILE, "w"))
    return jsonify(ok=True)

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
