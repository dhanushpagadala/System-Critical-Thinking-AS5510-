#!/usr/bin/env python3
"""Create per-snapshot predicted rankings and compare them with binary ground truth.

Input: held-out predictions CSV with columns:
  case_id, step, model, agent_id, predicted_score, ground_truth

`ground_truth` must contain the actual Critical/Non-critical label (or 1/0).
This script does NOT create a separate ground-truth rank: the dataset supplies
binary criticality labels, not an independent ordinal ranking.

Example:
  python ground_truth_ranking.py --input fair_rf_vs_graphsage_test_predictions.csv \
      --output AE5510_Literal_Ranking_vs_Ground_Truth.xlsx
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REQUIRED = {"case_id", "step", "model", "agent_id", "predicted_score", "ground_truth"}


def normalise_label(value):
    if pd.isna(value):
        raise ValueError("ground_truth contains missing labels")
    s = str(value).strip().lower()
    if s in {"critical", "1", "true", "yes"}:
        return "Critical"
    if s in {"non-critical", "noncritical", "0", "false", "no"}:
        return "Non-critical"
    raise ValueError(f"Unrecognised ground_truth label: {value!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Held-out prediction CSV")
    ap.add_argument("--output", default="AE5510_Literal_Ranking_vs_Ground_Truth.xlsx")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    if args.top_k < 1:
        ap.error("--top-k must be at least 1")

    df = pd.read_csv(args.input)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    df = df[list(REQUIRED)].copy()
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Input contains missing values in required columns")
    df["ground_truth"] = df["ground_truth"].map(normalise_label)
    df["predicted_score"] = pd.to_numeric(df["predicted_score"], errors="raise")
    df["step"] = pd.to_numeric(df["step"], errors="raise").astype(int)
    df["agent_id"] = df["agent_id"].astype(str)

    # Rank independently within every snapshot and model; highest score is rank 1.
    df = df.sort_values(
        ["case_id", "step", "model", "predicted_score", "agent_id"],
        ascending=[True, True, True, False, True], kind="mergesort"
    )
    df["rank"] = df.groupby(["case_id", "step", "model"]).cumcount() + 1

    full = df[["case_id", "step", "model", "rank", "agent_id", "predicted_score", "ground_truth"]]
    top = df[df["rank"] <= args.top_k]
    wide = top.pivot(index=["case_id", "step", "rank"], columns="model",
                     values=["agent_id", "predicted_score", "ground_truth"])
    wide.columns = [f"{model}_{field}" for field, model in wide.columns]
    wide = wide.reset_index()
    # Friendly RF / GraphSAGE column order when both models are present.
    preferred = ["case_id", "step", "rank"]
    for model in ["RF", "GraphSAGE"]:
        for field in ["agent_id", "predicted_score", "ground_truth"]:
            col = f"{model}_{field}"
            if col in wide.columns:
                preferred.append(col)
    preferred += [c for c in wide.columns if c not in preferred]
    wide = wide[preferred]

    # Ranking metrics averaged over snapshots, reported per model.
    metric_rows = []
    for model, part in df.groupby("model"):
        aps, pks, rks = [], [], []
        for (_, _), g in part.groupby(["case_id", "step"]):
            g = g.sort_values("rank")
            y = (g["ground_truth"] == "Critical").astype(int).to_numpy()
            npos = int(y.sum())
            if npos == 0:
                continue
            k = min(args.top_k, len(g))
            hits = y[:k]
            precision_at_r = (hits.cumsum() / (pd.Series(range(1, k + 1)).to_numpy()))
            ap_score = float((precision_at_r * hits).sum() / npos)
            aps.append(ap_score)
            pks.append(float(hits.sum() / k))
            rks.append(float(hits.sum() / npos))
        metric_rows.append({
            "model": model,
            "snapshots_evaluated": len(aps),
            "macro_average_precision": sum(aps) / len(aps) if aps else float("nan"),
            f"macro_precision_at_{args.top_k}": sum(pks) / len(pks) if pks else float("nan"),
            f"macro_recall_at_{args.top_k}": sum(rks) / len(rks) if rks else float("nan"),
        })
    metrics = pd.DataFrame(metric_rows)
    notes = pd.DataFrame([
        ("Input", str(Path(args.input))),
        ("Ranking", "Within each case_id + step + model, descending predicted_score; rank 1 is highest."),
        ("Ground truth", "Binary Critical/Non-critical label from ground_truth; no independent numerical ground-truth rank is available."),
        ("Top-K", args.top_k),
        ("Rows", len(df)),
        ("Snapshots", df[["case_id", "step"]].drop_duplicates().shape[0]),
        ("Interpretation", "Precision@K and Recall@K use actual Critical labels among the highest-scored agents."),
    ], columns=["Item", "Details"])
    out = Path(args.output)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        wide.to_excel(writer, sheet_name=f"Top-{args.top_k} per Snapshot", index=False)
        full.to_excel(writer, sheet_name="Full Rankings", index=False)
        metrics.to_excel(writer, sheet_name="Ranking Metrics", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"Saved ranking comparison: {out.resolve()}")
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
