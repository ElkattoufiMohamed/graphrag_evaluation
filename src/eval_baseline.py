from __future__ import annotations
import json
import pandas as pd
from collections import defaultdict

from src.metrics import best_of_many

def main(path="outputs/baseline_predictions.jsonl"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pred = obj.get("pred_baseline")
            if not pred:  # skip missing predictions (Gemini failed)
                continue
            f1, rl = best_of_many(pred, obj["answers"])
            rows.append({
                "subset": obj["subset"],
                "sample_id": obj["sample_id"],
                "f1": f1,
                "rougeL": rl,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No completed predictions yet (pred_baseline missing).")
        return

    summary = df.groupby("subset")[["f1", "rougeL"]].mean().reset_index()
    summary["f1"] = summary["f1"].round(4)
    summary["rougeL"] = summary["rougeL"].round(4)

    out_csv = "outputs/baseline_results.csv"
    summary.to_csv(out_csv, index=False)
    print(summary)
    print(f"\nWrote: {out_csv}")

if __name__ == "__main__":
    main()
