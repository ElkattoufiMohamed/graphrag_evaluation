from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.metrics import best_of_many


def _load_scored_rows(path: Path, pred_key: str) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame(rows)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pred = obj.get(pred_key)
            if not pred:
                continue
            f1, rouge_l = best_of_many(pred, obj["answers"])
            rows.append(
                {
                    "subset": obj["subset"],
                    "sample_id": obj["sample_id"],
                    "f1": f1,
                    "rougeL": rouge_l,
                }
            )

    return pd.DataFrame(rows)


def _improvement(graph_val: float, base_val: float) -> float:
    if base_val == 0:
        return 0.0
    return ((graph_val - base_val) / base_val) * 100.0


def main(
    baseline_path: str = "outputs/baseline_predictions.jsonl",
    graphrag_path: str = "outputs/graphrag_predictions.jsonl",
    out_csv: str = "outputs/performance_comparison.csv",
):
    baseline_df = _load_scored_rows(Path(baseline_path), pred_key="pred_baseline")
    graphrag_df = _load_scored_rows(Path(graphrag_path), pred_key="pred_graphrag")

    if baseline_df.empty or graphrag_df.empty:
        print("Need both baseline and GraphRAG prediction files with non-empty predictions.")
        return

    baseline_mean = baseline_df.groupby("subset")[["f1", "rougeL"]].mean()
    graphrag_mean = graphrag_df.groupby("subset")[["f1", "rougeL"]].mean()

    rows = []
    for subset in sorted(set(baseline_mean.index).intersection(graphrag_mean.index)):
        base_f1 = float(baseline_mean.loc[subset, "f1"])
        base_rouge = float(baseline_mean.loc[subset, "rougeL"])
        graph_f1 = float(graphrag_mean.loc[subset, "f1"])
        graph_rouge = float(graphrag_mean.loc[subset, "rougeL"])

        rows.append(
            {
                "dataset": subset,
                "metric": "F1-Score",
                "standard_baseline_mean": round(base_f1, 4),
                "graphrag_mean": round(graph_f1, 4),
                "improvement_delta_percent": round(_improvement(graph_f1, base_f1), 2),
            }
        )
        rows.append(
            {
                "dataset": subset,
                "metric": "ROUGE-L",
                "standard_baseline_mean": round(base_rouge, 4),
                "graphrag_mean": round(graph_rouge, 4),
                "improvement_delta_percent": round(_improvement(graph_rouge, base_rouge), 2),
            }
        )

    out_df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(out_df)
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
