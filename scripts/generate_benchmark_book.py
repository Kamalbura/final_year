from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured LaTeX benchmark book from local + Kaggle outputs.")
    parser.add_argument(
        "--output",
        default="outputs/book_assets/tex/aqi_model_benchmark_book.tex",
        help="LaTeX output path",
    )
    return parser.parse_args()


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def discover_summaries(root: Path) -> list[Path]:
    files = list(root.glob("outputs/**/benchmark_summary.csv"))
    files.extend(root.glob("kaggle/**/benchmark_summary.csv"))
    filtered: list[Path] = []
    for file in files:
        if file.exists() and file.stat().st_size > 10:
            filtered.append(file)
    return sorted(set(filtered))


def source_kind(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()
    if "kaggle_kernel_latest" in text:
        return "kaggle_kernel_latest"
    if "kaggle_benchmarks" in text:
        return "kaggle_benchmarks"
    if "smoke_bench" in text:
        return "local_smoke_bench"
    if "kaggle/" in text:
        return "kaggle_misc"
    return "local_other"


def source_env(kind: str) -> str:
    return "kaggle" if kind.startswith("kaggle") else "local_pc"


def load_all_summaries(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []

    for path in discover_summaries(root):
        frame = pd.read_csv(path, keep_default_na=False)
        kind = source_kind(path)
        mtime = path.stat().st_mtime
        frame["source_kind"] = kind
        frame["source_path"] = str(path)
        frame["source_mtime"] = mtime
        rows.append(frame)

        inventory.append(
            {
                "source_kind": kind,
                "source_env": source_env(kind),
                "path": str(path),
                "mtime": pd.to_datetime(mtime, unit="s"),
                "rows": int(len(frame)),
                "ok": int((frame["status"] == "ok").sum()) if "status" in frame.columns else 0,
                "skipped": int((frame["status"] == "skipped").sum()) if "status" in frame.columns else 0,
                "failed": int((frame["status"] == "failed").sum()) if "status" in frame.columns else 0,
            }
        )

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    all_rows = pd.concat(rows, ignore_index=True)
    inv = pd.DataFrame(inventory).sort_values("mtime", ascending=False)
    return all_rows, inv


def load_individual_trainers(root: Path) -> pd.DataFrame:
    files = list(root.glob("outputs/individual_trainers/**/results.json"))
    payloads: list[dict[str, object]] = []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = data.get("config", {})
        short_config = " ".join(
            [
                f"h={config.get('hidden_dim', config.get('model_dim', '-'))}",
                f"layers={config.get('num_layers', config.get('layers', '-'))}",
                f"heads={config.get('heads', '-')}",
                f"lr={config.get('lr', '-')}",
                f"drop={config.get('dropout', '-')}",
                f"epochs={config.get('epochs', '-')}",
            ]
        )
        ts = str(data.get("timestamp", "")).replace("T", " ")
        if "." in ts:
            ts = ts.split(".", 1)[0]
        payloads.append(
            {
                "city": data.get("city", ""),
                "model": data.get("model", ""),
                "timestamp": ts,
                "rmse": _as_float(data.get("metrics", {}).get("rmse")),
                "mae": _as_float(data.get("metrics", {}).get("mae")),
                "config": short_config,
                "path": str(file),
            }
        )
    return pd.DataFrame(payloads)


def audit_skipped_with_artifacts(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    checks: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        status = str(row.get("status", ""))
        family = str(row.get("family", ""))
        if status != "skipped" or family != "deep_learning":
            continue
        source_path = Path(str(row.get("source_path", "")))
        city = str(row.get("city", "")).strip().lower()
        model = str(row.get("model", ""))
        candidate = source_path.parent / city / f"{safe_model_name(model)}.pt"
        checks.append(
            {
                "source_kind": str(row.get("source_kind", "")),
                "city": city,
                "model": model,
                "status": status,
                "artifact_found": candidate.exists(),
                "artifact_path": str(candidate),
                "reason": str(row.get("reason", ""))[:140],
            }
        )
    return pd.DataFrame(checks)


def latest_rows_per_source(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    keep_sources = []
    for source in rows["source_kind"].unique():
        part = rows[rows["source_kind"] == source]
        latest = part["source_mtime"].max()
        keep_sources.append(part[part["source_mtime"] == latest])
    return pd.concat(keep_sources, ignore_index=True)


def source_inventory_table(inv: pd.DataFrame) -> str:
    if inv.empty:
        return r"\textit{No benchmark summary files were discovered.}"
    lines = [
        r"\begin{longtable}{@{}L{3.1cm}L{1.8cm}L{2.3cm}L{0.9cm}L{0.9cm}L{1.1cm}L{1.1cm}@{}}",
        r"\toprule",
        r"Source & Env & Timestamp & Rows & OK & Skipped & Failed \\",
        r"\midrule",
    ]
    for _, row in inv.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row["source_kind"]),
                    latex_escape(row["source_env"]),
                    latex_escape(pd.to_datetime(row["mtime"]).strftime("%Y-%m-%d %H:%M")),
                    str(int(row["rows"])),
                    str(int(row["ok"])),
                    str(int(row["skipped"])),
                    str(int(row["failed"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def metrics_table(rows: pd.DataFrame) -> str:
    if rows.empty:
        return r"\textit{No benchmark metric rows available.}"
    view = rows.copy()
    view["rmse_num"] = view["rmse"].map(_as_float)
    view["mae_num"] = view["mae"].map(_as_float)
    view["r2_num"] = view["r2"].map(_as_float)
    view = view.sort_values(["source_kind", "city", "phase", "rmse_num"], na_position="last")
    lines = [
        r"\begin{longtable}{@{}L{2.3cm}L{1.4cm}L{2.5cm}L{2.2cm}L{1.0cm}L{0.9cm}L{0.9cm}L{0.9cm}@{}}",
        r"\toprule",
        r"Source & City & Phase & Model & Status & RMSE & MAE & R2 \\",
        r"\midrule",
    ]
    for _, row in view.iterrows():
        rmse = row["rmse_num"]
        mae = row["mae_num"]
        r2 = row["r2_num"]
        lines.append(
            " & ".join(
                [
                    latex_escape(row["source_kind"]),
                    latex_escape(row.get("city", "")),
                    latex_escape(row.get("phase", "")),
                    latex_escape(row.get("model", "")),
                    latex_escape(row.get("status", "")),
                    f"{rmse:.2f}" if rmse is not None else "--",
                    f"{mae:.2f}" if mae is not None else "--",
                    f"{r2:.2f}" if r2 is not None else "--",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def skipped_audit_table(audit: pd.DataFrame) -> str:
    if audit.empty:
        return r"\textit{No skipped deep-learning rows found for audit.}"
    lines = [
        r"\begin{longtable}{@{}L{2.7cm}L{1.4cm}L{2.6cm}L{1.4cm}L{1.8cm}L{4.1cm}@{}}",
        r"\toprule",
        r"Source & City & Model & Status & Artifact Found & Reason (truncated) \\",
        r"\midrule",
    ]
    for _, row in audit.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row["source_kind"]),
                    latex_escape(row["city"]),
                    latex_escape(row["model"]),
                    latex_escape(row["status"]),
                    "yes" if bool(row["artifact_found"]) else "no",
                    latex_escape(row["reason"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def individual_table(individual: pd.DataFrame) -> str:
    if individual.empty:
        return r"\textit{No individual trainer result files found under outputs/individual\_trainers.}"
    view = individual.sort_values("timestamp", ascending=False)
    lines = [
        r"\begin{longtable}{@{}L{1.5cm}L{1.8cm}L{2.8cm}L{0.9cm}L{0.9cm}L{5.9cm}@{}}",
        r"\toprule",
        r"City & Model & Timestamp & RMSE & MAE & Hyperparameter Config \\",
        r"\midrule",
    ]
    for _, row in view.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row["city"]),
                    latex_escape(row["model"]),
                    latex_escape(str(row["timestamp"])),
                    f"{row['rmse']:.2f}" if row["rmse"] is not None else "--",
                    f"{row['mae']:.2f}" if row["mae"] is not None else "--",
                    latex_escape(str(row["config"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def generate(rows: pd.DataFrame, inv: pd.DataFrame, audit: pd.DataFrame, individual: pd.DataFrame) -> str:
    total_rows = int(len(rows))
    ok_count = int((rows["status"] == "ok").sum()) if "status" in rows.columns else 0
    skipped_count = int((rows["status"] == "skipped").sum()) if "status" in rows.columns else 0
    failed_count = int((rows["status"] == "failed").sum()) if "status" in rows.columns else 0
    latest_rows = latest_rows_per_source(rows)

    return rf"""\documentclass[11pt,a4paper,oneside]{{book}}
\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs}}
\usepackage{{geometry}}
\usepackage{{hyperref}}
\usepackage{{longtable}}
\usepackage{{amsmath}}
\usepackage{{array}}
\usepackage{{xurl}}
\usepackage{{microtype}}
\usepackage{{lmodern}}
\geometry{{margin=1in}}
\newcolumntype{{L}}[1]{{>{{\raggedright\arraybackslash}}p{{#1}}}}
\setlength{{\LTleft}}{{0pt}}
\setlength{{\LTright}}{{0pt}}
\setlength{{\tabcolsep}}{{4pt}}
\setlength{{\emergencystretch}}{{3em}}
\sloppy
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=cyan, pdftitle={{AQI Benchmark Book (Local + Kaggle)}}}}

\title{{\Huge\textbf{{AQI Model Benchmark Book}}\\\Large Local PC + Kaggle Execution Audit}}
\author{{Final Year AQI Platform}}
\date{{\today}}

\begin{{document}}
\maketitle
\tableofcontents

\chapter{{Executive Summary}}
This report now includes both Kaggle outputs and locally executed training outputs from this PC.  
Key totals across all discovered benchmark summary files:
\begin{{itemize}}
  \item Total benchmark rows discovered: {total_rows}
  \item status=ok rows: {ok_count}
  \item status=skipped rows: {skipped_count}
  \item status=failed rows: {failed_count}
\end{{itemize}}
F1-score is not included because these are regression tasks (continuous AQI forecasting), not classification.

\chapter{{Run Source Inventory}}
\small
{source_inventory_table(inv)}
\normalsize

\chapter{{Execution Audit: Did Models Actually Run?}}
This section cross-checks skip records against expected artifact paths for deep-learning models.
\small
{skipped_audit_table(audit)}
\normalsize

\chapter{{Latest Rows Per Source (Readable Metrics)}}
\small
{metrics_table(latest_rows)}
\normalsize

\chapter{{All Discovered Benchmark Rows}}
\small
{metrics_table(rows)}
\normalsize

\chapter{{Local Individual Trainer Runs (Non-Kaggle)}}
These are dedicated local runs in \texttt{{outputs/individual\_trainers}} with per-city tuned configs.
\small
{individual_table(individual)}
\normalsize

\chapter{{Interpretation and Validation Notes}}
\begin{{itemize}}
  \item If a run is marked \texttt{{skipped}} and no matching artifact exists, it likely did not execute for that model in that run context.
  \item If artifacts exist under a different folder/run, that indicates successful execution elsewhere (different run profile or hardware).
  \item Use this report's source-specific tables instead of mixing rows from different runs.
  \item For publication-quality comparison, pin one canonical run family for each environment (e.g., latest Kaggle + latest local smoke/individual run).
\end{{itemize}}

\chapter{{References}}
\begin{{itemize}}
  \item \href{{https://github.com/josephmisiti/awesome-machine-learning}}{{Awesome Machine Learning}}
  \item \href{{https://github.com/ChristosChristofidis/awesome-deep-learning}}{{Awesome Deep Learning}}
\end{{itemize}}

\end{{document}}
"""


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    rows, inv = load_all_summaries(root)
    audit = audit_skipped_with_artifacts(rows)
    individual = load_individual_trainers(root)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate(rows, inv, audit, individual), encoding="utf-8")
    print(
        {
            "latex": str(output_path),
            "summary_rows": int(len(rows)),
            "sources": int(inv["source_kind"].nunique()) if not inv.empty else 0,
            "individual_runs": int(len(individual)),
        }
    )


if __name__ == "__main__":
    main()
