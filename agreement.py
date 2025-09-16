"""Compute transcription agreement statistics from existing ASR outputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yaml

from transcription_agreement import enrich_dataframe

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV file with ASR transcripts.")
    parser.add_argument("config", type=Path, help="YAML file describing ASR columns and threshold.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("updated_agreement.csv"),
        help="Path for the enriched CSV output.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar while processing rows.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in CSV: {', '.join(missing)}")
    return list(columns)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()
    LOGGER.info("Loading configuration from %s", args.config)
    config = load_config(args.config)

    model_columns = config.get("asr_models", [])
    if not model_columns:
        raise ValueError("The configuration must define an 'asr_models' list.")

    threshold = float(config.get("threshold", 0.0))

    LOGGER.info("Reading transcripts from %s", args.csv)
    df = pd.read_csv(args.csv)
    ensure_columns(df, model_columns)

    LOGGER.info("Computing agreement across %d models", len(model_columns))
    enriched = enrich_dataframe(
        df,
        model_columns,
        threshold,
        show_progress=not args.no_progress,
    )

    LOGGER.info("Writing enriched dataset to %s", args.output)
    enriched.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
