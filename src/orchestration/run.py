"""
run.py
------
CLI entry point for running the AML investigation pipeline directly.

Usage:
    python src/orchestration/run.py \
        --file data/raw/transactions.csv \
        --account ACC_000123 \
        --hops 2 \
        --window 30
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AML Investigation Pipeline — Team 38"
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the raw transaction CSV file.",
    )
    parser.add_argument(
        "--account", "-a",
        required=True,
        help="Account ID to investigate.",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Graph expansion hop radius (default: 2).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Time window in days (default: 30).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Optional path to save the final report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate input file
    if not Path(args.file).exists():
        logger.error(f"Transaction file not found: {args.file}")
        sys.exit(1)

    # Import here so env vars are loaded first
    from src.orchestration.state import initial_state
    from src.orchestration.graph import aml_pipeline

    state = initial_state(
        raw_transaction_path=args.file,
        account_id=args.account,
        hop_radius=args.hops,
        time_window_days=args.window,
    )

    logger.info(
        f"Starting AML investigation | account={args.account} | "
        f"hops={args.hops} | window={args.window}d"
    )

    result = aml_pipeline.invoke(state)

    # Print errors if any
    if result.get("errors"):
        logger.warning(f"Pipeline completed with errors: {result['errors']}")

    # Display final report
    report = result.get("final_report", {})
    print("\n" + "=" * 60)
    print("INVESTIGATION COMPLETE")
    print("=" * 60)
    print(f"Account:    {report.get('account_id')}")
    print(f"Risk Score: {report.get('risk_score', 0):.4f}")
    print(f"Risk Tier:  {report.get('risk_tier')}")
    print(f"Patterns:   {', '.join(report.get('detected_patterns', [])) or 'None'}")

    if report.get("sar_narrative"):
        print("\n--- SAR NARRATIVE ---")
        print(report["sar_narrative"])
    elif report.get("exit_summary"):
        print(f"\n{report['exit_summary']}")

    print("=" * 60 + "\n")

    # Optionally save to JSON
    output_path = args.output or f"data/reports/{args.account}_report.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Remove non-serialisable objects before saving
    report_to_save = {
        k: v for k, v in report.items()
        if not hasattr(v, "__dataframe__")  # exclude DataFrames
    }
    with open(output_path, "w") as f:
        json.dump(report_to_save, f, indent=2, default=str)

    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()