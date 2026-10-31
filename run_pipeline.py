#!/usr/bin/env python3
"""
RetailIQ — Full Pipeline Runner
================================
Runs the complete location intelligence pipeline:
  1. Data ingestion (H3 grid + POIs + buildings + roads)
  2. Feature engineering (26 spatial features per hexagon)
  3. Demand estimation (Gradient Boosted Regression)
  4. Network gap analysis
  5. Composite site scoring (0–100, A–E grading)

Usage:
    python run_pipeline.py            # Run full pipeline
    python run_pipeline.py --live     # Use live OSM data (requires network)
    streamlit run src/dashboard.py    # Launch dashboard after pipeline
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_ingestion import run_ingestion
from feature_engineering import build_feature_matrix
from models import run_modeling


def main():
    parser = argparse.ArgumentParser(description="RetailIQ Pipeline")
    parser.add_argument("--live", action="store_true",
                       help="Use live OSM data via Overpass API")
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("  RetailIQ — Location Intelligence for Retail Site Selection")
    print("█" * 60 + "\n")

    # Step 1: Data Ingestion
    grid, pois, buildings, roads = run_ingestion(use_live_osm=args.live)

    # Step 2: Feature Engineering
    feature_matrix = build_feature_matrix(grid, pois, buildings, roads)

    # Step 3: Modeling & Scoring
    results = run_modeling(feature_matrix)

    print("\n" + "█" * 60)
    print("  Pipeline Complete!")
    print("█" * 60)
    print("\n  Outputs:")
    print("    data/processed/h3_grid.geojson      — H3 hexagonal grid")
    print("    data/processed/feature_matrix.parquet — 26-feature matrix")
    print("    data/processed/site_scores.parquet   — Site scores & grades")
    print("    outputs/demand_model.pkl             — Trained GBR model")
    print("    outputs/scaler.pkl                   — Feature scaler")
    print("\n  Launch dashboard:")
    print("    streamlit run src/dashboard.py")
    print()

    return results


if __name__ == "__main__":
    main()
