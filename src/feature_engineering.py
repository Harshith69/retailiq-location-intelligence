"""
RetailIQ — Feature Engineering
==============================
Indexes all spatial data to H3 hexagons and builds the feature matrix
used by the demand estimation and site scoring models.

Features per hexagon:
  - Population density proxy (building count × levels)
  - POI count by category (8 categories)
  - Catchment population (rings at 0.5, 1, 2, 3, 5 km)
  - Road density and accessibility score
  - Competition index (POI saturation relative to population)
  - POI diversity (Shannon entropy across categories)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from pathlib import Path

from config import (
    H3_RESOLUTION, CATCHMENT_RINGS_KM, POI_CATEGORIES,
    CITY_CENTER,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def _h3_distance_km(h1: str, h2: str) -> float:
    """Approximate distance in km between two H3 cell centers."""
    lat1, lon1 = h3.cell_to_latlng(h1)
    lat2, lon2 = h3.cell_to_latlng(h2)
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def index_pois_to_hex(grid: gpd.GeoDataFrame,
                       pois: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Count POIs per H3 hexagon, broken out by category.
    Returns a DataFrame indexed by hex_id with columns like
    poi_restaurant_count, poi_retail_count, etc.
    """
    print("  Indexing POIs to hexagons...")
    pois = pois.copy()
    pois["hex_id"] = pois.apply(
        lambda r: h3.latlng_to_cell(r.lat, r.lon, H3_RESOLUTION), axis=1
    )

    # Pivot: count per category per hex
    poi_counts = (
        pois.groupby(["hex_id", "category"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=lambda c: f"poi_{c}_count")
    )

    # Total POI count
    poi_counts["poi_total_count"] = poi_counts.sum(axis=1)

    # Merge with grid
    result = grid[["hex_id"]].set_index("hex_id").join(poi_counts, how="left").fillna(0)
    print(f"    → {len(poi_counts)} hexagons have POIs")
    return result


def index_buildings_to_hex(grid: gpd.GeoDataFrame,
                            buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute population density proxy per hexagon from building data.
    Proxy = count of buildings × average levels (floor area proxy).
    """
    print("  Indexing buildings to hexagons...")
    buildings = buildings.copy()
    buildings["hex_id"] = buildings.apply(
        lambda r: h3.latlng_to_cell(r.lat, r.lon, H3_RESOLUTION), axis=1
    )

    bld_stats = buildings.groupby("hex_id").agg(
        building_count=("levels", "count"),
        avg_levels=("levels", "mean"),
        total_floor_area_proxy=("levels", "sum"),
    )

    # Population proxy: total floor area × scaling factor
    # In reality, ~15 people per 1000 sqm residential floor area
    bld_stats["pop_density_proxy"] = bld_stats["total_floor_area_proxy"] * 3.5

    result = grid[["hex_id"]].set_index("hex_id").join(bld_stats, how="left").fillna(0)
    print(f"    → {(result['building_count'] > 0).sum()} hexagons have buildings")
    return result


def index_roads_to_hex(grid: gpd.GeoDataFrame,
                        roads: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute road density per hexagon (total road length in meters).
    """
    print("  Computing road density per hexagon...")
    # Project to UTM for length in meters
    roads_utm = roads.to_crs(epsg=32643)  # UTM 43N covers Bangalore

    road_features = {}
    grid_utm = grid.to_crs(epsg=32643)

    for _, hex_row in grid_utm.iterrows():
        hex_id = hex_row["hex_id"]
        hex_geom = hex_row.geometry

        # Find roads that intersect this hex
        intersecting = roads_utm[roads_utm.intersects(hex_geom)]
        if len(intersecting) > 0:
            clipped = intersecting.intersection(hex_geom)
            total_length = clipped.length.sum()
            # Weight by road type
            type_weights = {"primary": 3, "secondary": 2, "tertiary": 1,
                           "motorway": 4, "trunk": 3.5}
            weighted_length = 0
            for _, road in intersecting.iterrows():
                clip = road.geometry.intersection(hex_geom)
                w = type_weights.get(road.road_type, 1)
                weighted_length += clip.length * w

            road_features[hex_id] = {
                "road_length_m": total_length,
                "road_density": total_length / hex_geom.area * 1e6,
                "weighted_road_access": weighted_length,
            }
        else:
            road_features[hex_id] = {
                "road_length_m": 0, "road_density": 0, "weighted_road_access": 0,
            }

    df = pd.DataFrame.from_dict(road_features, orient="index")
    df.index.name = "hex_id"
    print(f"    → {(df['road_length_m'] > 0).sum()} hexagons have roads")
    return df


def compute_catchment_population(grid: gpd.GeoDataFrame,
                                  pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each hexagon, sum population within concentric rings.
    Uses H3 k-ring neighbors for efficient spatial lookup.
    """
    print("  Computing catchment population rings...")
    hex_ids = grid["hex_id"].values
    pop_dict = pop_df["pop_density_proxy"].to_dict()

    # Approximate k values for each ring radius at res 8
    # Edge length at res 8 ≈ 0.46 km
    edge_km = 0.461
    ring_k = {r: max(1, int(r / edge_km)) for r in CATCHMENT_RINGS_KM}

    catchment = {}
    for hid in hex_ids:
        row = {}
        for radius_km, k in ring_k.items():
            neighbors = h3.grid_disk(hid, k)
            total_pop = sum(pop_dict.get(n, 0) for n in neighbors)
            row[f"catchment_pop_{radius_km}km"] = total_pop
        catchment[hid] = row

    df = pd.DataFrame.from_dict(catchment, orient="index")
    df.index.name = "hex_id"
    print(f"    → Computed {len(CATCHMENT_RINGS_KM)} ring radii for {len(df)} hexagons")
    return df


def compute_competition_index(poi_df: pd.DataFrame,
                               pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Competition index = POI density relative to population.
    High competition = many POIs per capita (saturated market).
    """
    print("  Computing competition index...")
    pop = pop_df[["pop_density_proxy"]].copy()
    pois = poi_df[["poi_total_count"]].copy()

    merged = pop.join(pois, how="left").fillna(0)
    # Avoid division by zero
    merged["competition_index"] = np.where(
        merged["pop_density_proxy"] > 0,
        merged["poi_total_count"] / (merged["pop_density_proxy"] / 100),
        0,
    )
    # Normalize to 0-1
    cmax = merged["competition_index"].quantile(0.99)
    merged["competition_index"] = np.clip(merged["competition_index"] / cmax, 0, 1)
    return merged[["competition_index"]]


def compute_poi_diversity(poi_df: pd.DataFrame) -> pd.DataFrame:
    """
    POI diversity using Shannon entropy across categories.
    Higher entropy = more diverse commercial ecosystem.
    """
    print("  Computing POI diversity (Shannon entropy)...")
    cat_cols = [c for c in poi_df.columns if c.startswith("poi_") and c.endswith("_count")
                and c != "poi_total_count"]

    counts = poi_df[cat_cols].copy()
    totals = counts.sum(axis=1)

    # Shannon entropy
    entropy = pd.Series(0.0, index=counts.index)
    for col in cat_cols:
        p = counts[col] / totals.replace(0, 1)
        entropy -= np.where(p > 0, p * np.log2(p.replace(0, 1)), 0)

    # Normalize by max possible entropy
    max_entropy = np.log2(len(cat_cols))
    diversity = entropy / max_entropy

    return pd.DataFrame({"poi_diversity_score": diversity}, index=counts.index)


def compute_distance_to_center(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """Distance from each hex center to CBD."""
    print("  Computing distance to city center...")
    distances = {}
    for _, row in grid.iterrows():
        lat, lon = row.center_lat, row.center_lon
        R = 6371
        dlat = np.radians(lat - CITY_CENTER[0])
        dlon = np.radians(lon - CITY_CENTER[1])
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(CITY_CENTER[0])) * np.cos(np.radians(lat)) *
             np.sin(dlon / 2) ** 2)
        d = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances[row.hex_id] = {"dist_to_center_km": d}
    df = pd.DataFrame.from_dict(distances, orient="index")
    df.index.name = "hex_id"
    return df


def build_feature_matrix(grid: gpd.GeoDataFrame,
                          pois: gpd.GeoDataFrame,
                          buildings: gpd.GeoDataFrame,
                          roads: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Master function: build the complete feature matrix for all hexagons.
    """
    print(f"\n{'='*60}")
    print("RetailIQ Feature Engineering")
    print(f"{'='*60}\n")

    # 1. Index POIs
    print("[1/6] POI Features")
    poi_features = index_pois_to_hex(grid, pois)

    # 2. Index buildings (population proxy)
    print("\n[2/6] Population Proxy")
    pop_features = index_buildings_to_hex(grid, buildings)

    # 3. Road density
    print("\n[3/6] Road Accessibility")
    road_features = index_roads_to_hex(grid, roads)

    # 4. Catchment population
    print("\n[4/6] Catchment Population")
    catchment_features = compute_catchment_population(grid, pop_features)

    # 5. Competition index
    print("\n[5/6] Competition Index")
    competition = compute_competition_index(poi_features, pop_features)

    # 6. POI diversity
    print("\n[6/6] POI Diversity")
    diversity = compute_poi_diversity(poi_features)

    # Distance to center
    dist_center = compute_distance_to_center(grid)

    # Merge everything
    print("\n  Assembling feature matrix...")
    feature_matrix = (
        grid[["hex_id", "center_lat", "center_lon"]]
        .set_index("hex_id")
        .join(pop_features)
        .join(poi_features)
        .join(road_features)
        .join(catchment_features)
        .join(competition)
        .join(diversity)
        .join(dist_center)
        .fillna(0)
    )

    # Save
    feature_matrix.to_parquet(DATA_DIR / "processed" / "feature_matrix.parquet")
    print(f"\n  Feature matrix: {feature_matrix.shape[0]} hexagons × {feature_matrix.shape[1]} features")
    print(f"  Saved: feature_matrix.parquet")

    print(f"\n{'='*60}")
    print("Feature engineering complete!")
    print(f"{'='*60}")

    return feature_matrix


if __name__ == "__main__":
    # Load data
    grid = gpd.read_file(DATA_DIR / "processed" / "h3_grid.geojson")
    pois = gpd.read_file(DATA_DIR / "raw" / "pois.geojson")
    buildings = gpd.read_file(DATA_DIR / "raw" / "buildings.geojson")
    roads = gpd.read_file(DATA_DIR / "raw" / "roads.geojson")

    fm = build_feature_matrix(grid, pois, buildings, roads)
    print(f"\nTop features by variance:")
    print(fm.var().sort_values(ascending=False).head(10))
