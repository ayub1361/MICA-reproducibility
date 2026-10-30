"""Per-city point-of-interest catalogue construction."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

TOURISM_KEYWORDS = [
    "Museums", "Art Galleries", "Landmarks", "Historical Tours", "Parks",
    "Zoos", "Aquariums", "Beaches", "Hiking", "Botanical Gardens",
    "Wineries", "Breweries", "Distilleries", "Tours", "Observatories",
    "Amusement Parks", "Bowling", "Theaters", "Performing Arts", "Music Venues",
    "Restaurants", "Cafes", "Bakeries", "Ice Cream & Frozen Yogurt",
]

GOOGLE_TOURISM_KEYWORDS = [
    "restaurant", "cafe", "coffee shop", "bakery", "sandwich shop",
    "ice cream", "frozen yogurt", "deli", "diner", "bistro", "food court",
    "bar", "pub", "brewery", "winery", "distillery", "taproom",
    "museum", "art gallery", "gallery", "theater", "theatre", "opera house",
    "concert hall", "live music venue", "night club", "performing arts",
    "landmark", "historical place", "monument", "observatory",
    "park", "garden", "zoo", "aquarium", "beach", "hiking area", "trail",
    "nature preserve", "scenic spot",
    "tourist attraction", "amusement", "bowling", "tour", "sightseeing",
    "arcade", "stadium", "arena",
]

EARTH_KM = 6371.0

def haversine_matrix(lat, lon):
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

_GOOGLE_TOURISM = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in GOOGLE_TOURISM_KEYWORDS) + r")\b")


def is_tourism(categories: str, dataset: str = "yelp") -> bool:
    if not isinstance(categories, str):
        return False
    if dataset == "google":
        return bool(_GOOGLE_TOURISM.search(categories.lower()))
    return any(k.lower() in categories.lower() for k in TOURISM_KEYWORDS)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--business", type=Path, default=Path("data/interim/business.parquet"))
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--tag", type=str, default="pre_pandemic")
    p.add_argument("--dataset", choices=["yelp", "google"], default="yelp",
                   help="which category vocabulary to filter POIs with; the two corpora do not share one")
    p.add_argument("--cities", nargs="+", default=None,
                   help="explicit city list; default = top-N by eligible POI count")
    p.add_argument("--top-cities", type=int, default=2)
    p.add_argument("--min-reviews", type=int, default=20)
    p.add_argument("--max-pois-per-city", type=int, default=400)
    p.add_argument("--knn", type=int, default=10)
    p.add_argument("--max-transition-km", type=float, default=8.0)
    a = p.parse_args()

    business = pd.read_parquet(a.business)
    roster = json.loads((a.processed_dir / f"warm_roster_{a.tag}.json").read_text(encoding="utf-8"))
    warm_pois = set(roster["pois"])

    frame = business.loc[
        business["business_id"].isin(warm_pois)
        & (business["is_open"].fillna(0) == 1)
        & (business["review_count"].fillna(0) >= a.min_reviews)
        & business["categories"].map(lambda c: is_tourism(c, a.dataset))
        & business["hours"].notna()
    ].copy()
    print(f"eligible POIs after operational filters: {len(frame):,}")

    counts = frame["city"].value_counts()
    cities = a.cities or counts.head(a.top_cities).index.tolist()
    print("\nTop cities by eligible POI count:")
    print(counts.head(10).to_string())
    print(f"\nselected: {cities}")

    out_dir = a.processed_dir / "cities"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for city in cities:
        sub = frame.loc[frame["city"] == city].copy()
        sub = sub.sort_values("review_count", ascending=False).head(a.max_pois_per_city)
        sub = sub.reset_index(drop=True)
        if len(sub) < 20:
            print(f"  skip {city}: only {len(sub)} POIs")
            continue

        distance = haversine_matrix(sub["latitude"].to_numpy(), sub["longitude"].to_numpy())
        np.fill_diagonal(distance, np.inf)

        edges = []
        for i in range(len(sub)):
            order = np.argsort(distance[i])[: a.knn]
            for j in order:
                if distance[i, j] <= a.max_transition_km:
                    edges.append((sub.at[i, "business_id"], sub.at[j, "business_id"],
                                  float(distance[i, j])))

        slug = city.lower().replace(" ", "_")
        sub.to_parquet(out_dir / f"pois_{slug}_{a.tag}.parquet", index=False)
        pd.DataFrame(edges, columns=["from_id", "to_id", "km"]).to_parquet(
            out_dir / f"edges_{slug}_{a.tag}.parquet", index=False)

        finite = distance[np.isfinite(distance)]
        summary[city] = {
            "pois": int(len(sub)),
            "edges": int(len(edges)),
            "median_pairwise_km": float(np.median(finite)),
            "mean_review_count": float(sub["review_count"].mean()),
        }
        print(f"  {city}: {len(sub)} POIs, {len(edges)} edges, "
              f"median pairwise {np.median(finite):.2f} km")

    (out_dir / f"city_summary_{a.tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
