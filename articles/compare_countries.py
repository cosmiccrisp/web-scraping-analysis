#!/usr/bin/env python3
"""
Script to compare country names between geo_analysis.json and countries-110m.json
"""

import json
import re


def extract_geo_countries():
    """Extract country names from geo_analysis.json"""
    with open(
        "/Users/paigelee/Desktop/web-scraping-analysis/visualization/geo_analysis.json",
        "r",
    ) as f:
        data = json.load(f)

    countries = set()
    for country_name in data.get("countries", {}).keys():
        countries.add(country_name)

    return countries


def extract_map_countries():
    """Extract country names from countries-110m.json"""
    with open(
        "/Users/paigelee/Desktop/web-scraping-analysis/visualization/countries-110m.json",
        "r",
    ) as f:
        data = json.load(f)

    countries = set()
    for feature in data.get("features", []):
        country_name = feature.get("properties", {}).get("name")
        if country_name:
            countries.add(country_name)

    return countries


def normalize_name(name):
    """Normalize country names for comparison"""
    # Convert to lowercase and remove common variations
    normalized = name.lower().strip()

    # Handle common variations
    variations = {
        "usa": "united states",
        "uk": "united kingdom",
        "uae": "united arab emirates",
        "dr congo": "democratic republic of the congo",
        "drc": "democratic republic of the congo",
        "rd congo": "democratic republic of the congo",
        "congo, democratic republic of the": "democratic republic of the congo",
        "côte d'ivoire": "ivory coast",
        "brasil": "brazil",
        "korea": "south korea",
        "north korea": "north korea",
        "south korea": "south korea",
        "republic of the congo": "congo",
        "congo": "congo",
        "central african republic": "central african republic",
        "burkina faso": "burkina faso",
        "côte d'ivoire": "ivory coast",
        "ivory coast": "ivory coast",
        "united arab emirates": "united arab emirates",
        "united states": "united states",
        "united kingdom": "united kingdom",
        "south africa": "south africa",
        "south sudan": "south sudan",
        "north korea": "north korea",
        "south korea": "south korea",
        "new zealand": "new zealand",
        "czech republic": "czech republic",
        "bosnia and herzegovina": "bosnia and herzegovina",
        "vatican city": "vatican city",
        "state of palestine": "palestine",
        "palestinian territories": "palestine",
        "western sahara": "western sahara",
        "cape verde": "cape verde",
        "costa rica": "costa rica",
        "dominican republic": "dominican republic",
        "el salvador": "el salvador",
        "trinidad and tobago": "trinidad",
        "trinidad": "trinidad",
        "sri lanka": "sri lanka",
        "czech republic": "czech republic",
        "slovakia": "slovakia",
        "myanmar": "myanmar",
        "kazakhstan": "kazakhstan",
        "vatican city": "vatican city",
        "suriname": "suriname",
        "kosovo": "kosovo",
        "norway": "norway",
        "laos": "laos",
        "georgia": "georgia",
        "aruba": "aruba",
        "panama": "panama",
        "paraguay": "paraguay",
        "honduras": "honduras",
        "guatemala": "guatemala",
        "nicaragua": "nicaragua",
        "chile": "chile",
        "nauru": "nauru",
        "guyana": "guyana",
        "finland": "finland",
        "european union": "european union",
        "eu": "european union",
        "america": "united states",
        "britain": "united kingdom",
        "england": "united kingdom",
        "scotland": "united kingdom",
        "wales": "united kingdom",
        "northern ireland": "united kingdom",
        "catalonia": "spain",
        "tibet": "china",
        "papua": "papua new guinea",
        "caribbean": "caribbean",
        "latin america": "latin america",
        "east africa": "east africa",
        "africa": "africa",
        "europe": "europe",
        "asian": "asia",
        "african": "africa",
        "none": "none",
        "sahrawi": "western sahara",
        "somaliland": "somalia",
        "mayotte": "mayotte",
        "martinique": "martinique",
        "aruba": "aruba",
        "northern cyprus": "cyprus",
        "cyprus": "cyprus",
        "french southern and antarctic lands": "french southern and antarctic lands",
        "antarctica": "antarctica",
    }

    return variations.get(normalized, normalized)


def main():
    print("Extracting country names from both files...")

    geo_countries = extract_geo_countries()
    map_countries = extract_map_countries()

    print(f"Found {len(geo_countries)} countries in geo_analysis.json")
    print(f"Found {len(map_countries)} countries in countries-110m.json")

    # Normalize all names for comparison
    geo_normalized = {normalize_name(name): name for name in geo_countries}
    map_normalized = {normalize_name(name): name for name in map_countries}

    print("\n=== COUNTRIES IN GEO_ANALYSIS.JSON BUT NOT IN COUNTRIES-110M.JSON ===")
    geo_only = set(geo_normalized.keys()) - set(map_normalized.keys())
    for norm_name in sorted(geo_only):
        original_name = geo_normalized[norm_name]
        print(f"  - {original_name}")

    print(f"\nTotal: {len(geo_only)} countries")

    print("\n=== COUNTRIES IN COUNTRIES-110M.JSON BUT NOT IN GEO_ANALYSIS.JSON ===")
    map_only = set(map_normalized.keys()) - set(geo_normalized.keys())
    for norm_name in sorted(map_only):
        original_name = map_normalized[norm_name]
        print(f"  - {original_name}")

    print(f"\nTotal: {len(map_only)} countries")

    print("\n=== POTENTIAL MATCHES (SIMILAR NAMES) ===")
    # Look for potential matches with slight differences
    for geo_norm in geo_normalized:
        if geo_norm not in map_normalized:
            for map_norm in map_normalized:
                if geo_norm in map_norm or map_norm in geo_norm:
                    if geo_norm != map_norm:
                        print(
                            f"  Potential match: '{geo_normalized[geo_norm]}' <-> '{map_normalized[map_norm]}'"
                        )


if __name__ == "__main__":
    main()
