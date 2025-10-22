#!/usr/bin/env python3
"""
Geographic Analysis Script for Article Titles

This script reads combined_articles.csv or parquet files and analyzes article titles
to extract countries and cities using GPT-4o, then organizes them by continent.
The LLM handles multilingual text and maps cities to countries and countries to continents.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Set, Tuple
from openai import OpenAI
import argparse
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


def extract_geographic_entities_batch(texts: List[str]) -> List[List[Dict[str, str]]]:
    """
    Use GPT-4o to extract countries and cities from multiple texts in a single API call.
    Returns a list of entity lists, one for each input text.
    """
    try:
        # Create a single prompt for all texts
        texts_str = "\n\n".join([f"Text {i+1}: {text}" for i, text in enumerate(texts)])

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a geographic entity extraction expert. Extract countries and cities from the given texts and map them to their continents.
                    Return ONLY a JSON object with this exact structure:
                    {
                        "results": [
                            {
                                "text_index": 0,
                                "entities": [
                                    {
                                        "name": "country or city name in ENGLISH",
                                        "type": "country" or "city",
                                        "country": "country name in ENGLISH (if entity is a city, otherwise same as name)",
                                        "continent": "continent name in ENGLISH"
                                    }
                                ]
                            }
                        ]
                    }
                    
                    Rules:
                    - Process each text separately and return results in order
                    - ALWAYS return entity names in ENGLISH, even if the original text is in another language
                    - For cities, provide the country they belong to in ENGLISH
                    - For countries, country field should be the same as name (both in ENGLISH)
                    - Map each entity to its correct continent in ENGLISH
                    - Be comprehensive but accurate
                    - Return empty entities array if no geographic entities found
                    - Use standard English names (e.g., "Kenya" not "كينيا", "India" not "भारत", "Qatar" not "قطر")
                    - Handle multiple countries/cities in the same text
                    - If you can't determine the country for a city, set country to "Unknown"
                    - If you can't determine the continent, set continent to "Unknown"
                    
                    Continents: Africa, Asia, Europe, North America, South America, Oceania""",
                },
                {
                    "role": "user",
                    "content": f"Extract countries and cities from these texts and map to continents:\n\n{texts_str}",
                },
            ],
            max_completion_tokens=4000,  # Increased for batch processing
        )

        content = response.choices[0].message.content

        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.endswith("```"):
            content = content[:-3]  # Remove ```
        content = content.strip()

        result = json.loads(content)
        results = result.get("results", [])

        # Ensure we return results in the same order as input texts
        entity_lists = []
        for i in range(len(texts)):
            # Find the result for this text index
            text_result = next(
                (r for r in results if r.get("text_index") == i), {"entities": []}
            )
            entity_lists.append(text_result.get("entities", []))

        return entity_lists

    except json.JSONDecodeError as e:
        print(f"JSON decode error in batch processing: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        return [[] for _ in texts]  # Return empty lists for all texts
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return [[] for _ in texts]  # Return empty lists for all texts


def analyze_articles(
    file_path: str, sample_size: int = None, batch_size: int = 10
) -> Dict:
    """
    Analyze articles for geographic mentions and organize by continent.
    """
    print(f"Loading data from {file_path}...")

    # Load data
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    print(f"Loaded {len(df)} articles")

    # Sample data if specified
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Using sample of {len(df)} articles")

    # Initialize results
    results = {
        "total_articles": len(df),
        "articles_analyzed": 0,
        "countries": {},
    }

    # Process articles in batches for speed
    BATCH_SIZE = batch_size
    titles = []
    article_data = []

    # Prepare data for batch processing
    for idx, row in df.iterrows():
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        if title.strip():
            titles.append(title)
            article_data.append({"index": idx, "row": row})

    print(f"Processing {len(titles)} articles in batches of {BATCH_SIZE}...")

    # Process in batches
    for i in tqdm(range(0, len(titles), BATCH_SIZE), desc="Processing batches"):
        batch_titles = titles[i : i + BATCH_SIZE]
        batch_data = article_data[i : i + BATCH_SIZE]

        # Extract entities for this batch
        batch_entities = extract_geographic_entities_batch(batch_titles)

        # Process results for this batch
        for j, (entities, data) in enumerate(zip(batch_entities, batch_data)):
            idx = data["index"]
            row = data["row"]
            title = batch_titles[j]

            # Process each geographic entity
            for entity in entities:
                country = entity.get("country", "Unknown")
                continent = entity.get("continent", "Unknown")
                entity_name = entity.get("name", "")
                entity_type = entity.get("type", "unknown")

                # Skip if country is unknown
                if country == "Unknown":
                    continue

                # Initialize country data if not exists
                if country not in results["countries"]:
                    results["countries"][country] = {
                        "mentions": 0,
                        "articles": [],
                        "continent": continent,
                        "cities": set(),  # Track cities mentioned for this country
                    }

                # Update country data
                results["countries"][country]["mentions"] += 1
                results["countries"][country]["articles"].append(
                    {
                        "article_id": row.get("article_id", idx),
                        "title": title,
                        "url": row.get("url", ""),
                        "site_name": row.get("site_name", ""),
                        "entity_name": entity_name,
                        "entity_type": entity_type,
                    }
                )

                # Track cities for this country
                if entity_type == "city" and entity_name:
                    results["countries"][country]["cities"].add(entity_name)

            results["articles_analyzed"] += 1

        # Small delay between batches to avoid rate limiting
        time.sleep(0.5)

    # Convert sets to lists for JSON serialization
    for country, data in results["countries"].items():
        # Convert cities set to list for JSON serialization
        data["cities"] = list(data["cities"])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze geographic mentions in article titles"
    )
    parser.add_argument(
        "--file",
        "-f",
        default="data/combined_articles.csv",
        help="Path to CSV or parquet file",
    )
    parser.add_argument(
        "--sample", "-s", type=int, default=None, help="Sample size (for testing)"
    )
    parser.add_argument(
        "--output", "-o", default="geographic_analysis.json", help="Output JSON file"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)",
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        return

    # Run analysis
    results = analyze_articles(args.file, args.sample, args.batch_size)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis complete!")
    print(f"Articles analyzed: {results['articles_analyzed']}")
    print(f"Countries found: {len(results['countries'])}")
    print(f"Results saved to: {args.output}")

    # Print summary by country
    print("\nSummary by Country:")
    # Sort countries by mention count
    sorted_countries = sorted(
        results["countries"].items(), key=lambda x: x[1]["mentions"], reverse=True
    )

    for country, data in sorted_countries[:10]:  # Show top 10 countries
        print(f"{country}: {data['mentions']} mentions ({data['continent']})")
        if data["cities"]:
            print(
                f"  Cities: {', '.join(data['cities'][:3])}{'...' if len(data['cities']) > 3 else ''}"
            )


if __name__ == "__main__":
    main()
