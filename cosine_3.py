import numpy as np
import faiss
import networkx as nx
import json
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import time

# ------------------
# Load embeddings and article dataframe
# ------------------
embeddings = np.load("embeddings/combined_embeddings.npy").astype("float32")
faiss.normalize_L2(embeddings)

# Try to load CSV first, fallback to parquet
try:
    df = pd.read_csv("data/combined_articles.csv")
    print("✅ Loaded articles from CSV")
except FileNotFoundError:
    try:
        df = pd.read_parquet("data/combined_articles.parquet")
        print("✅ Loaded articles from Parquet")
    except FileNotFoundError:
        print("❌ Could not find combined articles file")
        exit(1)

# ------------------
# Load site region mapping
# ------------------
try:
    sites_df = pd.read_csv("sites.csv")
    # Create mapping from domain to region using the "News Websites" column
    site_to_region = dict(zip(sites_df["News Websites"], sites_df["Region"]))
    print("✅ Loaded site region mapping")
    print(f"📊 Sample mappings: {dict(list(site_to_region.items())[:5])}")
except FileNotFoundError:
    print("⚠️  sites.csv not found, skipping region mapping")
    site_to_region = {}


# ------------------
# Add region information to articles
# ------------------
# Extract domain from full URL for mapping
def extract_domain(url):
    """Extract domain from full URL"""
    if pd.isna(url) or not url:
        return None
    # Remove https:// or http:// and get domain
    domain = str(url).replace("https://", "").replace("http://", "").split("/")[0]
    return domain


df["domain"] = df["site_name"].apply(extract_domain)
df["region"] = df["domain"].map(site_to_region).fillna("Unknown")
print(f"📊 Articles by region: {df['region'].value_counts().to_dict()}")


# ------------------
# LLM Labeling Function
# ------------------
def generate_cluster_label(titles, client):
    """
    Generate a descriptive label for a cluster based on its article titles.
    """
    if not titles or len(titles) == 0:
        return "Unknown Topic"

    # Randomly sample up to 20 titles to get better context
    import random

    sample_size = min(20, len(titles))
    sample_titles = random.sample(titles, sample_size)
    titles_text = "\n".join([f"- {title}" for title in sample_titles])

    prompt = f"""You are a political discourse expert and media analyst specializing in international news patterns and geopolitical narratives. Analyze the following article titles from various news sources and identify the core political theme or discourse.

{titles_text}

As a political discourse expert, provide a precise label that captures:
- The primary political actors, nations, or organizations involved
- The geopolitical or domestic political significance
- The underlying political narrative or discourse pattern
- Regional or global political implications
- Specific policy areas or political conflicts

The label should be:
- Politically precise and specific (up to 10 words)
- Include proper names of countries, leaders, or organizations when relevant
- Reflect the political significance and discourse framing
- Capture the geopolitical context and implications

Examples of expert-level labels:
- "US-China Strategic Competition Over Semiconductor Technology and Trade"
- "Middle East Diplomatic Realignment Following Abraham Accords Implementation"
- "European Migration Policy Crisis and Border Security Challenges"
- "African Democratic Backsliding and Authoritarian Resurgence Patterns"
- "Latin American Left-Wing Resurgence and Populist Movements"
- "Russian Energy Geopolitics and European Energy Security Dependencies"
- "Indian Ocean Security Dynamics and Regional Power Competition"
- "Global Climate Policy Implementation and Green Energy Transitions"
- "Cybersecurity Threats and State-Sponsored Digital Warfare Campaigns"
- "International Trade Wars and Economic Decoupling Strategies"

Political Discourse Analysis:"""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        label = response.choices[0].message.content.strip()
        # Clean up the label
        label = label.replace('"', "").replace("'", "").strip()
        return label if label else "General News"
    except Exception as e:
        print(f"⚠️  Error generating label: {e}")
        return "General News"


# ------------------
# Sentiment Analysis Function
# ------------------
def analyze_cluster_sentiment(titles, client):
    """
    Analyze the sentiment of a cluster based on its article titles.
    Returns average sentiment score from -1 (very negative) to 1 (very positive).
    """
    if not titles or len(titles) == 0:
        return 0.0

    # Adaptive sampling for sentiment analysis based on cluster size
    import random

    # For small clusters (≤10), use all articles
    # For medium clusters (11-50), sample 10-20 articles
    # For large clusters (50+), sample 20-30 articles
    if len(titles) <= 10:
        sample_size = len(titles)  # Use all articles for small clusters
    elif len(titles) <= 50:
        sample_size = min(20, len(titles))  # Sample up to 20 for medium clusters
    else:
        sample_size = min(30, len(titles))  # Sample up to 30 for large clusters

    sample_titles = random.sample(titles, sample_size)
    titles_text = "\n".join([f"- {title}" for title in sample_titles])

    # Log sampling strategy for debugging
    if len(titles) != sample_size:
        print(
            f"   Sampling {sample_size}/{len(titles)} articles for sentiment analysis"
        )

    prompt = f"""You are a sentiment analysis expert specializing in news media and political discourse. Analyze the sentiment of the following news article titles and provide a sentiment score.

{titles_text}

For each title, consider:
- Overall emotional tone and framing
- Positive vs negative language
- Optimistic vs pessimistic outlook
- Constructive vs destructive framing
- Hope vs fear messaging
- Success vs failure narratives

Provide a single sentiment score from -1.0 to 1.0 where:
- -1.0 = Very negative sentiment (fear, anger, crisis, failure, conflict)
- 0.0 = Neutral sentiment (factual, balanced, informational)
- 1.0 = Very positive sentiment (hope, success, progress, optimism)

Consider the overall sentiment across all titles in the cluster. Respond with only a single number between -1.0 and 1.0, rounded to 2 decimal places.

Sentiment Score:"""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        sentiment_text = response.choices[0].message.content.strip()

        # Extract numeric value from response
        import re

        sentiment_match = re.search(r"-?\d+\.?\d*", sentiment_text)
        if sentiment_match:
            sentiment_score = float(sentiment_match.group())
            # Clamp to valid range
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            return round(sentiment_score, 2)
        else:
            print(f"⚠️  Could not parse sentiment score: {sentiment_text}")
            return 0.0

    except Exception as e:
        print(f"⚠️  Error analyzing sentiment: {e}")
        return 0.0


# ------------------
# Batched Analysis Function
# ------------------
def analyze_cluster_batch(cluster_batch, client):
    """
    Analyze multiple clusters in a single LLM call for both labels and sentiment.
    cluster_batch: List of dicts with 'cluster_id', 'titles', and 'sample_size'
    Returns: List of dicts with 'cluster_id', 'label', 'sentiment_score'
    """
    if not cluster_batch:
        return []

    # Prepare the batch prompt
    batch_content = []
    for i, cluster in enumerate(cluster_batch):
        cluster_id = cluster["cluster_id"]
        titles = cluster["titles"]
        sample_size = cluster["sample_size"]

        # Sample titles for this cluster
        import random

        sample_titles = random.sample(titles, sample_size)
        titles_text = "\n".join([f"  - {title}" for title in sample_titles])

        batch_content.append(f"CLUSTER {cluster_id}:\n{titles_text}")

    batch_text = "\n\n".join(batch_content)

    # Get the actual cluster IDs for this batch
    actual_cluster_ids = [cluster["cluster_id"] for cluster in cluster_batch]

    prompt = f"""You are a political discourse expert and media analyst. Analyze the following {len(cluster_batch)} news article clusters and provide both a descriptive label and sentiment score for each.

{batch_text}

For each cluster, provide:
1. A precise political label (up to 10 words) that captures the core political theme, actors, and geopolitical significance
2. A sentiment score from -1.0 to 1.0 where:
   - -1.0 = Very negative (fear, anger, crisis, failure, conflict)
   - 0.0 = Neutral (factual, balanced, informational)  
   - 1.0 = Very positive (hope, success, progress, optimism)

Respond in this exact JSON format with the EXACT cluster IDs from the input:
{{
  "clusters": [
    {{
      "cluster_id": {actual_cluster_ids[0]},
      "label": "US-China Strategic Competition Over Semiconductor Technology",
      "sentiment_score": -0.3
    }},
    {{
      "cluster_id": {actual_cluster_ids[1] if len(actual_cluster_ids) > 1 else actual_cluster_ids[0]}, 
      "label": "European Migration Policy Crisis and Border Security",
      "sentiment_score": -0.7
    }}
  ]
}}

CRITICAL: Use the EXACT cluster IDs from the input: {actual_cluster_ids}. Do NOT use sequential numbering (1, 2, 3...)."""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.choices[0].message.content.strip()

        # Parse JSON response
        import json
        import re

        # Extract JSON from response (in case there's extra text)
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())

            # Validate and process results
            processed_results = []
            for cluster_result in result.get("clusters", []):
                cluster_id = cluster_result.get("cluster_id")
                label = cluster_result.get("label", "General News")
                sentiment_score = cluster_result.get("sentiment_score", 0.0)

                # Clamp sentiment score to valid range
                sentiment_score = max(-1.0, min(1.0, float(sentiment_score)))

                processed_results.append(
                    {
                        "cluster_id": cluster_id,
                        "label": label,
                        "sentiment_score": round(sentiment_score, 2),
                    }
                )

            return processed_results
        else:
            print(f"⚠️  Could not parse JSON from response: {response_text[:200]}...")
            return []

    except Exception as e:
        print(f"⚠️  Error in batch analysis: {e}")
        return []


# ------------------
# Initialize OpenAI client for labeling
# ------------------
load_dotenv()
try:
    openai_client = OpenAI()
    print("✅ OpenAI client initialized for labeling")
except Exception as e:
    print(f"⚠️  OpenAI client not available: {e}")
    openai_client = None

# ------------------
# Build FAISS index
# ------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product ~ cosine
index.add(embeddings)

# ------------------
# Find neighbors above threshold
# ------------------
k = 100  # number of neighbors to check per article
D, I = index.search(embeddings, k)

threshold = 0.75
pairs = []
for i in range(len(embeddings)):
    for j, score in zip(I[i], D[i]):
        if i < j and score >= threshold:
            pairs.append((i, j, float(score)))

print(f"Found {len(pairs)} near-duplicate pairs (cos >= {threshold})")

# ------------------
# Build graph and clusters
# ------------------
G = nx.Graph()
G.add_nodes_from(range(len(embeddings)))
G.add_edges_from([(i, j) for i, j, score in pairs])

clusters = list(nx.connected_components(G))
clusters = [list(c) for c in clusters if len(c) > 1]

print(f"Formed {len(clusters)} clusters of near-duplicates")

# ------------------
# Enrich clusters with metadata
# ------------------
clusters_out = {
    "num_articles": len(embeddings),
    "num_clusters": len(clusters),
    "threshold": threshold,
    "clusters": [],
}

for cid, cl in enumerate(clusters):
    cluster_articles = []
    for idx in cl:
        row = df.iloc[idx]
        cluster_articles.append(
            {
                "article_id": int(row["article_id"]),
                "title": str(row.get("title", "")),
                "url": str(row.get("url", "")),
                "site_name": str(row.get("site_name", "")),
                "index": int(idx),  # to allow exact join back if needed
            }
        )
    clusters_out["clusters"].append({"cluster_id": cid, "articles": cluster_articles})

# ------------------
# Save JSON
# ------------------
with open("data/combined_near_duplicate_clusters.json", "w", encoding="utf-8") as f:
    json.dump(clusters_out, f, indent=2, ensure_ascii=False)

print("✅ Saved data/combined_near_duplicate_clusters.json with titles and site names")

# ------------------
# Compute diversity per cluster and create CSV
# ------------------
print("📊 Analyzing clusters and generating labels...")
cluster_stats = []

for i, c in enumerate(clusters_out["clusters"]):
    sites = [a["site_name"] for a in c["articles"] if a.get("site_name")]
    urls = [a["url"] for a in c["articles"] if a.get("url")]
    titles = [a["title"] for a in c["articles"]]
    unique_sites = set(sites)

    # Get regions for this cluster
    cluster_article_ids = [a["article_id"] for a in c["articles"]]
    cluster_regions = df[df["article_id"].isin(cluster_article_ids)]["region"].tolist()
    unique_regions = list(set(cluster_regions))

    # Will generate labels and sentiment after sorting
    llm_label = "Pending"
    sentiment_score = 0.0

    cluster_stats.append(
        {
            "cluster_id": c["cluster_id"],
            "num_articles": len(c["articles"]),
            "num_sites": len(unique_sites),
            "sites": list(unique_sites),
            "regions": unique_regions,
            "num_regions": len(unique_regions),
            "urls": urls,  # List of all article URLs in this cluster
            "titles": titles,
            "llm_label": llm_label,
            "sentiment_score": sentiment_score,
        }
    )

# Turn into DataFrame for easy exploration
df_clusters = pd.DataFrame(cluster_stats)

# ------------------
# Sort by number of articles (largest first), then by diversity
# ------------------
df_clusters = df_clusters.sort_values(
    by=["num_sites", "num_articles"], ascending=[False, False]
)

# ------------------
# Generate LLM labels and sentiment analysis for ALL clusters (BATCHED)
# ------------------
print(
    f"🤖 Generating LLM labels and sentiment analysis for ALL {len(df_clusters)} clusters using batching..."
)


# Calculate adaptive sample sizes for each cluster
def get_sample_size(titles):
    if len(titles) <= 10:
        return len(titles)  # Use all articles for small clusters
    elif len(titles) <= 50:
        return min(
            15, len(titles)
        )  # Sample up to 15 for medium clusters (reduced for speed)
    else:
        return min(
            20, len(titles)
        )  # Sample up to 20 for large clusters (reduced for speed)


# Prepare cluster data for batching
cluster_batches = []
BATCH_SIZE = 10  # Process 10 clusters at a time
MIN_CLUSTER_SIZE = 1

# Filter out very small clusters if desired
if MIN_CLUSTER_SIZE > 1:
    df_clusters_filtered = df_clusters[df_clusters["num_articles"] >= MIN_CLUSTER_SIZE]
    print(
        f"📊 Filtering clusters: {len(df_clusters)} → {len(df_clusters_filtered)} (skipping clusters with < {MIN_CLUSTER_SIZE} articles)"
    )
    df_clusters = df_clusters_filtered

for i in range(0, len(df_clusters), BATCH_SIZE):
    batch = []
    for j in range(i, min(i + BATCH_SIZE, len(df_clusters))):
        titles = df_clusters.iloc[j]["titles"]
        sample_size = get_sample_size(titles)
        batch.append(
            {
                "cluster_id": df_clusters.iloc[j]["cluster_id"],
                "titles": titles,
                "sample_size": sample_size,
            }
        )
    cluster_batches.append(batch)

print(
    f"📦 Processing {len(cluster_batches)} batches of up to {BATCH_SIZE} clusters each"
)
print(
    f"⏱️  Estimated time: ~{len(cluster_batches) * 2 / 60:.1f} minutes (assuming 2s per batch)"
)

# Process each batch
for batch_idx, cluster_batch in enumerate(cluster_batches):
    if openai_client:
        print(
            f"🤖 Processing batch {batch_idx+1}/{len(cluster_batches)} ({len(cluster_batch)} clusters)..."
        )

        # Process the batch
        results = analyze_cluster_batch(cluster_batch, openai_client)

        # Update the dataframe with results
        for result in results:
            cluster_id = result["cluster_id"]
            label = result["label"]
            sentiment_score = result["sentiment_score"]

            # Find the row with this cluster_id
            mask = df_clusters["cluster_id"] == cluster_id
            if mask.any():
                df_clusters.loc[mask, "llm_label"] = label
                df_clusters.loc[mask, "sentiment_score"] = sentiment_score
                print(
                    f"   Cluster {cluster_id}: {label} (sentiment: {sentiment_score})"
                )

        # Rate limiting between batches
        time.sleep(1.0)  # 1 second between batches (reduced for speed)
    else:
        # If no OpenAI client, mark all clusters in this batch as disabled
        for cluster in cluster_batch:
            cluster_id = cluster["cluster_id"]
            mask = df_clusters["cluster_id"] == cluster_id
            if mask.any():
                df_clusters.loc[mask, "llm_label"] = "Labeling Disabled"
                df_clusters.loc[mask, "sentiment_score"] = 0.0

# Save the sorted version as CSV
df_clusters.to_csv("data/combined_df_clusters_75_100.csv", index=False)

# Also save as JSON for compatibility
with open("data/combined_df_clusters_75_100.json", "w", encoding="utf-8") as f:
    json.dump(df_clusters.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

print(
    "✅ Saved data/combined_df_clusters_75_100.csv and data/combined_df_clusters_75_100.json sorted by num_sites, then num_articles"
)
print(
    f"📊 Created {len(df_clusters)} cluster records with article URLs, regions, LLM labels, and sentiment scores"
)
print(f"🌍 Regions found: {sorted(df['region'].unique())}")
print(
    f"🤖 LLM labels and sentiment analysis generated for ALL {len(df_clusters)} clusters"
)
