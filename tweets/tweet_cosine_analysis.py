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
# Configuration
# ------------------
MAX_TEXT_LENGTH_FOR_CSV = 2000  # Truncate entire text columns to this total length for CSV compatibility

# ------------------
# Load embeddings and tweet dataframe
# ------------------
embeddings = np.load("../embeddings/clean_and_complete_tweets_embeddings.npy").astype("float32")
print(f"✅ Loaded embeddings: {embeddings.shape}")

# ------------------
# Clean and validate embeddings
# ------------------
print("🔍 Validating embeddings...")

# Check for NaN values
nan_count = np.isnan(embeddings).sum()
if nan_count > 0:
    print(f"⚠️  Found {nan_count} NaN values in embeddings")
    # Replace NaN with zeros
    embeddings = np.nan_to_num(embeddings, nan=0.0)

# Check for infinite values
inf_count = np.isinf(embeddings).sum()
if inf_count > 0:
    print(f"⚠️  Found {inf_count} infinite values in embeddings")
    # Replace infinite values with zeros
    embeddings = np.nan_to_num(embeddings, posinf=0.0, neginf=0.0)

# Check for very large values that might cause overflow
max_val = np.max(np.abs(embeddings))
if max_val > 1e6:
    print(f"⚠️  Found very large values (max: {max_val:.2e}), normalizing...")
    # Normalize to prevent overflow
    embeddings = embeddings / np.max(np.abs(embeddings))

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

print(f"✅ Embeddings cleaned and normalized: {embeddings.shape}")
print(f"📊 Value range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
print(f"📊 Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")

# Load tweets from Excel file
try:
    df = pd.read_excel("../scraped_data/Tweets.xlsx")
    print("✅ Loaded tweets from Excel")
except FileNotFoundError:
    print("❌ Could not find tweets Excel file")
    exit(1)

# Load cleaned texts mapping
try:
    with open("../embeddings/clean_tweet_texts_by_id.json", "r", encoding="utf-8") as f:
        tweet_id_to_cleaned_text = json.load(f)
    print(f"✅ Loaded cleaned texts mapping for {len(tweet_id_to_cleaned_text)} tweets")
except FileNotFoundError:
    print("⚠️  Could not find cleaned texts mapping, using original texts")
    tweet_id_to_cleaned_text = {}

# ------------------
# Basic tweet metadata
# ------------------
if "author_id" in df.columns:
    print(f"📊 Tweets by author: {df['author_id'].value_counts().head().to_dict()}")
else:
    print("⚠️  No author_id column found")


# ------------------
# LLM Labeling Function
# ------------------
def generate_cluster_label(tweets, client):
    """
    Generate a descriptive label for a cluster based on its tweet texts.
    """
    if not tweets or len(tweets) == 0:
        return "Unknown Topic"

    # Randomly sample up to 20 tweets to get better context
    import random

    sample_size = min(20, len(tweets))
    sample_tweets = random.sample(tweets, sample_size)
    tweets_text = "\n".join([f"- {tweet}" for tweet in sample_tweets])

    prompt = f"""You are a social media discourse expert and political analyst specializing in Twitter/X conversations and political narratives. Analyze the following tweets and identify the core political theme or discourse. If they are an artifact of Twitter (emojis, retweets, following other accounts, etc.), ignore them., etc. please note that.

{tweets_text}

As a social media discourse expert, provide a precise label that captures:
- The primary political actors, nations, or organizations discussed
- The political sentiment and narrative framing
- The underlying political discourse pattern
- Regional or global political implications
- Specific policy areas or political conflicts

The label should be:
- Politically precise and specific (up to 10 words)
- Include proper names of countries, leaders, or organizations when relevant
- Reflect the political significance and discourse framing
- Capture the social media context and implications

Examples of expert-level labels:
- "US-China Trade War Discourse and Economic Competition"
- "Middle East Peace Process and Diplomatic Developments"
- "European Migration Crisis and Border Policy Debates"
- "Climate Change Policy and Environmental Activism"
- "Election Security and Democratic Process Concerns"
- "Healthcare Policy and Public Health Discussions"
- "Immigration Reform and Border Security Debates"
- "Technology Regulation and Digital Rights Issues"
- "International Relations and Diplomatic Tensions"
- "Economic Policy and Financial Market Discussions"

Social Media Political Discourse Analysis:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
def analyze_cluster_sentiment(tweets, client):
    """
    Analyze the sentiment of a cluster based on its tweet texts.
    Returns average sentiment score from -1 (very negative) to 1 (very positive).
    """
    if not tweets or len(tweets) == 0:
        return 0.0

    # Adaptive sampling for sentiment analysis based on cluster size
    import random

    # For small clusters (≤10), use all tweets
    # For medium clusters (11-50), sample 10-20 tweets
    # For large clusters (50+), sample 20-30 tweets
    if len(tweets) <= 10:
        sample_size = len(tweets)  # Use all tweets for small clusters
    elif len(tweets) <= 50:
        sample_size = min(20, len(tweets))  # Sample up to 20 for medium clusters
    else:
        sample_size = min(30, len(tweets))  # Sample up to 30 for large clusters

    sample_tweets = random.sample(tweets, sample_size)
    tweets_text = "\n".join([f"- {tweet}" for tweet in sample_tweets])

    # Log sampling strategy for debugging
    if len(tweets) != sample_size:
        print(
            f"   Sampling {sample_size}/{len(tweets)} tweets for sentiment analysis"
        )

    prompt = f"""You are a sentiment analysis expert specializing in social media and political discourse. Analyze the sentiment of the following tweets and provide a sentiment score.

{tweets_text}

For each tweet, consider:
- Overall emotional tone and framing
- Positive vs negative language
- Optimistic vs pessimistic outlook
- Constructive vs destructive framing
- Hope vs fear messaging
- Success vs failure narratives
- Social media context and informal language

Provide a single sentiment score from -1.0 to 1.0 where:
- -1.0 = Very negative sentiment (fear, anger, crisis, failure, conflict)
- 0.0 = Neutral sentiment (factual, balanced, informational)
- 1.0 = Very positive sentiment (hope, success, progress, optimism)

Consider the overall sentiment across all tweets in the cluster. Respond with only a single number between -1.0 and 1.0, rounded to 2 decimal places.

Sentiment Score:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
    cluster_batch: List of dicts with 'cluster_id', 'tweets', and 'sample_size'
    Returns: List of dicts with 'cluster_id', 'label', 'sentiment_score'
    """
    if not cluster_batch:
        return []

    # Prepare the batch prompt
    batch_content = []
    for i, cluster in enumerate(cluster_batch):
        cluster_id = cluster["cluster_id"]
        tweets = cluster["tweets"]
        sample_size = cluster["sample_size"]

        # Sample tweets for this cluster
        import random

        sample_tweets = random.sample(tweets, sample_size)
        tweets_text = "\n".join([f"  - {tweet}" for tweet in sample_tweets])

        batch_content.append(f"CLUSTER {cluster_id}:\n{tweets_text}")

    batch_text = "\n\n".join(batch_content)

    # Get the actual cluster IDs for this batch
    actual_cluster_ids = [cluster["cluster_id"] for cluster in cluster_batch]

    prompt = f"""You are a social media discourse expert and political analyst. Analyze the following {len(cluster_batch)} tweet clusters and provide both a descriptive label and sentiment score for each.

{batch_text}

For each cluster, provide:
1. A precise political label (up to 10 words) that captures the core political theme, actors, and social media discourse significance
2. A sentiment score from -1.0 to 1.0 where:
   - -1.0 = Very negative (fear, anger, crisis, failure, conflict)
   - 0.0 = Neutral (factual, balanced, informational)  
   - 1.0 = Very positive (hope, success, progress, optimism)

Respond in this exact JSON format with the EXACT cluster IDs from the input:
{{
  "clusters": [
    {{
      "cluster_id": {actual_cluster_ids[0]},
      "label": "US-China Trade War Discourse and Economic Competition",
      "sentiment_score": -0.3
    }},
    {{
      "cluster_id": {actual_cluster_ids[1] if len(actual_cluster_ids) > 1 else actual_cluster_ids[0]}, 
      "label": "Climate Change Policy and Environmental Activism",
      "sentiment_score": 0.2
    }}
  ]
}}

CRITICAL: Use the EXACT cluster IDs from the input: {actual_cluster_ids}. Do NOT use sequential numbering (1, 2, 3...)."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
k = 100  # number of neighbors to check per tweet
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
    "num_tweets": len(embeddings),
    "num_clusters": len(clusters),
    "threshold": threshold,
    "clusters": [],
}

for cid, cl in enumerate(clusters):
    cluster_tweets = []
    for idx in cl:
        row = df.iloc[idx]
        cluster_tweets.append(
            {
                "tweet_id": int(row["tweet_id"]),
                "text": str(row.get("text", "")),
                "author_id": str(row.get("author_id", "")),
                "username": str(row.get("username", "")),
                "created_at": str(row.get("created_at", "")),
                "index": int(idx),  # to allow exact join back if needed
            }
        )
    clusters_out["clusters"].append({"cluster_id": cid, "tweets": cluster_tweets})

# ------------------
# Create output directory and save JSON
# ------------------
output_dir = f"../data/cosine_{int(threshold*100)}"
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/cosine_{int(threshold*100)}_clusters.json", "w", encoding="utf-8") as f:
    json.dump(clusters_out, f, indent=2, ensure_ascii=False)

print(f"✅ Saved cosine_{int(threshold*100)}_clusters.json to {output_dir}/")

# ------------------
# Compute diversity per cluster and create CSV
# ------------------
print("📊 Analyzing clusters and generating labels...")
cluster_stats = []

for i, c in enumerate(clusters_out["clusters"]):
    authors = [a["author_id"] for a in c["tweets"] if a.get("author_id")]
    usernames = [a["username"] for a in c["tweets"] if a.get("username")]
    tweets = [a["text"] for a in c["tweets"]]
    
    # Get cleaned texts for this cluster
    cleaned_tweets = []
    for tweet in c["tweets"]:
        tweet_id = str(tweet["tweet_id"])
        if tweet_id in tweet_id_to_cleaned_text:
            cleaned_tweets.append(tweet_id_to_cleaned_text[tweet_id])
        else:
            cleaned_tweets.append(tweet["text"])  # Fallback to original text
    
    unique_authors = set(authors)

    # Get tweet IDs for this cluster
    cluster_tweet_ids = [a["tweet_id"] for a in c["tweets"]]

    # Will generate labels and sentiment after sorting
    llm_label = "Pending"
    sentiment_score = 0.0

    # Truncate long text columns for CSV compatibility
    max_column_length = MAX_TEXT_LENGTH_FOR_CSV
    
    def truncate_text_list(text_list, max_column_length):
        """Truncate the entire text list to max_column_length characters total"""
        # Join all texts and truncate the entire column
        full_text = " | ".join(text_list)  # Use | as separator between tweets
        if len(full_text) > max_column_length:
            truncated_text = full_text[:max_column_length] + "..."
            return [truncated_text]  # Return as single item since we truncated the whole column
        else:
            return text_list  # Return original if under limit
    
    cluster_stats.append(
        {
            "cluster_id": c["cluster_id"],
            "num_tweets": len(c["tweets"]),
            "num_authors": len(unique_authors),
            "authors": list(unique_authors),
            "usernames": list(set(usernames)),
            "tweet_ids": cluster_tweet_ids,  # List of all tweet IDs in this cluster
            "tweets": truncate_text_list(tweets, max_column_length),  # Truncated original tweets
            "cleaned_tweets": truncate_text_list(cleaned_tweets, max_column_length),  # Truncated cleaned tweets
            "llm_label": llm_label,
            "sentiment_score": sentiment_score,
        }
    )

# Turn into DataFrame for easy exploration
df_clusters = pd.DataFrame(cluster_stats)

# ------------------
# Sort by number of tweets (largest first), then by diversity
# ------------------
df_clusters = df_clusters.sort_values(
    by=["num_authors", "num_tweets"], ascending=[False, False]
)

# ------------------
# Generate LLM labels and sentiment analysis for ALL clusters (BATCHED)
# ------------------
print(
    f"🤖 Generating LLM labels and sentiment analysis for ALL {len(df_clusters)} clusters using batching..."
)


# Calculate adaptive sample sizes for each cluster
def get_sample_size(tweets):
    if len(tweets) <= 10:
        return len(tweets)  # Use all tweets for small clusters
    elif len(tweets) <= 50:
        return min(
            15, len(tweets)
        )  # Sample up to 15 for medium clusters (reduced for speed)
    else:
        return min(
            20, len(tweets)
        )  # Sample up to 20 for large clusters (reduced for speed)


# Prepare cluster data for batching
cluster_batches = []
BATCH_SIZE = 10  # Process 10 clusters at a time
MIN_CLUSTER_SIZE = 1

# Filter out very small clusters if desired
if MIN_CLUSTER_SIZE > 1:
    df_clusters_filtered = df_clusters[df_clusters["num_tweets"] >= MIN_CLUSTER_SIZE]
    print(
        f"📊 Filtering clusters: {len(df_clusters)} → {len(df_clusters_filtered)} (skipping clusters with < {MIN_CLUSTER_SIZE} tweets)"
    )
    df_clusters = df_clusters_filtered

for i in range(0, len(df_clusters), BATCH_SIZE):
    batch = []
    for j in range(i, min(i + BATCH_SIZE, len(df_clusters))):
        # Use cleaned tweets for LLM analysis
        tweets = df_clusters.iloc[j]["cleaned_tweets"]
        sample_size = get_sample_size(tweets)
        batch.append(
            {
                "cluster_id": df_clusters.iloc[j]["cluster_id"],
                "tweets": tweets,
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
df_clusters.to_csv(f"{output_dir}/cosine_{int(threshold*100)}_df_clusters.csv", index=False)

# Also save as JSON for compatibility
with open(f"{output_dir}/cosine_{int(threshold*100)}_df_clusters.json", "w", encoding="utf-8") as f:
    json.dump(df_clusters.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

print(
    f"✅ Saved cosine_{int(threshold*100)}_df_clusters.csv and cosine_{int(threshold*100)}_df_clusters.json to {output_dir}/ sorted by num_authors, then num_tweets"
)
print(f"📝 Note: Text columns (tweets, cleaned_tweets) are truncated to {MAX_TEXT_LENGTH_FOR_CSV} characters in CSV for Excel compatibility")
print(
    f"📊 Created {len(df_clusters)} cluster records with tweet IDs, authors, LLM labels, and sentiment scores"
)
print(f"👥 Authors found: {len(df['author_id'].unique()) if 'author_id' in df.columns else 'Unknown'}")
print(
    f"🤖 LLM labels and sentiment analysis generated for ALL {len(df_clusters)} clusters"
)
