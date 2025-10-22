import numpy as np
import json
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openai import OpenAI
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt

# ------------------
# Load embeddings and tweet dataframe
# ------------------
embeddings = np.load("../embeddings/clean_and_complete_tweets_embeddings.npy").astype("float32")
print(f"✅ Loaded embeddings: {embeddings.shape}")

# Load tweets from Excel file
try:
    df = pd.read_excel("../scraped_data/Tweets.xlsx")
    print("✅ Loaded tweets from Excel")
    print(f"📊 Loaded {len(df)} tweets")
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

    prompt = f"""You are a social media discourse expert and political analyst specializing in Twitter/X conversations and political narratives. Analyze the following tweets and identify the core political theme or discourse.

{tweets_text}

As a social media discourse expert, provide a precise label that MUST include at least 2 of the following 3 elements:

1. **SPECIFIC REGION/COUNTRY**: Specific geographic areas, nations, or regions mentioned
2. **SPECIFIC LEADER/PERSON**: Names of political leaders, public figures, or key individuals
3. **SPECIFIC POLITICAL TOPIC**: Specific policy areas, political issues, or governance themes

The label should be:
- Politically precise and specific (up to 12 words)
- Include proper names of countries, leaders, or organizations when relevant
- Reflect the political significance and discourse framing
- Capture the social media context and implications
- MUST contain at least 2 of: region/country, leader/person, political topic

Examples of expert-level labels (showing required elements):
- "DRC-Rwanda Conflict and M23 Rebel Group" (Region + Political Topic)
- "Félix Tshisekedi and Democratic Republic of Congo Elections" (SpecificLeader + Region)
- "Nairobi Opposition Meeting and Political Unity" (Region + Political Topic)
- "Patrick Muyaya and RDC Government Communications" (Specific Leader + Political Topic)
- "Eastern Congo Security Crisis and FARDC Operations" (Region + Political Topic)
- "Kigali Government and Regional Diplomacy" (Region + Political Topic)
- "Economic Development and US-RDC Partnership" (Political Topic + Region)

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
1. A precise political label (up to 12 words) that MUST include at least 2 of: region/country, leader/person, political topic
2. A sentiment score from -1.0 to 1.0 where:
   - -1.0 = Very negative (fear, anger, crisis, failure, conflict)
   - 0.0 = Neutral (factual, balanced, informational)  
   - 1.0 = Very positive (hope, success, progress, optimism)

Respond in this exact JSON format with the EXACT cluster IDs from the input:
{{
  "clusters": [
    {{
      "cluster_id": {actual_cluster_ids[0]},
      "label": "DRC-Rwanda Conflict and M23 Rebel Group",
      "sentiment_score": -0.3
    }},
    {{
      "cluster_id": {actual_cluster_ids[1] if len(actual_cluster_ids) > 1 else actual_cluster_ids[0]}, 
      "label": "Félix Tshisekedi and Democratic Republic of Congo Elections",
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
# Configuration - Fixed number of clusters
# ------------------
FIXED_K = 25  # Use exactly 49 clusters as requested


# ------------------
# Perform K-means clustering
# ------------------
print("🤖 Starting K-means clustering analysis...")

# Use fixed number of clusters (skip optimal k search)
print(f"🎯 Performing K-means clustering with k={FIXED_K}...")
kmeans = KMeans(n_clusters=FIXED_K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

print(f"✅ Clustering completed! Created {FIXED_K} clusters")

# ------------------
# Analyze cluster composition
# ------------------
print("📊 Analyzing cluster composition...")

# Create clusters dictionary
clusters = {}
for i, label in enumerate(cluster_labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(i)

# Convert to list format for consistency with cosine similarity approach
clusters_list = [list(cluster_indices) for cluster_indices in clusters.values()]

print(f"📈 Cluster size distribution:")
cluster_sizes = [len(cluster) for cluster in clusters_list]
print(f"   Min: {min(cluster_sizes)}")
print(f"   Max: {max(cluster_sizes)}")
print(f"   Mean: {np.mean(cluster_sizes):.1f}")
print(f"   Median: {np.median(cluster_sizes):.1f}")

# ------------------
# Enrich clusters with metadata
# ------------------
clusters_out = {
    "num_tweets": len(embeddings),
    "num_clusters": len(clusters_list),
    "clustering_method": "kmeans",
    "fixed_k": FIXED_K,
    "clusters": [],
}

for cid, cl in enumerate(clusters_list):
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
output_dir = f"../data/kmeans_{FIXED_K}"
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/kmeans_{FIXED_K}_clusters.json", "w", encoding="utf-8") as f:
    json.dump(clusters_out, f, indent=2, ensure_ascii=False)

print(f"✅ Saved kmeans_{FIXED_K}_clusters.json to {output_dir}/")

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

    cluster_stats.append(
        {
            "cluster_id": c["cluster_id"],
            "num_tweets": len(c["tweets"]),
            "num_authors": len(unique_authors),
            "authors": list(unique_authors),
            "usernames": list(set(usernames)),
            "tweet_ids": cluster_tweet_ids,  # List of all tweet IDs in this cluster
            "tweets": tweets,  # Original tweets
            "cleaned_tweets": cleaned_tweets,  # Cleaned tweets (no @username mentions)
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
df_clusters.to_csv(f"{output_dir}/kmeans_{FIXED_K}_df_clusters.csv", index=False)

# Also save as JSON for compatibility
with open(f"{output_dir}/kmeans_{FIXED_K}_df_clusters.json", "w", encoding="utf-8") as f:
    json.dump(df_clusters.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

print(
    f"✅ Saved kmeans_{FIXED_K}_df_clusters.csv and kmeans_{FIXED_K}_df_clusters.json to {output_dir}/ sorted by num_authors, then num_tweets"
)
print(
    f"📊 Created {len(df_clusters)} cluster records with tweet IDs, authors, LLM labels, and sentiment scores"
)
print(f"👥 Authors found: {len(df['author_id'].unique()) if 'author_id' in df.columns else 'Unknown'}")
print(
    f"🤖 LLM labels and sentiment analysis generated for ALL {len(df_clusters)} clusters"
)

# ------------------
# Create visualization of clustering results
# ------------------
print("📊 Creating clustering visualization...")

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f"Tweet K-means Clustering Analysis (k={FIXED_K})", fontsize=16)

# 1. Cluster size distribution (bar chart)
cluster_ids = list(range(len(cluster_sizes)))
axes[0, 0].bar(cluster_ids, cluster_sizes, alpha=0.7, edgecolor="black")
axes[0, 0].set_xlabel("Cluster ID")
axes[0, 0].set_ylabel("Number of Tweets")
axes[0, 0].set_title(f"Cluster Size Distribution (k={FIXED_K})")
axes[0, 0].grid(True, alpha=0.3)

# 2. Cluster size distribution (histogram)
axes[0, 1].hist(cluster_sizes, bins=20, alpha=0.7, edgecolor="black")
axes[0, 1].set_xlabel("Cluster Size")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Cluster Size Distribution")
axes[0, 1].axvline(x=np.mean(cluster_sizes), color="red", linestyle="--", label=f"Mean: {np.mean(cluster_sizes):.1f}")
axes[0, 1].axvline(x=np.median(cluster_sizes), color="green", linestyle="--", label=f"Median: {np.median(cluster_sizes):.1f}")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Author diversity per cluster
author_diversity = [len(set([df.iloc[idx]['author_id'] for idx in cluster if 'author_id' in df.columns])) for cluster in clusters_list]
axes[1, 0].bar(cluster_ids, author_diversity, alpha=0.7, edgecolor="black", color='orange')
axes[1, 0].set_xlabel("Cluster ID")
axes[1, 0].set_ylabel("Number of Unique Authors")
axes[1, 0].set_title("Author Diversity per Cluster")
axes[1, 0].grid(True, alpha=0.3)

# 4. Tweets per author in clusters
author_counts = df_clusters["authors"].explode().value_counts()
axes[1, 1].bar(range(len(author_counts)), author_counts.values)
axes[1, 1].set_xlabel("Author")
axes[1, 1].set_ylabel("Number of Tweets")
axes[1, 1].set_title("Tweets per Author in Clusters")
axes[1, 1].set_xticks(range(len(author_counts)))
axes[1, 1].set_xticklabels(author_counts.index, rotation=45)
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(f"{output_dir}/kmeans_{FIXED_K}_clustering_analysis.png", dpi=300, bbox_inches="tight")
print(f"✅ Saved clustering visualization to {output_dir}/kmeans_{FIXED_K}_clustering_analysis.png")

print("\n🎉 Tweet K-means clustering analysis completed!")
print(f"📊 Summary:")
print(f"   - Total tweets: {len(embeddings)}")
print(f"   - Fixed clusters: {FIXED_K}")
print(f"   - Clustering method: K-means")
print(
    f"   - Files saved: {output_dir}/kmeans_{FIXED_K}_clusters.json, {output_dir}/kmeans_{FIXED_K}_df_clusters.csv, {output_dir}/kmeans_{FIXED_K}_df_clusters.json"
)
print(f"   - Visualization: {output_dir}/kmeans_{FIXED_K}_clustering_analysis.png")
