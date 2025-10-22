#!/usr/bin/env python3
"""
K-means Clustering Progression Analysis

This script performs k-means clustering for k values from 1 to 49 and saves
each result to separate JSON files. This allows you to see how clusters
evolve as the number of clusters increases.

Note: This script skips LLM labeling for faster processing and focuses on
clustering structure and clean text data.

Usage:
    python3 kmeans_progression_analysis.py
"""

import numpy as np
import json
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time

# ------------------
# Configuration
# ------------------
MAX_K = 49  # Maximum number of clusters to test
MIN_K = 1   # Minimum number of clusters to test
OUTPUT_BASE_DIR = "../data/kmeans_progression"

# ------------------
# Load embeddings and tweet dataframe
# ------------------
print("🔍 Loading data...")
embeddings = np.load("../embeddings/clean_and_complete_tweets_embeddings.npy").astype("float32")
print(f"✅ Loaded embeddings: {embeddings.shape}")

# Load tweets from Excel file
try:
    df = pd.read_excel("../scraped_data/Tweets.xlsx")
    print(f"✅ Loaded {len(df)} tweets from Excel")
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
# Create output directory structure
# ------------------
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
print(f"📁 Created output directory: {OUTPUT_BASE_DIR}")

# ------------------
# Helper function to create cluster data
# ------------------
def create_cluster_data(k, cluster_labels, embeddings, df, tweet_id_to_cleaned_text):
    """
    Create cluster data structure for a given k value.
    """
    # Create clusters dictionary
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    # Convert to list format
    clusters_list = [list(cluster_indices) for cluster_indices in clusters.values()]
    
    # Create cluster data with cleaned texts
    cluster_data = {
        "k": k,
        "num_tweets": len(embeddings),
        "num_clusters": len(clusters_list),
        "clustering_method": "kmeans",
        "clusters": []
    }
    
    for cid, cluster_indices in enumerate(clusters_list):
        cluster_original_texts = []
        cluster_cleaned_texts = []
        
        for idx in cluster_indices:
            row = df.iloc[idx]
            tweet_id = int(row["tweet_id"])
            
            # Get original text
            original_text = str(row.get("text", ""))
            cluster_original_texts.append(original_text)
            
            # Get cleaned text if available
            cleaned_text = ""
            if str(tweet_id) in tweet_id_to_cleaned_text:
                cleaned_text = tweet_id_to_cleaned_text[str(tweet_id)]
            else:
                cleaned_text = original_text  # Fallback to original if no cleaned version
            
            cluster_cleaned_texts.append(cleaned_text)
        
        # Calculate basic cluster statistics
        cluster_size = len(cluster_original_texts)
        
        cluster_data["clusters"].append({
            "cluster_id": cid,
            "cluster_size": cluster_size,
            "original_texts": cluster_original_texts,  # List of original tweet texts
            "cleaned_texts": cluster_cleaned_texts     # List of cleaned tweet texts
        })
    
    return cluster_data

# ------------------
# Perform clustering for each k value
# ------------------
print(f"🤖 Starting k-means clustering progression analysis (k={MIN_K} to {MAX_K})...")
print(f"📊 This will create {MAX_K - MIN_K + 1} different clustering results")

# Track performance metrics
inertias = []
silhouette_scores = []
processing_times = []

for k in range(MIN_K, MAX_K + 1):
    print(f"\n🎯 Processing k={k}...")
    start_time = time.time()
    
    try:
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        inertias.append(inertia)
        
        # Calculate silhouette score (only for k > 1)
        if k > 1 and len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
        
        # Create cluster data
        cluster_data = create_cluster_data(k, cluster_labels, embeddings, df, tweet_id_to_cleaned_text)
        
        # Save to JSON file
        output_file = f"{OUTPUT_BASE_DIR}/kmeans_k{k}_clusters.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(cluster_data, f, indent=2, ensure_ascii=False)
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        print(f"   ✅ Completed k={k} in {processing_time:.2f}s")
        print(f"   📊 Inertia: {inertia:.2f}, Silhouette: {silhouette_avg:.3f}")
        print(f"   💾 Saved to: {output_file}")
        
    except Exception as e:
        print(f"   ❌ Error processing k={k}: {e}")
        inertias.append(0)
        silhouette_scores.append(0)
        processing_times.append(0)

# ------------------
# Create summary analysis
# ------------------
print(f"\n📊 Creating summary analysis...")

# Create summary data
summary_data = {
    "analysis_type": "kmeans_progression",
    "total_tweets": len(embeddings),
    "k_range": list(range(MIN_K, MAX_K + 1)),
    "inertias": inertias,
    "silhouette_scores": silhouette_scores,
    "processing_times": processing_times,
    "best_silhouette_k": MIN_K + np.argmax(silhouette_scores[1:]) + 1 if len(silhouette_scores) > 1 else MIN_K,
    "best_silhouette_score": max(silhouette_scores) if silhouette_scores else 0,
    "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

# Save summary
summary_file = f"{OUTPUT_BASE_DIR}/kmeans_progression_summary.json"
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

# ------------------
# Create cluster size analysis for each k
# ------------------
print("📈 Analyzing cluster size distributions...")

cluster_size_analysis = {}
for k in range(MIN_K, MAX_K + 1):
    try:
        # Load the cluster data for this k
        cluster_file = f"{OUTPUT_BASE_DIR}/kmeans_k{k}_clusters.json"
        if os.path.exists(cluster_file):
            with open(cluster_file, "r", encoding="utf-8") as f:
                cluster_data = json.load(f)
            
            # Analyze cluster sizes
            cluster_sizes = [cluster["cluster_size"] for cluster in cluster_data["clusters"]]
            
            cluster_size_analysis[k] = {
                "cluster_sizes": cluster_sizes,
                "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
                "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                "mean_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
                "median_cluster_size": np.median(cluster_sizes) if cluster_sizes else 0,
                "std_cluster_size": np.std(cluster_sizes) if cluster_sizes else 0,
                "total_texts": sum(cluster_sizes)
            }
    except Exception as e:
        print(f"   ⚠️  Error analyzing k={k}: {e}")

# Save cluster size analysis
size_analysis_file = f"{OUTPUT_BASE_DIR}/cluster_size_analysis.json"
with open(size_analysis_file, "w", encoding="utf-8") as f:
    json.dump(cluster_size_analysis, f, indent=2, ensure_ascii=False)

# ------------------
# Print final summary
# ------------------
print(f"\n🎉 K-means progression analysis completed!")
print(f"📊 Summary:")
print(f"   - Total k values tested: {MAX_K - MIN_K + 1}")
print(f"   - Best silhouette score: {max(silhouette_scores):.3f} at k={summary_data['best_silhouette_k']}")
print(f"   - Total processing time: {sum(processing_times):.2f}s")
print(f"   - Average time per k: {np.mean(processing_times):.2f}s")

print(f"\n📁 Files created:")
print(f"   - Individual cluster files: {OUTPUT_BASE_DIR}/kmeans_k{{1-49}}_clusters.json")
print(f"   - Summary analysis: {summary_file}")
print(f"   - Cluster size analysis: {size_analysis_file}")

print(f"\n🔍 Key insights:")
print(f"   - Silhouette scores range: {min(silhouette_scores):.3f} to {max(silhouette_scores):.3f}")
print(f"   - Inertia decreases from {inertias[0]:.2f} to {inertias[-1]:.2f}")
print(f"   - Best clustering performance at k={summary_data['best_silhouette_k']}")

print(f"\n💡 Usage tips:")
print(f"   - Compare cluster quality across different k values")
print(f"   - Look for the 'elbow' in the inertia curve")
print(f"   - Check silhouette scores for optimal clustering")
print(f"   - Examine cluster size distributions for balance")
