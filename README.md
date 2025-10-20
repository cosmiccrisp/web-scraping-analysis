# Web Scraping Analysis

A comprehensive pipeline for scraping 8,913 news articles from multiple websites, generating embeddings, and performing similarity analysis to identify near-duplicate content across different sources.

## Overview

This project implements a three-stage pipeline:

1. **Web Scraping** - Asynchronously scrapes articles from news websites
2. **Embedding Generation** - Creates vector embeddings using OpenAI's text-embedding-3-large model
3. **K-means Clustering** - Uses K-means clustering to group articles by topic similarity

## Architecture

### Stage 1: Web Scraping (`scraping_1.py`)

**Purpose**: Efficiently scrape articles from multiple news websites using async HTTP requests.

**Key Features**:
- **Concurrent Processing**: Uses `httpx` with async/await for high-performance scraping
- **Rate Limiting**: Implements per-host and global concurrency limits to respect server resources
- **Retry Logic**: Automatic retry with exponential backoff for failed requests
- **Content Extraction**: Parses HTML to extract structured article data (title, author, content, metadata)
- **Category Filtering**: Focuses on specific content categories (currently technology)

**Configuration**:
- `PER_HOST_LIMIT = 2` - Max concurrent requests per domain
- `GLOBAL_LIMIT = 20` - Max total concurrent requests
- `SITES_CONCURRENCY = 5` - Max sites processed simultaneously
- `REQUEST_TIMEOUT = 20.0` - Request timeout in seconds

**Output**: JSON file containing scraped articles with metadata

### Stage 2: Embedding Generation (`embedding_2.py`)

**Purpose**: Convert article text into high-dimensional vector representations for similarity analysis.

**Key Features**:
- **OpenAI Integration**: Uses `text-embedding-3-large` model (3072 dimensions)
- **Batch Processing**: Processes articles in batches to respect API rate limits
- **Text Preparation**: Combines article titles and content for comprehensive representation
- **Error Handling**: Graceful handling of API failures with fallback embeddings
- **Memory Efficient**: Saves embeddings as numpy arrays for fast loading

**Configuration**:
- `BATCH_SIZE = 100` - Articles processed per API call
- `DEFAULT_MODEL = "text-embedding-3-large"` - OpenAI embedding model
- `include_content = True` - Whether to include article content beyond titles

**Output**: `.npy` file containing normalized embeddings array

### Stage 3: K-means Clustering (`kmeans_clustering_3.py`)

**Purpose**: Group articles by topic similarity using K-means clustering on vector embeddings.

**Key Features**:
- **Optimal K Selection**: Uses elbow method and silhouette analysis to find optimal number of clusters
- **K-means Clustering**: Uses scikit-learn's KMeans to partition articles into topic-based clusters
- **Complete Coverage**: Every article is assigned to exactly one cluster (no articles left out)
- **Metadata Enrichment**: Adds site regions and generates cluster labels using LLM
- **Diversity Analysis**: Tracks which sites and regions appear in each cluster
- **Visualization**: Creates comprehensive analysis plots showing clustering performance

**Configuration**:
- `max_k = 30` - Maximum number of clusters to test
- `min_k = 2` - Minimum number of clusters to test
- `BATCH_SIZE = 10` - Clusters processed per LLM batch
- Automatic optimal k selection based on silhouette score

**Output**: 
- JSON file with cluster metadata and article mappings
- CSV file with cluster statistics sorted by diversity
- PNG visualization of clustering analysis

## Data Flow

```
sites.csv → scraping_1.py → scraped_articles.json
                ↓
scraped_articles.json → embedding_2.py → embeddings.npy
                ↓
embeddings.npy + scraped_articles.json → kmeans_clustering_3.py → clusters.json + clusters.csv
```

## Usage

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Configure Sites
Edit `sites.csv` with your target news websites. The scraper expects columns:
- `News Websites`: Base URL of the news site
- `Region`: Geographic region for analysis
- `Outlet Name`: Site identifier

### 3. Run Scraping
```bash
python scraping_1.py
```

### 4. Generate Embeddings
```bash
python embedding_2.py scraped_articles.json
```

### 5. Perform K-means Clustering
```bash
python kmeans_clustering_3.py
```

## Output Files

- `scraped_articles.json` - Raw scraped article data
- `data/combined_embeddings.npy` - Vector embeddings
- `data/kmeans_clusters.json` - K-means cluster definitions
- `data/kmeans_df_clusters.csv` - Cluster statistics and metadata
- `data/kmeans_clustering_analysis.png` - Clustering analysis visualization

## Key Metrics

The pipeline tracks several important metrics:
- **Scraping Success Rate**: Percentage of successfully scraped articles
- **Cluster Diversity**: Number of unique sites per cluster
- **Geographic Distribution**: Regional analysis of content clustering
- **Optimal K Selection**: Automatic determination of best number of clusters
- **Silhouette Score**: Quality measure of cluster separation

## Performance

- **Scraping**: ~20 articles/second with proper rate limiting
- **Embeddings**: ~100 articles per batch with OpenAI API
- **Clustering**: Handles 10,000+ articles efficiently with scikit-learn K-means

## Configuration

Key parameters can be adjusted in each script:
- Concurrency limits in `scraping_1.py`
- Batch sizes in `embedding_2.py` 
- K-means parameters in `kmeans_clustering_3.py`

This pipeline is designed for analyzing content clustering across news sources, helping identify topic-based groupings and content patterns across different regions and outlets.
