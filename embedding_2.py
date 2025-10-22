#!/usr/bin/env python3
"""
Standalone embeddings generation script for article texts.

This script loads articles from JSON files and generates embeddings using OpenAI's
text-embedding-3-large model. It saves the embeddings as numpy arrays for later use.
"""

import os
import json
import time
import logging
import warnings
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("embeddings.log")],
)
log = logging.getLogger("embeddings")

# Configuration
DEFAULT_MODEL = "text-embedding-3-large"
BATCH_SIZE = 100  # Process articles in batches to avoid rate limits
DEFAULT_OUTPUT_DIR = "data"


def load_articles(file_path: str) -> pd.DataFrame:
    """Load articles from JSON file or tweets from Excel file and return as DataFrame."""
    log.info(f"📂 Loading data from {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check file extension to determine how to load
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        # Check if file has multiple sheets
        xl_file = pd.ExcelFile(file_path)
        sheet_names = xl_file.sheet_names
        log.info(f"📊 Found {len(sheet_names)} sheets: {sheet_names}")
        
        if len(sheet_names) > 1:
            # Load and merge all sheets
            dfs = []
            for sheet_name in sheet_names:
                sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
                log.info(f"📄 Loaded {len(sheet_df)} tweets from sheet '{sheet_name}'")
                dfs.append(sheet_df)
            
            # Concatenate all dataframes
            df = pd.concat(dfs, ignore_index=True)
            log.info(f"✅ Merged {len(dfs)} sheets into {len(df)} total tweets")
        else:
            # Load single sheet
            df = pd.read_excel(file_path)
            log.info(f"✅ Loaded {len(df)} tweets from Excel file")
        
        # Ensure tweet_id is preserved as the first column
        if "tweet_id" in df.columns:
            # Move tweet_id to first position
            cols = ["tweet_id"] + [col for col in df.columns if col != "tweet_id"]
            df = df[cols]
        else:
            log.warning("⚠️  No tweet_id column found, creating sequential IDs")
            df.insert(0, "tweet_id", range(1, len(df) + 1))
    else:
        # Load JSON file (articles)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        log.info(f"✅ Loaded {len(df)} articles from JSON file")
        
        # Add article ID if not present
        if "article_id" not in df.columns:
            df.insert(0, "article_id", range(1, len(df) + 1))

    return df


def clean_tweet_text(text):
    """
    Clean tweet text by removing @username mentions, RT prefixes, and extra spaces.
    """
    if pd.isna(text) or not text:
        return ""
    
    # Remove @username mentions (including @username at start, middle, or end)
    text = re.sub(r'@\w+\s*', '', str(text))
    
    # Remove "RT :" prefix (retweet indicator)
    text = re.sub(r'^RT\s*:\s*', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def prepare_texts(df: pd.DataFrame, include_content: bool = True) -> List[str]:
    """
    Prepare texts for embedding by combining title and optionally content for articles,
    or using cleaned tweet text for tweets.

    Args:
        df: DataFrame with articles or tweets
        include_content: Whether to include article content in addition to title (ignored for tweets)

    Returns:
        List of prepared text strings
    """
    log.info("📝 Preparing texts for embedding...")

    texts = []
    for _, row in df.iterrows():
        # Check if this is a tweet (has tweet_id column)
        if "tweet_id" in df.columns:
            # For tweets, clean the text and use it
            tweet_text = clean_tweet_text(row.get("text", ""))
            print(row.get('text',''))
            print(tweet_text)
            print()
            if tweet_text:
                texts.append(tweet_text)
            else:
                # Fallback if no text after cleaning
                texts.append("No text available")
        else:
            # For articles, use title and optionally content
            text_parts = [str(row.get("title", "")).strip()]

            # Add content if requested and available
            if include_content and "content" in row and pd.notna(row["content"]):
                content = str(row["content"]).strip()
                if content:
                    text_parts.append(content)

            # Join parts with space and clean up
            full_text = " ".join(text_parts).strip()
            if full_text:
                texts.append(full_text)
            else:
                # Fallback to just the URL if no text content
                texts.append(str(row.get("url", "")))

    log.info(f"✅ Prepared {len(texts)} texts for embedding")
    return texts


def generate_embeddings(
    texts: List[str],
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using OpenAI's embedding model.

    Args:
        texts: List of text strings to embed
        client: OpenAI client instance
        model: Embedding model to use
        batch_size: Number of texts to process in each batch

    Returns:
        Numpy array of embeddings with shape (n_texts, embedding_dim)
    """
    log.info(f"🤖 Generating embeddings with {model}...")
    log.info(f"📊 Processing {len(texts)} texts in batches of {batch_size}")

    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i : i + batch_size]

        try:
            response = client.embeddings.create(model=model, input=batch)

            # Extract embeddings from response
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)

            # Small delay to respect rate limits
            time.sleep(0.1)

        except Exception as e:
            log.error(
                f"❌ Error processing batch {i//batch_size + 1}/{total_batches}: {e}"
            )
            # Add zero embeddings for failed batch
            embedding_dim = 3072 if "3-large" in model else 1536
            batch_embeddings = [[0.0] * embedding_dim for _ in batch]
            embeddings.extend(batch_embeddings)

    embeddings_array = np.array(embeddings, dtype=np.float32)
    log.info(f"✅ Generated embeddings: {embeddings_array.shape}")

    return embeddings_array


def save_embeddings(embeddings: np.ndarray, output_path: str, df: pd.DataFrame = None, cleaned_texts: List[str] = None) -> None:
    """Save embeddings to numpy file and optionally save mapping with IDs and cleaned texts."""
    log.info(f"💾 Saving embeddings to {output_path}...")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, embeddings)
    log.info(f"✅ Saved embeddings: {embeddings.shape}")
    
    # If DataFrame is provided and has tweet_id or article_id, save mapping
    if df is not None:
        id_column = None
        if "tweet_id" in df.columns:
            id_column = "tweet_id"
        elif "article_id" in df.columns:
            id_column = "article_id"
            
        if id_column:
            mapping_path = output_path.replace('.npy', '_mapping.json')
            mapping_data = {
                "ids": df[id_column].tolist(),
                "embeddings_shape": embeddings.shape,
                "total_items": len(df)
            }
            
            # Add cleaned texts if provided
            if cleaned_texts is not None:
                # Create tweet_id to cleaned_text mapping
                tweet_id_to_cleaned_text = dict(zip(df[id_column].tolist(), cleaned_texts))
                mapping_data["tweet_id_to_cleaned_text"] = tweet_id_to_cleaned_text
                mapping_data["cleaned_texts"] = cleaned_texts  # Keep as list too for compatibility
                log.info(f"✅ Including {len(cleaned_texts)} cleaned texts with tweet_id mapping")
            
            with open(mapping_path, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            log.info(f"✅ Saved ID mapping to {mapping_path}")
            
            # Also save cleaned texts as a separate file for easy access
            if cleaned_texts is not None:
                texts_path = output_path.replace('.npy', '_cleaned_texts.json')
                with open(texts_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_texts, f, indent=2, ensure_ascii=False)
                log.info(f"✅ Saved cleaned texts to {texts_path}")
                
                # Save tweet_id to cleaned_text mapping as a separate file
                mapping_texts_path = output_path.replace('.npy', '_tweet_id_to_cleaned_text.json')
                with open(mapping_texts_path, 'w', encoding='utf-8') as f:
                    json.dump(tweet_id_to_cleaned_text, f, indent=2, ensure_ascii=False)
                log.info(f"✅ Saved tweet_id to cleaned_text mapping to {mapping_texts_path}")


def analyze_embeddings(embeddings: np.ndarray) -> None:
    """Analyze the generated embeddings."""
    log.info("\n" + "=" * 60)
    log.info("🔍 EMBEDDINGS ANALYSIS")
    log.info("=" * 60)

    log.info(f"📊 Shape: {embeddings.shape}")
    log.info(f"📊 Data type: {embeddings.dtype}")
    log.info(f"📊 Memory usage: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    log.info(f"📊 Value range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    log.info(f"📊 Mean: {embeddings.mean():.4f}")
    log.info(f"📊 Std: {embeddings.std():.4f}")

    # Check for any NaN or infinite values
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()

    if nan_count > 0:
        log.warning(f"⚠️  Found {nan_count} NaN values in embeddings")
    if inf_count > 0:
        log.warning(f"⚠️  Found {inf_count} infinite values in embeddings")

    if nan_count == 0 and inf_count == 0:
        log.info("✅ Embeddings look clean (no NaN or infinite values)")


def main(
    input_file: str,
    output_file: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    include_content: bool = True,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Main function to generate embeddings for articles.

    Args:
        input_file: Path to JSON file with articles
        output_file: Path to save embeddings (default: data/embeddings.npy)
        model: OpenAI embedding model to use
        include_content: Whether to include article content
        batch_size: Batch size for processing
    """
    start_time = time.time()

    # Load environment variables
    load_dotenv()

    # Initialize OpenAI client
    try:
        client = OpenAI()
        log.info("✅ OpenAI client initialized")
    except Exception as e:
        log.error(f"❌ Failed to initialize OpenAI client: {e}")
        return

    # Set default output file if not provided
    if output_file is None:
        input_name = Path(input_file).stem
        output_file = os.path.join(DEFAULT_OUTPUT_DIR, f"{input_name}_embeddings.npy")

    try:
        # Load articles
        df = load_articles(input_file)

        # Prepare texts
        texts = prepare_texts(df, include_content=include_content)
        
        # Generate embeddings
        embeddings = generate_embeddings(
            texts, client, model=model, batch_size=batch_size
        )

        # Analyze embeddings
        analyze_embeddings(embeddings)

        # Save embeddings with cleaned texts
        save_embeddings(embeddings, output_file, df, texts)

        # Summary
        elapsed_time = time.time() - start_time
        log.info("\n" + "=" * 60)
        log.info("📊 SUMMARY")
        log.info("=" * 60)
        log.info(f"📂 Input file: {input_file}")
        log.info(f"💾 Output file: {output_file}")
        log.info(f"📊 Articles processed: {len(df)}")
        log.info(f"📊 Embeddings shape: {embeddings.shape}")
        log.info(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        log.info(f"🤖 Model used: {model}")
        log.info(f"📝 Content included: {include_content}")
        log.info("=" * 60)

    except Exception as e:
        log.error(f"❌ Error in main process: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate embeddings for article texts"
    )
    parser.add_argument("input_file", help="Path to JSON file containing articles")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path for embeddings (default: data/{input_name}_embeddings.npy)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI embedding model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-content", action="store_true", help="Only use titles, not article content"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for processing (default: {BATCH_SIZE})",
    )

    args = parser.parse_args()

    main(
        input_file=args.input_file,
        output_file=args.output,
        model=args.model,
        include_content=not args.no_content,
        batch_size=args.batch_size,
    )
