#!/usr/bin/env python3
"""
Tweet Language Detection using LLM

This script uses OpenAI's GPT-4o-mini to detect the language of tweets from Excel files.
It cleans the text by removing usernames and "RT :" prefixes before processing.
It handles cases where language cannot be determined (links, emojis only, etc.) by returning "No language detected".
"""

import json
import csv
import os
import time
import re
import pandas as pd
from typing import Dict, List, Tuple
import openai
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_tweet_text(text: str) -> str:
    """
    Clean tweet text by removing usernames, RT prefixes, and extra whitespace.
    
    Args:
        text: Raw tweet text
        
    Returns:
        Cleaned tweet text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove RT : prefix (more comprehensive pattern)
    text = re.sub(r'^RT\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^RT\s+', '', text, flags=re.IGNORECASE)
    
    # Remove @username mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class TweetLanguageDetector:
    def __init__(self, api_key: str = None):
        """
        Initialize the language detector with OpenAI API key.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
        """
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Using GPT-4o-mini for cost efficiency
        
    def detect_languages_batch(self, tweets: List[str]) -> List[str]:
        """
        Detect languages for multiple tweets in a single API call.
        
        Args:
            tweets: List of tweet texts
            
        Returns:
            List of detected languages (same order as input)
        """
        if not tweets:
            return []
        
        # Send all tweets to LLM for processing - let it decide what's special content
        processed_tweets = tweets
        
        try:
            # Create the batch prompt
            tweets_text = "\n".join([f"{i+1}. {tweet}" for i, tweet in enumerate(processed_tweets)])
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a language detection expert. Your task is to identify the primary language of multiple tweets.

CRITICAL RULES - BE VERY AGGRESSIVE:
1. Return a JSON array with the detected language for each tweet in the same order
2. Use standard language names in English (e.g., "English", "Spanish", "French", "Arabic", "Swahili", "Lingala", etc.)
3. ONLY return "No language detected" for tweets that are:
   - ONLY URLs (e.g., "https://t.co/abc123")
   - ONLY emojis (e.g., "🚀🔥💯")
   - Empty or just whitespace
4. If a tweet has ANY readable text, detect the language - even if it's mixed with URLs/emojis
5. Be EXTREMELY AGGRESSIVE - detect language for almost everything
6. Short phrases, single words, hashtags with text - ALL should be detected
7. Mixed content (text + URLs + emojis) should detect the language of the text part
8. Even incomplete sentences should be detected

Examples:
- "Hello world" → "English"
- "Hola mundo" → "Spanish" 
- "https://t.co/abc123" → "No language detected"
- "🚀🔥💯" → "No language detected"
- "RETOUR" → "French"
- "Nazo yebisa yo" → "Lingala"
- "Hii ni mfano" → "Swahili"
- "🇺🇸🇷🇺 Peace in Ukraine" → "English"
- "Are you ready for Riyadh Fashion Week?🥳 https://t.co/PvPi1bdYKV" → "English"
- "soon they'll fly ! https://t.co/xcITeFSbo6" → "English"
- "respect" → "English"
- "#RDC #BANDAL – Une promesse te..." → "French"
- "The immediate and unconditiona..." → "English"
- "Come visit! 🍕❤️" → "English"
- "very interesting https://t.co/..." → "English"
- "Rick Owens' Los Angeles home p..." → "English"
- "😀" → "No language detected"
- "😂😂😂😂😂😂 https://t.co/NyKX9IcDcR" → "No language detected"

Return format: ["English", "Spanish", "No language detected", "French"]"""
                    },
                    {
                        "role": "user",
                        "content": f"Detect the languages of these tweets:\n{tweets_text}\n\nReturn a JSON array of languages in the same order."
                    }
                ]
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON array from the response
            import re
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                try:
                    detected_languages = json.loads(json_text)
                    # Validate that we got a list
                    if not isinstance(detected_languages, list):
                        logger.warning("JSON response is not a list, using fallback parser")
                        detected_languages = self._parse_fallback_response(response_text, len(processed_tweets))
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error: {e}, using fallback parser")
                    # Fallback: try to parse line by line
                    detected_languages = self._parse_fallback_response(response_text, len(processed_tweets))
            else:
                logger.warning("No JSON array found in response, using fallback parser")
                detected_languages = self._parse_fallback_response(response_text, len(processed_tweets))
            
            # Ensure we have the right number of results
            if len(detected_languages) != len(processed_tweets):
                logger.warning(f"Expected {len(processed_tweets)} languages, got {len(detected_languages)}")
                # Pad with "No language detected" for missing results
                while len(detected_languages) < len(processed_tweets):
                    detected_languages.append("No language detected")
                detected_languages = detected_languages[:len(processed_tweets)]
            
            # Clean up the results
            final_results = []
            for language in detected_languages:
                if isinstance(language, str):
                    # Clean up the language name
                    language = language.strip().strip('"\'')
                    if language.lower() in ["no language detected", "no language", "unable to detect", "error"]:
                        language = "No language detected"
                else:
                    language = "No language detected"
                final_results.append(language)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error detecting languages for batch: {e}")
            return ["No language detected"] * len(tweets)
    
    def _parse_fallback_response(self, response_text: str, expected_count: int) -> List[str]:
        """Fallback parser for when JSON parsing fails."""
        import re
        
        # Try to extract language names from the response
        languages = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for numbered items or language names
            if re.match(r'^\d+\.?\s*', line):
                # Remove numbering
                line = re.sub(r'^\d+\.?\s*', '', line)
            
            # Extract potential language name
            if line and not line.startswith('[') and not line.startswith('{'):
                # Clean up the line
                line = line.strip('"\'.,')
                if line and line not in ['[', ']', '{', '}']:
                    languages.append(line)
        
        # If we still don't have enough, pad with "No language detected"
        while len(languages) < expected_count:
            languages.append("No language detected")
        
        return languages[:expected_count]
    
    def _is_only_special_content(self, text: str) -> bool:
        """
        Check if text contains only URLs, emojis, hashtags, or special characters.
        This function is more conservative - only returns True for obvious non-language content.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text is only special content, False otherwise
        """
        import re
        
        # Handle empty or very short text
        if not text or len(text.strip()) < 2:
            return True
        
        # Remove URLs but keep track of what we removed
        url_pattern = r'https?://\S+|www\.\S+'
        text_no_urls = re.sub(url_pattern, '', text)
        
        # Remove hashtags
        text_no_hashtags = re.sub(r'#\w+', '', text_no_urls)
        
        # Remove mentions
        text_no_mentions = re.sub(r'@\w+', '', text_no_hashtags)
        
        # Remove extra whitespace
        text_clean = text_no_mentions.strip()
        
        # If nothing left after removing URLs/hashtags/mentions, it's special content
        if not text_clean:
            return True
            
        # Remove emojis to check for meaningful text
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]'
        text_no_emojis = re.sub(emoji_pattern, '', text_clean)
        
        # Remove punctuation but keep letters and numbers
        text_final = re.sub(r'[^\w\s]', '', text_no_emojis).strip()
        
        # Only consider it special content if there are very few meaningful characters
        # This is more conservative - allows short words like "respect", "soon", etc.
        return len(text_final) < 2  # Less than 2 meaningful characters
    
    def process_tweets_batch(self, tweets_data: Dict[str, str], batch_size: int = 20, delay: float = 2.0) -> List[Tuple[str, str]]:
        """
        Process tweets in batches to detect languages using batch API calls.
        
        Args:
            tweets_data: Dictionary of tweet_id -> tweet_text
            batch_size: Number of tweets to process in each batch
            delay: Delay between API calls in seconds
            
        Returns:
            List of tuples (tweet_id, detected_language)
        """
        results = []
        total_tweets = len(tweets_data)
        
        logger.info(f"Processing {total_tweets} tweets in batches of {batch_size}")
        
        tweet_items = list(tweets_data.items())
        
        for i in range(0, total_tweets, batch_size):
            batch = tweet_items[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_tweets + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tweets)")
            
            try:
                # Extract tweet texts and IDs
                tweet_ids = [item[0] for item in batch]
                tweet_texts = [item[1] for item in batch]
                
                # Detect languages for the entire batch
                languages = self.detect_languages_batch(tweet_texts)
                
                # Combine results
                for tweet_id, tweet_text, language in zip(tweet_ids, tweet_texts, languages):
                    results.append((tweet_id, language))
                    # Log first 30 characters of cleaned tweet text for better visibility
                    tweet_preview = tweet_text[:30].replace('\n', ' ').replace('\r', ' ')
                    logger.info(f"Tweet '{tweet_preview}{'...' if len(tweet_text) > 30 else ''}': {language}")
                
                # Add delay between batches
                if i + batch_size < total_tweets:
                    logger.info(f"Waiting {delay} seconds before next batch...")
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Add "No language detected" results for this batch
                for tweet_id, _ in batch:
                    results.append((tweet_id, "No language detected"))
        
        return results
    
    def save_results_to_csv(self, results: List[Tuple[str, str]], output_file: str):
        """
        Save language detection results to CSV file.
        
        Args:
            results: List of tuples (tweet_id, detected_language)
            output_file: Path to output CSV file
        """
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['tweet_id', 'language'])
            
            for tweet_id, language in results:
                writer.writerow([tweet_id, language])
        
        logger.info(f"Results saved to {output_file}")


def main():
    """Main function to run the language detection process."""
    
    # File paths
    input_file = "/Users/paigelee/Desktop/web-scraping-analysis/scraped_data/Tweets from Accounts(1).xlsx"
    output_file = "/Users/paigelee/Desktop/web-scraping-analysis/tweets/data/tweet_languages.csv"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Load tweet data from Excel file
        logger.info(f"Loading tweet data from {input_file}")
        df = pd.read_excel(input_file)
        
        # Check if 'text' column exists
        if 'text' not in df.columns:
            raise ValueError("'text' column not found in Excel file. Available columns: " + str(df.columns.tolist()))
        
        # Log available columns for debugging
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Determine the ID column to use
        id_column = None
        possible_id_columns = ['id', 'tweet_id', 'ID', 'tweetid', 'tweetId']
        for col in possible_id_columns:
            if col in df.columns:
                id_column = col
                break
        
        if id_column:
            logger.info(f"Using '{id_column}' column for tweet IDs")
        else:
            logger.warning("No ID column found, using row index as tweet ID")
        
        # Clean the text and create tweets_data dictionary
        tweets_data = {}
        empty_tweets = {}  # Store empty tweets separately
        cleaned_count = 0
        empty_count = 0
        
        for idx, row in df.iterrows():
            # Get tweet ID from appropriate column or use index
            if id_column:
                tweet_id = str(row[id_column]) if pd.notna(row[id_column]) else str(idx)
            else:
                tweet_id = str(idx)
            
            raw_text = str(row['text']) if pd.notna(row['text']) else ""
            
            # Clean the text
            cleaned_text = clean_tweet_text(raw_text)
            
            # Debug logging for first few tweets
            if cleaned_count < 5:
                logger.info(f"Tweet ID: {tweet_id}, Original: '{raw_text[:50]}...' -> Cleaned: '{cleaned_text[:50]}...'")
            
            # Include all tweets, but mark empty ones
            if cleaned_text and len(cleaned_text.strip()) > 0:
                tweets_data[tweet_id] = cleaned_text
                cleaned_count += 1
            else:
                empty_tweets[tweet_id] = ""  # Empty string for empty tweets
                empty_count += 1
        
        logger.info(f"Loaded {len(tweets_data)} tweets with content after cleaning")
        logger.info(f"Found {len(empty_tweets)} empty tweets to mark as 'No language detected'")
        logger.info(f"Total tweets to process: {len(tweets_data) + len(empty_tweets)}")
        
        # Initialize language detector
        detector = TweetLanguageDetector()
        
        # Process tweets with content
        results = []
        if tweets_data:
            results = detector.process_tweets_batch(
                tweets_data, 
                batch_size=10,  # Very small batch size for maximum accuracy
                delay=0.5  # Small delay to respect rate limits
            )
        
        # Add empty tweets with "No language detected"
        for tweet_id in empty_tweets:
            results.append((tweet_id, "No language detected"))
        
        logger.info(f"Processed {len(tweets_data)} tweets with content and {len(empty_tweets)} empty tweets")
        
        # Save results
        detector.save_results_to_csv(results, output_file)
        
        # Print summary
        language_counts = {}
        for _, language in results:
            language_counts[language] = language_counts.get(language, 0) + 1
        
        logger.info("Language detection summary:")
        for language, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {language}: {count} tweets")
        
        logger.info(f"Language detection completed! Results saved to {output_file}")
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
    except Exception as e:
        logger.error(f"Error during language detection: {e}")


if __name__ == "__main__":
    main()
