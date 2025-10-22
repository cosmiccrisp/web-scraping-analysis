#!/usr/bin/env python3
"""
Example script demonstrating tweet language detection.

This script shows how to use the TweetLanguageDetector class for individual tweets
or small batches, useful for testing or processing specific tweets.
"""

import os
import sys
from tweet_language_detection import TweetLanguageDetector

def test_individual_tweets():
    """Test language detection on individual example tweets."""
    
    # Example tweets in different languages
    test_tweets = [
        "Hello world, this is a test tweet in English",
        "Hola mundo, este es un tweet de prueba en español", 
        "Bonjour le monde, ceci est un tweet de test en français",
        "Hii ni mfano wa jinsi tunavyoweza kutoa msaada wa kigeni",
        "مرحبا بالعالم، هذه تغريدة تجريبية باللغة العربية",
        "https://t.co/abc123",  # URL only
        "🚀🔥💯",  # Emojis only
        "🇺🇸🇷🇺 Peace in Ukraine",  # Emojis + text
        "",  # Empty
        "I cannot wait",  # Short English
    ]
    
    try:
        # Initialize detector
        detector = TweetLanguageDetector()
        
        print("Testing batch tweet language detection:")
        print("=" * 50)
        
        # Process all tweets in a single batch
        languages = detector.detect_languages_batch(test_tweets)
        
        for i, (tweet, language) in enumerate(zip(test_tweets, languages), 1):
            print(f"{i:2d}. Tweet: '{tweet[:50]}{'...' if len(tweet) > 50 else ''}'")
            print(f"    Language: {language}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your OPENAI_API_KEY environment variable")

def process_sample_tweets():
    """Process a small sample of tweets from the JSON file."""
    
    import json
    
    # Load a small sample of tweets
    json_file = "/Users/paigelee/Desktop/web-scraping-analysis/embeddings/clean_tweet_texts_by_id.json"
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            all_tweets = json.load(f)
        
        # Take first 5 tweets as sample
        sample_tweets = dict(list(all_tweets.items())[:5])
        
        print(f"Processing sample of {len(sample_tweets)} tweets:")
        print("=" * 50)
        
        detector = TweetLanguageDetector()
        
        # Process as a batch
        tweet_ids = list(sample_tweets.keys())
        tweet_texts = list(sample_tweets.values())
        languages = detector.detect_languages_batch(tweet_texts)
        
        for tweet_id, tweet_text, language in zip(tweet_ids, tweet_texts, languages):
            print(f"Tweet ID: {tweet_id}")
            print(f"Text: {tweet_text[:100]}{'...' if len(tweet_text) > 100 else ''}")
            print(f"Language: {language}")
            print("-" * 30)
            
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Tweet Language Detection Examples")
    print("=" * 40)
    print()
    
    # Test with individual tweets
    test_individual_tweets()
    
    print("\n" + "=" * 50)
    print()
    
    # Test with sample from JSON file
    process_sample_tweets()
