#!/usr/bin/env python3
"""
Extract mention data from tweets.xlsx and create two spreadsheets:
1. Accounts that mention others (by type and total)
2. Accounts that get mentioned (by type and total)
"""

import pandas as pd
import re
import os
from pathlib import Path
from collections import defaultdict, Counter

def extract_mentions_from_text(text):
    """Extract @mentions from tweet text"""
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # Find all @mentions in the text
    all_mentions = re.findall(r'@(\w+)', text)
    
    # Filter out mentions that are followed by "..." (truncated)
    filtered_mentions = []
    for mention in all_mentions:
        # Check if this mention is followed by "..." in the original text
        mention_pattern = f'@{re.escape(mention)}\\.\\.\\.'
        if not re.search(mention_pattern, text):
            filtered_mentions.append(mention)
    
    return filtered_mentions

def load_tweets_data():
    """Load tweets data from Excel file"""
    tweets_file = Path("../scraped_data/Tweets.xlsx")
    
    if not tweets_file.exists():
        raise FileNotFoundError(f"Tweets file not found: {tweets_file}")
    
    print(f"Loading tweets from {tweets_file}...")
    df = pd.read_excel(tweets_file)
    
    print(f"Loaded {len(df)} tweets")
    print(f"Columns: {list(df.columns)}")
    
    return df

def analyze_mentions(df):
    """Analyze mentions from tweets data"""
    
    # Initialize data structures
    author_mentions = defaultdict(lambda: defaultdict(int))  # author -> mentioned_account -> count
    mentioned_accounts = defaultdict(int)  # mentioned_account -> total_count
    author_mention_counts = defaultdict(int)  # author -> total_mentions_made
    author_unique_mentions = defaultdict(set)  # author -> set of mentioned accounts
    author_mention_tweet_ids = defaultdict(lambda: defaultdict(set))  # author -> mentioned_account -> set of tweet_ids
    mentioned_account_tweet_ids = defaultdict(set)  # mentioned_account -> set of tweet_ids
    
    # Debug: Check first few tweets
    print("Sample tweet texts:")
    for i, (_, row) in enumerate(df.head(5).iterrows()):
        text = row.get('text', '')
        print(f"Tweet {i+1}: {text[:100]}...")
        mentions = extract_mentions_from_text(text)
        print(f"Mentions found: {mentions}")
    
    # Process each tweet
    for _, row in df.iterrows():
        # Get tweet author
        author = row.get('username', '')  # Changed from 'author_username' to 'username'
        if not author:
            continue
            
        # Get tweet text
        text = row.get('text', '')
        if not text:
            continue
            
        # Get tweet type
        tweet_type = row.get('tweet_type', 'post')  # Default to 'post' if not specified
        
        # Get tweet ID - try different possible column names
        tweet_id = None
        possible_id_columns = ['id', 'tweet_id', 'ID', 'tweetid', 'tweetId']
        for col in possible_id_columns:
            if col in df.columns and pd.notna(row.get(col)):
                tweet_id = str(row.get(col))
                break
        
        # If no ID column found, use row index
        if tweet_id is None:
            tweet_id = str(row.name)
        
        # Extract mentions
        mentions = extract_mentions_from_text(text)
        
        # Update counters
        # Track unique accounts mentioned in this tweet
        unique_mentions_in_tweet = set(mentions)
        
        for mentioned_account in unique_mentions_in_tweet:
            # Author mentions this account (count once per tweet)
            author_mentions[author][mentioned_account] += 1
            author_unique_mentions[author].add(mentioned_account)
            author_mention_tweet_ids[author][mentioned_account].add(tweet_id)
            
            # This account gets mentioned (count once per tweet)
            mentioned_accounts[mentioned_account] += 1
            mentioned_account_tweet_ids[mentioned_account].add(tweet_id)
        
        # Update total mention count for author
        author_mention_counts[author] += len(unique_mentions_in_tweet)
    
    return author_mentions, mentioned_accounts, author_mention_counts, author_unique_mentions, author_mention_tweet_ids, mentioned_account_tweet_ids

def create_mentioning_accounts_sheet(df, author_mentions, author_mention_counts, author_unique_mentions, author_mention_tweet_ids):
    """Create spreadsheet for accounts that mention others"""
    
    # Get unique authors and their data
    authors_data = []
    
    for author in author_mention_counts.keys():
        # Get mention counts by type for this author
        type_counts = defaultdict(int)
        type_unique_counts = defaultdict(set)
        
        # Count mentions by type for this author
        for _, row in df.iterrows():
            if row.get('username') == author:  # Fixed: use 'username' not 'author_username'
                tweet_type = row.get('tweet_type', 'post')
                text = row.get('text', '')
                
                if text:
                    mentions = extract_mentions_from_text(text)
                    # Count unique mentions per tweet (consistent with main analysis)
                    unique_mentions_in_tweet = set(mentions)
                    type_counts[tweet_type] += len(unique_mentions_in_tweet)
                    type_unique_counts[tweet_type].update(unique_mentions_in_tweet)
        
        # Calculate unique accounts mentioned by type
        type_unique_accounts = {t: len(s) for t, s in type_unique_counts.items()}
        
        # Collect all tweet_ids for this author's mentions
        all_tweet_ids = set()
        for mentioned_account in author_unique_mentions[author]:
            all_tweet_ids.update(author_mention_tweet_ids[author][mentioned_account])
        
        authors_data.append({
            'author_id': '',  # We don't have author IDs in the data
            'username': author,
            'total_mentions': author_mention_counts[author],
            'unique_accounts_mentioned_total': len(author_unique_mentions[author]),
            'post_mentions': type_counts.get('post', 0),
            'post_unique_accounts': type_unique_accounts.get('post', 0),
            'reply_mentions': type_counts.get('reply', 0),
            'reply_unique_accounts': type_unique_accounts.get('reply', 0),
            'quote_mentions': type_counts.get('quote', 0),
            'quote_unique_accounts': type_unique_accounts.get('quote', 0),
            'retweet_mentions': type_counts.get('retweet', 0),
            'retweet_unique_accounts': type_unique_accounts.get('retweet', 0),
            'tweet_ids': ', '.join(sorted(all_tweet_ids))
        })
    
    # Create DataFrame
    mentioning_df = pd.DataFrame(authors_data)
    
    # Check if DataFrame is empty
    if len(mentioning_df) == 0:
        print("Warning: No authors found with mentions")
        return mentioning_df
    
    # Sort by total mentions (descending)
    mentioning_df = mentioning_df.sort_values('total_mentions', ascending=False)
    
    return mentioning_df

def load_lkm_accounts():
    """Load LKM accounts from the Excel file"""
    lkm_file = Path("../scraped_data/X Commenters _ LKM (work doc).xlsx")
    
    if not lkm_file.exists():
        print(f"Warning: LKM file not found: {lkm_file}")
        return set()
    
    try:
        # Read the "List of LKM Sites and X Account" sheet
        lkm_df = pd.read_excel(lkm_file, sheet_name="List of LKM Sites and X Account")
        
        # Get the X (Twitter) Accounts column
        if 'X (Twitter) Accounts' in lkm_df.columns:
            lkm_accounts = set(lkm_df['X (Twitter) Accounts'].dropna().astype(str).tolist())
            # Remove @ symbols if present and convert to lowercase
            lkm_accounts = {acc.replace('@', '').lower() for acc in lkm_accounts if acc and acc != 'nan'}
            print(f"Loaded {len(lkm_accounts)} LKM accounts")
            return lkm_accounts
        else:
            print("Warning: 'X (Twitter) Accounts' column not found in LKM file")
            return set()
    except Exception as e:
        print(f"Error loading LKM accounts: {e}")
        return set()

def create_mentioned_accounts_sheet(mentioned_accounts, df, mentioned_account_tweet_ids):
    """Create spreadsheet for accounts that get mentioned"""
    
    # Get unique authors from the dataset
    unique_authors = set(df['username'].dropna().tolist())
    
    # Load LKM accounts
    lkm_accounts = load_lkm_accounts()
    
    # Get mention counts by type for each mentioned account
    mentioned_data = []
    
    for account, total_mentions in mentioned_accounts.items():
        # Count mentions by type for this account
        type_counts = defaultdict(int)
        mentioned_by_authors = set()
        
        for _, row in df.iterrows():
            text = row.get('text', '')
            if text and f'@{account}' in text:
                tweet_type = row.get('tweet_type', 'post')
                author = row.get('username', '')
                
                # Count mentions per tweet (not per occurrence within tweet)
                mentions_in_tweet = extract_mentions_from_text(text)
                if account in mentions_in_tweet:
                    type_counts[tweet_type] += 1  # Count once per tweet
                    
                    # Track who mentioned this account
                    if author:
                        mentioned_by_authors.add(author)
        
        # Check if this account is also an author in the dataset
        is_author = account in unique_authors
        
        # Check if this account is in LKM dataset (compare in lowercase)
        is_in_lkm = account.lower() in lkm_accounts
        
        mentioned_data.append({
            'account': account,
            'total_mentions': total_mentions,
            'unique_authors_mentioned_by': len(mentioned_by_authors),
            'is_author_in_dataset': 'Yes' if is_author else 'No',
            'mentioned_by': ', '.join(sorted(mentioned_by_authors)) if mentioned_by_authors else '',
            'in_lkm_dataset': 'Yes' if is_in_lkm else 'No',
            'post_mentions': type_counts.get('post', 0),
            'reply_mentions': type_counts.get('reply', 0),
            'quote_mentions': type_counts.get('quote', 0),
            'retweet_mentions': type_counts.get('retweet', 0),
            'tweet_ids': ', '.join(sorted(mentioned_account_tweet_ids[account]))
        })
    
    # Create DataFrame
    mentioned_df = pd.DataFrame(mentioned_data)
    
    # Check if DataFrame is empty
    if len(mentioned_df) == 0:
        print("Warning: No mentioned accounts found")
        return mentioned_df
    
    # Sort by total mentions (descending)
    mentioned_df = mentioned_df.sort_values('total_mentions', ascending=False)
    
    return mentioned_df

def main():
    """Main function to extract mentions and create spreadsheets"""
    print("Loading tweets data...")
    df = load_tweets_data()
    
    print("Analyzing mentions...")
    author_mentions, mentioned_accounts, author_mention_counts, author_unique_mentions, author_mention_tweet_ids, mentioned_account_tweet_ids = analyze_mentions(df)
    
    print(f"Found {len(author_mention_counts)} authors who made mentions")
    print(f"Found {len(mentioned_accounts)} accounts that were mentioned")
    
    # Debug: Check tweet types
    print(f"Tweet types in data: {df['tweet_type'].value_counts().to_dict()}")
    
    # Debug: Check a specific author
    if author_mention_counts:
        sample_author = list(author_mention_counts.keys())[0]
        print(f"Sample author: {sample_author}")
        author_tweets = df[df['username'] == sample_author]
        print(f"Tweets by {sample_author}: {len(author_tweets)}")
        print(f"Tweet types for {sample_author}: {author_tweets['tweet_type'].value_counts().to_dict()}")
    
    print("Creating spreadsheet for accounts that mention others...")
    mentioning_df = create_mentioning_accounts_sheet(df, author_mentions, author_mention_counts, author_unique_mentions, author_mention_tweet_ids)
    
    print("Creating spreadsheet for accounts that get mentioned...")
    mentioned_df = create_mentioned_accounts_sheet(mentioned_accounts, df, mentioned_account_tweet_ids)
    
    # Save the spreadsheets
    output_dir = Path("data/mentions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mentioning_file = output_dir / "accounts_that_mention_others.csv"
    mentioned_file = output_dir / "accounts_that_get_mentioned.csv"
    
    mentioning_df.to_csv(mentioning_file, index=False)
    mentioned_df.to_csv(mentioned_file, index=False)
    
    print(f"\nSpreadsheets created successfully!")
    print(f"1. Accounts that mention others: {mentioning_file}")
    print(f"2. Accounts that get mentioned: {mentioned_file}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"- {len(mentioning_df)} accounts that mention others")
    print(f"- {len(mentioned_df)} accounts that get mentioned")
    if len(mentioning_df) > 0:
        print(f"- Top mentioner: {mentioning_df.iloc[0]['username']} ({mentioning_df.iloc[0]['total_mentions']} mentions)")
    if len(mentioned_df) > 0:
        print(f"- Most mentioned: {mentioned_df.iloc[0]['account']} ({mentioned_df.iloc[0]['total_mentions']} mentions)")
    
    # Show sample of each spreadsheet
    print(f"\nSample of accounts that mention others:")
    print(mentioning_df.head(10)[['username', 'total_mentions', 'post_mentions', 'reply_mentions', 'quote_mentions', 'retweet_mentions']].to_string())
    
    print(f"\nSample of accounts that get mentioned:")
    print(mentioned_df.head(10)[['account', 'total_mentions', 'unique_authors_mentioned_by', 'is_author_in_dataset', 'in_lkm_dataset', 'post_mentions', 'reply_mentions', 'quote_mentions', 'retweet_mentions']].to_string())

if __name__ == "__main__":
    main()
