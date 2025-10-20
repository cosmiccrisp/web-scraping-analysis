#!/usr/bin/env python3
"""
Async web scraper for news sites that extracts articles from specific categories.
Uses httpx with concurrency for much faster scraping.
Visits each site in the CSV and scrapes articles from:
- category/arts%20and%20entertainment
- category/business
- category/environment
- category/health
- category/politics
"""

import csv
import asyncio
import httpx
from bs4 import BeautifulSoup
import time
import json
import re
from urllib.parse import urljoin, urlparse
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Concurrency settings
PER_HOST_LIMIT = 2  # max in-flight requests per domain
GLOBAL_LIMIT = 20  # max in-flight requests overall
REQUEST_TIMEOUT = 20.0
RETRIES = 2
INITIAL_BACKOFF = 0.5
SITES_CONCURRENCY = 5  # max sites to process concurrently


def host_of(url: str) -> str:
    """Extract hostname from URL"""
    return urlparse(url).netloc.lower()


class AsyncNewsScraper:
    def __init__(
        self, csv_file="sites.csv", output_file="scraped_articles_9-10-politics.json"
    ):
        self.csv_file = csv_file
        self.output_file = output_file
        self.categories = [
            # "arts%20and%20entertainment",
            # "business",
            # "environment",
            # "health",
            "politics",
            # "technology"
        ]
        self.scraped_articles = []
        self.failed_urls = []
        self.max_articles_per_category = None  # None = no limit

    def configure_article_limit(self, max_articles_per_category=None):
        """Configure maximum articles to scrape per category"""
        self.max_articles_per_category = max_articles_per_category
        if max_articles_per_category:
            logger.info(
                f"Article limit set to {max_articles_per_category} per category"
            )
        else:
            logger.info("No article limit - will scrape all articles found")

    async def fetch_with_retry(
        self, client: httpx.AsyncClient, url: str, semaphores: dict
    ) -> Tuple[str, str]:
        """Fetch a single URL with retry logic and per-host concurrency control"""
        host = host_of(url)
        sem = semaphores.setdefault(host, asyncio.Semaphore(PER_HOST_LIMIT))

        backoff = INITIAL_BACKOFF
        for attempt in range(1, RETRIES + 2):
            async with sem:  # per-host concurrency gate
                try:
                    response = await client.get(url, timeout=REQUEST_TIMEOUT)
                    response.raise_for_status()
                    return url, response.text
                except (
                    httpx.ReadTimeout,
                    httpx.ConnectTimeout,
                    httpx.RemoteProtocolError,
                ) as e:
                    if attempt <= RETRIES:
                        logger.warning(
                            f"⏱️  Timeout on {url}, retrying in {backoff}s (attempt {attempt}/{RETRIES + 1})"
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2
                    else:
                        logger.error(
                            f"💥 Failed to fetch {url} after {RETRIES + 1} attempts: {e}"
                        )
                        self.failed_urls.append((url, str(e)))
                        return url, f"ERROR: {type(e).__name__}: {e}"
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code
                    if status in (429, 500, 502, 503, 504) and attempt <= RETRIES:
                        logger.warning(
                            f"🔄 HTTP {status} on {url}, retrying in {backoff}s (attempt {attempt}/{RETRIES + 1})"
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2
                    else:
                        logger.error(f"💥 HTTP {status} error on {url}")
                        self.failed_urls.append((url, f"HTTP {status}"))
                        return url, f"ERROR: HTTP {status}"
                except Exception as e:
                    logger.error(f"Unexpected error fetching {url}: {e}")
                    self.failed_urls.append((url, str(e)))
                    return url, f"ERROR: {type(e).__name__}: {e}"

    def extract_article_links(
        self, soup: BeautifulSoup, base_url: str
    ) -> List[Dict[str, str]]:
        """Extract article links from a category page"""
        article_links = []

        # Ensure base_url has proper protocol
        if not base_url.startswith(("http://", "https://")):
            base_url = "https://" + base_url

        # Look for links with class "hover:underline" as specified
        links = soup.find_all("a", class_="hover:underline")

        for link in links:
            href = link.get("href")
            if href:
                full_url = urljoin(base_url, href)
                title = link.get_text(strip=True)
                if title and "/article/" in href:
                    article_links.append({"url": full_url, "title": title})

        # Also look for other common article link patterns
        other_links = soup.find_all("a", href=re.compile(r"/article/"))
        for link in other_links:
            href = link.get("href")
            if href and not any(
                art["url"] == urljoin(base_url, href) for art in article_links
            ):
                full_url = urljoin(base_url, href)
                title = link.get_text(strip=True)
                if title:
                    article_links.append({"url": full_url, "title": title})

        return article_links

    def extract_article_content(
        self, soup: BeautifulSoup, url: str, site_name: str
    ) -> Dict[str, str]:
        """Extract article content from the article page"""
        article_data = {
            "url": url,
            "site_name": site_name,
            "title": "",
            "author": "",
            "publish_datetime": "",
            "category": "",
            "content": "",
            "tags": [],
            "scraped_at": datetime.now().isoformat(),
        }

        # Extract title
        title_elem = soup.find("h1")
        if title_elem:
            article_data["title"] = title_elem.get_text(strip=True)

        # Extract author - look for the specific pattern you mentioned
        author_elem = soup.find("div", class_="mt-12 p-6 bg-secondary/50 rounded-lg")
        if author_elem:
            author_name_elem = author_elem.find("p", class_="font-medium")
            if author_name_elem:
                # Extract author name with original spacing and capitalization
                author_text = author_name_elem.get_text(strip=True)
                # Handle the specific case where HTML comments are between words
                # First, add spaces before capital letters that follow lowercase letters
                author_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", author_text)
                # Then clean up any remaining HTML comment artifacts
                author_text = re.sub(r"\s*<!--\s*-->\s*", " ", author_text)
                # Clean up multiple consecutive spaces
                author_text = re.sub(r"\s+", " ", author_text)
                article_data["author"] = author_text.strip()
        else:
            # Fallback to original method
            author_elem = soup.find("span", string=re.compile(r"Par|By|Author"))
            if author_elem:
                author_text = author_elem.get_text(strip=True)
                if "Par" in author_text:
                    author_name = author_text.split("Par")[-1].strip()
                elif "By" in author_text:
                    author_name = author_text.split("By")[-1].strip()
                else:
                    author_name = author_text
                article_data["author"] = author_name.strip()

        # Extract publish datetime
        time_elem = soup.find("time")
        if time_elem:
            # Get the datetime attribute
            datetime_attr = time_elem.get("datetime")
            if datetime_attr:
                article_data["publish_datetime"] = datetime_attr

        # Extract category
        category_elem = soup.find("span", class_=re.compile(r"bg-primary|category|tag"))
        if category_elem:
            article_data["category"] = category_elem.get_text(strip=True)

        # Extract content
        content_elem = soup.find("div", class_="prose")
        if content_elem:
            # Remove script and style elements
            for script in content_elem(["script", "style"]):
                script.decompose()

            # Extract content with proper formatting
            content_parts = []
            for element in content_elem.find_all(
                [
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6",
                    "p",
                    "ul",
                    "ol",
                    "li",
                    "blockquote",
                    "div",
                    "br",
                ]
            ):
                if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    # Add newlines before and after headings
                    content_parts.append(f"\n{element.get_text(strip=True)}\n")
                elif element.name == "p":
                    # Add newlines after paragraphs
                    content_parts.append(f"{element.get_text(strip=True)}\n")
                elif element.name in ["ul", "ol"]:
                    # Add newlines before and after lists
                    content_parts.append(f"\n{element.get_text(strip=True)}\n")
                elif element.name == "li":
                    # Add bullet points and newlines for list items
                    content_parts.append(f"• {element.get_text(strip=True)}\n")
                elif element.name == "blockquote":
                    # Add newlines and indentation for blockquotes
                    content_parts.append(f"\n> {element.get_text(strip=True)}\n")
                elif element.name == "br":
                    # Add newlines for line breaks
                    content_parts.append("\n")
                else:
                    # For other elements, just add the text
                    content_parts.append(element.get_text(strip=True))

            # Join all parts and clean up extra whitespace
            article_data["content"] = "\n".join(content_parts).strip()

        # Extract hashtags - look for the specific pattern you mentioned
        hashtag_container = soup.find("div", class_="flex flex-wrap gap-2")
        if hashtag_container:
            hashtag_spans = hashtag_container.find_all(
                "span",
                class_="px-2 py-1 text-xs bg-secondary text-secondary-foreground rounded",
            )
            for span in hashtag_spans:
                tag_text = span.get_text(strip=True)
                if tag_text and tag_text.startswith("#"):
                    # Clean up the hashtag (remove HTML comment artifacts)
                    clean_tag = re.sub(r"\s*<!--\s*-->\s*", "", tag_text)
                    article_data["tags"].append(clean_tag)
        else:
            # Fallback to original method
            tag_elems = soup.find_all("span", class_=re.compile(r"bg-secondary|tag"))
            for tag_elem in tag_elems:
                tag_text = tag_elem.get_text(strip=True)
                if tag_text and not tag_text.startswith("#"):
                    article_data["tags"].append(tag_text)

        return article_data

    async def scrape_category(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        category: str,
        site_name: str,
        semaphores: dict,
    ) -> List[Dict[str, str]]:
        """Scrape all articles from a specific category"""
        category_url = f"{base_url}/category/{category}"
        logger.info(f"🔍 Scraping category: {category} from {site_name}")

        # Fetch category page
        url, html = await self.fetch_with_retry(client, category_url, semaphores)
        if html.startswith("ERROR"):
            logger.error(f"❌ Failed to fetch category page: {html}")
            return []

        soup = BeautifulSoup(html, "html.parser")
        article_links = self.extract_article_links(soup, base_url)
        logger.info(f"📄 Found {len(article_links)} articles in {category} category")

        # Apply article limit if configured
        articles_to_scrape = article_links
        if self.max_articles_per_category:
            articles_to_scrape = article_links[: self.max_articles_per_category]
            logger.info(
                f"🔢 Limiting to {self.max_articles_per_category} articles per category"
            )

        # Fetch all articles concurrently
        article_tasks = []
        for article_link in articles_to_scrape:
            task = asyncio.create_task(
                self.fetch_with_retry(client, article_link["url"], semaphores)
            )
            article_tasks.append((task, article_link["title"]))

        # Process results as they complete
        category_articles = []
        successful = 0
        failed = 0

        for i, (task, title) in enumerate(article_tasks, 1):
            url, html = await task
            if not html.startswith("ERROR"):
                soup = BeautifulSoup(html, "html.parser")
                article_data = self.extract_article_content(soup, url, site_name)
                category_articles.append(article_data)
                successful += 1
                logger.info(
                    f"✅ [{i}/{len(articles_to_scrape)}] Scraped: {article_data['title'][:50]}..."
                )
            else:
                failed += 1
                logger.error(
                    f"❌ [{i}/{len(articles_to_scrape)}] Failed: {title[:50]}... - {html}"
                )

        logger.info(
            f"📊 Category {category} complete: {successful} successful, {failed} failed"
        )
        return category_articles

    async def scrape_site(
        self, site_info: Dict[str, str], semaphores: dict
    ) -> List[Dict[str, str]]:
        """Scrape all categories for a single site"""
        base_url = site_info["url"]
        site_name = site_info.get("name", base_url)

        logger.info(f"🌐 Starting site: {base_url}")
        start_time = time.time()

        # Create HTTP client for this site
        limits = httpx.Limits(
            max_connections=GLOBAL_LIMIT, max_keepalive_connections=GLOBAL_LIMIT
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        async with httpx.AsyncClient(
            http2=True, headers=headers, limits=limits, follow_redirects=True
        ) as client:
            site_articles = []

            # Scrape all categories concurrently
            category_tasks = []
            for category in self.categories:
                task = asyncio.create_task(
                    self.scrape_category(
                        client, base_url, category, site_name, semaphores
                    )
                )
                category_tasks.append((task, category))

            # Wait for all categories to complete
            successful_categories = 0
            failed_categories = 0

            for task, category in category_tasks:
                try:
                    result = await task
                    if isinstance(result, list):
                        site_articles.extend(result)
                        successful_categories += 1
                        logger.info(f"✅ Category {category}: {len(result)} articles")
                    else:
                        failed_categories += 1
                        logger.error(f"❌ Category {category} failed: {result}")
                except Exception as e:
                    failed_categories += 1
                    logger.error(f"❌ Category {category} exception: {e}")

        elapsed = time.time() - start_time
        logger.info(
            f"🏁 Site {base_url} complete: {len(site_articles)} articles in {elapsed:.1f}s ({successful_categories}/{len(self.categories)} categories successful)"
        )
        return site_articles

    async def scrape_sites_concurrently(self, sites: List[Dict[str, str]]) -> None:
        """Scrape multiple sites concurrently with controlled concurrency"""
        semaphores = {}
        site_semaphore = asyncio.Semaphore(SITES_CONCURRENCY)

        logger.info(
            f"🚀 Starting concurrent scraping of {len(sites)} sites (max {SITES_CONCURRENCY} concurrent)"
        )

        async def scrape_site_with_semaphore(site_info):
            async with site_semaphore:
                return await self.scrape_site(site_info, semaphores)

        # Create tasks for all sites
        site_tasks = [
            asyncio.create_task(scrape_site_with_semaphore(site)) for site in sites
        ]

        # Process results as they complete
        completed_sites = 0
        total_articles = 0

        for i, task in enumerate(asyncio.as_completed(site_tasks), 1):
            try:
                site_articles = await task
                self.scraped_articles.extend(site_articles)
                completed_sites += 1
                total_articles += len(site_articles)
                logger.info(
                    f"📈 Progress: {completed_sites}/{len(sites)} sites completed, {total_articles} total articles"
                )
            except Exception as e:
                completed_sites += 1
                logger.error(f"💥 Site {i}/{len(sites)} failed: {e}")

        logger.info(
            f"🎯 All sites completed: {total_articles} articles from {completed_sites} sites"
        )

    def read_sites_from_csv(self) -> List[Dict[str, str]]:
        """Read sites from CSV file"""
        sites = []
        try:
            with open(self.csv_file, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    site_url = row.get("News Websites", "").strip()
                    if site_url:
                        # Ensure URL has protocol
                        if not site_url.startswith(("http://", "https://")):
                            site_url = "https://" + site_url

                        sites.append(
                            {
                                "url": site_url,
                                "name": site_url,
                                "region": row.get("Region", ""),
                                "outlet_name": row.get("Outlet Name", ""),
                                "ip_address": row.get("IP Address", ""),
                                "location": row.get("Location", ""),
                                "registration_date": row.get("Registration Date", ""),
                            }
                        )
        except FileNotFoundError:
            logger.error(f"CSV file {self.csv_file} not found")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")

        return sites

    def save_results(self):
        """Save scraped articles to JSON file"""
        try:
            with open(self.output_file, "w", encoding="utf-8") as file:
                json.dump(self.scraped_articles, file, indent=2, ensure_ascii=False)
            logger.info(
                f"Saved {len(self.scraped_articles)} articles to {self.output_file}"
            )
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def show_summary(self):
        """Show a clean summary of scraped articles by site"""
        if not self.scraped_articles:
            logger.info("No articles were scraped.")
            return

        # Group articles by site
        sites_summary = {}
        for article in self.scraped_articles:
            site_name = article.get("site_name", "Unknown")
            if site_name not in sites_summary:
                sites_summary[site_name] = {
                    "count": 0,
                    "articles": [],
                    "site_url": (
                        article.get("url", "Unknown").split("/article/")[0]
                        if "/article/" in article.get("url", "")
                        else "Unknown"
                    ),
                }
            sites_summary[site_name]["count"] += 1
            sites_summary[site_name]["articles"].append(
                article.get("title", "No title")
            )

        # Display summary
        print("\n" + "=" * 80)
        print("SCRAPING SUMMARY")
        print("=" * 80)
        print(f"Total articles scraped: {len(self.scraped_articles)}")
        print(f"Total sites processed: {len(sites_summary)}")
        print("-" * 80)

        for site_name, data in sites_summary.items():
            print(f"\n📰 {site_name}")
            print(f"   URL: {data['site_url']}")
            print(f"   Articles: {data['count']}")

            # Show first 3 article titles as examples
            if data["articles"]:
                print("   Examples:")
                for i, title in enumerate(data["articles"][:3], 1):
                    # Truncate long titles
                    short_title = title[:60] + "..." if len(title) > 60 else title
                    print(f"     {i}. {short_title}")

                if len(data["articles"]) > 3:
                    print(f"     ... and {len(data['articles']) - 3} more articles")

        print("\n" + "=" * 80)

    def get_failed_urls_report(self):
        """Get a report of failed URLs"""
        if not self.failed_urls:
            return "No URLs failed during scraping."

        report = f"Failed URLs ({len(self.failed_urls)}):\n"
        for url, error in self.failed_urls:
            report += f"  - {url}: {error}\n"
        return report

    async def run(self):
        """Main method to run the scraper"""
        logger.info("🚀 Starting async web scraper...")
        start_time = time.time()

        # Read sites from CSV
        sites = self.read_sites_from_csv()
        if not sites:
            logger.error("❌ No sites found in CSV file")
            return

        logger.info(f"📋 Found {len(sites)} sites to scrape")
        logger.info(
            f"⚙️  Concurrency settings: {SITES_CONCURRENCY} sites, {PER_HOST_LIMIT} per host, {GLOBAL_LIMIT} global limit"
        )

        # Scrape all sites concurrently
        await self.scrape_sites_concurrently(sites)

        # Save results and show summary
        self.save_results()
        self.show_summary()

        # Show failed URLs report
        failed_report = self.get_failed_urls_report()
        if "No URLs failed" not in failed_report:
            logger.warning(f"⚠️  {failed_report}")

        elapsed_time = time.time() - start_time
        logger.info(f"🏆 Scraping completed in {elapsed_time:.2f} seconds")
        logger.info(
            f"📊 Average speed: {len(self.scraped_articles)/elapsed_time:.2f} articles/second"
        )
        logger.info(f"💾 Results saved to: {self.output_file}")


def main():
    """Main function to run the scraper"""
    scraper = AsyncNewsScraper(
        csv_file="sites.csv", output_file="scraped_articles_9-10-tech.json"
    )
    scraper.configure_article_limit(None)  # No limit - scrape all articles

    # Run the async scraper
    asyncio.run(scraper.run())


if __name__ == "__main__":
    main()
