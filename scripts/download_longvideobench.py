"""Download LongVideoBench samples using camoufox for browsing + direct downloads."""

import json
from pathlib import Path

from camoufox.sync_api import Camoufox

DATA_DIR = Path("data/longvideobench")
META_DIR = DATA_DIR / "meta"
SAMPLES_DIR = DATA_DIR / "samples"


def download_meta_parquet(page):
    """Download the parquet files from LongVideoBench-Meta."""
    print("\n=== Downloading Meta parquet files ===")
    
    # Navigate to the data directory
    page.goto(
        "https://huggingface.co/datasets/longvideobench/LongVideoBench-Meta/tree/main/data",
        timeout=30000,
    )
    page.wait_for_timeout(3000)
    
    # Find all parquet file links
    links = page.query_selector_all("a[href*='.parquet']")
    parquet_urls = []
    for link in links:
        href = link.get_attribute("href")
        text = link.inner_text().strip()
        if href and ".parquet" in href:
            # Convert to raw download URL
            raw_url = href.replace("/blob/", "/resolve/")
            if not raw_url.startswith("http"):
                raw_url = "https://huggingface.co" + raw_url
            parquet_urls.append((text, raw_url))
            print(f"  Found: {text} -> {raw_url}")
    
    return parquet_urls


def download_file(page, url: str, dest: Path):
    """Download a file using the browser."""
    import urllib.request
    
    print(f"  Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Done: {dest} ({dest.stat().st_size / 1024:.1f} KB)")


def explore_dataset_card(page):
    """Get dataset structure info from the card."""
    print("\n=== Exploring dataset structure ===")
    page.goto(
        "https://huggingface.co/datasets/longvideobench/LongVideoBench-Meta",
        timeout=30000,
    )
    page.wait_for_timeout(3000)
    
    # Get the API endpoint for the dataset viewer
    api_url = "https://datasets-server.huggingface.co/rows?dataset=longvideobench%2FLongVideoBench-Meta&config=default&split=validation&offset=0&length=10"
    
    print(f"  Fetching rows from API: {api_url}")
    page.goto(api_url, timeout=30000)
    page.wait_for_timeout(2000)
    
    body_text = page.inner_text("body")
    return body_text


def main():
    META_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    with Camoufox(headless=True) as browser:
        page = browser.new_page()
        
        # 1. Get parquet file URLs
        parquet_urls = download_meta_parquet(page)
        
        # 2. Download parquet files
        for name, url in parquet_urls:
            fname = name if name.endswith(".parquet") else url.split("/")[-1]
            dest = META_DIR / fname
            if not dest.exists():
                download_file(page, url, dest)
            else:
                print(f"  Already exists: {dest}")
        
        # 3. Get sample rows via API
        print("\n=== Fetching sample rows via HF API ===")
        api_url = "https://datasets-server.huggingface.co/rows?dataset=longvideobench%2FLongVideoBench-Meta&config=default&split=validation&offset=0&length=20"
        page.goto(api_url, timeout=30000)
        page.wait_for_timeout(2000)
        body_text = page.inner_text("body")
        
        # Save raw API response
        api_dest = META_DIR / "sample_rows_validation.json"
        api_dest.write_text(body_text)
        print(f"  Saved API response to {api_dest}")
        
        # Also get test split
        api_url_test = "https://datasets-server.huggingface.co/rows?dataset=longvideobench%2FLongVideoBench-Meta&config=default&split=test&offset=0&length=20"
        page.goto(api_url_test, timeout=30000)
        page.wait_for_timeout(2000)
        body_text_test = page.inner_text("body")
        api_dest_test = META_DIR / "sample_rows_test.json"
        api_dest_test.write_text(body_text_test)
        print(f"  Saved API response to {api_dest_test}")
        
    # 4. Parse and display sample info
    print("\n=== Sample validation entries ===")
    try:
        data = json.loads(api_dest.read_text())
        rows = data.get("rows", [])
        for i, row_wrapper in enumerate(rows[:5]):
            row = row_wrapper.get("row", row_wrapper)
            print(f"\n--- Sample {i+1} ---")
            print(f"  video_id: {row.get('video_id')}")
            print(f"  id: {row.get('id')}")
            print(f"  video_path: {row.get('video_path')}")
            print(f"  duration: {row.get('duration')}s (group: {row.get('duration_group')})")
            print(f"  category: {row.get('question_category')} / {row.get('topic_category')}")
            q = row.get("question", "")
            print(f"  question: {q[:120]}...")
            print(f"  correct_choice: {row.get('correct_choice')}")
            for j in range(5):
                opt = row.get(f"option{j}", "N/A")
                marker = " ✓" if j == row.get("correct_choice") else ""
                print(f"  option{j}: {str(opt)[:80]}{marker}")
    except Exception as e:
        print(f"  Error parsing: {e}")
    
    print(f"\n=== Done! Files saved to {DATA_DIR} ===")


if __name__ == "__main__":
    main()
