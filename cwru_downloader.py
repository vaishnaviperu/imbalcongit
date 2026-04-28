import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import re

DATASET_ROOT = "data"

PAGES = {
    "normal": "https://engineering.case.edu/bearingdatacenter/normal-baseline-data",
    "drive_end_12k": "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data",
    "drive_end_48k": "https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data",
    "fan_end_12k": "https://engineering.case.edu/bearingdatacenter/12k-fan-end-bearing-fault-data"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

RETRY_COUNT = 3
TIMEOUT = 30

def create_folders():
    """Create the dataset directory structure under DATASET_ROOT."""
    print("\nCreating dataset structure...\n")
    os.makedirs(DATASET_ROOT, exist_ok=True)

    for category in PAGES:
        path = os.path.join(DATASET_ROOT, category)
        os.makedirs(path, exist_ok=True)
        print(f"  {path}")

def fetch_page(url: str) -> str | None:
    """
    Fetch the HTML content of a URL.

    Parameters
    ----------
    url : str
        Target URL.

    Returns
    -------
    str or None
        Response body as text, or None on failure.
    """
    print(f"Fetching: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  Failed to fetch page: {e}")
        return None


def clean_filename(text: str) -> str | None:
    """
    Validate and normalise a link label as a .mat filename.

    Only labels that begin with a recognised CWRU prefix (IR, B, OR, Normal)
    are accepted. Whitespace is stripped.

    Parameters
    ----------
    text : str
        Raw anchor text from the page.

    Returns
    -------
    str or None
        Cleaned filename stem, or None if the label is not a dataset file.
    """
    text = text.strip()
    text = re.sub(r"\s+", "", text)

    if re.match(r"(IR|B|OR|Normal)", text):
        return text

    return None


def extract_named_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """
    Extract labeled .mat download links from a CWRU dataset page.

    Parameters
    ----------
    html : str
        Page HTML content.
    base_url : str
        Base URL used to resolve relative hrefs.

    Returns
    -------
    list of (filename, url) tuples
        Only links whose anchor text matches a recognised CWRU file label.
    """
    print("  Extracting labeled .mat links...")

    soup = BeautifulSoup(html, "html.parser")
    results = []

    for link in soup.find_all("a"):
        href = link.get("href")
        text = link.text

        if not href or ".mat" not in href:
            continue

        name = clean_filename(text)
        if name is None:
            continue

        full_url = urljoin(base_url, href)

        if not name.endswith(".mat"):
            name += ".mat"

        results.append((name, full_url))

    print(f"  Found {len(results)} valid dataset files")
    return results


def download_file(url: str, save_path: str) -> bool:
    """
    Download a file from a URL to disk with retry logic.

    Streams the response in 8 KB chunks to keep memory usage flat
    regardless of file size. Retries up to RETRY_COUNT times on failure
    with a 2-second backoff between attempts.

    Parameters
    ----------
    url : str
        Source URL.
    save_path : str
        Destination file path.

    Returns
    -------
    bool
        True if the file was downloaded successfully, False otherwise.
    """
    file_name = os.path.basename(save_path)

    for attempt in range(RETRY_COUNT):
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"    OK  {file_name}")
            return True

        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    print(f"    FAILED: {file_name}")
    return False

def main():
    """
    Entry point. Iterates all configured CWRU dataset pages, extracts
    .mat download links, and downloads each file into the appropriate
    category subfolder. Files that already exist are skipped. Any files
    that fail all retry attempts are collected and retried once more at
    the end.
    """
    print("\nCWRU BEARING DATASET DOWNLOADER\n")

    create_folders()

    failed_files = []

    for category, url in PAGES.items():
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print("="*60)

        html = fetch_page(url)
        if html is None:
            continue

        files = extract_named_links(html, url)
        save_dir = os.path.join(DATASET_ROOT, category)

        for i, (name, link) in enumerate(files):
            save_path = os.path.join(save_dir, name)

            if os.path.exists(save_path):
                print(f"  Skipping (exists): {name}")
                continue

            print(f"\n  [{i+1}/{len(files)}] {name}")
            success = download_file(link, save_path)

            if not success:
                failed_files.append((name, link, save_dir))

    if failed_files:
        print("\nRetrying failed downloads...\n")
        for name, link, save_dir in failed_files:
            save_path = os.path.join(save_dir, name)
            print(f"  Retrying: {name}")
            download_file(link, save_path)

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
