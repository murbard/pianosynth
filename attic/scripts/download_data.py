import re
import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_URL = "https://theremin.music.uiowa.edu/"
HTML_FILE = "iowa_piano.html"
OUTPUT_DIR = Path("data/raw")

def parse_html_for_links(html_file):
    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Regex to find .aiff links for Piano samples
    # Looking for pattern: href="sound files/MIS/Piano_Other/piano/Piano.<dynamic>.<note>.aiff"
    pattern = r'href="(sound files/MIS/Piano_Other/piano/Piano\.(pp|mf|ff)\.([A-Za-z0-9]+)\.aiff)"'
    matches = re.findall(pattern, content)
    
    # matches is list of (full_rel_path, dynamic, note)
    return matches

def download_file(args):
    rel_path, dynamic, note = args
    # Construct proper URL (handling spaces)
    url_path = rel_path.replace(" ", "%20")
    url = BASE_URL + url_path
    
    # Output path
    save_dir = OUTPUT_DIR / dynamic
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{note}.aiff"
    save_path = save_dir / filename
    
    if save_path.exists():
        return # Skip if already exists
        
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    if not os.path.exists(HTML_FILE):
        print(f"Error: {HTML_FILE} not found.")
        return

    links = parse_html_for_links(HTML_FILE)
    print(f"Found {len(links)} samples.")
    
    # Create tasks
    # Deduplicate based on dynamic+note (just in case)
    unique_links = {}
    for rel, dyn, note in links:
        key = (dyn, note)
        if key not in unique_links:
            unique_links[key] = (rel, dyn, note)
    
    tasks = list(unique_links.values())
    print(f"Downloading {len(tasks)} unique samples...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(download_file, tasks), total=len(tasks)))

    print("Download complete.")

if __name__ == "__main__":
    main()
