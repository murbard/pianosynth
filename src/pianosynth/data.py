"""
Data Management Module for PianoSynth

This module handles the downloading and organization of audio datasets, specifically the
University of Iowa Piano samples. It checks for the existence of the dataset index
(downloading it if necessary) and then fetches the individual .aiff audio files
for different dynamic levels (pp, mf, ff).
"""
import os
import re
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://theremin.music.uiowa.edu/"
PIANO_URL = "http://theremin.music.uiowa.edu/MISpiano.html"
HTML_FILENAME = "iowa_piano.html"

def download_file(url, save_path):
    """Downloads a file from a URL to a local path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def parse_html_for_links(html_content):
    """
    Parses HTML content to find piano sample links.
    Returns a list of (relative_url, dynamic, note).
    """
    # Regex to find .aiff links for Piano samples
    # Pattern: href="sound files/MIS/Piano_Other/piano/Piano.<dynamic>.<note>.aiff"
    pattern = r'href="(sound files/MIS/Piano_Other/piano/Piano\.(pp|mf|ff)\.([A-Za-z0-9]+)\.aiff)"'
    matches = re.findall(pattern, html_content)
    return matches

def download_iowa_piano_data(output_dir="data/raw"):
    """
    Downloads the University of Iowa Piano samples.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    html_path = output_path / HTML_FILENAME
    
    # 1. Ensure HTML index exists
    if not html_path.exists():
        print(f"Downloading index from {PIANO_URL}...")
        try:
            response = requests.get(PIANO_URL)
            response.raise_for_status()
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
        except Exception as e:
            print(f"Error downloading index: {e}")
            return

    # 2. Parse HTML
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    links = parse_html_for_links(content)
    print(f"Found {len(links)} samples in index.")

    # 3. Download Samples
    unique_links = {}
    for rel, dyn, note in links:
        key = (dyn, note)
        if key not in unique_links:
            unique_links[key] = (rel, dyn, note)
    
    tasks = []
    for rel, dyn, note in unique_links.values():
        # Construct proper URL (handling spaces)
        url_path = rel.replace(" ", "%20")
        url = BASE_URL + url_path
        
        save_dir = output_path / dyn
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{note}.aiff"
        save_path = save_dir / filename
        
        if not save_path.exists():
            tasks.append((url, save_path))

    if not tasks:
        print("All samples already downloaded.")
        return

    print(f"Downloading {len(tasks)} new samples...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Define helper to unpack args
        def _do_download(args):
            download_file(*args)
            
        list(tqdm(executor.map(_do_download, tasks), total=len(tasks)))
    
    print("Download complete.")
