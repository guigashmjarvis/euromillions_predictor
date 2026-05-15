#!/usr/bin/env python3
"""Fetch historical EuroMillions draw results from the web."""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "euromillions_history.json"


def sanitize_number(s):
    """Extract integer from messy string."""
    if not s:
        return None
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else None


def fetch_from_official():
    """Scrape EuroMillions official historical data."""
    url = "https://www.euro-millions.com/results/history"
    
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; EuroMillionsAnalyzer/1.0)',
        'Accept': 'text/html,application/xhtml+xml',
    })
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode('utf-8')
    except Exception as e:
        print(f"[fetch_data] Official site failed: {e}")
        return None

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Parse the results table
    results = []
    
    # Method 1: Try to find table rows with draw data
    for table in soup.find_all('table'):
        rows = table.find_all('tr', class_='results')
        if not rows:
            rows = table.find_all('tr')[1:]  # skip header
        
        for row in rows:
            try:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 3:
                    continue
                
                # Date cell
                date_text = cells[0].get_text(strip=True)
                # Extract date
                date_matches = re.findall(r'(\d{1,2}\s+\w+\s+\d{4})', date_text)
                if not date_matches:
                    date_matches = re.findall(r'(\w+\s+\d{1,2}\s*,\s*\d{4})', date_text)
                
                # Numbers and stars
                nums = []
                stars = []
                balls = row.find_all(['span', 'em-ball', 'div'], class_=True)
                
                for ball in balls:
                    num = sanitize_number(ball.get_text(strip=True))
                    if num and 1 <= num <= 50 and len(nums) < 5:
                        nums.append(num)
                    elif num and 1 <= num <= 12 and len(stars) < 2 and num not in nums:
                        stars.append(num)
                
                if len(date_matches) >= 1:
                    date_str = date_matches[0]
                    if len(nums) == 5 and len(stars) == 2:
                        results.append({
                            'date': date_str,
                            'numbers': sorted(nums),
                            'stars': sorted(stars),
                            'draw_number': None,
                        })
                        
                        if len(results) > 2000:
                            break
            except Exception:
                continue
    
    results.sort(key=lambda x: x.get('date', ''))
    return results


def fetch_from_lottery_net():
    """Alternative: try lottery.net or similar APIs."""
    url = "https://www.lotterycritic.com/euromillions/results/"
    # This is a backup scraper - official site is preferred
    return None


def save_data(results):
    """Save draw results to JSON."""
    DATA_DIR.mkdir(exist_ok=True)
    
    # Load existing data to avoid duplicates
    existing = {}
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            for item in json.load(f):
                key = f"{item.get('date')}_{item.get('numbers')}_{item.get('stars')}"
                existing[key] = item
    
    # Add new results
    for r in results:
        key = f"{r.get('date')}_{r.get('numbers')}_{r.get('stars')}"
        if key not in existing:
            existing[key] = r
    
    all_results = sorted(existing.values(), key=lambda x: x.get('date', ''))
    
    # Assign draw numbers
    for i, r in enumerate(all_results, 1):
        if r.get('draw_number') is None:
            r['draw_number'] = i
    
    with open(DATA_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved {len(all_results)} draws to {DATA_FILE}")
    return all_results


def load_data():
    """Load cached draw data."""
    if not DATA_FILE.exists():
        return []
    with open(DATA_FILE) as f:
        return json.load(f)


def main():
    print("=" * 50)
    print("  EuroMillions Historical Data Fetcher")
    print("=" * 50)
    print()
    
    results = fetch_from_official()
    
    if results:
        print(f"Fetched {len(results)} draws from official site")
        save_data(results)
    else:
        print("WARNING: No data from official site. Check internet/connection.")
        sys.exit(1)
    
    return load_data()


if __name__ == "__main__":
    main()
