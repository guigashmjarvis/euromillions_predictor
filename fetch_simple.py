#!/usr/bin/env python3
"""Simple EuroMillions data fetcher - stdlib only, no dependencies."""

import json
import re
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "draws.json"

def http_get(url):
    """GET request with user-agent."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def parse_draws_from_html(html):
    """Parse draw results from euro-millions.com HTML."""
    draws = []
    
    # Find all draws by looking for date patterns followed by ball numbers
    # The HTML structure has dates and then resultBall elements
    
    # Split by date patterns
    date_pattern = r'(\d{2}-\d{2}-\d{4})'
    parts = re.split(date_pattern, html)
    
    for i in range(1, len(parts) - 1, 2):
        date_str = parts[i]
        content_after_date = parts[i + 1]
        
        # Extract all ball numbers after this date (up to next date)
        # Look for resultBall elements
        ball_pattern = r'resultBall[^>]*>(\d+)'
        balls = re.findall(ball_pattern, content_after_date[:2000])  # Look at next 2000 chars
        
        if len(balls) >= 7:
            try:
                nums = [int(b) for b in balls[:5]]
                stars = [int(b) for b in balls[5:7]]
                
                # Validate ranges
                if all(1 <= n <= 50 for n in nums) and all(1 <= s <= 12 for s in stars):
                    draws.append({
                        "date": date_str,
                        "numbers": sorted(set(nums)),
                        "stars": sorted(set(stars))
                    })
            except (ValueError, IndexError):
                continue
    
    return draws

def load_existing_draws():
    """Load existing draws from file."""
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            return json.load(f)
    return []

def save_draws(draws):
    """Save draws to file."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, 'w') as f:
        json.dump(draws, f, indent=2)

def fetch_recent_draws():
    """Fetch recent draws from euro-millions.com."""
    print("Fetching recent EuroMillions draws...")
    
    all_draws = []
    
    # Try the main results page
    url = "https://www.euro-millions.com/results"
    html = http_get(url)
    if html:
        draws = parse_draws_from_html(html)
        print(f"Found {len(draws)} draws on main page")
        all_draws.extend(draws)
    
    # Try paginated results
    for page in range(2, 6):
        url = f"https://www.euro-millions.com/results?page={page}"
        html = http_get(url)
        if html:
            draws = parse_draws_from_html(html)
            if draws:
                print(f"Found {len(draws)} draws on page {page}")
                all_draws.extend(draws)
            else:
                break  # No more results
    
    return all_draws

def merge_draws(existing, new):
    """Merge new draws with existing, avoiding duplicates."""
    # Create a set of existing (date, numbers, stars) tuples
    existing_set = set()
    for d in existing:
        key = (d["date"], tuple(d["numbers"]), tuple(d["stars"]))
        existing_set.add(key)
    
    # Add new draws that don't exist
    merged = existing.copy()
    for d in new:
        key = (d["date"], tuple(d["numbers"]), tuple(d["stars"]))
        if key not in existing_set:
            merged.append(d)
            print(f"Added new draw: {d['date']}")
    
    # Sort by date (convert DD-MM-YYYY to sortable format)
    def date_key(draw):
        parts = draw["date"].split("-")
        return f"{parts[2]}-{parts[1]}-{parts[0]}"
    
    merged.sort(key=date_key)
    return merged

def main():
    # Load existing draws
    existing = load_existing_draws()
    print(f"Loaded {len(existing)} existing draws")
    
    if existing:
        last_draw = existing[-1]
        print(f"Last draw in database: {last_draw['date']} (#{last_draw.get('draw_number', '?')})")
    
    # Fetch new draws
    new_draws = fetch_recent_draws()
    
    if not new_draws:
        print("No new draws found or unable to fetch")
        return
    
    # Merge and save
    merged = merge_draws(existing, new_draws)
    
    # Add draw numbers if missing
    for i, draw in enumerate(merged):
        if "draw_number" not in draw:
            draw["draw_number"] = i + 1
    
    save_draws(merged)
    print(f"\nTotal draws saved: {len(merged)}")
    
    if len(merged) > len(existing):
        print(f"Added {len(merged) - len(existing)} new draws")
        print(f"Latest draw: {merged[-1]['date']} (#{merged[-1]['draw_number']})")

if __name__ == "__main__":
    main()
