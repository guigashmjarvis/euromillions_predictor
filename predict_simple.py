#!/usr/bin/env python3
"""Generate predictions for next draw using frequency analysis - stdlib only."""

import json
import random
import math
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_FILE = Path(__file__).parent / "data" / "draws.json"
RECORDS_FILE = Path(__file__).parent / "predictions" / "records.json"
CONFIG_FILE = Path(__file__).parent / "models" / "config.json"

def load_json(fp, default=None):
    if default is None:
        default = [] if fp.name != "config.json" else {}
    if fp.exists():
        with open(fp) as f:
            return json.load(f)
    return default

def save_json(fp, data):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, 'w') as f:
        json.dump(data, f, indent=2)

def calculate_frequencies(draws, lookback=30):
    """Calculate number frequencies from recent draws."""
    recent = draws[-lookback:] if len(draws) >= lookback else draws
    
    main_freq = Counter()
    star_freq = Counter()
    
    for draw in recent:
        for num in draw["numbers"]:
            main_freq[num] += 1
        for star in draw["stars"]:
            star_freq[star] += 1
    
    return main_freq, star_freq

def softmax_sample(freq_dict, temperature=1.5, exclude=None):
    """Sample from frequency distribution with temperature."""
    if exclude is None:
        exclude = set()
    
    # Calculate softmax probabilities
    max_freq = max(freq_dict.values()) if freq_dict else 1
    logits = {}
    for k, v in freq_dict.items():
        if k not in exclude:
            logits[k] = math.log(v + 0.1) / temperature
    
    # Normalize to probabilities
    total = sum(math.exp(v) for v in logits.values())
    probs = {k: math.exp(v) / total for k, v in logits.items()}
    
    # Sample
    r = random.random()
    cumsum = 0
    for k, p in sorted(probs.items()):
        cumsum += p
        if r <= cumsum:
            return k
    return list(probs.keys())[-1] if probs else 1

def generate_prediction(main_freq, star_freq, n_sets=5):
    """Generate n_sets of predictions."""
    predictions = []
    
    for i in range(n_sets):
        # Sample 5 main numbers (1-50)
        nums = set()
        temp = 1.0 + (i * 0.3)  # Increase temperature for diversity
        while len(nums) < 5:
            num = softmax_sample(main_freq, temperature=temp, exclude=nums)
            nums.add(num)
        
        # Sample 2 stars (1-12)
        stars = set()
        while len(stars) < 2:
            star = softmax_sample(star_freq, temperature=temp, exclude=stars)
            stars.add(star)
        
        predictions.append({
            "numbers": sorted(list(nums)),
            "stars": sorted(list(stars))
        })
    
    return predictions

def main():
    draws = load_json(DATA_FILE, [])
    records = load_json(RECORDS_FILE, [])
    
    if len(draws) < 10:
        print("Not enough historical data for predictions (need at least 10 draws)")
        return
    
    # Determine next draw number
    last_draw = draws[-1]
    next_draw_num = last_draw["draw_number"] + 1
    
    # Check if predictions already exist for this draw
    existing = [r for r in records if r.get("for_draw_number") == next_draw_num and not r.get("checked")]
    if existing:
        print(f"✓ Predictions already exist for draw #{next_draw_num}")
        print(f"  Generated at: {existing[0].get('predicted_at', 'unknown')}")
        print(f"  Skipping prediction generation")
        return
    
    print(f"Generating predictions for draw #{next_draw_num}")
    print(f"Last draw: {last_draw['date']} ({last_draw['numbers']} Stars: {last_draw['stars']})")
    
    # Calculate frequencies from recent draws
    main_freq, star_freq = calculate_frequencies(draws, lookback=30)
    
    print(f"\nUsing last 30 draws for frequency analysis")
    print(f"Most common numbers: {main_freq.most_common(5)}")
    print(f"Most common stars: {star_freq.most_common(3)}")
    
    # Generate predictions
    predictions = generate_prediction(main_freq, star_freq, n_sets=5)
    
    print(f"\nGenerated {len(predictions)} prediction sets:")
    for i, pred in enumerate(predictions):
        print(f"  #{i+1}: {pred['numbers']} Stars: {pred['stars']}")
    
    # Record the prediction
    new_record = {
        "predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "for_draw_number": next_draw_num,
        "after_draw": last_draw["date"],
        "predictions": predictions,
        "checked": False,
        "actual_draw": None,
        "accuracy": None
    }
    
    records.append(new_record)
    save_json(RECORDS_FILE, records)
    
    print(f"\n✓ Predictions recorded for draw #{next_draw_num}")
    print(f"  Will be checked after the actual draw results are available")

if __name__ == "__main__":
    main()
