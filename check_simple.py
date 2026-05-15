#!/usr/bin/env python3
"""Check past predictions against actual draws - stdlib only."""

import json
from pathlib import Path
from datetime import datetime

DATA_FILE = Path(__file__).parent / "data" / "draws.json"
RECORDS_FILE = Path(__file__).parent / "predictions" / "records.json"

def load_json(fp):
    if fp.exists():
        with open(fp) as f:
            return json.load(f)
    return []

def save_json(fp, data):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, 'w') as f:
        json.dump(data, f, indent=2)

def check_prediction(pred_record, actual_draw):
    """Check a prediction record against an actual draw."""
    results = []
    for pred in pred_record["predictions"]:
        pred_nums = set(pred["numbers"])
        pred_stars = set(pred["stars"])
        actual_nums = set(actual_draw["numbers"])
        actual_stars = set(actual_draw["stars"])
        
        matched_nums = len(pred_nums & actual_nums)
        matched_stars = len(pred_stars & actual_stars)
        
        results.append({
            "numbers": pred["numbers"],
            "stars": pred["stars"],
            "matched_numbers": matched_nums,
            "matched_stars": matched_stars
        })
    
    return results

def main():
    draws = load_json(DATA_FILE)
    records = load_json(RECORDS_FILE)
    
    if not draws:
        print("No draw data available")
        return
    
    if not records:
        print("No prediction records to check")
        return
    
    # Create a lookup of draws by draw_number
    draws_by_num = {d["draw_number"]: d for d in draws}
    
    updated_count = 0
    for record in records:
        if record.get("checked"):
            continue  # Already checked
        
        draw_num = record.get("for_draw_number")
        if draw_num in draws_by_num:
            actual_draw = draws_by_num[draw_num]
            record["checked"] = True
            record["actual_draw"] = {
                "date": actual_draw["date"],
                "numbers": actual_draw["numbers"],
                "stars": actual_draw["stars"]
            }
            
            # Check each prediction
            results = check_prediction(record, actual_draw)
            record["accuracy"] = results
            
            print(f"✓ Checked draw #{draw_num} ({actual_draw['date']})")
            for i, r in enumerate(results):
                print(f"  Prediction #{i+1}: {r['matched_numbers']}/5 numbers, {r['matched_stars']}/2 stars")
            
            updated_count += 1
        else:
            print(f"✗ Draw #{draw_num} not found in data (actual draw not yet available)")
    
    if updated_count > 0:
        save_json(RECORDS_FILE, records)
        print(f"\nUpdated {updated_count} prediction records")
    else:
        print("\nNo predictions to check")
    
    # Show summary of checked predictions
    checked = [r for r in records if r.get("checked")]
    pending = [r for r in records if not r.get("checked")]
    
    print(f"\nSummary:")
    print(f"  Checked predictions: {len(checked)}")
    print(f"  Pending predictions: {len(pending)}")

if __name__ == "__main__":
    main()
