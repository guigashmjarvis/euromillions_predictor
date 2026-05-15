#!/usr/bin/env python3
"""Fix date sorting and renumber draws in chronological order."""
import json
from datetime import datetime

DATA_FILE = '/home/guigashmjarvis/.hermes/euromillions_predictor/data/draws.json'

with open(DATA_FILE) as f:
    draws = json.load(f)

# Convert DD-MM-YYYY to proper datetime for sorting
def parse_date(d):
    return datetime.strptime(d['date'], '%d-%m-%Y')

draws.sort(key=parse_date)

# Remove duplicates (same date appearing more than once)
seen = set()
unique = []
for d in draws:
    if d['date'] not in seen:
        seen.add(d['date'])
        unique.append(d)
draws = unique

# Renumber draws sequentially
for i, d in enumerate(draws, 1):
    d['draw_number'] = i

with open(DATA_FILE, 'w') as f:
    json.dump(draws, f, indent=2)

print(f"Fixed! {len(draws)} draws in chronological order")
print(f"From: {draws[0]['date']} (#{draws[0]['draw_number']}) -> {draws[-1]['date']} (#{draws[-1]['draw_number']})")

# Show last 5
print("\nLast 5 draws:")
for d in draws[-5:]:
    print(f"  #{d['draw_number']} {d['date']}: {d['numbers']} + {d['stars']}")

# Show first 3
print("\nFirst 3 draws:")
for d in draws[:3]:
    print(f"  #{d['draw_number']} {d['date']}: {d['numbers']} + {d['stars']}")

# Check 2025-2026
y25 = [d for d in draws if d['date'].endswith('2025')]
y26 = [d for d in draws if d['date'].endswith('2026')]
print(f"\n2025: {len(y25)} draws")
print(f"2026: {len(y26)} draws")
