#!/usr/bin/env python3
"""EuroMillions cron job runner - fetch, check, and predict."""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

BOT_DIR = Path(__file__).parent.resolve()

def run_script(name):
    """Run a script and capture output."""
    script = BOT_DIR / name
    if not script.exists():
        return f"Script not found: {name}", False
    
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BOT_DIR),
        capture_output=True,
        text=True
    )
    
    return result.stdout + result.stderr, result.returncode == 0

def main():
    print("=" * 60)
    print(f"EuroMillions Automation - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print()
    
    # Step 1: Fetch latest draws
    print("📥 Step 1: Fetching latest draws...")
    output, success = run_script("fetch_simple.py")
    print(output)
    if not success:
        print("⚠️ Warning: Fetch had issues, continuing anyway...")
    print()
    
    # Step 2: Check past predictions
    print("✅ Step 2: Checking past predictions...")
    output, success = run_script("check_simple.py")
    print(output)
    print()
    
    # Step 3: Generate new predictions
    print("🎯 Step 3: Generating new predictions...")
    output, success = run_script("predict_simple.py")
    print(output)
    print()
    
    print("=" * 60)
    print("Automation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
