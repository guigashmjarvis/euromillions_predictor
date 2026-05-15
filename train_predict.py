#!/usr/bin/env python3
"""
EuroMillions LSTM - Main application for training and prediction.

Usage:
    python3 train_predict.py train          # Fetch data, train, and save model
    python3 train_predict.py predict        # Load model and generate predictions
    python3 train_predict.py check          # Compare stored predictions with actual draws
"""

import json
import numpy as np
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler

from lstm_model import LSTMNetwork

# ── Paths ────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
PREDICTIONS_DIR = APP_DIR / "predictions"
MODELS_DIR = APP_DIR / "models"
DATA_FILE = DATA_DIR / "euromillions_history.json"
RESULTS_FILE = PREDICTIONS_DIR / "prediction_results.json"

for d in [DATA_DIR, PREDICTIONS_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True)

# ── EuroMillions Configuration ───────────────────────────────────────────────
NUMBERS_MIN, NUMBERS_MAX = 1, 50    # 5 numbers from 1-50
STARS_MIN, STARS_MAX = 1, 12         # 2 stars from 1-12
NUM_NUMBERS = 5
NUM_STARS = 2

# ── Data Fetching ────────────────────────────────────────────────────────────

def fetch_euromillions_data():
    """Fetch historical EuroMillions results from euro-millions.com."""
    print("[fetch] Fetching historical EuroMillions data...")
    
    url = "https://www.euro-millions.com/results/history"
    
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    })
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode('utf-8')
    except Exception as e:
        print(f"[fetch] ERROR: Could not fetch data: {e}")
        return None
    
    soup = BeautifulSoup(html, 'html.parser')
    draws = []
    
    # The site has a table with class "table table-draws"
    table = soup.find('table', class_='table-draws')
    if not table:
        # Try any table
        table = soup.find('table')
    
    if table:
        rows = table.find_all('tr')
        for row in rows:
            # Skip header rows
            if row.find('th'):
                continue
            
            td_list = row.find_all('td')
            if len(td_list) < 3:
                continue
            
            try:
                # Column 0: Date
                date_td = td_list[0]
                date_text = date_td.get_text(strip=True)
                # Convert to standard date format
                # Typical format: "Friday 4th April 2025"
                date_text_clean = re.sub(r'(\d)(st|nd|rd|th)', r'\1', date_text)
                
                date_obj = None
                for fmt in [
                    '%A %d %B %Y',
                    '%A %B %d %Y',
                    '%d %B %Y',
                    '%B %d %Y',
                ]:
                    try:
                        date_obj = datetime.strptime(date_text_clean, fmt)
                        break
                    except ValueError:
                        continue
                
                if date_obj is None:
                    continue
                
                date_str = date_obj.strftime('%Y-%m-%d')
                
                # Column 1: numbers
                nums_td = td_list[1]
                num_balls = nums_td.find_all('span', class_=True)
                # Get text from span elements that contain numbers
                nums = []
                for ball in num_balls:
                    n = re.search(r'\d+', ball.get_text())
                    if n:
                        val = int(n.group())
                        if 1 <= val <= NUMBERS_MAX:
                            nums.append(val)
                
                # Column 2: stars
                stars_td = td_list[2]
                star_balls = stars_td.find_all('span', class_=True)
                stars = []
                for ball in star_balls:
                    n = re.search(r'\d+', ball.get_text())
                    if n:
                        val = int(n.group())
                        if 1 <= val <= STARS_MAX:
                            stars.append(val)
                
                if len(nums) == NUM_NUMBERS and len(stars) == NUM_STARS:
                    draw_number = None
                    # Sometimes the draw number is embedded in date text or URL
                    # We'll assign them later when sorting
                    draws.append({
                        'date': date_str,
                        'draw_number': draw_number,
                        'numbers': sorted(nums),
                        'stars': sorted(stars),
                    })
            except Exception:
                continue
    
    # Sort by date and assign draw numbers
    draws.sort(key=lambda x: x['date'])
    for i, d in enumerate(draws, 1):
        if d['draw_number'] is None:
            d['draw_number'] = i
    
    # Load existing data and merge
    existing_draws = []
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            existing_draws = json.load(f)
        existing_keys = {d['date'] for d in existing_draws}
        for d in draws:
            if d['date'] not in existing_keys:
                existing_draws.append(d)
                print(f"[fetch] New draw found: {d['date']}")
        draws = existing_draws
        # Re-sort and re-number
        draws.sort(key=lambda x: x['date'])
        for i, d in enumerate(draws, 1):
            d['draw_number'] = i
    
    # Save
    with open(DATA_FILE, 'w') as f:
        json.dump(draws, f, indent=2)
    
    print(f"[fetch] Loaded {len(draws)} draws from {draws[0]['date']} to {draws[-1]['date']}")
    return draws


# ── Data Preprocessing ───────────────────────────────────────────────────────

def prepare_sequences(draws, seq_len=10):
    """
    Convert draw history into sequences for LSTM training.
    
    Input: list of draw dicts with 'numbers' and 'stars'
    Output: (X, y_numbers, y_stars) where:
        X has shape (num_samples, seq_len, input_dim)
        y_numbers, y_stars are target values for the draw after the sequence
    
    We normalize numbers to [0,1] range for training.
    """
    all_numbers = []
    all_stars = []
    
    for d in draws:
        # Normalize: numbers to [0, 1] relative to max (50)
        norm_nums = [n / NUMBERS_MAX for n in d['numbers']]
        norm_stars = [s / STARS_MAX for s in d['stars']]
        all_numbers.append(norm_nums)
        all_stars.append(norm_stars)
    
    all_numbers = np.array(all_numbers, dtype=np.float32)
    all_stars = np.array(all_stars, dtype=np.float32)
    
    # Create sequences
    X = []
    y_nums = []
    y_strs = []
    
    for i in range(len(draws) - seq_len):
        # Input: seq_len draws of features
        # Features per draw: [5 normalized_numbers, 2 normalized_stars]
        seq = np.concatenate([all_numbers[i:i+seq_len], all_stars[i:i+seq_len]], axis=1)
        # seq shape: (seq_len, 7)  -- 5 nums + 2 stars per timestep
        X.append(seq)
        
        # Target: next draw's numbers and stars
        y_nums.append(all_numbers[i + seq_len])
        y_strs.append(all_stars[i + seq_len])
    
    X = np.array(X, dtype=np.float32)
    y_nums = np.array(y_nums, dtype=np.float32)
    y_strs = np.array(y_strs, dtype=np.float32)
    
    # Also store metadata for each sample (the draw number it predicts)
    sample_meta = []
    for i in range(len(draws) - seq_len):
        target_idx = i + seq_len
        target_draw = draws[target_idx]
        sample_meta.append({
            'date': target_draw['date'],
            'draw_number': target_draw['draw_number'],
            'actual_numbers': target_draw['numbers'],
            'actual_stars': target_draw['stars'],
        })
    
    return X, y_nums, y_strs, sample_meta, all_numbers, all_stars


def generate_latest_sequence(all_numbers, all_stars, seq_len=10):
    """
    Create the most recent sequence to predict the next draw.
    Returns: (seq,) shape array and metadata
    """
    # Use the last seq_len draws
    nums = all_numbers[-seq_len:]  # (seq_len, 5)
    stars = all_stars[-seq_len:]   # (seq_len, 2)
    
    seq = np.concatenate([nums, stars], axis=1)  # (seq_len, 7)
    return seq


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(epochs=100, learning_rate=0.001, seq_len=10, verbose=True):
    """Train LSTM model on historical EuroMillions data."""
    draws = fetch_euromillions_data()
    if draws is None:
        print("[train] No data available. Cannot train.")
        return None
    
    print(f"\n[train] Preparing sequences (seq_len={seq_len})...")
    X, y_nums, y_strs, sample_meta, all_numbers, all_stars = prepare_sequences(draws, seq_len)
    print(f"[train] Training sequences: {len(X)}</div>  ")
    print(f"[train] Each sequence: ({seq_len} draws × 7 features) → 5 numbers + 2 stars")
    
    if len(X) < 20:
        print("[train] WARNING: Not enough data for training. Need at least 20 sequences.")
        return None
    
    input_dim = 7  # 5 numbers + 2 stars
    hidden_dim = 64
    num_layers = 2
    
    print(f"\n[train] Initializing LSTM: input={input_dim}, hidden={hidden_dim}, layers={num_layers}")
    model = LSTMNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_len=seq_len,
    )
    
    # Training loop with Adam-ish SGD
    print(f"\n[train] Training for {epochs} epochs (lr={learning_rate})...")
    batch_size = min(32, len(X) // 4)
    best_loss = float('inf')
    
    np.random.seed(42)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X))
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, len(X), batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) == 0:
                continue
            
            batch_x = X[batch_idx]
            batch_y_num = y_nums[batch_idx]
            batch_y_str = y_strs[batch_idx]
            
            # Forward pass for regression (not classification)
            # We treat this as regression on normalized values
            num_logits, star_logits = model.forward_sequence(batch_x)
            
            # MSE loss for regression
            num_loss = np.mean((num_logits - batch_y_num) ** 2)
            str_loss = np.mean((star_logits - batch_y_str) ** 2)
            loss = num_loss + str_loss
            epoch_loss += loss
            n_batches += 1
            
            # Backward pass
            # d(output)/d(logits) = 2 * (logits - targets) / batch_size
            d_num = 2.0 * (num_logits - batch_y_num) / (batch_size * NUM_NUMBERS)
            d_str = 2.0 * (star_logits - batch_y_str) / (batch_size * NUM_STARS)
            
            # Gradients for output layer
            batch_h_final = model._h_states[-1][-1]  # (batch, hidden)
            model.grads_dict['W_num'] = batch_h_final.T @ d_num
            model.grads_dict['b_num'] = d_num.sum(axis=0)
            model.grads_dict['W_star'] = batch_h_final.T @ d_str
            model.grads_dict['b_star'] = d_str.sum(axis=0)
            
            # Backprop through time for LSTM layers
            dh = batch_h_final @ model.W_num.T + batch_h_final @ model.W_star.T
            dh = d_num @ model.W_num.T / (batch_size * NUM_NUMBERS) + d_str @ model.W_star.T / (batch_size * NUM_STARS)
            dc = np.zeros_like(dh)
            
            for i in reversed(range(seq_len)):
                x_t = batch_x[:, i, :]
                model.layers[-1].cache.update({'x': x_t})
                
                for l in reversed(range(model.num_layers)):
                    dx, dh, dc = model.layers[l].backward(dh, dc)
                    if l > 0:
                        dh = model.layers[l-1]._h_states[-1][l-1] if hasattr(model.layers[l-1], '_h_states') else dh
                        # We need the previous layer's output as input gradient to this layer
            
            model.clip_gradients(max_norm=5.0)
            
            # SGD update
            for name, param in model.params_dict.items():
                grad = model.grads_dict[name]
                param -= learning_rate * grad
            
            for layer in model.layers:
                for name, param in zip(layer.param_names, layer.params):
                    param -= learning_rate * layer.grads[name]
            
            model.zero_grads()
        
        avg_loss = epoch_loss / max(n_batches, 1)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:4d}/{epochs} |  Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save(MODELS_DIR / "model_best.npz")
    
    print(f"\n[train] Training complete. Best loss: {best_loss:.4f}")
    model.save(MODELS_DIR / "model_latest.npz")
    
    # Store model config for later
    config = {
        'seq_len': seq_len,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'input_dim': input_dim,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'training_date': datetime.now().isoformat(),
        'training_samples': len(X),
    }
    with open(MODELS_DIR / "model_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    return model


# ── Prediction ────────────────────────────────────────────────────────────────

def generate_predictions(model=None, num_predictions=5):
    """
    Generate predictions for the next EuroMillions draw.
    
    Returns a list of prediction dicts, each with:
        numbers: list of 5 numbers (1-50)
        stars: list of 2 stars (1-12)
        confidence: prediction confidence score
    """
    # Load data
    if not DATA_FILE.exists():
        print("[predict] No data found. Run training first.")
        return []
    
    with open(DATA_FILE) as f:
        draws = json.load(f)
    
    _, _, _, _, all_numbers, all_stars = prepare_sequences(draws, 10)
    
    if model is None:
        # Load best model
        model_path = MODELS_DIR / "model_best.npz"
        if model_path.exists():
            model = LSTMNetwork.load(model_path)
        else:
            print("[predict] No model found. Run training first.")
            return []
    
    seq_len = model.seq_len
    predictions = []
    
    # Get latest sequence
    latest_seq = generate_latest_sequence(all_numbers, all_stars, seq_len)
    
    # Generate multiple predictions with different sampling strategies
    for p in range(num_predictions):
        if p == 0:
            # Standard prediction
            x_input = latest_seq[np.newaxis, :, :]
            pred_nums, pred_stars, _, _ = model.predict(x_input)
            confidence = "standard"
        else:
            # Slightly varied predictions (add noise to input)
            noise = np.random.normal(0, 0.02, latest_seq.shape)
            x_input = (latest_seq + noise)[np.newaxis, :, :]
            pred_nums, pred_stars, _, _ = model.predict(x_input)
            confidence = f"variation_{p}"
        
        prediction = {
            'numbers': sorted(set(pred_nums))[:NUM_NUMBERS],  # Ensure unique numbers, 5 of them
            'stars': sorted(set(pred_stars))[:NUM_STARS],     # Ensure unique stars, 2 of them
            'method': confidence,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # Ensure we have exactly 5 numbers and 2 stars
        while len(prediction['numbers']) < NUM_NUMBERS:
            # Add random numbers if model didn't give enough unique ones
            existing = set(prediction['numbers'])
            for n in range(1, NUMBERS_MAX + 1):
                if n not in existing:
                    prediction['numbers'].append(n)
                    existing.add(n)
                    if len(prediction['numbers']) == NUM_NUMBERS:
                        break
        
        prediction['numbers'] = sorted(prediction['numbers'][:NUM_NUMBERS])
        prediction['stars'] = sorted(prediction['stars'][:NUM_STARS])
        
        # Fill with random if still not enough
        while len(prediction['numbers']) < NUM_NUMBERS:
            existing = set(prediction['numbers'])
            for n in range(1, NUMBERS_MAX + 1):
                if n not in existing:
                    prediction['numbers'].append(n)
                    existing.add(n)
                    if len(prediction['numbers']) == NUM_NUMBERS:
                        break
        
        while len(prediction['stars']) < NUM_STARS:
            existing = set(prediction['stars'])
            for s in range(1, STARS_MAX + 1):
                if s not in existing:
                    prediction['stars'].append(s)
                    existing.add(s)
                    if len(prediction['stars']) == NUM_STARS:
                        break
        
        prediction['numbers'] = sorted(prediction['numbers'][:NUM_NUMBERS])
        prediction['stars'] = sorted(prediction['stars'][:NUM_STARS])
        
        predictions.append(prediction)
    
    return predictions


def record_predictions(predictions, draw_number=None):
    """Record predictions with metadata for later comparison."""
    if not RESULTS_FILE.exists():
        results = []
    else:
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    
    # Find the next expected draw number
    if draw_number is None:
        # Get the last prediction to find the next draw number
        if results:
            last_draw_num = max(r.get('target_draw_number', 0) for r in results)
            draw_number = last_draw_num + 1
        else:
            # Use the last historical draw number + 1
            if DATA_FILE.exists():
                with open(DATA_FILE) as f:
                    draws = json.load(f)
                draw_number = draws[-1]['draw_number'] + 1 if draws else 1
            else:
                draw_number = 1
    
    # Calculate expected date (next Tuesday or Friday after now)
    today = datetime.now()
    day_of_week = today.weekday()  # Monday=0
    
    if day_of_week < 1:  # Monday
        days_ahead = 1  # Tuesday
    elif day_of_week < 4:  # Tuesday or Wednesday or Thursday
        days_ahead = 4 - day_of_week  # Friday
    elif day_of_week == 4:  # Friday
        days_ahead = 4  # Next Tuesday
    else:  # Saturday or Sunday
        days_ahead = 2 - day_of_week  # Next Tuesday
    
    expected_date = (today).strftime('%Y-%m-%d')
    
    record = {
        'prediction_group_id': f"pred_{len(results) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'target_draw_number': draw_number,
        'predicted_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'target_date_range': expected_date,
        'predictions': predictions,
        'status': 'pending',  # Will be updated to 'compared' when results are checked
        'matched_numbers': None,
        'matched_stars': None,
        'actual_draw': None,
    }
    
    results.append(record)
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[record] Predictions saved for draw #{draw_number}")
    return record


def check_predictions():
    """Compare stored predictions with actual draw results."""
    if not RESULTS_FILE.exists():
        print("[check] No predictions to check.")
        return
    
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    
    if not DATA_FILE.exists():
        print("[check] No historical data to compare against.")
        return
    
    with open(DATA_FILE) as f:
        draws = json.load(f)
    
    draws_by_number = {d['draw_number']: d for d in draws}
    
    updated_count = 0
    for record in results:
        if record['status'] == 'compared':
            continue
        
        target_num = record['target_draw_number']
        if target_num in draws_by_number:
            actual_draw = draws_by_number[target_num]
            actual_numbers = set(actual_draw['numbers'])
            actual_stars = set(actual_draw['stars'])
            
            record['actual_draw'] = {
                'date': actual_draw['date'],
                'draw_number': actual_draw['draw_number'],
                'numbers': actual_draw['numbers'],
                'stars': actual_draw['stars'],
            }
            
            record['matched_numbers'] = 0
            record['matched_stars'] = 0
            prediction_details = []
            
            for i, pred in enumerate(record['predictions']):
                pred_nums = set(pred['numbers'])
                pred_stars = set(pred['stars'])
                
                matched_n = len(pred_nums & actual_numbers)
                matched_s = len(pred_stars & actual_stars)
                
                record['matched_numbers'] = max(record['matched_numbers'], matched_n)
                record['matched_stars'] = max(record['matched_stars'], matched_s)
                
                prediction_details.append({
                    'prediction': i + 1,
                    'predicted_numbers': pred['numbers'],
                    'predicted_stars': pred['stars'],
                    'matched_numbers': matched_n,
                    'matched_stars': matched_s,
                    'matched_numbers_values': sorted(list(pred_nums & actual_numbers)),
                    'matched_stars_values': sorted(list(pred_stars & actual_stars)),
                })
            
            record['prediction_details'] = prediction_details
            record['status'] = 'compared'
            updated_count += 1
            
            print(f"\n[check] Draw #{target_num} ({actual_draw['date']})")
            print(f"  Actual: Numbers={actual_draw['numbers']}, Stars={actual_draw['stars']}")
            for detail in prediction_details:
                print(f"  Pred #{detail['prediction']}: Numbers={detail['predicted_numbers']}, "
                      f"Stars={detail['predicted_stars']}")
                print(f"    → Matched: {detail['matched_numbers']} numbers "
                      f"({detail['matched_numbers_values']}), "
                      f"{detail['matched_stars']} stars "
                      f"({detail['matched_stars_values']})")
    
    if updated_count == 0:
        print("[check] No new draw results available to compare with pending predictions.")
    else:
        print(f"\n[check] Updated {updated_count} predictions.")
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


# ── CLI Interface ────────────────────────────────────────────────────────────

def print_status():
    """Print current status of the application."""
    print("=" * 60)
    print("  EuroMillions LSTM Prediction System")
    print("=" * 60)
    print()
    
    # Data status
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            draws = json.load(f)
        print(f"  Data: {len(draws)} draws from {draws[0]['date']} to {draws[-1]['date']}")
        print(f"  Latest draw number: {draws[-1]['draw_number']}")
    else:
        print("  Data: No data. Run 'train' to fetch data.")
    
    print()
    
    # Model status
    if (MODELS_DIR / "model_best.npz").exists():
        print("  Model: Trained ✓")
        config_path = MODELS_DIR / "model_config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            print(f"    Trained: {cfg.get('training_date', 'unknown')}")
            print(f"    Samples: {cfg.get('training_samples', 'unknown')}")
    else:
        print("  Model: Not trained. Run 'train' first.")
    
    print()
    
    # Predictions status
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        pending = sum(1 for r in results if r['status'] == 'pending')
        compared = sum(1 for r in results if r['status'] == 'compared')
        print(f"  Predictions: {len(results)} total ({pending} pending, {compared} compared)")
        
        # Show accuracy if any compared
        if compared > 0:
            matched_numbers = [r.get('matched_numbers', 0) for r in results if r['status'] == 'compared']
            matched_stars = [r.get('matched_stars', 0) for r in results if r['status'] == 'compared']
            avg_nums = sum(matched_numbers) / len(matched_numbers) if matched_numbers else 0
            avg_stars = sum(matched_stars) / len(matched_stars) if matched_stars else 0
            print(f"    Avg matched numbers: {avg_nums:.1f}/5")
            print(f"    Avg matched stars: {avg_stars:.1f}/2")
    else:
        print("  Predictions: None yet. Run 'predict' to generate.")
    
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 train_predict.py [status|train|predict|check|help]")
        print()
        print("Commands:")
        print("  status   - Show current status")
        print("  train    - Fetch data and train LSTM model")
        print("  predict  - Generate predictions and record them")
        print("  check    - Compare recorded predictions with actual draws")
        print("  help     - Show this help message")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "help" or command == "--help" or command == "-h":
        print("Usage: python3 train_predict.py [command]")
        print()
        print("Commands:")
        print("  status              - Show current application status")
        print("  train [epochs]      - Fetch data and train LSTM model (default: 100 epochs)")
        print("  predict [n]         - Generate n predictions and record them (default: 5)")
        print("  check               - Compare recorded predictions with actual draws")
        print()
        print("Examples:")
        print("  python3 train_predict.py train 50        # Train for 50 epochs")
        print("  python3 train_predict.py predict 3       # Generate 3 predictions")
        print("  python3 train_predict.py check           # Check results")
        
    elif command == "status":
        print_status()
        
    elif command == "train":
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        print(f"\n[train] Training for {epochs} epochs...")
        model = train_model(epochs=epochs)
        if model:
            print("\n[train] Done! You can now run 'predict' to generate predictions.")
        else:
            print("\n[train] Training failed.")
            sys.exit(1)
            
    elif command == "predict":
        num_preds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        
        # Load or train model
        model_path = MODELS_DIR / "model_best.npz"
        if model_path.exists():
            model = LSTMNetwork.load(str(model_path))
        else:
            print("[predict] No model found. Training one first...")
            model = train_model(epochs=100)
            if not model:
                print("[predict] Training failed.")
                sys.exit(1)
        
        # Generate predictions
        predictions = generate_predictions(model, num_preds)
        
        if not predictions:
            print("[predict] Failed to generate predictions.")
            sys.exit(1)
        
        print("\n" + "=" * 50)
        print("  EuroMillions Predictions")
        print("=" * 50)
        for i, pred in enumerate(predictions, 1):
            print(f"\n  Prediction {i} ({pred['method']}):")
            print(f"    Numbers: {pred['numbers']}")
            print(f"    Stars:   {pred['stars']}")
        
        print(f"\n  Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Record the predictions
        draw_num = None
        if len(sys.argv) > 3:
            try:
                draw_num = int(sys.argv[3])
            except ValueError:
                pass
        
        record = record_predictions(predictions, draw_num)
        print(f"\n  Recorded for draw #{record['target_draw_number']}")
        print(f"  Results saved to: {RESULTS_FILE}")
        print(f"\n  To check results later, run: python3 train_predict.py check")
        
    elif command == "check":
        print("[check] Checking predictions against actual draws...")
        check_predictions()
        print()
        print_status()
        
    else:
        print(f"Unknown command: {command}")
        print("Run 'python3 train_predict.py help' for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
