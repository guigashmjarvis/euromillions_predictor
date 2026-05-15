#!/usr/bin/env python3
"""
LSTM Model from scratch using pure NumPy.

Predicts EuroMillions draws: 5 main numbers (1-50) + 2 star numbers (1-12).
Built for prediction purposes — lottery numbers are random by design,
so this is for entertainment/educational purposes.
"""

import numpy as np
import json
import os
import time
from datetime import datetime
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def softmax(x):
    """Stable softmax."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-10)

def sigmoid(x):
    """Stable sigmoid."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    x = np.clip(x, -500, 500)
    return np.tanh(x)

def one_hot(indices, vocab_size):
    """Convert integer indices to one-hot encoding."""
    result = np.zeros((len(indices), vocab_size), dtype=np.float32)
    result[np.arange(len(indices)), indices] = 1.0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Cell
# ─────────────────────────────────────────────────────────────────────────────

class LSTMCell:
    """Single LSTM layer cell with forward and backward pass."""
    
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        
        # Forget gate
        self.Wf = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.Uf = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.bf = np.zeros(hidden_dim, dtype=np.float32)
        
        # Input gate
        self.Wi = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.Ui = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.bi = np.zeros(hidden_dim, dtype=np.float32)
        
        # Cell candidate
        self.Wc = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.Uc = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.bc = np.zeros(hidden_dim, dtype=np.float32)
        
        # Output gate
        self.Wo = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.Uo = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.bo = np.zeros(hidden_dim, dtype=np.float32)
        
        self.params = [self.Wf, self.Uf, self.bf,
                       self.Wi, self.Ui, self.bi,
                       self.Wc, self.Uc, self.bc,
                       self.Wo, self.Uo, self.bo]
        self.param_names = ['Wf', 'Uf', 'bf', 'Wi', 'Ui', 'bi',
                           'Wc', 'Uc', 'bc', 'Wo', 'Uo', 'bo']
        
        # Gradient accumulators
        self.grads = {name: np.zeros_like(p) for name, p in zip(self.param_names, self.params)}
        
        # Cache for backward
        self.cache = {}
    
    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x: (batch, input_dim)
            h_prev: (batch, hidden_dim)
            c_prev: (batch, hidden_dim)
        Returns:
            h_next, c_next
        """
        self.cache['x'] = x
        self.cache['h_prev'] = h_prev
        self.cache['c_prev'] = c_prev
        
        # Gates
        f = sigmoid(x @ self.Wf + h_prev @ self.Uf + self.bf)
        self.cache['f'] = f
        
        i = sigmoid(x @ self.Wi + h_prev @ self.Ui + self.bi)
        self.cache['i'] = i
        
        c_tilde = tanh(x @ self.Wc + h_prev @ self.Uc + self.bc)
        self.cache['c_tilde'] = c_tilde
        
        c_next = f * c_prev + i * c_tilde
        self.cache['c_next'] = c_next
        
        o = sigmoid(x @ self.Wo + h_prev @ self.Uo + self.bo)
        self.cache['o'] = o
        
        h_next = o * tanh(c_next)
        
        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        """Backpropagation through LSTM cell."""
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        c_prev = self.cache['c_prev']
        f = self.cache['f']
        i = self.cache['i']
        c_tilde = self.cache['c_tilde']
        c_next = self.cache['c_next']
        o = self.cache['o']
        
        tanh_c = tanh(c_next)
        
        # Output gate
        do = dh_next * tanh_c
        do_raw = o * (1 - o) * do
        
        # Cell state
        dc = dc_next + dh_next * o * (1 - tanh_c ** 2)
        
        # Forget gate
        df = dc * c_prev
        df_raw = f * (1 - f) * df
        
        # Input gate
        di = dc * c_tilde
        di_raw = i * (1 - i) * di
        
        # Cell candidate
        dc_tilde = dc * i
        dc_tilde_raw = (1 - c_tilde ** 2) * dc_tilde
        
        # Gradients for parameters
        self.grads['Wf'] += x.T @ df_raw
        self.grads['Uf'] += h_prev.T @ df_raw
        self.grads['bf'] += df_raw.sum(axis=0)
        
        self.grads['Wi'] += x.T @ di_raw
        self.grads['Ui'] += h_prev.T @ di_raw
        self.grads['bi'] += di_raw.sum(axis=0)
        
        self.grads['Wc'] += x.T @ dc_tilde_raw
        self.grads['Uc'] += h_prev.T @ dc_tilde_raw
        self.grads['bc'] += dc_tilde_raw.sum(axis=0)
        
        self.grads['Wo'] += x.T @ do_raw
        self.grads['Uo'] += h_prev.T @ do_raw
        self.grads['bo'] += do_raw.sum(axis=0)
        
        # Gradients for previous state and input
        dx = df_raw @ self.Wf.T + di_raw @ self.Wi.T + dc_tilde_raw @ self.Wc.T + do_raw @ self.Wo.T
        dh_prev = df_raw @ self.Uf.T + di_raw @ self.Ui.T + dc_tilde_raw @ self.Uc.T + do_raw @ self.Uo.T
        dc_prev = dc * f
        
        return dx, dh_prev, dc_prev


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Network
# ─────────────────────────────────────────────────────────────────────────────

class LSTMNetwork:
    """Multi-layer LSTM with output projection for prediction."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, 
                 num_numbers=50, num_stars=12, seq_len=10):
        """
        Args:
            input_dim: dimension of input features per timestep
            hidden_dim: hidden state dimension
            num_layers: number of LSTM layers
            num_numbers: vocabulary size for main numbers (50 for EuroMillions)
            num_stars: vocabulary size for star numbers (12 for EuroMillions)
            seq_len: sequence length for training
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_numbers = num_numbers
        self.num_stars = num_stars
        self.seq_len = seq_len
        
        # LSTM layers
        self.layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(LSTMCell(in_dim, hidden_dim))
        
        # Output projections: predict 5 numbers + 2 stars
        # Number head: predicts distribution over 50 numbers
        self.W_num = np.random.randn(hidden_dim, num_numbers).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b_num = np.zeros(num_numbers, dtype=np.float32)
        
        # Star head: predicts distribution over 12 stars
        self.W_star = np.random.randn(hidden_dim, num_stars).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b_star = np.zeros(num_stars, dtype=np.float32)
        
        self.params_dict = {
            'W_num': self.W_num, 'b_num': self.b_num,
            'W_star': self.W_star, 'b_star': self.b_star,
        }
        for i, layer in enumerate(self.layers):
            for name, param in zip(layer.param_names, layer.params):
                self.params_dict[f'layer{i}_{name}'] = param
        
        self.grads_dict = {}
        self.grads_dict['W_num'] = np.zeros_like(self.W_num)
        self.grads_dict['b_num'] = np.zeros_like(self.b_num)
        self.grads_dict['W_star'] = np.zeros_like(self.W_star)
        self.grads_dict['b_star'] = np.zeros_like(self.b_star)
    
    def forward_sequence(self, x_seq):
        """
        Forward pass through entire sequence.
        
        Args:
            x_seq: (batch, seq_len, input_dim)
        Returns:
            num_logits: (batch, num_numbers) - logits for number prediction
            star_logits: (batch, num_stars) - logits for star prediction
        """
        batch, seq_len, _ = x_seq.shape
        
        # Storage for backward
        self._x_seq = x_seq
        self._h_states = []  # list of (num_layers, batch, hidden_dim)
        self._c_states = []
        
        h = [np.zeros((batch, self.hidden_dim), dtype=np.float32) for _ in range(self.num_layers)]
        c = [np.zeros((batch, self.hidden_dim), dtype=np.float32) for _ in range(self.num_layers)]
        self._h_states.append([hi.copy() for hi in h])
        self._c_states.append([ci.copy() for ci in c])
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            for i, layer in enumerate(self.layers):
                h[i], c[i] = layer.forward(x_t, h[i], c[i])
                x_t = h[i]
            self._h_states.append([hi.copy() for hi in h])
            self._c_states.append([ci.copy() for ci in c])
        
        # Final output
        h_final = h[-1]  # last layer's hidden state
        num_logits = h_final @ self.W_num + self.b_num
        star_logits = h_final @ self.W_star + self.b_star
        
        return num_logits, star_logits
    
    def predict(self, x_seq):
        """
        Run inference (no training state).
        
        Args:
            x_seq: (seq_len, input_dim) or (batch, seq_len, input_dim)
        Returns:
            numbers: list of top-5 number indices (sorted, 1-based)
            stars: list of top-2 star indices (sorted, 1-based)
            num_probs: (num_numbers,) probability distribution
            star_probs: (num_stars,) probability distribution
        """
        if x_seq.ndim == 2:
            x_seq = x_seq[np.newaxis, :]
        
        num_logits, star_logits = self.forward_sequence(x_seq)
        
        num_probs = softmax(num_logits[0])
        star_probs = softmax(star_logits[0])
        
        # Mask previously selected numbers to encourage diversity
        top_nums = []
        probs_copy = num_probs.copy()
        for _ in range(5):
            idx = np.argmax(probs_copy)
            top_nums.append(int(idx))
            probs_copy[idx] = -1e10
        
        top_stars = np.argsort(star_probs)[-2:][::-1].tolist()
        
        return (
            sorted([n + 1 for n in top_nums]),
            sorted([s + 1 for s in top_stars]),
            num_probs,
            star_probs,
        )
    
    def clip_gradients(self, max_norm=5.0):
        """Clip all gradients to prevent explosion."""
        for name, grad in self.grads_dict.items():
            norm = np.sqrt(np.sum(grad ** 2))
            if norm > max_norm:
                self.grads_dict[name] = grad * (max_norm / norm)
        
        for layer in self.layers:
            for name, grad in layer.grads.items():
                norm = np.sqrt(np.sum(grad ** 2))
                if norm > max_norm:
                    layer.grads[name] = grad * (max_norm / norm)
    
    def zero_grads(self):
        """Reset all gradients."""
        for name in self.grads_dict:
            self.grads_dict[name].fill(0)
        for layer in self.layers:
            for name in layer.grads:
                layer.grads[name].fill(0)
    
    def save(self, filepath=None):
        """Save model weights."""
        if filepath is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = MODEL_DIR / f"model_{ts}.npz"
        else:
            filepath = Path(filepath)
        
        data = {}
        for name, param in self.params_dict.items():
            data[name] = param
        
        for i, layer in enumerate(self.layers):
            for name, param in zip(layer.param_names, layer.params):
                data[f'layer{i}_{name}'] = param
        
        data['config'] = json.dumps({
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_numbers': self.num_numbers,
            'num_stars': self.num_stars,
            'seq_len': self.seq_len,
        })
        
        np.savez(filepath, **data)
        print(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """Load model from file."""
        data = np.load(filepath, allow_pickle=True)
        config = json.loads(str(data['config']))
        
        model = cls(**config)
        
        model.W_num = data['W_num'].astype(np.float32)
        model.b_num = data['b_num'].astype(np.float32)
        model.W_star = data['W_star'].astype(np.float32)
        model.b_star = data['b_star'].astype(np.float32)
        
        for i, layer in enumerate(model.layers):
            for name in layer.param_names:
                key = f'layer{i}_{name}'
                if key in data:
                    layer_param = getattr(layer, name)
                    layer_param[:] = data[key].astype(np.float32)
        
        model.params_dict = {
            'W_num': model.W_num, 'b_num': model.b_num,
            'W_star': model.W_star, 'b_star': model.b_star,
        }
        for i, layer in enumerate(model.layers):
            for name, param in zip(layer.param_names, layer.params):
                model.params_dict[f'layer{i}_{name}'] = param
        
        print(f"Model loaded from {filepath}")
        return model
