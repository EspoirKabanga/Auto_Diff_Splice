import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time

# Add the parent directory so we can import indirect_models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indirect_models import create_spliceator_model
from diffusion_feynman import load_and_encode_sequences, train_model_with_timing, generate_full_sequence_enhanced

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_proxy_classifier(model, optimizer, criterion, X_train, Y_train, epochs=2):
    model.train()
    batch_size = 32
    num_samples = len(X_train)
    for epoch in range(epochs):
        permutation = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], Y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

def evaluate_classifier(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)
        acc = (preds == Y_test).float().mean().item()
    return acc

def encode_strings(seq_list):
    seq_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0.25, 0.25, 0.25, 0.25]}
    encoded = []
    for seq in seq_list:
        if len(seq) == 402:
            encoded.append([seq_dict.get(nt, seq_dict['N']) for nt in seq])
    return torch.tensor(encoded).float().to(device)

def load_text_seqs(filepath, max_items):
    seqs = []
    with open(filepath, 'r') as f:
        for line in f:
            seq = line.strip().upper()
            if len(seq) == 402 and 'N' not in seq:
                seqs.append(seq)
            if len(seqs) >= max_items:
                break
    return seqs

def main():
    print("Evaluating current diffusion_feynman.py...")
    # 1. Dataset variables
    species = "homo"
    seq_type = "acceptor"
    pos_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/{species}_{seq_type}_positive.txt"
    neg_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/{species}_{seq_type}_negative.txt"
    
    n_train = 500
    n_test = 500
    n_gen = 500
    
    # Load real lists
    real_pos_train = load_text_seqs(pos_file, n_train)
    real_neg_train = load_text_seqs(neg_file, n_train)
    
    all_real_pos = load_text_seqs(pos_file, n_train + n_test)
    all_real_neg = load_text_seqs(neg_file, n_train + n_test)
    
    real_pos_test = all_real_pos[-n_test:]
    real_neg_test = all_real_neg[-n_test:]
    
    X_test_real = encode_strings(real_pos_test + real_neg_test)
    Y_test_real = torch.tensor([1]*len(real_pos_test) + [0]*len(real_neg_test)).to(device)
    
    # 2. Baseline Real/Real Evaluation
    print("Training Real/Real Baseline Spliceator proxy...")
    X_train_real = encode_strings(real_pos_train + real_neg_train)
    Y_train_real = torch.tensor([1]*len(real_pos_train) + [0]*len(real_neg_train)).to(device)
    
    model_rr, optim_rr, crit = create_spliceator_model(device)
    model_rr = train_proxy_classifier(model_rr, optim_rr, crit, X_train_real, Y_train_real, epochs=3)
    baseline_acc = evaluate_classifier(model_rr, X_test_real, Y_test_real)
    print(f"Baseline Real/Real Accuracy: {baseline_acc:.4f}")
    
    # 3. Generate Synthetic Sequences
    print("Training Diffusion Model for Synthesis...")
    training_data = load_and_encode_sequences(pos_file, n_train, seq_type)
    diff_model, prev_dist, next_dist = train_model_with_timing(training_data, seq_type, pos_file)
    
    print(f"Generating {n_gen} synthetic sequences...")
    synth_seqs = []
    for i in range(n_gen):
        # blend_weight=0.0 implies no previous lambda lambda weighting, purely relying on the updated Feynman-Kac hook
        seq, _ = generate_full_sequence_enhanced(prev_dist, next_dist, diff_model, seq_type, blend_weight=0.0)
        synth_seqs.append(seq)
        if (i+1) % 100 == 0:
            print(f"Generated {i+1}/{n_gen}...")
            
    correct_motif = sum([1 for seq in synth_seqs if seq[200:202] == ("GT" if seq_type=="donor" else "AG")])
    motif_acc = correct_motif / len(synth_seqs) if synth_seqs else 0
    print(f"Generated motif accuracy: {motif_acc:.4f}")
    
    # 4. Train Synthetic/Test Real Evaluation
    print("Training Synthetic/Real Spliceator proxy...")
    X_train_synth = encode_strings(synth_seqs + real_neg_train)
    Y_train_synth = torch.tensor([1]*len(synth_seqs) + [0]*len(real_neg_train)).to(device)
    
    model_sr, optim_sr, _ = create_spliceator_model(device)
    model_sr = train_proxy_classifier(model_sr, optim_sr, crit, X_train_synth, Y_train_synth, epochs=3)
    synth_acc = evaluate_classifier(model_sr, X_test_real, Y_test_real)
    print(f"Train Synthetic/Test Real Accuracy: {synth_acc:.4f}")
    
    # 5. Output quality score for autoresearch
    # Quality score is an average between structural viability (Motif) and predictive surrogate accuracy
    # A multiplier is applied so it's a robust metric that doesn't collapse easily
    quality_score = 0.5 * motif_acc + 0.5 * (synth_acc / baseline_acc if baseline_acc > 0 else synth_acc)
    print("---")
    print(f"quality_score: {quality_score:.6f}")

if __name__ == '__main__':
    main()
