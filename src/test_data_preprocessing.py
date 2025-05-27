import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from data_preprocessing import NusaXSentimentDataProcessor

def analyze_data(x_train, y_train, x_val, y_val, x_test, y_test, data_processor):
    """
    Analyze and visualize the preprocessed data
    """
    print("\n===== Data Analysis =====")
    
    print("\n--- Data Shapes ---")
    print(f"Training data: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Validation data: {x_val.shape}")
    print(f"Validation labels: {y_val.shape}")
    print(f"Test data: {x_test.shape}")
    print(f"Test labels: {y_test.shape}")
    
    print("\n--- Data Types ---")
    print(f"Training data type: {x_train.dtype}")
    print(f"Training labels type: {y_train.dtype}")
    print(f"Sample training labels: {y_train[:10]}")
    
    print("\n--- Label Mapping ---")
    print("Label mapping:")
    for label, idx in data_processor.label_mapping.items():
        print(f"  {label} -> {idx}")
    
    print("\n--- Label Distribution ---")
    train_label_dist = np.bincount(y_train.astype(int))
    val_label_dist = np.bincount(y_val.astype(int))
    test_label_dist = np.bincount(y_test.astype(int))
    
    idx_to_label = {v: k for k, v in data_processor.label_mapping.items()}
    
    for i in range(len(train_label_dist)):
        label_name = idx_to_label.get(i, f"Unknown-{i}")
        print(f"Class {i} ({label_name}): Train={train_label_dist[i]} ({train_label_dist[i]/len(y_train):.2%}), "
              f"Val={val_label_dist[i] if i < len(val_label_dist) else 0} "
              f"({val_label_dist[i]/len(y_val):.2%} if i < len(val_label_dist) else 0.0), "
              f"Test={test_label_dist[i] if i < len(test_label_dist) else 0} "
              f"({test_label_dist[i]/len(y_test):.2%} if i < len(test_label_dist) else 0.0)")
    
    print("\n--- Sequence Length Distribution ---")
    train_seq_lengths = np.sum(x_train > 0, axis=1)
    print(f"Mean sequence length: {np.mean(train_seq_lengths):.2f}")
    print(f"Median sequence length: {np.median(train_seq_lengths):.2f}")
    print(f"Min sequence length: {np.min(train_seq_lengths)}")
    print(f"Max sequence length: {np.max(train_seq_lengths)}")
    
    print("\n--- Vocabulary Information ---")
    vocab_size = data_processor.get_vocabulary_size()
    print(f"Vocabulary size: {vocab_size}")
    
    vocab = data_processor.vectorize_layer.get_vocabulary()
    print("Top 20 vocabulary items:")
    for i, word in enumerate(vocab[:20]):
        print(f"  {i}: {word}")
    
    print("\n--- Sample Data ---")
    num_samples = min(5, len(x_train))
    
    for i in range(num_samples):
        print(f"\nSample {i+1}:")
        label_idx = int(y_train[i])
        label_name = idx_to_label.get(label_idx, f"Unknown-{label_idx}")
        print(f"Label: {label_idx} ({label_name})")
        tokens = x_train[i]
        non_zero_tokens = tokens[tokens > 0]
        print(f"Sequence length: {len(non_zero_tokens)}")
        print("Tokens:", non_zero_tokens[:10], "..." if len(non_zero_tokens) > 10 else "")
        if len(vocab) > 1:
            print("First few tokens decoded:")
            for token_idx in non_zero_tokens[:10]:
                if token_idx < len(vocab):
                    print(f"  {token_idx}: {vocab[token_idx]}")
    
    plt.figure(figsize=(15, 5))
    
    label_names = [idx_to_label.get(i, f"Unknown-{i}") for i in range(max(len(train_label_dist), len(val_label_dist), len(test_label_dist)))]
    
    plt.subplot(1, 3, 1)
    plt.bar(range(len(train_label_dist)), train_label_dist)
    plt.title('Training Set Label Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(range(len(train_label_dist)), label_names[:len(train_label_dist)])
    
    plt.subplot(1, 3, 2)
    plt.bar(range(len(val_label_dist)), val_label_dist)
    plt.title('Validation Set Label Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(range(len(val_label_dist)), label_names[:len(val_label_dist)])
    
    plt.subplot(1, 3, 3)
    plt.bar(range(len(test_label_dist)), test_label_dist)
    plt.title('Test Set Label Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(range(len(test_label_dist)), label_names[:len(test_label_dist)])
    
    plt.tight_layout()
    os.makedirs('plotting', exist_ok=True)
    plt.savefig('plotting/label_distribution.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.hist(train_seq_lengths, bins=20)
    plt.title('Sequence Length Distribution in Training Set')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.savefig('plotting/sequence_length_distribution.png')
    plt.close()
    
    print("\nPlots saved to plotting/label_distribution.png and plotting/sequence_length_distribution.png")

def main():
    parser = argparse.ArgumentParser(description='Test data preprocessing for NusaX-Sentiment dataset')
    parser.add_argument('--data_dir', type=str, default='indonesian',
                        help='Directory containing train.csv, valid.csv, and test.csv')
    args = parser.parse_args()
    data_processor = NusaXSentimentDataProcessor(args.data_dir)
    print("Preparing data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data_processor.prepare_data()
    analyze_data(x_train, y_train, x_val, y_val, x_test, y_test, data_processor)


if __name__ == "__main__":
    main() 