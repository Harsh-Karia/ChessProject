#!/usr/bin/env python3
"""
Create Data Split

This script creates train/test split indices for Leela Chess Zero .gz files.
Instead of physically separating files, it creates index files specifying
which .gz files belong to training and testing sets.

Usage:
    python src/scripts/create_data_split.py --data_dir data/leela_data --output_file data/splits.json --test_ratio 0.2
"""

import argparse
import glob
import json
import os
import random


def main():
    parser = argparse.ArgumentParser(
        description="Create train/test split indices for Leela Chess Zero data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing Leela Chess Zero .gz files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data_splits.json",
        help="Output JSON file to save split indices",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Ratio of data to use for testing (0.0-1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        return

    if args.test_ratio < 0.0 or args.test_ratio > 1.0:
        print(f"Error: Test ratio must be between 0.0 and 1.0, got {args.test_ratio}")
        return

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Find all .gz files in the data directory
    print(f"Scanning for .gz files in {args.data_dir}...")
    data_files = []

    # Get all .gz files recursively
    for file_path in glob.glob(
        os.path.join(args.data_dir, "**", "*.gz"), recursive=True
    ):
        # Store relative path from the data directory
        rel_path = os.path.relpath(file_path, args.data_dir)
        data_files.append(rel_path)

    if not data_files:
        print(f"Error: No .gz files found in {args.data_dir}")
        return

    file_count = len(data_files)
    print(f"Found {file_count} .gz files in {args.data_dir}")

    # Shuffle the files
    random.shuffle(data_files)

    # Calculate split
    test_size = int(file_count * args.test_ratio)
    train_size = file_count - test_size

    # Create split
    train_files = data_files[:train_size]
    test_files = data_files[train_size:]

    # Create file indices
    file_indices = {"train": train_files, "test": test_files}

    # Save split indices to JSON file
    with open(args.output_file, "w") as f:
        json.dump(file_indices, f, indent=2)

    print(
        f"Split created with {len(train_files)} training files ({len(train_files) / file_count:.1%}) and "
        f"{len(test_files)} test files ({len(test_files) / file_count:.1%})"
    )
    print(f"Split indices saved to {args.output_file}")

    if args.verbose:
        print("\nFirst 5 training files:")
        for file in train_files[:5]:
            print(f"  {file}")

        print("\nFirst 5 test files:")
        for file in test_files[:5]:
            print(f"  {file}")


if __name__ == "__main__":
    main()
