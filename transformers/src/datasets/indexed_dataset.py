"""
Indexed Dataset

This module provides a dataset implementation that uses index files for train/test splits
instead of physically separating data files into different directories.
"""

import json
import os
import sys

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.train import V4FormatConverter


class IndexedLeelaDataset(Dataset):
    """Dataset for loading Leela Chess Zero data using split index files."""

    def __init__(self, data_dir, split_file, split="train", verbose=False):
        """
        Initialize the dataset.

        Args:
            data_dir: Base directory containing all .gz files
            split_file: JSON file containing train/test file indices
            split: "train" or "test" split to load
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.data_dir = data_dir
        self.data_paths = []
        self.positions = []
        self.converter = V4FormatConverter(verbose=self.verbose)

        # Verify the data directory exists
        if not os.path.exists(self.data_dir):
            print(f"ERROR: Data directory {self.data_dir} does not exist.")
            print(f"Checked path: {os.path.abspath(self.data_dir)}")
            return

        # Verify the split file exists
        if not os.path.exists(split_file):
            print(f"ERROR: Split file {split_file} does not exist.")
            return

        # Load the split file
        with open(split_file, "r") as f:
            try:
                split_data = json.load(f)
            except json.JSONDecodeError:
                print(f"ERROR: Failed to parse JSON from {split_file}")
                return

        # Check for required split in split data
        if split not in split_data:
            print(
                f"ERROR: Split '{split}' not found in split file. Available splits: {list(split_data.keys())}"
            )
            return

        # Get the list of file paths for this split
        relative_paths = split_data[split]

        if not relative_paths:
            print(f"WARNING: No files found for split '{split}' in {split_file}")
            return

        # Convert relative paths to absolute paths
        self.data_paths = [
            os.path.join(self.data_dir, rel_path) for rel_path in relative_paths
        ]

        # Verify all files exist
        missing_files = [path for path in self.data_paths if not os.path.exists(path)]
        if missing_files:
            print(
                f"WARNING: {len(missing_files)} files from split file don't exist in {self.data_dir}"
            )
            if self.verbose and len(missing_files) <= 10:
                for path in missing_files:
                    print(f"  - Missing: {path}")

            # Remove missing files from the list
            self.data_paths = [path for path in self.data_paths if os.path.exists(path)]

        if self.verbose:
            print(f"Found {len(self.data_paths)} .gz files for split '{split}'")
            for i, path in enumerate(sorted(self.data_paths)):
                if i < 5 or i >= len(self.data_paths) - 5:
                    if i == 5 and len(self.data_paths) > 10:
                        print(f"  ... and {len(self.data_paths) - 10} more files ...")
                    else:
                        print(f"  - {os.path.basename(path)}")

        # Index the files
        self._index_files()

    def _index_files(self):
        """Index the .gz files to build the dataset."""
        if not self.data_paths:
            return

        print(f"Indexing {len(self.data_paths)} .gz files...")
        positions_found = 0

        for file_path in tqdm(self.data_paths, disable=not self.verbose):
            try:
                # Count positions in this file
                record_count = self.converter.count_records_in_file(file_path)

                # Add all positions from this file to the index
                for i in range(record_count):
                    self.positions.append((file_path, i))

                positions_found += record_count

            except Exception as e:
                if self.verbose:
                    print(f"Error indexing {os.path.basename(file_path)}: {str(e)}")

        print(
            f"Indexed {len(self.positions)} positions from {len(self.data_paths)} files"
        )

    def __len__(self):
        """Return the number of positions in the dataset."""
        return len(self.positions)

    def __getitem__(self, idx):
        """
        Get the item at the given index.

        Args:
            idx: Index of the item to get

        Returns:
            Dictionary with input features and training targets
        """
        file_path, record_idx = self.positions[idx]

        # Load and convert the record
        record = self.converter.read_specific_record(file_path, record_idx)
        if record is None:
            # If record loading fails, return a dummy record
            # This is a safety measure - ideally this shouldn't happen after indexing
            if self.verbose:
                print(f"Warning: Failed to load record {record_idx} from {file_path}")
            return self._get_dummy_item()

        try:
            # Convert the record to a training sample
            converted = self.converter.convert_record(record)
            return converted
        except Exception as e:
            if self.verbose:
                print(
                    f"Error converting record {record_idx} from {file_path}: {str(e)}"
                )
            return self._get_dummy_item()

    def _get_dummy_item(self):
        """Return a dummy item when record loading fails."""
        # Create a simple dummy item with zeros
        return {
            "input_features": torch.zeros(112, 8, 8),
            "policy_targets": (torch.zeros(64, 64), torch.zeros(64, 64, 3)),
            "value_targets": {
                "wdl": torch.tensor([0.33, 0.34, 0.33]),
                "q": torch.tensor(0.0),
                "d": torch.tensor(0.34),
                "q_error": torch.tensor(0.0),
                "d_error": torch.tensor(0.0),
            },
        }
