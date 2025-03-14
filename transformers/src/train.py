#!/usr/bin/env python3

import argparse
import glob
import gzip
import logging
import os
import struct
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# weird import issue fix
try:
    # Try package import first
    from src.model.model import ChessformerModel
except ImportError:
    # If that fails, try relative import
    from model.model import ChessformerModel

# LEELA CHESS CONFIG
V4_VERSION = 4
V4_STRUCT_STRING = "4s7432s832sBBBBBBBbffff"
v4_struct = struct.Struct(V4_STRUCT_STRING)

BOARD_SQUARES = 64
INPUT_PLANES = 104
POLICY_VECTOR_SIZE = 1858

CHESSFORMER_PLANES = 112  # The input feature size needed for Chessformer
CHESSFORMER_HISTORY = 8  # Number of past positions to include


class V4FormatConverter:
    """Converter for Leela Chess Zero v4 format training data."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.conversion_stats = {
            "total_attempted": 0,
            "success": 0,
            "errors": {
                "unpacking": 0,
                "input_features": 0,
                "policy_conversion": 0,
                "value_conversion": 0,
            },
        }
        self.stats_report_freq = 10000
        self.move_map = self._create_move_map()
        self.reverse_map = {move: idx for idx, move in enumerate(self.move_map)}

    def _create_move_map(self) -> List[str]:
        files = "abcdefgh"
        ranks = "12345678"
        moves = []

        # Generate all possible moves (from square, to square)
        for from_file in range(8):
            for from_rank in range(8):
                for to_file in range(8):
                    for to_rank in range(8):
                        # Skip non-moves (same square)
                        if from_file == to_file and from_rank == to_rank:
                            continue

                        # Create the move
                        from_sq = files[from_file] + ranks[from_rank]
                        to_sq = files[to_file] + ranks[to_rank]
                        move = from_sq + to_sq

                        # Add promotion moves for 7th rank to 8th rank
                        if (
                            from_rank == 6 and to_rank == 7
                        ):  # Pawn on 7th rank moving to 8th
                            moves.append(move + "n")  # Knight promotion
                            moves.append(move + "b")  # Bishop promotion
                            moves.append(move + "r")  # Rook promotion
                            moves.append(move + "q")  # Queen promotion
                        else:
                            moves.append(move)

        return moves

    def decode_planes(self, planes_data: bytes) -> np.ndarray:
        """
        Decode the binary board planes data into a numpy array.

        Args:
            planes_data: Binary data representing the board state

        Returns:
            A numpy array of shape [INPUT_PLANES, 8, 8] for the input planes
        """
        try:
            # Convert binary data to bit representation
            bits = np.unpackbits(np.frombuffer(planes_data, dtype=np.uint8))

            # Ensure we have the right number of bits
            expected_bits = INPUT_PLANES * 64
            if len(bits) < expected_bits:
                # Pad with zeros if needed
                padding = np.zeros(expected_bits - len(bits), dtype=np.uint8)
                bits = np.concatenate([bits, padding])
            elif len(bits) > expected_bits:
                # Truncate if too many bits
                bits = bits[:expected_bits]

            # Reshape to planes
            planes = bits.reshape(INPUT_PLANES, 8, 8)
            return planes
        except Exception as e:
            print(f"Error decoding planes: {e}")
            # Return empty planes in case of error
            return np.zeros((INPUT_PLANES, 8, 8), dtype=np.uint8)

    def convert_input_features(self, record: Tuple) -> torch.Tensor:
        """
        Convert Leela Chess Zero input features to Chessformer format.

        Args:
            record: Unpacked record from V4 format

        Returns:
            Tensor of shape [64, 112] for Chessformer input
        """
        features = torch.zeros((BOARD_SQUARES, CHESSFORMER_PLANES), dtype=torch.float32)

        try:
            planes_data = record[2]
            castling_us_ooo = int(record[3])
            castling_us_oo = int(record[4])
            castling_them_ooo = int(record[5])
            castling_them_oo = int(record[6])
            side_to_move = int(record[7])
            rule50_count = float(record[8]) / 100.0  # Normalize to [0, 1]

            planes = self.decode_planes(planes_data)
            planes_reshaped = planes.reshape(INPUT_PLANES, -1).T
            planes_tensor = torch.from_numpy(planes_reshaped).float()

            features[:, :INPUT_PLANES] = planes_tensor

            # Set castling rights planes
            castling_plane_offset = INPUT_PLANES  # After the input planes
            features[:, castling_plane_offset].fill_(castling_us_ooo)
            features[:, castling_plane_offset + 1].fill_(castling_us_oo)
            features[:, castling_plane_offset + 2].fill_(castling_them_ooo)
            features[:, castling_plane_offset + 3].fill_(castling_them_oo)

            # Set side to move plane
            stm_plane_offset = INPUT_PLANES + 4  # After castling planes
            features[:, stm_plane_offset].fill_(side_to_move)

            # Set rule50 count
            rule50_plane_offset = INPUT_PLANES + 5  # After side to move plane
            features[:, rule50_plane_offset].fill_(rule50_count)

        except Exception as e:
            print(f"Error in convert_input_features: {e}")

        return features

    def convert_policy(self, policy_data: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Leela Chess Zero policy vector to Chessformer format.

        Args:
            policy_data: Binary policy data from V4 format

        Returns:
            Tuple of (regular_policy, promotion_policy)
        """
        regular_policy = torch.zeros((64, 64), dtype=torch.float32)
        promotion_policy = torch.zeros(
            (8, 8, 4), dtype=torch.float32
        )

        try:
            policy_size = len(policy_data) // 4
            policy_probs = struct.unpack(f"{policy_size}f", policy_data)
            # Convert struct output directly to list of Python floats
            policy_probs = [float(p) for p in policy_probs]

            # Map Leela policy indices to Chessformer format
            for idx, prob in enumerate(policy_probs):
                if idx >= len(self.move_map) or prob <= 0:
                    continue

                move = self.move_map[idx]

                # Check if it's a promotion move
                if len(move) == 5:  # e.g., "e7e8q"
                    from_file = ord(move[0]) - ord("a")
                    from_rank = int(move[1]) - 1
                    to_file = ord(move[2]) - ord("a")
                    to_rank = int(move[3]) - 1

                    # Map promotion piece to index (n=0, b=1, r=2, q=3)
                    piece_map = {"n": 0, "b": 1, "r": 2, "q": 3}
                    piece_idx = piece_map[move[4]]

                    if 0 <= from_file < 8 and 0 <= to_file < 8 and 0 <= piece_idx < 4:
                        promotion_policy[from_file, to_file, piece_idx] = prob
                else:
                    # Regular move
                    from_file = ord(move[0]) - ord("a")
                    from_rank = int(move[1]) - 1
                    to_file = ord(move[2]) - ord("a")
                    to_rank = int(move[3]) - 1

                    from_sq = from_rank * 8 + from_file
                    to_sq = to_rank * 8 + to_file

                    if 0 <= from_sq < 64 and 0 <= to_sq < 64:
                        # Use float(prob) to ensure we're assigning a Python float
                        regular_policy[from_sq, to_sq] = prob

            # Normalize
            total = regular_policy.sum() + promotion_policy.sum()
            if total > 0:
                regular_policy /= total
                promotion_policy /= total
        except Exception as e:
            print(f"Error in convert_policy: {e}")

        return regular_policy, promotion_policy

    def convert_value_target(
        self, root_q: float, best_q: float, root_d: float, best_d: float
    ) -> Dict[str, torch.Tensor]:
        """
        Convert Leela Chess Zero Q and D values to Chessformer value targets.

        Args:
            root_q: Root node Q value
            best_q: Best move Q value
            root_d: Root node D (draw) value
            best_d: Best move D value

        Returns:
            Dictionary with value targets
        """
        try:
            # Convert Q,D to WDL probabilities
            def qd_to_wdl(q, d):
                w = (q + 1) * (1 - d) / 2
                l = (1 - q) * (1 - d) / 2
                d_prob = d
                return np.array([w, d_prob, l], dtype=np.float32)

            # Calculate WDL values
            root_wdl = qd_to_wdl(root_q, root_d)
            best_wdl = qd_to_wdl(best_q, best_d)

            # Calculate differences (error = best - root)
            wdl_error = best_wdl - root_wdl
            q_error = best_q - root_q
            d_error = best_d - root_d

            # Convert to tensors
            value_targets = {
                "wdl": torch.from_numpy(root_wdl),
                "best_wdl": torch.from_numpy(best_wdl),
                "error": torch.from_numpy(wdl_error),
                "q": torch.tensor([root_q], dtype=torch.float32),
                "best_q": torch.tensor([best_q], dtype=torch.float32),
                "q_error": torch.tensor([q_error], dtype=torch.float32),
                "d": torch.tensor([root_d], dtype=torch.float32),
                "best_d": torch.tensor([best_d], dtype=torch.float32),
                "d_error": torch.tensor([d_error], dtype=torch.float32),
            }

            return value_targets
        except Exception as e:
            print(f"Error in convert_value_target: {e}")
            # Return default empty values in case of error
            return {
                "wdl": torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32),
                "best_wdl": torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32),
                "error": torch.zeros(3, dtype=torch.float32),
                "q": torch.zeros(1, dtype=torch.float32),
                "best_q": torch.zeros(1, dtype=torch.float32),
                "q_error": torch.zeros(1, dtype=torch.float32),
                "d": torch.zeros(1, dtype=torch.float32),
                "best_d": torch.zeros(1, dtype=torch.float32),
                "d_error": torch.zeros(1, dtype=torch.float32),
            }

    def convert_record(
        self, record_data: bytes
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Convert a Leela Chess Zero V4 record to Chessformer format.

        Args:
            record_data: Raw record data in V4 format

        Returns:
            Tuple of (input_features, policy_targets, value_targets)
        """
        self.conversion_stats["total_attempted"] += 1

        try:
            # Unpack the record
            try:
                unpacked = v4_struct.unpack(record_data)
            except struct.error as e:
                self.conversion_stats["errors"]["unpacking"] += 1
                if self.verbose:
                    print(f"Error unpacking record: {e}")
                    print(
                        f"Record data length: {len(record_data)}, expected: {v4_struct.size}"
                    )
                raise ValueError(f"Failed to unpack record: {e}")

            # Extract data
            version_bytes = unpacked[0]
            policy_data = unpacked[1]
            planes_data = unpacked[2]

            # Validate field indices
            if len(unpacked) < 15:
                self.conversion_stats["errors"]["general"] += 1
                if self.verbose:
                    print(
                        f"Record has too few fields: {len(unpacked)}, expected at least 15"
                    )
                raise ValueError(f"Record has insufficient fields: {len(unpacked)}")

            root_q = float(unpacked[11])
            best_q = float(unpacked[12])
            root_d = float(unpacked[13])
            best_d = float(unpacked[14])

            # Convert to Chessformer format
            try:
                input_features = self.convert_input_features(unpacked)
            except Exception as e:
                self.conversion_stats["errors"]["input_features"] += 1
                if self.verbose:
                    print(f"Error converting input features: {e}")
                raise ValueError(f"Failed to convert input features: {e}")

            try:
                regular_policy, promotion_policy = self.convert_policy(policy_data)
            except Exception as e:
                self.conversion_stats["errors"]["policy_conversion"] += 1
                if self.verbose:
                    print(f"Error converting policy: {e}")
                raise ValueError(f"Failed to convert policy: {e}")

            try:
                value_targets = self.convert_value_target(
                    root_q, best_q, root_d, best_d
                )
            except Exception as e:
                self.conversion_stats["errors"]["value_conversion"] += 1
                if self.verbose:
                    print(f"Error converting value targets: {e}")
                raise ValueError(f"Failed to convert value targets: {e}")

            # Create policy targets dictionary
            policy_targets = {
                "policy": regular_policy,
                "promotion_policy": promotion_policy,
            }

            self.conversion_stats["success"] += 1
            return input_features, policy_targets, value_targets

        except Exception as e:
            self.conversion_stats["errors"]["general"] += 1
            if self.verbose:
                print(f"Error converting record: {e}")

            # Periodically print conversion stats if verbose
            if (
                self.verbose
                and self.conversion_stats["total_attempted"] % self.stats_report_freq
                == 0
            ):
                self._print_conversion_stats()

            # Return empty data in case of error
            empty_features = torch.zeros(
                (BOARD_SQUARES, CHESSFORMER_PLANES), dtype=torch.float32
            )
            empty_policy = {
                "policy": torch.zeros((64, 64), dtype=torch.float32),
                "promotion_policy": torch.zeros((8, 8, 4), dtype=torch.float32),
            }
            empty_value = {
                "wdl": torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32),
                "error": torch.zeros(3, dtype=torch.float32),
                "q": torch.zeros(1, dtype=torch.float32),
                "best_q": torch.zeros(1, dtype=torch.float32),
                "q_error": torch.zeros(1, dtype=torch.float32),
            }
            return empty_features, empty_policy, empty_value

    def _print_conversion_stats(self):
        """Print current conversion statistics"""
        stats = self.conversion_stats
        success_rate = (
            stats["success"] / stats["total_attempted"] * 100
            if stats["total_attempted"] > 0
            else 0
        )

        print("\nRecord Conversion Statistics:")
        print(f"  Total records attempted: {stats['total_attempted']}")
        print(f"  Successfully converted: {stats['success']} ({success_rate:.2f}%)")

        if sum(stats["errors"].values()) > 0:
            print("  Errors by category:")
            for category, count in stats["errors"].items():
                if count > 0:
                    error_pct = count / stats["total_attempted"] * 100
                    print(f"    - {category}: {count} ({error_pct:.2f}%)")

    def count_records_in_file(self, file_path):
        """
        Count the number of records in a .gz file.

        Args:
            file_path: Path to the .gz file

        Returns:
            Number of records in the file
        """
        try:
            record_count = 0
            with gzip.open(file_path, "rb") as f:
                # Read and verify version
                version = struct.unpack("i", f.read(4))[0]
                if version != V4_VERSION:
                    if self.verbose:
                        print(
                            f"Warning: {file_path} has version {version}, expected {V4_VERSION}"
                        )
                    return 0

                # Count records
                while True:
                    try:
                        # Try to read record header (size)
                        size_data = f.read(4)
                        if not size_data or len(size_data) < 4:
                            break  # End of file reached

                        # Get record size and skip to next record
                        size = struct.unpack("i", size_data)[0]
                        f.seek(size, 1)  # Skip the record data
                        record_count += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"Error counting records in {file_path}: {str(e)}")
                        break

            return record_count
        except Exception as e:
            if self.verbose:
                print(f"Error opening file {file_path}: {str(e)}")
            return 0

    def read_specific_record(self, file_path, record_idx):
        """
        Read a specific record from a .gz file.

        Args:
            file_path: Path to the .gz file
            record_idx: Index of the record to read (0-based)

        Returns:
            The unpacked record tuple, or None if the record cannot be read
        """
        try:
            with gzip.open(file_path, "rb") as f:
                # Read and verify version
                version = struct.unpack("i", f.read(4))[0]
                if version != V4_VERSION:
                    if self.verbose:
                        print(
                            f"Warning: {file_path} has version {version}, expected {V4_VERSION}"
                        )
                    return None

                # Skip to the desired record
                current_idx = 0
                while current_idx < record_idx:
                    # Read record size
                    size_data = f.read(4)
                    if not size_data or len(size_data) < 4:
                        return None  # End of file reached before the target record

                    # Skip this record
                    size = struct.unpack("i", size_data)[0]
                    f.seek(size, 1)
                    current_idx += 1

                # Read the target record
                size_data = f.read(4)
                if not size_data or len(size_data) < 4:
                    return None  # End of file reached

                size = struct.unpack("i", size_data)[0]
                record_data = f.read(size)

                # Unpack the record
                return self._unpack_v4_record(record_data)

        except Exception as e:
            if self.verbose:
                print(f"Error reading record {record_idx} from {file_path}: {str(e)}")
            return None

    def _unpack_v4_record(self, record_data):
        """
        Unpack a V4 format record.

        Args:
            record_data: Binary record data

        Returns:
            Unpacked record tuple, or None if unpacking fails
        """
        try:
            # Unpack the record according to V4 format structure
            record = struct.unpack(V4_STRUCT_STRING, record_data)
            return record
        except Exception as e:
            if self.verbose:
                print(f"Error unpacking record: {str(e)}")
            return None


class GzippedLeelaDataset(Dataset):
    """Dataset for loading Leela Chess Zero V4 training data directly from .gz files."""

    def __init__(self, data_dir, split="train", verbose=False, use_raw_dir=False):
        """
        Initialize the dataset.

        Args:
            data_dir: Base directory containing .gz files (or subdirectories with .gz files)
            split: "train" or "test" split to load. Use empty string to look directly in data_dir.
            verbose: Enable verbose logging
            use_raw_dir: If True, ignores the split parameter and loads all files from data_dir
        """
        self.verbose = verbose

        # If use_raw_dir is True, we ignore the split and use data_dir directly
        if use_raw_dir:
            self.data_dir = data_dir
            self.split_type = "raw"  # For logging purposes
        else:
            self.data_dir = data_dir if not split else os.path.join(data_dir, split)
            self.split_type = split or "direct"

        self.data_paths = []
        self.positions = []
        self.converter = V4FormatConverter(verbose=self.verbose)

        # Verify the data directory exists
        if not os.path.exists(self.data_dir):
            print(f"WARNING: Data directory {self.data_dir} does not exist.")
            print(f"Checked path: {os.path.abspath(self.data_dir)}")
            return

        # Find all .gz files in the data directory
        gz_pattern = os.path.join(self.data_dir, "**", "*.gz")
        recursive_paths = glob.glob(gz_pattern, recursive=True)

        # Try without recursion if no files found with recursion
        direct_paths = glob.glob(os.path.join(self.data_dir, "*.gz"))

        # Combine paths, ensuring no duplicates
        self.data_paths = list(set(recursive_paths + direct_paths))

        if not self.data_paths:
            print(f"WARNING: No .gz files found in {self.data_dir}")
            print(f"Checked recursive pattern: {gz_pattern}")
            print(f"Checked direct pattern: {os.path.join(self.data_dir, '*.gz')}")
            return

        if self.verbose:
            print(f"Found {len(self.data_paths)} .gz files in {self.data_dir}")
            for i, path in enumerate(sorted(self.data_paths)):
                if i < 5 or i >= len(self.data_paths) - 5:
                    print(f"  - {os.path.basename(path)}")
                elif i == 5:
                    print(f"  - ... ({len(self.data_paths) - 10} more files) ...")

        # Index positions
        self._index_positions()

        total_files = len(self.data_paths)
        files_with_positions = sum(
            1 for file_path, count in self.file_stats.items() if count > 0
        )

        print(f"Dataset summary for '{self.split_type}' split:")
        print(f"  - Directory: {self.data_dir}")
        print(f"  - Total .gz files found: {total_files}")
        print(f"  - Files with valid positions: {files_with_positions}")
        print(f"  - Total positions indexed: {len(self.positions)}")

        if total_files > 0 and files_with_positions == 0:
            print("\nWARNING: No valid positions found in any of the .gz files!")

            if self.attempted_files and self.verbose:
                print("\nDetailed error information:")
                for file_path, error in list(self.file_errors.items())[:5]:
                    print(f"  - {os.path.basename(file_path)}: {error}")
                if len(self.file_errors) > 5:
                    print(
                        f"  - ... and {len(self.file_errors) - 5} more files with errors"
                    )

    def _index_positions(self):
        """Index all positions in the data files."""
        self.file_stats = {}  # Track positions per file
        self.file_errors = {}  # Track errors per file
        self.attempted_files = set()  # Track which files we tried to process

        for file_idx, file_path in enumerate(self.data_paths):
            self.attempted_files.add(file_path)
            positions_in_file = 0

            try:
                with gzip.open(file_path, "rb") as f:
                    # Try reading the first few bytes to check if it's a valid gzip file
                    try:
                        f.read(4)
                        f.seek(0)  # Reset position
                    except Exception as e:
                        self.file_errors[file_path] = f"Not a valid gzip file: {str(e)}"
                        continue

                    data = f.read()

                    # Check version
                    try:
                        version = struct.unpack("i", data[0:4])[0]
                        if version != V4_VERSION:
                            self.file_errors[file_path] = (
                                f"Unsupported version: {version}"
                            )
                            if self.verbose:
                                print(
                                    f"Skipping file with unsupported version {version}: {file_path}"
                                )
                            continue
                    except struct.error as e:
                        self.file_errors[file_path] = (
                            f"Failed to unpack version: {str(e)}"
                        )
                        continue
                    except Exception as e:
                        self.file_errors[file_path] = (
                            f"Error checking version: {str(e)}"
                        )
                        continue

                    # Calculate record size
                    record_size = v4_struct.size

                    # Index positions
                    for offset in range(4, len(data), record_size):
                        if offset + record_size <= len(data):
                            self.positions.append((file_path, offset))
                            positions_in_file += 1

                    if positions_in_file == 0 and self.verbose:
                        print(
                            f"No valid positions found in {file_path} (file size: {len(data)} bytes)"
                        )

                    self.file_stats[file_path] = positions_in_file

                    # Progress indicator for large datasets
                    if (file_idx + 1) % 20 == 0 or file_idx == len(self.data_paths) - 1:
                        print(
                            f"Indexed {file_idx + 1}/{len(self.data_paths)} files, found {len(self.positions)} positions so far"
                        )

            except gzip.BadGzipFile as e:
                self.file_errors[file_path] = f"Bad gzip file: {str(e)}"
                if self.verbose:
                    print(f"Error: {file_path} is not a valid gzip file: {e}")
            except FileNotFoundError as e:
                self.file_errors[file_path] = f"File not found: {str(e)}"
                if self.verbose:
                    print(f"Error: {file_path} not found: {e}")
            except Exception as e:
                self.file_errors[file_path] = f"General error: {str(e)}"
                if self.verbose:
                    print(f"Error processing file {file_path}: {e}")

    def __len__(self):
        """Return the number of positions in the dataset."""
        return len(self.positions)

    def __getitem__(self, idx):
        """Get a position by index."""
        file_path, offset = self.positions[idx]

        try:
            with gzip.open(file_path, "rb") as f:
                f.seek(offset)
                record_data = f.read(v4_struct.size)

            # Convert the record to Chessformer format
            return self.converter.convert_record(record_data)
        except Exception as e:
            if self.verbose:
                print(f"Error reading record at {file_path}:{offset}: {e}")
            # Return empty data in case of error
            empty_features = torch.zeros(
                (BOARD_SQUARES, CHESSFORMER_PLANES), dtype=torch.float32
            )
            empty_policy = {
                "policy": torch.zeros((64, 64), dtype=torch.float32),
                "promotion_policy": torch.zeros((8, 8, 4), dtype=torch.float32),
            }
            empty_value = {
                "wdl": torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32),
                "error": torch.zeros(3, dtype=torch.float32),
                "q": torch.zeros(1, dtype=torch.float32),
                "best_q": torch.zeros(1, dtype=torch.float32),
                "q_error": torch.zeros(1, dtype=torch.float32),
            }
            return empty_features, empty_policy, empty_value


class ChessformerTrainer:
    """Trainer for the Chessformer model."""

    def __init__(self, config_path, data_dir, output_dir="checkpoints", verbose=False):
        """
        Initialize the trainer.

        Args:
            config_path: Path to configuration YAML file
            data_dir: Directory containing Leela Chess Zero .gz files
            output_dir: Directory to save checkpoints and logs
            verbose: Enable verbose output and debugging information
        """
        # Store verbose flag
        self.verbose = verbose

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

        # Create TensorBoard writer
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(
            log_dir=os.path.join(output_dir, "logs", current_time)
        )

        # Create model
        self.model = self.create_model()
        self.model.to(self.device)

        # Create optimizer and scheduler
        training_config = self.config.get("training", {})
        self.learning_rate = training_config.get("learning_rate", 1e-4)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=training_config.get("weight_decay", 1e-2),
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Create data loaders
        self.batch_size = training_config.get("batch_size", 128)

        # Set the number of workers for data loading
        dataset_config = self.config.get("dataset", {})
        preprocess_config = dataset_config.get("preprocess", {})
        self.num_workers = preprocess_config.get("num_workers", 4)

        self.train_loader, self.val_loader = self.create_data_loaders(data_dir)

        # Loss weights
        self.loss_weights = training_config.get(
            "loss_weights", {"policy": 1.0, "value": 1.0}
        )

        # Training parameters
        self.num_epochs = training_config.get("epochs", 100)
        self.save_freq = training_config.get("save_frequency", 10)
        self.grad_clip = training_config.get("grad_clip", 1.0)

    def create_model(self):
        """Create and initialize the model from config."""
        model_config = self.config.get("model", {})

        # Extract model parameters from config
        embed_dim = model_config.get("embed_dim", 1024)
        num_heads = model_config.get("num_heads", 32)
        ff_hidden_dim = model_config.get("ff_hidden_dim", 4096)
        num_layers = model_config.get("num_layers", 15)
        d_value = model_config.get("d_value", 32)
        value_embedding_dim = model_config.get("value_embedding_dim", 128)
        dropout = model_config.get("dropout", 0.1)

        # Create model
        model = ChessformerModel(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            num_layers=num_layers,
            d_value=d_value,
            value_embedding_dim=value_embedding_dim,
            dropout=dropout,
        )

        return model

    def create_data_loaders(self, data_dir):
        """Create training and validation data loaders."""
        # Create datasets
        print(f"Loading training data from {data_dir}/train")
        train_dataset = GzippedLeelaDataset(
            data_dir, split="train", verbose=self.verbose
        )

        print(f"Loading validation data from {data_dir}/test")
        val_dataset = GzippedLeelaDataset(data_dir, split="test", verbose=self.verbose)

        # Check if we have data
        if len(train_dataset) == 0:
            print(
                "WARNING: No training positions found. Training will not be possible."
            )
        else:
            print(f"Successfully loaded {len(train_dataset)} training positions")

        if len(val_dataset) == 0:
            print(
                "WARNING: No validation positions found. Evaluation will not be possible."
            )
        else:
            print(f"Successfully loaded {len(val_dataset)} validation positions")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def create_data_loaders_with_random_split(self, data_dir, train_ratio=0.8, seed=42):
        """
        Create training and validation data loaders using PyTorch's random_split.

        This method loads all data from the data_dir and uses PyTorch's random_split
        to create training and validation datasets.

        Args:
            data_dir: Directory containing all .gz files
            train_ratio: Ratio of data to use for training (default: 0.8 = 80%)
            seed: Random seed for reproducible splits

        Returns:
            train_loader, val_loader: DataLoader objects for training and validation
        """
        print(f"Loading all data from directory: {data_dir}")
        print(f"Using PyTorch's random_split with train_ratio: {train_ratio}")

        # Load all data without splitting
        full_dataset = GzippedLeelaDataset(
            data_dir, split="", verbose=self.verbose, use_raw_dir=True
        )

        if len(full_dataset) == 0:
            print(f"ERROR: No valid positions found in {data_dir}")
            # Return dummy loaders with empty datasets
            dummy_dataset = torch.utils.data.TensorDataset(
                torch.empty(0, 64, 112),
                torch.empty(0, 64, 64),
                torch.empty(0, 3),
            )
            train_loader = DataLoader(
                dummy_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            val_loader = train_loader
            return train_loader, val_loader

        # Split the dataset
        total_size = len(full_dataset)
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size

        # Set the generator for reproducibility
        generator = torch.Generator().manual_seed(seed)

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        print(f"Successfully split {total_size} positions into:")
        print(f"  - Training: {train_size} positions ({train_ratio:.0%})")
        print(f"  - Validation: {val_size} positions ({1 - train_ratio:.0%})")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def create_data_loaders(self, data_dir, train_ratio=0.8, random_seed=42):
        """
        Create training and validation data loaders by loading from a directory
        and using PyTorch's random_split.

        Args:
            data_dir: Directory containing data files
            train_ratio: Ratio of data to use for training
            random_seed: Random seed for reproducible splits

        Returns:
            train_loader, val_loader: DataLoader objects for training and validation
        """
        logging.info(f"Loading data from directory: {data_dir}")
        logging.info(f"Using PyTorch's random_split with train_ratio: {train_ratio}")

        # Use the same function for consistency
        return self.create_data_loaders_with_random_split(
            data_dir, train_ratio, random_seed
        )

    def calculate_policy_loss(self, predicted_policy, target_policy):
        """
        Calculate policy loss.

        Args:
            predicted_policy: Tuple of (move_logits, promotion_logits) from model
            target_policy: Dict with 'policy' and 'promotion_policy' targets

        Returns:
            Policy loss tensor
        """
        move_logits, promotion_logits = predicted_policy

        # Regular move loss - KL divergence loss
        regular_policy_target = target_policy["policy"]

        # Check for NaN values in the target policy and replace with zeros
        if torch.isnan(regular_policy_target).any():
            regular_policy_target = torch.zeros_like(regular_policy_target)
            # Use a uniform policy where every possible move has equal probability
            regular_policy_target.fill_(1.0 / (64 * 64))

        # Get log softmax of move logits
        log_softmax_moves = F.log_softmax(
            move_logits.reshape(-1, 64 * 64), dim=-1
        ).reshape(-1, 64, 64)
        regular_policy_loss = -torch.sum(
            regular_policy_target * log_softmax_moves, dim=(1, 2)
        ).mean()

        # Promotion move loss
        promotion_policy_target = target_policy["promotion_policy"]

        # Check for NaN values in the promotion policy and replace with zeros
        if torch.isnan(promotion_policy_target).any():
            promotion_policy_target = torch.zeros_like(promotion_policy_target)
            # Use a uniform policy where every possible promotion has equal probability
            promotion_policy_target.fill_(1.0 / (8 * 8 * 4))

        log_softmax_promotion = F.log_softmax(
            promotion_logits.reshape(-1, 8 * 8 * 4), dim=-1
        ).reshape(-1, 8, 8, 4)
        promotion_loss = -torch.sum(
            promotion_policy_target * log_softmax_promotion, dim=(1, 2, 3)
        ).mean()

        # Combine losses - weight promotion loss less since it's less common
        policy_loss = regular_policy_loss + 0.1 * promotion_loss

        return policy_loss

    def calculate_value_loss(self, predicted_value, target_value):
        """
        Calculate value loss.

        Args:
            predicted_value: Output from value head
            target_value: Dict with value targets

        Returns:
            Value loss tensor
        """
        # WDL prediction loss (cross-entropy)
        wdl_target = target_value["wdl"]
        wdl_loss = F.cross_entropy(predicted_value, torch.argmax(wdl_target, dim=1))

        return wdl_loss

    def train_epoch(self, epoch, train_loader):
        """Train for one epoch."""
        self.model.train()
        start_time = time.time()
        total_loss = 0
        policy_losses = 0
        value_losses = 0

        for i, (inputs, policy_targets, value_targets) in enumerate(train_loader):
            # Move data to device
            inputs = inputs.to(self.device)
            policy_targets["policy"] = policy_targets["policy"].to(self.device)
            policy_targets["promotion_policy"] = policy_targets["promotion_policy"].to(
                self.device
            )
            for key in value_targets:
                value_targets[key] = value_targets[key].to(self.device)

            # Forward pass
            policy_logits, value_logits = self.model(inputs)

            # Calculate loss
            policy_loss = self.calculate_policy_loss(policy_logits, policy_targets)
            value_loss = self.calculate_value_loss(value_logits, value_targets)
            loss = policy_loss + value_loss

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            policy_losses += policy_loss.item()
            value_losses += value_loss.item()

            # Print progress
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch {epoch}, Batch {i + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Policy: {policy_loss.item():.4f}, "
                    f"Value: {value_loss.item():.4f}"
                )

        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_policy_loss = policy_losses / len(train_loader)
        avg_value_loss = value_losses / len(train_loader)

        # Log metrics
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        self.writer.add_scalar("Policy_Loss/train", avg_policy_loss, epoch)
        self.writer.add_scalar("Value_Loss/train", avg_value_loss, epoch)

        # Print epoch summary
        elapsed_time = time.time() - start_time
        print(
            f"Epoch {epoch} completed in {elapsed_time:.2f}s, "
            f"Loss: {avg_loss:.4f}, "
            f"Policy: {avg_policy_loss:.4f}, "
            f"Value: {avg_value_loss:.4f}"
        )

        return avg_loss

    def validate(self, epoch, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        policy_losses = 0
        value_losses = 0

        with torch.no_grad():
            for inputs, policy_targets, value_targets in val_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                policy_targets["policy"] = policy_targets["policy"].to(self.device)
                policy_targets["promotion_policy"] = policy_targets[
                    "promotion_policy"
                ].to(self.device)
                for key in value_targets:
                    value_targets[key] = value_targets[key].to(self.device)

                # Forward pass
                policy_logits, value_logits = self.model(inputs)

                # Calculate loss
                policy_loss = self.calculate_policy_loss(policy_logits, policy_targets)
                value_loss = self.calculate_value_loss(value_logits, value_targets)
                loss = policy_loss + value_loss

                # Update metrics
                total_loss += loss.item()
                policy_losses += policy_loss.item()
                value_losses += value_loss.item()

        # Calculate average losses
        avg_loss = total_loss / len(val_loader)
        avg_policy_loss = policy_losses / len(val_loader)
        avg_value_loss = value_losses / len(val_loader)

        # Log metrics
        self.writer.add_scalar("Loss/val", avg_loss, epoch)
        self.writer.add_scalar("Policy_Loss/val", avg_policy_loss, epoch)
        self.writer.add_scalar("Value_Loss/val", avg_value_loss, epoch)

        # Print validation summary
        print(
            f"Validation - Loss: {avg_loss:.4f}, "
            f"Policy: {avg_policy_loss:.4f}, "
            f"Value: {avg_value_loss:.4f}"
        )

        return avg_loss

    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")

        # Create checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": self.config,
        }

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")

    def train(self, train_loader, val_loader):
        """Train the model for the specified number of epochs."""
        print(f"Starting training for {self.num_epochs} epochs")

        # Store data loaders for convenience
        self.train_loader = train_loader
        self.val_loader = val_loader

        best_val_loss = float("inf")

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")

            # Train
            if train_loader is None:
                logging.info("Test-only mode, skipping training")
                train_loss = 0
            else:
                train_loss = self.train_epoch(epoch, train_loader)

            # Validate
            if val_loader is None:
                logging.info("Test-only mode, skipping validation")
                val_loss = 0
            else:
                val_loss = self.validate(epoch, val_loader)

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            if epoch % self.save_freq == 0 or is_best or epoch == self.num_epochs:
                self.save_checkpoint(epoch, val_loss, is_best)

        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        self.writer.close()


def test_model_architecture(trainer):
    """
    Test the model architecture with a small batch to verify input/output dimensions.

    Args:
        trainer: ChessformerTrainer instance
    """
    print("Testing model architecture...")
    trainer.model.eval()

    # Create a small test batch
    batch_size = 2
    inputs = torch.zeros((batch_size, 64, 112), dtype=torch.float32).to(trainer.device)

    # Forward pass
    with torch.no_grad():
        policy_output, value_output = trainer.model(inputs)

    # Unpack policy output - handle both tuple and non-tuple returns
    if isinstance(policy_output, tuple):
        move_logits, promotion_logits = policy_output
        print(
            f"Move logits shape: {move_logits.shape}, expected: (batch_size=2, 64, 64)"
        )
        print(
            f"Promotion logits shape: {promotion_logits.shape}, expected: (batch_size=2, 8, 8, 4)"
        )

        model_valid = (
            move_logits.shape == (batch_size, 64, 64)
            and promotion_logits.shape == (batch_size, 8, 8, 4)
            and value_output.shape == (batch_size, 3)
        )
    else:
        # Single policy output case
        print(f"Policy output shape: {policy_output.shape}")
        model_valid = policy_output.shape[0] == batch_size and value_output.shape == (
            batch_size,
            3,
        )

    print(f"Value output shape: {value_output.shape}, expected: (batch_size=2, 3)")
    print("Model architecture test completed")

    return model_valid


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train the Chessformer model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with Leela Chess Zero data",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help="Ratio of data to use for training (default: 0.8 or from config)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for train-test split (default: 42 or from config)",
    )
    parser.add_argument(
        "--use_raw_dir",
        action="store_true",
        help="Use data_dir/raw for training data instead of data_dir/train and data_dir/test",
    )
    parser.add_argument("--test_only", action="store_true", help="Only test model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract model and training hyperparameters
    model_params = config.get("model", {})
    training_params = config.get("training", {})
    dataset_params = config.get("dataset", {})

    # Get train_ratio and random_seed from config if not specified in command line
    if args.train_ratio is None:
        args.train_ratio = dataset_params.get("train_ratio", 0.8)
    if args.random_seed is None:
        args.random_seed = dataset_params.get("random_seed", 42)

    logging.info(
        f"Using train_ratio: {args.train_ratio}, random_seed: {args.random_seed}"
    )

    # Set up logging and create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )

    # Test model architecture if requested
    if args.test_only:
        logging.info("Testing model architecture...")
        embed_dim = model_params.get("embed_dim", 128)
        num_heads = model_params.get("num_heads", 8)
        ff_hidden_dim = model_params.get("ff_hidden_dim", 512)
        num_layers = model_params.get("num_layers", 4)
        d_value = model_params.get("d_value", 32)
        value_embedding_dim = model_params.get("value_embedding_dim", 64)
        dropout = model_params.get("dropout", 0.1)

        model = ChessformerModel(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            num_layers=num_layers,
            d_value=d_value,
            value_embedding_dim=value_embedding_dim,
            dropout=dropout,
        )

        # Test forward pass with random input
        board_tensor = torch.rand(2, 8, 8, 15)
        meta_tensor = torch.rand(2, 7)
        policy_logits, value_logits = model(board_tensor, meta_tensor)

        logging.info(f"Policy logits shape: {policy_logits.shape}")
        logging.info(f"Value logits shape: {value_logits.shape}")
        logging.info("Model test completed successfully.")
        return

    # Create trainer
    try:
        batch_size = training_params.get("batch_size", 64)
        learning_rate = training_params.get("learning_rate", 1e-4)
        weight_decay = training_params.get("weight_decay", 0.01)
        epochs = training_params.get("epochs", 10)
        save_frequency = training_params.get("save_frequency", 1)
        grad_clip = training_params.get("grad_clip", 1.0)
        loss_weights = training_params.get(
            "loss_weights", {"policy": 1.0, "value": 1.0}
        )

        # Create trainer with model parameters
        trainer = ChessformerTrainer(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

        # Create datasets and dataloaders
        if args.test_only:
            logging.info("Test-only mode, skipping data loading")
            train_loader, val_loader = None, None
        else:
            if args.use_raw_dir:
                # Use data directly from data_dir/raw
                raw_dir = os.path.join(args.data_dir, "raw")
                logging.info(f"Using raw directory for data: {raw_dir}")
                train_loader, val_loader = trainer.create_data_loaders(
                    raw_dir, args.train_ratio, args.random_seed
                )
            else:
                logging.info(f"Using standard data directory: {args.data_dir}")
                train_loader, val_loader = trainer.create_data_loaders(
                    args.data_dir, args.train_ratio, args.random_seed
                )

        # Train the model
        if args.test_only:
            logging.info("Test-only mode, skipping training")
            # Optionally add test/evaluation code here
        else:
            trainer.train(train_loader, val_loader)

    except Exception as e:
        logging.error(f"ERROR: Failed to initialize trainer: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
