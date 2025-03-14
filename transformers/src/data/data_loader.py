import glob
import gzip
import os
import struct

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

# Constants for V7B format
V7B_VERSION = 170

# Chess move encoding constants
NUM_SQUARES = 64
POLICY_VECTOR_SIZE = 1858  # Total number of possible moves in chess

# Define the struct format sizes for V7B
STRUCT_SIZES = {
    "version": 4,  # int (4 bytes)
    "input_format": 4,  # int (4 bytes)
    "probabilities": 7432,  # float array (1858 * 4 bytes)
    "planes": 832,  # 104 bytes (13 planes * 8 squares per rank * 8 ranks / 8 bits per byte)
    "castling_us_ooo": 1,  # byte
    "castling_us_oo": 1,  # byte
    "castling_them_ooo": 1,  # byte
    "castling_them_oo": 1,  # byte
    "side_to_move": 1,  # byte
    "rule50_count": 1,  # byte
    "invariance_info": 1,  # byte
    "result": 1,  # byte (deprecated)
    "root_q": 4,  # float
    "best_q": 4,  # float
    "root_d": 4,  # float
    "best_d": 4,  # float
    "root_m": 4,  # float
    "best_m": 4,  # float
    "plies_left": 4,  # float
    "result_q": 4,  # float
    "result_d": 4,  # float
    "played_q": 4,  # float
    "played_d": 4,  # float
    "played_m": 4,  # float
    "orig_q": 4,  # float
    "orig_d": 4,  # float
    "orig_m": 4,  # float
    "visits": 4,  # int
    "played_idx": 2,  # short
    "best_idx": 2,  # short
    "policy_kld": 4,  # float
    "q_st": 4,  # float
    "d_st": 4,  # float
    "opp_played_idx": 2,  # short
    "next_played_idx": 2,  # short
    "extra_floats": 32,  # 8 floats (8 * 4 bytes)
    "opp_probs": 7432,  # float array (1858 * 4 bytes)
    "next_probs": 7432,  # float array (1858 * 4 bytes)
    "future_boards": 12
    * 8
    * 16,  # 12 planes * 8 squares per rank * 16 future positions
}


# Move encoding mapping from V7B policy format to 64x64 format
# This is a simplified mapping - in reality, we need a more complete implementation
def create_move_mapping():
    """Create a mapping from 1858 policy indices to (from_square, to_square) format."""
    # This is a placeholder - a real implementation would map each of the 1858 moves
    # to the corresponding source and destination squares
    mapping = {}
    idx = 0

    # Regular moves (queen moves from any square to any square)
    for from_square in range(64):
        for to_square in range(64):
            if from_square != to_square:  # Skip non-moves
                mapping[idx] = (
                    from_square,
                    to_square,
                    None,
                )  # (from, to, promotion_piece)
                idx += 1

    # Promotion moves (from 7th rank to 8th rank with promotion piece)
    promotion_pieces = ["n", "b", "r", "q"]  # knight, bishop, rook, queen
    for from_square in range(48, 56):  # 7th rank squares
        for to_square in range(56, 64):  # 8th rank squares
            # Only consider diagonal and forward moves for pawns
            if (
                to_square == from_square - 8
                or to_square == from_square - 7
                or to_square == from_square - 9
            ):
                for i, piece in enumerate(promotion_pieces):
                    mapping[idx] = (from_square, to_square, i)
                    idx += 1

    # Create reverse mapping for lookup
    reverse_mapping = {}
    for policy_idx, move_tuple in mapping.items():
        from_square, to_square, promotion = move_tuple
        if promotion is None:
            key = (from_square, to_square)
            reverse_mapping[key] = policy_idx
        else:
            key = (from_square, to_square, promotion)
            reverse_mapping[key] = policy_idx

    return mapping, reverse_mapping


# Global move mappings
MOVE_MAPPING, REVERSE_MOVE_MAPPING = create_move_mapping()


class ChessDataLoader:
    def __init__(self, config_path):
        """Initialize the data loader with configuration from a YAML file."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Extract parameters from config
        self.history_length = min(
            self.config.get("model", {}).get("history_length", 8), 8
        )
        self.input_paths = self._get_input_paths()
        self.batch_size = self.config.get("training", {}).get("batch_size", 256)
        self.num_workers = (
            self.config.get("dataset", {}).get("preprocess", {}).get("num_workers", 8)
        )

    def _get_input_paths(self):
        """Get list of input file paths from config."""
        paths = []
        dataset_config = self.config.get("dataset", {})

        # Get training data paths
        train_paths = dataset_config.get("input_train", [])
        for path_pattern in train_paths:
            # Expand glob patterns
            if "*" in path_pattern:
                expanded_paths = glob.glob(path_pattern)
                paths.extend(expanded_paths)
            elif os.path.isdir(path_pattern):
                # If directory, get all .gz files
                paths.extend(glob.glob(os.path.join(path_pattern, "*.gz")))
            elif os.path.isfile(path_pattern):
                paths.append(path_pattern)

        return paths

    def create_dataloader(self):
        """Create a PyTorch DataLoader from the configuration."""
        dataset = ChessDataset(self.input_paths, self.history_length)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return loader


class ChessDataset(Dataset):
    def __init__(self, file_paths, history_length=8):
        """
        Initialize chess dataset.

        Args:
            file_paths: List of paths to V7B format data files
            history_length: Number of past positions to include (1-8)
        """
        self.file_paths = file_paths
        self.history_length = min(history_length, 8)  # Cap at 8 (max in V7B format)

        # Index the files and positions
        self.positions = []
        for file_path in file_paths:
            try:
                with gzip.open(file_path, "rb") as f:
                    data = f.read()
                    version = struct.unpack("i", data[0:4])[0]

                    if version != V7B_VERSION:
                        print(
                            f"Skipping file with unsupported version {version}: {file_path}"
                        )
                        continue

                    # Calculate record size based on known struct format
                    record_size = sum(STRUCT_SIZES.values())

                    # Index each position in the file
                    for offset in range(0, len(data), record_size):
                        if offset + record_size <= len(data):
                            self.positions.append((file_path, offset))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        print(f"Loaded {len(self.positions)} positions from {len(file_paths)} files")

    def __len__(self):
        """Return number of positions in the dataset."""
        return len(self.positions)

    def __getitem__(self, idx):
        """Get a chess position by index."""
        file_path, offset = self.positions[idx]

        with gzip.open(file_path, "rb") as f:
            f.seek(offset)
            record_data = f.read(sum(STRUCT_SIZES.values()))

        # Process the record into a tensor
        input_tensor, policy_target, value_target = self._process_record(record_data)

        return input_tensor, policy_target, value_target

    def _process_record(self, record_data):
        """
        Process a V7B record into the tensor format described in the paper.

        Returns:
            input_tensor: Tensor of shape [64, 112] with features for each square
            policy_target: Move probabilities tensor shaped for model comparison
            value_target: Value target tensor
        """
        # Extract the planes data from the record
        # Starting offset for planes is after version, input_format, and probabilities
        planes_offset = (
            STRUCT_SIZES["version"]
            + STRUCT_SIZES["input_format"]
            + STRUCT_SIZES["probabilities"]
        )
        planes_data = record_data[
            planes_offset : planes_offset + STRUCT_SIZES["planes"]
        ]

        # Extract castling rights
        castling_offset = planes_offset + STRUCT_SIZES["planes"]
        castling_us_ooo = record_data[castling_offset]
        castling_us_oo = record_data[castling_offset + 1]
        castling_them_ooo = record_data[castling_offset + 2]
        castling_them_oo = record_data[castling_offset + 3]

        # Extract side to move and en passant
        side_to_move = record_data[castling_offset + 4]

        # Extract rule50 count
        rule50_count = record_data[castling_offset + 5]

        # Extract invariance info (contains repetition flag)
        invariance_info = record_data[castling_offset + 6]

        # Extract results
        result_offset = castling_offset + 8  # Skip deprecated result byte
        result_q = struct.unpack(
            "f", record_data[result_offset + 32 : result_offset + 36]
        )[0]
        result_d = struct.unpack(
            "f", record_data[result_offset + 36 : result_offset + 40]
        )[0]

        # Extract policy data (probabilities)
        policy_offset = STRUCT_SIZES["version"] + STRUCT_SIZES["input_format"]
        policy_data = record_data[
            policy_offset : policy_offset + STRUCT_SIZES["probabilities"]
        ]
        policy_probs = np.frombuffer(policy_data, dtype=np.float32)

        # Convert planes to bit representation (13 planes, 8x8 board)
        planes = np.unpackbits(np.frombuffer(planes_data, dtype=np.uint8))
        planes = planes.reshape(-1, 8, 8)  # Reshape to (13, 8, 8)

        # Create 64 tokens (one per square)
        input_tensor = np.zeros((64, 112), dtype=np.float32)

        # Process each square
        for square_idx in range(64):
            row, col = square_idx // 8, square_idx % 8

            # 1. Extract piece representation for current and past positions (8 one-hot vectors of length 12)
            # In V7B, first 12 planes represent pieces in current position
            piece_features = np.zeros(
                96, dtype=np.float32
            )  # 8 positions * 12 piece types

            for hist_pos in range(min(8, self.history_length)):
                # For each position in history, get 12 planes (pieces)
                # In actual implementation, you'd need to properly extract the piece planes for each historical position
                hist_offset = (
                    hist_pos * 12
                )  # Each position has 12 planes (6 white, 6 black pieces)
                for p in range(12):
                    if hist_pos == 0:  # Current position
                        piece_features[hist_offset + p] = planes[p, row, col]
                    else:
                        # For past positions, we'd need to extract from the appropriate place in the V7B struct
                        # This is a simplified placeholder
                        piece_features[hist_offset + p] = 0

            # 2. Add en passant and castling information
            # Castling rights
            castling_features = np.array(
                [
                    castling_us_ooo,
                    castling_us_oo,
                    castling_them_ooo,
                    castling_them_oo,
                    # En passant square (simplified)
                    planes[12, row, col],  # Plane 12 is en passant plane
                ],
                dtype=np.float32,
            )

            # 3. Add rule50 count (normalized by dividing by 100)
            rule50_feature = np.array([rule50_count / 100.0], dtype=np.float32)

            # 4. Add repetition flags
            # Extract repetition bit from invariance_info for current position
            # In real implementation, you'd extract repetition flags for all 8 positions
            repetition_flags = np.zeros(8, dtype=np.float32)
            # Check if highest bit is set (repetition flag in invariance_info)
            if invariance_info & 0x80:
                repetition_flags[0] = 1.0

            # Combine all features for this square
            square_features = np.concatenate(
                [
                    piece_features[
                        : self.history_length * 12
                    ],  # Use only requested history length
                    castling_features,
                    rule50_feature,
                    repetition_flags[
                        : self.history_length
                    ],  # Use only requested history length
                ]
            )

            # Store in the input tensor
            input_tensor[square_idx] = square_features

        # Convert policy target from V7B format (1858 probs) to 64x64 format
        # Initialize with small epsilon to avoid log(0) in loss functions
        policy_target_matrix = np.ones((64, 64), dtype=np.float32) * 1e-10

        # Map each move probability to the correct location in the matrix
        for policy_idx in range(min(len(policy_probs), len(MOVE_MAPPING))):
            if policy_idx in MOVE_MAPPING:
                from_square, to_square, promotion = MOVE_MAPPING[policy_idx]
                if promotion is None:  # Regular move
                    policy_target_matrix[from_square, to_square] = policy_probs[
                        policy_idx
                    ]
                # Promotion moves would need special handling in the actual model

        # We can either keep it as a matrix or flatten it, depending on how the model expects it
        policy_target = policy_target_matrix.flatten()

        # Calculate value target (win/draw/loss)
        # Convert Q and D values to WDL format
        w = (1 + result_q - result_d) / 2
        d = result_d
        l = (1 - result_q - result_d) / 2
        value_target = np.array([w, d, l], dtype=np.float32)

        # Convert to tensors
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        policy_target = torch.tensor(policy_target, dtype=torch.float32)
        value_target = torch.tensor(value_target, dtype=torch.float32)

        return input_tensor, policy_target, value_target
