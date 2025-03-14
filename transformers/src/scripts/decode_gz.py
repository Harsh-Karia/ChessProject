#!/usr/bin/env python3

import gzip
import math
import os
import struct
import sys

# Constants for V7B format based on what we found in chunkparser.py and data_loader.py
V7B_VERSION = 170
V7_VERSION = 7
V6_VERSION = 6
V5_VERSION = 5
V4_VERSION = 4
V3_VERSION = 3

# Define the struct formats for different versions
V7B_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHHfffHHffffffff7432s7432s1536s"
V7_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHHfffHHffffffff"
V6_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHHff"
V5_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffff"
V4_STRUCT_STRING = "4s7432s832sBBBBBBBbffff"
V3_STRUCT_STRING = "4s7432s832sBBBBBBBb"


def create_leela_move_map():
    """Create a mapping for Leela Chess Zero's move encoding."""
    # This is a simplified implementation based on the MOVES array in decode_training.py
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

                    # Check if the move is valid (no need to validate chess rules here)
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


# Create the struct formats
v7b_struct = struct.Struct(V7B_STRUCT_STRING)
v7_struct = struct.Struct(V7_STRUCT_STRING)
v6_struct = struct.Struct(V6_STRUCT_STRING)
v5_struct = struct.Struct(V5_STRUCT_STRING)
v4_struct = struct.Struct(V4_STRUCT_STRING)
v3_struct = struct.Struct(V3_STRUCT_STRING)

struct_sizes = {
    V7B_VERSION: v7b_struct.size,
    V7_VERSION: v7_struct.size,
    V6_VERSION: v6_struct.size,
    V5_VERSION: v5_struct.size,
    V4_VERSION: v4_struct.size,
    V3_VERSION: v3_struct.size,
}

# Create a mapping of move indices to algebraic notation
MOVE_MAP = create_leela_move_map()


def analyze_gz_file(file_path, max_records=3):
    """Analyze a .gz file to determine its format and content"""
    print(f"Analyzing file: {file_path}")

    try:
        with gzip.open(file_path, "rb") as f:
            # Read the whole file content
            data = f.read()

            # Check the version number (first 4 bytes)
            version_bytes = data[0:4]
            version = struct.unpack("i", version_bytes)[0]

            print(f"Version bytes: {version_bytes.hex()}")
            print(f"Version number: {version}")

            if version in struct_sizes:
                print(f"Recognized format: V{version}")
                record_size = struct_sizes[version]
                print(f"Record size: {record_size} bytes")

                # Count the number of records
                total_records = len(data) // record_size
                print(f"Total records in file: {total_records}")

                # Analyze a few records
                for i in range(min(max_records, total_records)):
                    record_start = i * record_size
                    record_data = data[record_start : record_start + record_size]
                    print(f"\nRecord {i + 1}/{total_records}:")

                    if version == V7B_VERSION:
                        analyze_v7b_record(record_data)
                    elif version == V7_VERSION:
                        analyze_v7_record(record_data)
                    elif version == V6_VERSION:
                        analyze_v6_record(record_data)
                    elif version == V5_VERSION:
                        analyze_v5_record(record_data)
                    elif version == V4_VERSION:
                        analyze_v4_record(record_data)
                    elif version == V3_VERSION:
                        analyze_v3_record(record_data)
            else:
                print(f"Unknown format version: {version}")
                # Try to detect the format based on record size
                for v, size in struct_sizes.items():
                    if len(data) % size == 0:
                        print(
                            f"File size ({len(data)} bytes) is divisible by V{v} record size ({size} bytes)"
                        )
    except Exception as e:
        print(f"Error analyzing file: {e}")


def analyze_v7b_record(record_data):
    """Analyze a V7B format record"""
    try:
        # Unpack the record using the V7B struct format
        unpacked = v7b_struct.unpack(record_data)

        version = struct.unpack("i", unpacked[0])[0]
        input_format = unpacked[1]

        # Probabilities (policy)
        probs_data = unpacked[2]
        policy_size = len(probs_data) // 4  # 4 bytes per float
        policy_probs = struct.unpack(f"{policy_size}f", probs_data)

        # Just print a few values for analysis
        print(f"  Policy distribution (first 5 values): {policy_probs[:5]}")
        print(f"  Policy distribution (sum): {sum(policy_probs)}")

        # Board planes data
        planes_data = unpacked[3]
        print(f"  Planes data size: {len(planes_data)} bytes")

        # Castling rights
        castling_us_ooo = unpacked[4]
        castling_us_oo = unpacked[5]
        castling_them_ooo = unpacked[6]
        castling_them_oo = unpacked[7]
        print(
            f"  Castling rights: us_OOO={castling_us_ooo}, us_OO={castling_us_oo}, them_OOO={castling_them_ooo}, them_OO={castling_them_oo}"
        )

        # Side to move and other metadata
        side_to_move = unpacked[8]
        rule50_count = unpacked[9]
        invariance_info = unpacked[10]
        result = unpacked[11]
        print(f"  Side to move: {side_to_move} (0=white, 1=black)")
        print(f"  Rule50 count: {rule50_count}")

        # Various Q values (evaluation scores)
        root_q = unpacked[12]
        best_q = unpacked[13]
        root_d = unpacked[14]
        best_d = unpacked[15]
        root_m = unpacked[16]
        best_m = unpacked[17]
        print(
            f"  Evaluations: root_q={root_q:.4f}, best_q={best_q:.4f}, root_d={root_d:.4f}, best_d={best_d:.4f}"
        )

        # Additional features
        visits = unpacked[31]
        played_idx = unpacked[32]
        best_idx = unpacked[33]
        print(f"  Visits: {visits}")
        print(f"  Played move index: {played_idx}, Best move index: {best_idx}")

        # Opponent and next move probabilities
        opp_probs_data = unpacked[45]
        next_probs_data = unpacked[46]
        print(f"  Opponent probs data size: {len(opp_probs_data)} bytes")
        print(f"  Next move probs data size: {len(next_probs_data)} bytes")

        # Future board states
        future_boards_data = unpacked[47]
        print(f"  Future boards data size: {len(future_boards_data)} bytes")

    except Exception as e:
        print(f"Error analyzing V7B record: {e}")


def analyze_v7_record(record_data):
    """Analyze a V7 format record"""
    try:
        # Unpack the record using the V7 struct format
        unpacked = v7_struct.unpack(record_data)

        version = struct.unpack("i", unpacked[0])[0]
        input_format = unpacked[1]

        print(f"  Version: {version}, Input format: {input_format}")
        # Other fields would be similar to V7B but without the future data
    except Exception as e:
        print(f"Error analyzing V7 record: {e}")


def analyze_v6_record(record_data):
    """Analyze a V6 format record"""
    try:
        # Unpack the record using the V6 struct format
        unpacked = v6_struct.unpack(record_data)

        version = struct.unpack("i", unpacked[0])[0]
        print(f"  Version: {version}")
        # Process other fields
    except Exception as e:
        print(f"Error analyzing V6 record: {e}")


def analyze_v5_record(record_data):
    """Analyze a V5 format record"""
    try:
        # Unpack the record using the V5 struct format
        unpacked = v5_struct.unpack(record_data)

        version = struct.unpack("i", unpacked[0])[0]
        print(f"  Version: {version}")
        # Process other fields
    except Exception as e:
        print(f"Error analyzing V5 record: {e}")


def analyze_v4_record(record_data):
    """Analyze a V4 format record"""
    try:
        # Unpack the record using the V4 struct format
        unpacked = v4_struct.unpack(record_data)

        # Version is the first 4 bytes but unpacked[0] is a bytes object
        version_bytes = unpacked[0]

        # Probabilities (policy)
        probs_data = unpacked[1]
        policy_size = len(probs_data) // 4  # 4 bytes per float
        policy_probs = struct.unpack(f"{policy_size}f", probs_data)

        # Filter out NaN values and get valid policy probabilities
        valid_probs = [
            (i, p) for i, p in enumerate(policy_probs) if not math.isnan(p) and p > 0
        ]
        non_zero_probs = len(valid_probs)

        # Calculate the sum of valid probabilities
        sum_probs = sum(p for _, p in valid_probs)

        print(f"  Policy distribution size: {policy_size} elements")
        print(f"  Non-zero policy probabilities: {non_zero_probs}")

        if valid_probs:
            print(
                f"  Policy distribution (first 5 non-zero values): {[(i, p) for i, p in valid_probs[:5]]}"
            )
            print(f"  Policy distribution (sum of valid probs): {sum_probs}")
        else:
            print("  No valid policy probabilities found")

        # Board planes data - this is the board state representation
        planes_data = unpacked[2]
        print(f"  Planes data size: {len(planes_data)} bytes")

        # Try to interpret the board planes
        print(f"  First 16 bytes of planes data: {planes_data[:16].hex()}")

        # Castling rights
        castling_us_ooo = unpacked[3]
        castling_us_oo = unpacked[4]
        castling_them_ooo = unpacked[5]
        castling_them_oo = unpacked[6]
        print(
            f"  Castling rights: us_OOO={castling_us_ooo}, us_OO={castling_us_oo}, them_OOO={castling_them_ooo}, them_OO={castling_them_oo}"
        )

        # Side to move and other metadata
        side_to_move = unpacked[7]
        rule50_count = unpacked[8]
        invariance_info = unpacked[9]
        deprecated_result = unpacked[10]
        print(f"  Side to move: {side_to_move} (0=white, 1=black)")
        print(f"  Rule50 count: {rule50_count}")
        print(f"  Invariance info: {invariance_info}")

        # Q and D values (win/draw/loss predictions)
        root_q = unpacked[11]
        best_q = unpacked[12]
        root_d = unpacked[13]
        best_d = unpacked[14]

        # Convert to win/draw/loss percentages
        def q_to_win_prob(q):
            return (q + 1) / 2

        def qd_to_wdl(q, d):
            w = (q + 1) * (1 - d) / 2
            l = (1 - q) * (1 - d) / 2
            return w, d, l

        root_win, root_draw, root_loss = qd_to_wdl(root_q, root_d)
        best_win, best_draw, best_loss = qd_to_wdl(best_q, best_d)

        print(f"  Root values (Q,D): ({root_q:.4f}, {root_d:.4f})")
        print(
            f"  Root WDL: Win={root_win:.1%}, Draw={root_draw:.1%}, Loss={root_loss:.1%}"
        )
        print(f"  Best values (Q,D): ({best_q:.4f}, {best_d:.4f})")
        print(
            f"  Best WDL: Win={best_win:.1%}, Draw={best_draw:.1%}, Loss={best_loss:.1%}"
        )

        # Look at top policy moves and if possible translate to algebraic notation
        if valid_probs:
            # Sort by probability
            sorted_valid_probs = sorted(valid_probs, key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in sorted_valid_probs[:5]]
            top_probs = [prob for _, prob in sorted_valid_probs[:5]]

            print(f"  Top 5 policy indices: {top_indices}")
            print(f"  Top 5 policy probabilities: {top_probs}")

            # Map indices to moves if we have the move mapping
            if len(MOVE_MAP) > 0:
                print("  Top policy moves:")
                for i, (idx, prob) in enumerate(sorted_valid_probs[:5]):
                    if idx < len(MOVE_MAP):
                        print(f"    {i + 1}. Index {idx}: {MOVE_MAP[idx]} ({prob:.4f})")
                    else:
                        print(f"    {i + 1}. Index {idx}: Unknown move ({prob:.4f})")

            # If there's a played_idx field in the record, try to extract it
            played_idx = (
                -1
            )  # This would need to be extracted from the record if available
            if played_idx >= 0 and played_idx < len(MOVE_MAP):
                print(f"  Played move: {MOVE_MAP[played_idx]}")

    except Exception as e:
        print(f"Error analyzing V4 record: {e}")
        import traceback

        traceback.print_exc()


def analyze_v3_record(record_data):
    """Analyze a V3 format record"""
    try:
        # Unpack the record using the V3 struct format
        unpacked = v3_struct.unpack(record_data)

        version = struct.unpack("i", unpacked[0])[0]
        print(f"  Version: {version}")
        # Process other fields
    except Exception as e:
        print(f"Error analyzing V3 record: {e}")


def main():
    # Get all .gz files in the training-run3 directory
    training_dir = "training-run3-20190614-2318"

    if len(sys.argv) > 1:
        # If a specific file is provided, analyze just that file
        analyze_gz_file(sys.argv[1])
    else:
        # List all .gz files in the directory
        gz_files = [
            os.path.join(training_dir, f)
            for f in os.listdir(training_dir)
            if f.endswith(".gz")
        ]

        if not gz_files:
            print(f"No .gz files found in {training_dir}")
            return

        # Analyze the first file
        analyze_gz_file(gz_files[0])


if __name__ == "__main__":
    main()
