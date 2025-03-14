# Chessformer

A transformer-based neural network for chess, implementing the architecture described in the paper "Chessformer: Leveraging Transformers for Chess" (see `prev-codebase/pdfs/` for details).

## Project Structure

```
chessformer/
├── configs/              # Configuration files for different model sizes
├── data/                 # Data directory
│   ├── processed/        # Processed data ready for training
│   └── raw/              # Raw data files
├── models/               # Saved model checkpoints
│   ├── small/            # Small model (6M parameters)
│   ├── medium/           # Medium model (42M parameters)
│   └── test/             # Test models
├── scripts/              # Utility scripts
│   └── convert_leela_data.py  # Convert Leela Chess Zero format to Chessformer format
├── src/                  # Source code
│   ├── data/             # Data loading and processing
│   ├── model/            # Model architecture
│   ├── train.py          # Training script
│   └── evaluate_model.py # Evaluation script
└── setup.py              # Installation script
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chessformer.git
cd chessformer

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install the package in development mode
pip install -e .
```

## Data Conversion

The model is trained on data from Leela Chess Zero, which needs to be converted to the Chessformer format:

```bash
python scripts/convert_leela_data.py --config configs/config.yaml --output data/processed
```

## Training

Train the model using the following command:

```bash
python src/train.py --config configs/small_model_config.yaml --data_dir data/processed --output_dir models/small/run1
```

For a quick test with a small dataset:

```bash
python src/train.py --config configs/test_config.yaml --data_dir data/processed --output_dir models/test/test_run --test_only
```

## Evaluation

Evaluate a trained model on test data:

```bash
python src/evaluate_model.py --model_path models/small/run1/best_model.pt --data_dir data/processed/test --batch_size 16
```

## Model Configurations

- **Small Model**: ~6M parameters (64 embedding dimension, 4 attention heads, 2 layers)
- **Medium Model**: ~42M parameters (128 embedding dimension, 8 attention heads, 4 layers)
- **Large Model**: ~240M parameters (1024 embedding dimension, 32 attention heads, 15 layers)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project implements the Chessformer architecture described in the paper "Chessformer: Leveraging Transformers for Chess" and builds upon the Leela Chess Zero project for training data.
