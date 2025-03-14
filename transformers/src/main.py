def main():
    """
    Main entry point for the Chessformer application.
    This function delegates to train.py's main function for training.
    """
    # Fix the import path issue
    if __name__ == "__main__":
        # If run directly, use relative import
        from train import main as train_main
    else:
        # If imported as a module, use package import
        from src.train import main as train_main

    # Call the training main function
    train_main()


if __name__ == "__main__":
    main()
