import os
import sys
sys.path.append(os.path.abspath('..'))  # Adds the parent directory to sys.path

import logging
from src.model import training

# Set up logging
logging.basicConfig(filename='../logs/pipeline.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # load data from Excel

    # train regression model
    logging.info("Training the model...")
    training()


if __name__ == "__main__":  # main called only in this file, not through import
    main()