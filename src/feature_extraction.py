# NOTE: Utilizing Librosa for audio feature extraction and analysis
import librosa
import numpy as np

import os
import shutil
from pathlib import Path


# Getting project root directory path (parents is zero-indexed)
ROOT_PATH = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT_PATH / 'data/raw/genres_original'

for folder in os.listdir(RAW_DATA_PATH):
    print(folder)
