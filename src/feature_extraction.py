# NOTE: Utilizing Librosa for audio feature extraction and analysis
import librosa
import numpy as np

import shutil
from pathlib import Path


# Getting project root directory path (parents is zero-indexed)
ROOT_PATH = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT_PATH / 'data/raw/genres_original'

# NOTE: ------------------------------- SETTING UP DATA ENVIRONMENT -------------------------------#

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 512

# NOTE: ------------------------------- FEATURE PREPROCESSING LOOP --------------------------------#
for folder in RAW_DATA_PATH.iterdir():
    if folder.is_dir():
        print('Preprocessing ' + str(folder.name))
        wave_files = []

        for wav_file in folder.iterdir():
            wave_files.append(wav_file.name)

            y, sr = librosa.load(wav_file)

            # time-series harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)

            tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)


            # Compute MFCC features from the raw signal
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

            mfcc_delta = librosa.feature.delta(mfcc)    # first-order differences (delta features)

            # Stack and synchronize between beat events
            beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)


            # Compute chroma features from the harmonic signal
            chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

            # Aggregate chroma features between beat events
            beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)


            # Vertically stack all features resulting in a feature matrix
            beat_features = np.vstack([beat_chroma, beat_mfcc_delta])


            # Let's make and display a mel-scaled power (energy-squared) spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)


        print(f'{wave_files}\n')
