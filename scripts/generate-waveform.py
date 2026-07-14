import librosa
import pandas as pd
import numpy as np

# --- SETTINGS ---
FILE_NAME = 'your_audio_file.wav'  # Change this to your filename
TARGET_POINTS = 300                # How many bars/points you want in Datawrapper

def generate_waveform():
    print(f"Reading {FILE_NAME}...")
    
    # Load audio (sr=None preserves original sampling rate)
    y, sr = librosa.load(FILE_NAME, sr=None)

    # Convert to absolute values (we want peaks, not raw vibrations)
    y_abs = np.abs(y)

    # Downsample the data so Datawrapper doesn't crash
    # We split the array into 'TARGET_POINTS' chunks and take the max of each
    chunks = np.array_split(y_abs, TARGET_POINTS)
    waveform_points = [np.max(chunk) for chunk in chunks]

    # Create symmetrical data: one positive, one negative for every point
    df = pd.DataFrame({
        'Time_Point': range(1, TARGET_POINTS + 1),
        'Top_Wave': waveform_points,
        'Bottom_Wave': [-x for x in waveform_points] # This mirrors the values
    })

    # Export to CSV
    output_file = 'datawrapper_waveform2.csv'
    df.to_csv(output_file, index=False)
    print(f"Success! Upload '{output_file}' to Datawrapper.")

if __name__ == "__main__":
    generate_waveform()
