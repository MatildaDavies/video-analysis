import cv2
import numpy as np
import pandas as pd
import os

from pathlib import Path

def generate_consolidated_motion_series(folder_name="videos"):
    # Path(__file__).resolve().parent gets the directory of this script
    folder_path = Path(__file__).resolve().parent / folder_name
    
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_name}' not found on Desktop.")
        return

    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
    
    if not video_files:
        print(f"❌ No .mp4 files found.")
        return

    # This dictionary will store our data: { 'filename': [list of motion values] }
    master_dict = {}
    max_length = 0

    print(f"📂 Found {len(video_files)} videos.")

    for filename in video_files:
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"⚠️ Warning: Could not open {filename}. Skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30 
        
        motion_per_second = []
        current_second_motions = []
        prev_gray = None
        frame_idx = 0

        print(f"--- Processing: {filename} ---")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                current_second_motions.append(np.mean(diff))
            
            prev_gray = gray
            frame_idx += 1

            # At the end of every second
            if frame_idx % int(fps) == 0:
                avg_motion = np.mean(current_second_motions) if current_second_motions else 0
                motion_per_second.append(round(avg_motion, 3))
                current_second_motions = []

        cap.release()
        
        # Store the list of seconds for this video
        master_dict[filename] = motion_per_second
        max_length = max(max_length, len(motion_per_second))
        print(f"✅ Finished {filename}")

    # Create the DataFrame
    # First, create the 'Second' column
    df_final = pd.DataFrame({'Second': range(1, max_length + 1)})

    # Add each video as a new column
    for filename, motion_list in master_dict.items():
        # If a video is shorter than the max_length, pandas automatically pads it with 'NaN'
        df_final[filename] = pd.Series(motion_list)

    # Save to Desktop
    output_path = os.path.join(desktop_path, "MASTER_MOTION_COMPARISON.csv")
    df_final.to_csv(output_path, index=False)
    print(f"Master file ready. Saved to Desktop as: MASTER_MOTION_COMPARISON.csv")

if __name__ == "__main__":
    generate_consolidated_motion_series()
