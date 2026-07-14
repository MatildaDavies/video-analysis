import cv2
import numpy as np
import pandas as pd
import os

def generate_consolidated_motion_series(folder_name="videos"):
    # Path(__file__).resolve().parent gets the directory of this script
    folder_path = Path(__file__).resolve().parent / folder_name
    
    return folder_path
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_name}' not found on Desktop.")
        return

    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
    
    if not video_files:
        print(f"❌ No .mp4 files found.")
        return

    master_dict = {}
    max_length = 0

    print(f"📂 Found {len(video_files)} videos. Analyzing Color Saturation...")

    for filename in video_files:
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"⚠️ Warning: Could not open {filename}. Skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30 
        
        sat_per_second = []
        current_second_sats = []
        frame_idx = 0

        print(f"--- Processing: {filename} ---")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to HSV (Hue, Saturation, Value)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Pull the Saturation channel (index 1)
            avg_sat_frame = np.mean(hsv[:,:,1])
            current_second_sats.append(avg_sat_frame)
            
            frame_idx += 1

            # At the end of every second
            if frame_idx % int(fps) == 0:
                avg_sat_sec = np.mean(current_second_sats) if current_second_sats else 0
                sat_per_second.append(round(avg_sat_sec, 3))
                current_second_sats = []

        cap.release()
        
        master_dict[filename] = sat_per_second
        max_length = max(max_length, len(sat_per_second))
        print(f"✅ Finished {filename}")

    # Create the DataFrame
    df_final = pd.DataFrame({'Second': range(1, max_length + 1)})

    for filename, sat_list in master_dict.items():
        df_final[filename] = pd.Series(sat_list)

    # Save to Desktop
    output_path = os.path.join(desktop_path, "MASTER_SATURATION_COMPARISON.csv")
    df_final.to_csv(output_path, index=False)
    print(f"\n🚀 DONE! Master Saturation file saved to Desktop.")

if __name__ == "__main__":
    generate_consolidated_saturation_series()
