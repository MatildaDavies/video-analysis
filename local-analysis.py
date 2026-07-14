import cv2
import numpy as np
import pandas as pd
import os

def analyze_local_videos(folder_name="videos"):
    summary_data = []
    desktop_path = os.path.expanduser("~/Documents/soundwaves/visuals")
    folder_path = os.path.join(desktop_path, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_name}' not found on Desktop.")
        return

    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
    
    print(f"📂 Found {len(video_files)} videos. Starting analysis...")

    for filename in video_files:
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)
        
        # FIX 1: Verify the video actually opened
        if not cap.isOpened():
            print(f"OpenCV could not open {filename}. It might be an unsupported format.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30 
        
        brightness_vals, saturation_vals, edge_vals, motion_vals = [], [], [], []
        flicker_count, prev_gray, prev_brightness, frame_idx = 0, None, None, 0

        print(f"--- Analyzing: {filename} ---")

        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            # Analyze roughly 2 times per second
            if frame_idx % int(max(1, fps // 2)) == 0:
                try:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Brightness & Saturation
                    current_brightness = np.mean(hsv[:,:,2])
                    brightness_vals.append(current_brightness)
                    saturation_vals.append(np.mean(hsv[:,:,1]))
                    
                    # Flicker Detection
                    if prev_brightness is not None:
                        if abs(current_brightness - prev_brightness) > 25.5:
                            flicker_count += 1
                    prev_brightness = current_brightness
                    
                    # Edge Density (Detail Level)
                    edges = cv2.Canny(gray, 100, 200)
                    edge_vals.append(np.sum(edges > 0) / (frame.shape[0] * frame.shape[1]))
                    
                    # Motion
                    if prev_gray is not None:
                        motion_vals.append(np.mean(cv2.absdiff(gray, prev_gray)))
                    prev_gray = gray
                except Exception as e:
                    print(f"  Error processing frame {frame_idx}: {e}")

            frame_idx += 1
            if frame_idx % 2000 == 0:
                print(f"  ... processed {frame_idx} frames")

        cap.release()

        # FIX 2: Ensure we actually have data before calculating averages
        if len(brightness_vals) > 0:
            duration_sec = frame_idx / fps
            num_cuts = len([m for m in motion_vals if m > 30])
            asl = duration_sec / (num_cuts + 1)

            summary_data.append({
                'Filename': filename,
                'Avg Brightness': round(np.mean(brightness_vals), 2),
                'Avg Saturation': round(np.mean(saturation_vals), 2),
                'Edge Density %': round(np.mean(edge_vals) * 100, 3),
                'Motion Intensity': round(np.mean(motion_vals), 2),
                'Flicker Events': flicker_count,
                'Flicker Rate (per min)': round(flicker_count / (max(0.1, duration_sec) / 60), 2),
                'Avg Shot Length (sec)': round(asl, 2)
            })
            print(f"✅ Finished: {filename} ({len(brightness_vals)} data points)")
        else:
            print(f"❌ Failed to extract any data from {filename}.")

    if summary_data:
        df = pd.DataFrame(summary_data)
        output_file = os.path.join(desktop_path, "final_research_results.csv")
        df.to_csv(output_file, index=False)
        print(f"\n🚀 ALL DONE! Results saved to Desktop.")
    else:
        print("\n❌ No data was collected. Check your video file formats.")

if __name__ == "__main__":
    analyze_local_videos()
    
