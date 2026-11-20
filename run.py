import argparse
import json
import io
import os
import statistics
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# Exact class mapping required by the brief (COCO dataset)
CLASSES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}

def process_video(model, video_bytes, video_id):
    """Processes a single video's bytes and returns the results."""
    results = []
    
    # 1. Read video from memory using OpenCV
    # Convert bytes to a numpy array for OpenCV
    import numpy as np
    temp_file = f"temp_{video_id.replace(':', '_')}.mp4"
    with open(temp_file, "wb") as f:
        f.write(video_bytes)
        
    cap = cv2.VideoCapture(temp_file)
    if not cap.isOpened():
        if os.path.exists(temp_file): os.remove(temp_file)
        return None, "could not decode video"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0 # Fallback if FPS is broken
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Dictionary to hold frame detections per second: {second: [frame_dict, frame_dict]}
    per_second_data = {}
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate which second this frame belongs to
        current_second = int(frame_idx / fps)
        
        # Run YOLO model (confidence >= 0.4, imgsz=640)
        result = model.predict(frame, conf=0.4, imgsz=640, verbose=False, classes=list(CLASSES.keys()))[0]
        
        # Count objects in this frame
        frame_counts = {name: 0 for name in CLASSES.values()}
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in CLASSES:
                frame_counts[CLASSES[cls_id]] += 1
                
        if current_second not in per_second_data:
            per_second_data[current_second] = []
        per_second_data[current_second].append(frame_counts)
        
        frame_idx += 1

    cap.release()
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # 2. Apply the exact math logic from the PDF (Section 5)
    for sec in sorted(per_second_data.keys()):
        frames_in_sec = per_second_data[sec]
        total_frames_in_sec = len(frames_in_sec)
        
        # Objects present in >= 50% of frames
        present_objects = []
        for obj_name in CLASSES.values():
            count_with_obj = sum(1 for f in frames_in_sec if f[obj_name] > 0)
            if count_with_obj >= (total_frames_in_sec / 2.0):
                present_objects.append(obj_name)
                
        # Median of people count (rounded down for .5)
        people_counts = [f["person"] for f in frames_in_sec]
        median_people = statistics.median(people_counts)
        final_people_count = int(median_people) # int() automatically rounds down for positive numbers

        results.append({
            "video": video_id,
            "second": sec,
            "objects": present_objects,
            "people": final_people_count
        })
        
    return results, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder with .parquet files")
    parser.add_argument("--output", required=True, help="Output results.jsonl path")
    args = parser.parse_args()

    # Load YOLO11L model (per brief instructions)
    print("Loading YOLO11L model...")
    model = YOLO("yolo11l.pt")

    input_folder = Path(args.input)
    parquet_files = sorted(input_folder.glob("*.parquet"))
    
    print(f"Found {len(parquet_files)} Parquet files.")
    
    with open(args.output, "w") as out_f:
        for pq_file in parquet_files:
            print(f"Processing {pq_file.name}...")
            
            try:
                # Read Parquet file
                df = pd.read_parquet(pq_file)
            except Exception as e:
                print(f"Failed to read {pq_file.name}: {e}")
                continue

            # Iterate through each row (video)
            for row_idx, row in tqdm(df.iterrows(), total=len(df)):
                video_id = f"{pq_file.name}:{row_idx}"
                video_bytes = row['mp4']
                
                # The 'mp4' column might be a dict containing 'bytes'
                if isinstance(video_bytes, dict):
                    video_bytes = video_bytes.get('bytes', b'')
                
                if not video_bytes:
                    out_f.write(json.dumps({"video": video_id, "error": "empty video bytes"}) + "\n")
                    continue

                try:
                    video_results, error = process_video(model, video_bytes, video_id)
                    
                    if error:
                        out_f.write(json.dumps({"video": video_id, "error": error}) + "\n")
                    else:
                        for res in video_results:
                            out_f.write(json.dumps(res) + "\n")
                            
                except Exception as e:
                    # Catch any unexpected crash and write error line
                    out_f.write(json.dumps({"video": video_id, "error": str(e)}) + "\n")

    print(f"✅ Processing complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
