import cv2
import numpy as np
import glob
import os
import h5py
import time

time1 = time.time()

def overlay_events(frame, events_path, bin_index):
    with h5py.File(events_path, 'r') as hdf:
        positive_events = hdf['data'][bin_index, :10, :, :]
        negative_events = hdf['data'][bin_index, 10:, :, :]

        positive_events_sum = np.sum(positive_events, axis=0)
        negative_events_sum = np.sum(negative_events, axis=0)

        # Resize event data to match the padded frame dimensions
        positive_events_sum_resized = cv2.resize(positive_events_sum.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        negative_events_sum_resized = cv2.resize(negative_events_sum.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        frame[positive_events_sum_resized > 0] = [255, 255, 255]
        frame[negative_events_sum_resized > 0] = [0, 0, 0]

    return frame


def process_frames(frame_dir, bbox_data, events_path, detection_path, output_video_path, use_rgb=True, use_events=True, use_detections=True, draw_label_bbox=True, confidence_threshold=0.8):
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')), key=lambda x: int(os.path.basename(x).split('_')[0]))
    detections = np.load(detection_path)
    video_out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (304, 240))

    for bin_index, frame_path in enumerate(frame_paths):
        if use_rgb:
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Failed to load frame: {frame_path}")
                continue
        else:
            frame = np.full((240, 304, 3), 127, dtype=np.uint8)  # Gray background if RGB is not used

        if use_events:
            frame_padded = cv2.copyMakeBorder(frame, 0, 0, 1, 1, cv2.BORDER_CONSTANT, value=[127, 127, 127])
            frame = overlay_events(frame_padded, events_path, bin_index)

        time_bin = bin_index * 50000

        if draw_label_bbox:
            current_bboxes = [bbox for bbox in bbox_data if bbox[0] == time_bin-1]
            for bbox in current_bboxes:
                _, x_min, y_min, width, height, class_id, _1, _2 = bbox
                x_max, y_max = x_min + width, y_min + height
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        if use_detections:
            for detection in detections:
                timestamp, x, y, w, h, label_value, _, confidence = detection
                if time_bin < timestamp <= time_bin + 50000 and confidence >= confidence_threshold:
                    x_max, y_max = int(x + w), int(y + h)
                    cv2.rectangle(frame, (int(x), int(y)), (x_max, y_max), (0, 0, 255), 2)
                    label_text = f'{label_value} {confidence:.2f}'
                    cv2.putText(frame, label_text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        frame = cv2.resize(frame, (304, 240))
        video_out.write(frame)

    video_out.release()
# Define paths and parameters
sequence_name = 'zurich_city_01_e'


base = r"C:\Users\lkxv3\OneDrive - National University of Singapore\DSEC\DSEC_MOD\training"
base_dir = os.path.join(base, sequence_name)
frame_dir = os.path.join(base_dir, 'train')
processed_dir = r'C:\Users\lkxv3\OneDrive\Desktop\CP5105\RVT\gen1\test'
seq =  os.path.join(processed_dir, sequence_name)
bbox_data = np.load(os.path.join(seq, 'labels_v2', 'labels.npz'))['labels']
events_path = os.path.join(seq, 'event_representations_v2', 'stacked_histogram_dt=50_nbins=10', 'event_representations.h5')


# detection_path = r"C:\Users\lkxv3\OneDrive\Desktop\CP5105\RVT\allpredictions.pkl"
detection_path = r"C:\Users\lkxv3\OneDrive\Desktop\CP5105\RVT\detections.npy"

output_video_path = 'all_01e.mp4'

use_rgb=True
use_events=True
use_detections=True
draw_label_bbox=True

# Process the frames
process_frames(frame_dir, bbox_data, events_path, detection_path, output_video_path, use_rgb, use_events, use_detections, draw_label_bbox, confidence_threshold=0.8)

time2 = time.time()
print('Time taken:', time2-time1)
