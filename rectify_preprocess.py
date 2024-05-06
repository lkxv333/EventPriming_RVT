import h5py
import numpy as np
import time
import os

time1 = time.time()

'''
    Data prerpocessing with cropping in the middle of the image.
    The original image size is 640x480, and the cropped image size is 304x240.
    The bounding boxes are adjusted accordingly.
    The events are also adjusted accordingly.
'''

def create_label_timestamps(all_label, output_path):
    # sorted list of all label timestamps
    all_ts = np.sort(np.unique(all_label['t'] +1))

    labels_ts_path = os.path.join(output_path, 'labels_v2')  # New directory for label files
    if not os.path.exists(labels_ts_path):
        os.makedirs(labels_ts_path)

    output = os.path.join(labels_ts_path, 'timestamps_us.npy')
    np.save(output, all_ts)

    return all_ts


def create_objframe_2_repridx(all_ts, output_path):
    # Save the array to a .npy file
    objframe = all_ts//50000 -1# Convert timestamps to time bin indices

    event_repr_path = os.path.join(output_path, 'event_representations_v2', 'stacked_histogram_dt=50_nbins=10')  # New directory
    if not os.path.exists(event_repr_path):
        os.makedirs(event_repr_path)

    output = os.path.join(event_repr_path, 'objframe_idx_2_repr_idx.npy')
    np.save(output, objframe)


def create_time_bin_intervals(num_time_bins, output_path, time_window=50000):

    start_times = np.arange(0, num_time_bins * time_window, time_window)
    end_times = start_times + time_window
    time_bin_intervals = np.column_stack((start_times, end_times))

    event_repr_path = os.path.join(output_path, 'event_representations_v2', 'stacked_histogram_dt=50_nbins=10')  # New directory
    if not os.path.exists(event_repr_path):
        os.makedirs(event_repr_path)

    output = os.path.join(event_repr_path, 'timestamps_us.npy')
    np.save(output, time_bin_intervals)


def create_label(label_path, output_path, time_window=50000, target_width=304, target_height=240, min_diagonal=30, min_dimension=10):
    dtype = [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'),
             ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4')]
    all_labels = []

    # Initialize a unique label identifier
    unique_label_id = 0

    timestamp_to_index = {}

    with open(label_path, 'r') as file:
        lines = file.readlines()

    # Sort lines by ascending frame numbers
    lines.sort(key=lambda line: int(line.strip().split()[0].split('_')[0]))

    for line in lines:
        parts = line.strip().split()
        frame_number = int(parts[0].split('_')[0])  # Extract frame number from the file name
        timestamp = frame_number * time_window

        for part in parts[1:]:
            bbox = list(map(float, part.split(',')))
            x_min, y_min, x_max, y_max, class_id = bbox
            width, height = x_max - x_min, y_max - y_min

            # Check if the bounding box meets the minimum size requirements
            if width >= min_dimension and height >= min_dimension:
                diagonal = (width ** 2 + height ** 2) ** 0.5
                if diagonal >= min_diagonal:
                    if timestamp not in timestamp_to_index:
                        timestamp_to_index[timestamp] = len(all_labels)

                    label_entry = (timestamp-1, x_min, y_min, width, height, class_id, 1.0, unique_label_id)
                    all_labels.append(label_entry)

                    unique_label_id += 1
 
    all_data = np.array(all_labels, dtype=dtype) if all_labels else np.array([], dtype=dtype)

    unique_ts = create_label_timestamps(all_data, output_path)
    
    objframe_idx_2_label_idx = np.array(list(timestamp_to_index.values()), dtype='i4')

    create_objframe_2_repridx(unique_ts, output_path)
    
    labels_v2_path = os.path.join(output_path, 'labels_v2')
    if not os.path.exists(labels_v2_path):
        os.makedirs(labels_v2_path)

    output = os.path.join(labels_v2_path, 'labels.npz')
    np.savez(output, labels=all_data, objframe_idx_2_label_idx=objframe_idx_2_label_idx)


import h5py
import numpy as np
import os

def create_er_objframe_ts(events_path, output_path, rectify_map_path):
    original_height, original_width = 480, 640
    target_height, target_width = 240, 304
    crop_x_start = (original_width - target_width) // 2
    crop_y_start = (original_height - target_height) // 2

    # Load event data
    with h5py.File(events_path, 'r') as hdf:
        events_group = hdf['events']
        p = np.array(events_group['p'])
        t = np.array(events_group['t'])
        x = np.array(events_group['x'])
        y = np.array(events_group['y'])

    # Load the rectification map
    with h5py.File(rectify_map_path, 'r') as hdf:
        rectify_map = hdf['rectify_map'][:]

    # Parameters for processing
    time_window = 50000  # 50ms time window
    num_sub_bins = 10  # Number of sub-bins per polarity
    total_sub_bins = num_sub_bins * 2  # Total sub-bins for both polarities
    total_time = t.max() - t.min()  # Total time span of the dataset
    num_time_bins = total_time // time_window + 1  # Number of 50ms time bins

    create_time_bin_intervals(num_time_bins, output_path)

    event_representation = np.zeros((num_time_bins, total_sub_bins, target_height, target_width), dtype=np.int8)
    t_adjusted = t - t.min()

    # Set up batching
    batch_size = 1000000  # Adjust based on your system's memory capacity
    num_batches = (len(x) + batch_size - 1) // batch_size

    # Process events in batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(x))

        # Apply rectification for the current batch
        x_batch, y_batch, p_batch, t_batch = x[start_idx:end_idx], y[start_idx:end_idx], p[start_idx:end_idx], t[start_idx:end_idx]
        rectified_positions = rectify_map[y_batch, x_batch]
        x_rectified, y_rectified = np.round(rectified_positions[..., 0]).astype(int), np.round(rectified_positions[..., 1]).astype(int)

        # Filter out-of-frame events for the current batch
        within_frame = (x_rectified >= 0) & (x_rectified < original_width) & (y_rectified >= 0) & (y_rectified < original_height)
        x_filtered, y_filtered, p_filtered, t_filtered = x_rectified[within_frame], y_rectified[within_frame], p_batch[within_frame], t_batch[within_frame]

        # Cropping and further processing
        x_cropped = np.clip(x_filtered - crop_x_start, 0, target_width - 1)
        y_cropped = np.clip(y_filtered - crop_y_start, 0, target_height - 1)
        t_cropped = t_filtered - t.min()

        # Calculate bin indices for the current batch
        chunk_time_bin_indices = (t_cropped // time_window).astype(int)
        chunk_sub_bin_indices = ((t_cropped % time_window) / (time_window / num_sub_bins)).astype(int)
        chunk_sub_bin_indices[p_filtered == 0] += num_sub_bins  # Adjust for negative polarity

        # Accumulate counts in the event representation array for the current batch
        np.add.at(event_representation, (chunk_time_bin_indices, chunk_sub_bin_indices, y_cropped, x_cropped), 1)

    # Save the processed data
    event_repr_path = os.path.join(output_path, 'event_representations_v2', 'stacked_histogram_dt=50_nbins=10')
    if not os.path.exists(event_repr_path):
        os.makedirs(event_repr_path)
    output_file = os.path.join(event_repr_path, 'event_representations.h5')
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('data', data=event_representation, compression="gzip", compression_opts=4)



# --------------------------------- Single file Usage ---------------------------------
        
# data_input = r"C:\Users\lkxv3\OneDrive - National University of Singapore\DSEC\DSEC_MOD\training\zurich_city_00_b"
# label_path = os.path.join(data_input, 'train', '_annotations.txt')
# events_path = os.path.join(data_input, 'events', 'left', 'events.h5')
# # rectify_map_path = os.path.join(data_input, 'events', 'left', 'rectify_map.h5')

# # create folder for the output folder in the C:\Users\lkxv3\OneDrive\Desktop\CP5105\RVT\MOD_transform
# save_path = r"C:\Users\lkxv3\OneDrive\Desktop\CP5105\RVT\MOD_transform"
# output_path = os.path.join(save_path, 'autolabelled')

# # create_er_objframe_ts(events_path, output_path)
# create_er_objframe_ts(events_path, output_path)
# create_label(label_path, output_path, time_window=50000)


# ---------------------------------Process whole directory---------------------------------

def process_sequence(data_input, save_path):
    # this label_path is the path to the _annotations.txt file that is created from RGB inference.
    label_path = os.path.join(data_input, 'train', '_annotations.txt')
    events_path = os.path.join(data_input, 'events', 'left', 'events.h5')
    rectify_map_path = os.path.join(data_input, 'events', 'left', 'rectify_map.h5')
    sequence_name = os.path.basename(data_input)  # Extract the name of the current sequence directory
    output_path = os.path.join(save_path, sequence_name)

    # Create directory for processed files if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #   Pass the rectify_map to the function
    create_er_objframe_ts(events_path, output_path, rectify_map_path)

    # create_er_objframe_ts(events_path, output_path)
    create_label(label_path, output_path, time_window=50000)

def main(data_directory, save_directory):

    # Iterate over each directory in the data_directory
    for dir_name in os.listdir(data_directory):
        full_dir_path = os.path.join(data_directory, dir_name)
        if os.path.isdir(full_dir_path):  # Check if it is a directory
            print(f"Processing sequence: {dir_name}")
            process_sequence(full_dir_path, save_directory)

# Usage
data_directory = r"C:\Users\lkxv3\OneDrive - National University of Singapore\DSEC\DSEC_MOD\training"
save_directory = r"C:\Users\lkxv3\OneDrive\Desktop\CP5105\RVT\MOD_transform\rectify"

main(data_directory, save_directory)


time2 = time.time()
print('Total time taken:', time2 - time1, 's')
