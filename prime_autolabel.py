import numpy as np
import h5py
import os


def create_time_bin_intervals(num_time_bins, output_path, time_window=50000):

    start_times = np.arange(0, num_time_bins * time_window, time_window)
    end_times = start_times + time_window
    time_bin_intervals = np.column_stack((start_times, end_times))

    event_repr_path = os.path.join(output_path, 'event_representations_v2', 'stacked_histogram_dt=50_nbins=10')  # New directory
    if not os.path.exists(event_repr_path):
        os.makedirs(event_repr_path)

    output = os.path.join(event_repr_path, 'timestamps_us.npy')
    np.save(output, time_bin_intervals)


def modify_event_representations(base_path, frequency):
    # List all directories in the base path
    test_folders = next(os.walk(base_path))[1]

    for folder in test_folders:
        data_path = os.path.join(base_path, folder)

        # Paths to the necessary files within each folder
        objframe_ind = os.path.join(data_path, 'event_representations_v2', 'stacked_histogram_dt=50_nbins=10', 'objframe_idx_2_repr_idx.npy')
        events_path = os.path.join(data_path, 'event_representations_v2', 'stacked_histogram_dt=50_nbins=10', 'event_representations.h5')
        labels_path = os.path.join(data_path, 'labels_v2', 'labels.npz')

        print(f'------------------ Processing {folder} ------------------')
        
        # Load event representations
        with h5py.File(events_path, 'r') as hdf:
            event_representations = hdf['data'][:]


        num_time_bins = event_representations.shape[0]
        create_time_bin_intervals(num_time_bins, data_path, time_window=50000)

        timestamp_path_er = os.path.join(data_path, 'event_representations_v2', 'stacked_histogram_dt=50_nbins=10', 'timestamps_us.npy')

        original_dtype = event_representations.dtype
        truth_data = np.load(labels_path)
        labels = truth_data['labels']

        objframe_index = np.load(objframe_ind)
        timestamps = np.load(timestamp_path_er)

        # Create a copy of event representations to apply changes
        modified_event_representations = event_representations.copy()

        # Apply filtering at the specified frequency
        for bin_index in objframe_index:
            # Skip frames that do not meet the frequency criteria
            if bin_index % frequency != 0:
                continue

            # Find the time frame for the current event bin
            event_start, event_end = timestamps[bin_index]
            
            # Initialize the mask with zeros to filter out all events initially
            mask = np.zeros_like(modified_event_representations[bin_index])
            
            # Iterate through labels to find those that fall within the current time frame
            for label in labels:
                timestamp, x, y, w, h, label_value, confidence, no = label
                
                # Check if the label's timestamp is within the current event bin's time frame
                if event_start <= timestamp <= event_end:
                    x_min, y_min = max(0, int(x)), max(0, int(y))
                    x_max, y_max = min(mask.shape[2], int(x + w)), min(mask.shape[1], int(y + h))

                    # Set the labeled region in the mask to 1
                    mask[:, y_min:y_max, x_min:x_max] = 1

            # Apply the mask to the current frame
            modified_event_representations[bin_index] *= mask

        # Overwrite the existing event representations with the modified data
        with h5py.File(events_path, 'w') as hdf:
            hdf.create_dataset('data', data=modified_event_representations, compression='gzip', compression_opts=4)
            print(f'Modified event representations saved to {events_path}')


base_path = r'C:\Users\lkxv3\OneDrive\Desktop\CP5105\RVT\MOD_transform\rectify_primed_f20'
frequency = 20 # Adjust priming frequency as needed
modify_event_representations(base_path, frequency)
