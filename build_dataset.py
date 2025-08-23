import os
import mne
import pandas as pd
def prepare_EEG(run_data, word_onset, duration):
    ## run_data : (n_sensors, n_timepoints)
    first_onset = word_onset[0] / 4 # convert time in seconds 
    temp_data = [run_data[:, first_onset + idx * duration] for idx in range(len(word_onset))]
    EEG_data = np.stack(temp_data)
    return EEG_data

def build_dataset(data_dir):
    subjects = os.listdir(data_dir)
    dataset_list = list()
    
    for subject in subjects:
        subj_dir = os.path.join(data_dir, subject, "ses-littleprince")
        eeg_dir = subj_dir
        acq_tsv_path = os.path.join(subj_dir, f"{subject}_ses-littleprince_scans.tsv")
        acq_df = pd.read_csv(acq_tsv_path, sep="\t")
        
        acq_files = acq_df["filename"]
        subject_data = list()
        
        for acq_file in acq_files:
            acq_file_path = os.path.join(eeg_dir, acq_file)
            raw_data = mne.io.read_raw_brainvision(acq_file_path, preload=True, verbose=False)
            data_array = raw_data.get_data()
            subject_data.append(data_array.astype(np.float16))
        dataset_list.append(subject_data)
    
    return dataset_list
