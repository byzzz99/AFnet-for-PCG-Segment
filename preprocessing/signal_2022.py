import os
import glob
import numpy as np
import pandas as pd
import librosa as lb
from scipy.signal import resample_poly, butter, lfilter, filtfilt
import natsort
from tqdm import trange

def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.max(np.abs(signal))

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, fs, lowcut, highcut, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def resample_audio(wav_path, seg_path, resampling, low=None, high=None, lowpass=None,
                   downsampling=False, bandpass=False, up_=None, down_=None):
    wav, orig_sr = lb.load(wav_path, sr=None)
    
    if downsampling:
        wav = resample_poly(wav, up=up_, down=down_)
    
    wav = normalize_signal(wav)
    
    if bandpass:
        wav = butter_bandpass_filter(wav, fs=resampling, lowcut=low, highcut=high, order=5)
    else:
        wav = butter_lowpass_filter(wav, cutoff=lowpass, fs=resampling, order=5)
    
    df = pd.read_csv(seg_path, sep='\t', header=None)
    seg = np.zeros_like(wav)
    
    for idx in range(len(df)):
        start_idx = int(df.iloc[idx, 0] * resampling)
        end_idx = int(df.iloc[idx, 1] * resampling)
        seg[start_idx:end_idx] = df.iloc[idx, 2]
    
    nonzero_indices = np.where(seg != 0)[0]
    if nonzero_indices.size == 0:
        print(f"No non-zero segment found for file {wav_path}")
        return None, None
    wav_new = wav[nonzero_indices[0]:nonzero_indices[-1]]
    seg_new = seg[nonzero_indices[0]:nonzero_indices[-1]]
    
    return wav_new, seg_new

def process_data_resample(low_, high_, sampling_, feature_length, wav_paths, seg_paths):
    up = sampling_
    down = 4000  # Original sampling rate for PhysioNet2022 is 4000 Hz
    count_with_zeros = 0
    problem_count = 0
    total_resample = []
    
    for i in trange(len(seg_paths)):
        try:
            wav, seg = resample_audio(wav_paths[i], seg_paths[i], resampling=sampling_,
                                      bandpass=True, downsampling=True, low=low_, high=high_, up_=up, down_=down)
            if wav is None or seg is None:
                problem_count += 1
                continue
            # Skip files where segmentation still contains zeros
            if np.any(seg == 0):
                count_with_zeros += 1
                continue
            total_resample.append({
                'wav': wav,
                'seg': seg,
                'fname': os.path.basename(wav_paths[i])
            })
        except Exception as e:
            print(f"Exception occurred for file {wav_paths[i]}: {e}")
            problem_count += 1
    
    print('Total valid files:', len(total_resample))
    print('Files with zeros in segmentation skipped:', count_with_zeros)
    print('Files with problems:', problem_count)
    
    total_new = [t for t in total_resample if len(t['wav']) >= feature_length]
    print('Total files after filtering by feature length:', len(total_new))
    
    output_filename = f'PhysioNet2022_{sampling_}Hz_{low_}_{high_}_fe_{feature_length}.npy'
    np.save(output_filename, total_new)
    print(f"Saved preprocessed PhysioNet2022 data to {output_filename}")

def preprocess_physionet2022_data():
    """
    The function expects the following directory structure:
        /path/to/PhysioNet2022_DB/*.wav
        /path/to/PhysioNet2022_Modified_TSV/*.tsv
    """
    wav_paths = natsort.natsorted(glob.glob('/home/xcy/zby/pcgfs/data/training_data/*.wav'))
    seg_paths = natsort.natsorted(glob.glob('/home/xcy/zby/pcgfs/data/training_data/*.tsv'))
    
    print(f"Found {len(wav_paths)} WAV files and {len(seg_paths)} segmentation files.")
    
    resample_params = [
        {'low_': 20, 'high_': 250},
        {'low_': 20, 'high_': 400},
        {'low_': 15, 'high_': 250},
        {'low_': 15, 'high_': 400},
    ]
    
    num_values = [(1000, 6144)]
    
    for sampling_, feature_length in num_values:
        print(f'Processing with sampling rate: {sampling_} Hz, feature length: {feature_length}')
        for params in resample_params:
            process_data_resample(**params, sampling_=sampling_, feature_length=feature_length,
                                    wav_paths=wav_paths, seg_paths=seg_paths)

def main():
    print("Starting PhysioNet2022 data preprocessing...")
    preprocess_physionet2022_data()

if __name__ == '__main__':
    main()
