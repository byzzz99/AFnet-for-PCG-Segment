import os
import glob
import numpy as np
import scipy
import scipy.io
import librosa as lb
from scipy.signal import butter, lfilter, filtfilt, resample_poly
from tqdm import trange


# 设计带通滤波器
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq  # 归一化低频截止频率
    high = highcut / nyq  # 归一化高频截止频率
    b, a = butter(order, [low, high], btype='band')  # 设计带通滤波器
    return b, a


# 应用带通滤波器
def butter_bandpass_filter(data, fs, lowcut=25, highcut=400, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)  # 获取滤波器系数
    y = lfilter(b, a, data)  # 应用滤波器
    return y


# 应用低通滤波器
def butter_lowpass_filter(data, fs, cutoff, order):
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # 设计低通滤波器
    y = filtfilt(b, a, data)  # 应用零相位滤波
    return y


# 加载PhysioNet 2016数据集
def load_data_2016(wav_path, seg_path, resampling, low=None, high=None, lowpass=None,
                   downsampling=False, bandpass=False):

    wav, orig_sr = lb.load(wav_path, sr=None)  # 加载音频文件，保留原始采样率

    # 降采样处理
    if downsampling:
        wav = resample_poly(wav, up=resampling, down=orig_sr)  # 重采样到目标采样率

    # 滤波处理
    if bandpass:
        wav = butter_bandpass_filter(wav, fs=resampling, lowcut=low, highcut=high, order=5)  # 带通滤波
    else:
        wav = butter_lowpass_filter(wav, fs=resampling, cutoff=lowpass, order=5)  # 低通滤波

    # 加载分割标注数据
    df = scipy.io.loadmat(seg_path)['state_ans']
    seg = np.zeros_like(wav)# 创建与音频等长的标签数组
    # 解析标注数据
    for idx in range(len(df) - 1):
        start_raw = float(df[idx, 0][0, 0])
        end_raw = float(df[idx + 1, 0][0, 0])
        start = int(start_raw * float(resampling) / float(orig_sr))
        end = int(end_raw * float(resampling) / float(orig_sr))

        cls = df[idx, 1][0, 0][0]  # 获取当前时间段的类别标签

        # 将文本标签转换为数字编码
        if cls == 'diastole':
            cls = 1
        elif cls == 'S1':
            cls = 2
        elif cls == 'systole':
            cls = 3
        elif cls == 'S2':
            cls = 4
        seg[start:end] = int(cls)  # 将对应时间段标记为类别编码

    # 去除前后端无效数据
    nonzero = np.where(seg != 0)[0]
    if nonzero.size == 0:
        raise ValueError("Segmentation array is empty after filtering.")
    seg = seg[nonzero[0]:nonzero[-1]]  # 截取有标注的部分
    wav = wav[nonzero[0]:nonzero[-1]]  # 同步截取音频部分

    return wav, seg


# 预处理PhysioNet 2016数据集
def preprocess_2016_data():
    # 定义训练数据目录
    data_dirs = [
        '/home/xcy/zby/pcgfs/data/data2016/train/training-a',
        '/home/xcy/zby/pcgfs/data/data2016/train/training-b',
        '/home/xcy/zby/pcgfs/data/data2016/train/training-c',
        '/home/xcy/zby/pcgfs/data/data2016/train/training-d',
        '/home/xcy/zby/pcgfs/data/data2016/train/training-e',
        '/home/xcy/zby/pcgfs/data/data2016/train/training-f'
    ]

    # 定义标注数据目录
    annot_dirs = [
        '/home/xcy/zby/pcgfs/data/data2016/ana/training-a_StateAns',
        '/home/xcy/zby/pcgfs/data/data2016/ana/training-b_StateAns',
        '/home/xcy/zby/pcgfs/data/data2016/ana/training-c_StateAns',
        '/home/xcy/zby/pcgfs/data/data2016/ana/training-d_StateAns',
        '/home/xcy/zby/pcgfs/data/data2016/ana/training-e_StateAns',
        '/home/xcy/zby/pcgfs/data/data2016/ana/training-f_StateAns'
    ]

    wav_files_group = []  # 存储各组音频文件路径
    seg_files_group = []  # 存储各组标注文件路径

    # 遍历各组数据目录
    for i in range(len(data_dirs)):
        wav_files = glob.glob(os.path.join(data_dirs[i], '*.wav'))  # 获取所有wav文件
        seg_files = glob.glob(os.path.join(annot_dirs[i], '*.mat'))  # 获取所有标注文件
        print(f"Group {i}: Found {len(wav_files)} wav files and {len(seg_files)} annotation files.")

        # 提取文件名ID
        wav_ids = [os.path.splitext(os.path.basename(s))[0] for s in wav_files]
        seg_ids = [os.path.basename(s).split('_')[0] for s in seg_files]

        # 找出没有对应标注的音频文件
        missing_ids = [num for num in wav_ids if num not in seg_ids]
        print(f"Group {i}: {len(missing_ids)} wav files missing annotations.")

        # 移除没有标注的音频文件
        for mid in missing_ids:
            wav_path = os.path.join(data_dirs[i], f'{mid}.wav')
            if wav_path in wav_files:
                wav_files.remove(wav_path)

        wav_files_group.append(wav_files)  # 保存有效音频文件路径
        seg_files_group.append(seg_files)  # 保存有效标注文件路径
        print(f"Group {i}: {len(wav_files)} wav files remain after removal.")

    # 展平文件列表
    wav_files_flat = sorted([f for sublist in wav_files_group for f in sublist])
    seg_files_flat = sorted([f for sublist in seg_files_group for f in sublist])

    # 验证文件有效性
    valid_wav_files = []
    valid_seg_files = []
    for wav_path, seg_path in zip(wav_files_flat, seg_files_flat):
        try:
            # 尝试加载数据，验证是否能成功处理
            _ = load_data_2016(wav_path, seg_path, resampling=1000, low=20, high=250,
                               bandpass=True, downsampling=True)
            valid_wav_files.append(wav_path)
            valid_seg_files.append(seg_path)
        except Exception as e:
            print(f"Skipping {wav_path} due to error: {e}")

    # 处理所有有效数据
    total_data = []
    for wav_path, seg_path in zip(valid_wav_files, valid_seg_files):
        try:
            wav, seg = load_data_2016(wav_path, seg_path, resampling=1000, low=20, high=250,
                                      bandpass=True, downsampling=True)
            total_data.append({'wav': wav, 'seg': seg, 'fname': os.path.basename(wav_path)})
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

    feature_length = 6144  # 定义特征序列最小长度
    # 筛选长度符合要求的数据
    total_feature_length_data = [d for d in total_data if len(d['wav']) >= feature_length]
    print(f"2016 Data: {len(total_feature_length_data)} samples after filtering (feature length >= {feature_length}).")

    # 保存预处理后的数据
    output_path = f'/home/xcy/zby/pcgfs/data/PhysioNet2016_1000Hz_20_250_fe_{feature_length}.npy'
    np.save(output_path, total_feature_length_data)
    print(f"Saved preprocessed 2016 data to {output_path}.")


# 主函数
def main():
    print("Starting preprocessing for 2016 data...")
    preprocess_2016_data()  # 调用预处理函数


if __name__ == '__main__':
    main()  # 程序入口点