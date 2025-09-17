import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Configure Training & Inference Arguments (8头注意力版)')


    parser.add_argument('--gpu', type=str, default='1', 
                        help='GPU device ID (e.g., "0,1" for multi-GPU)')
    parser.add_argument('--ver', type=int, default=45, 
                        help='Experiment version (for log folder naming)')


    parser.add_argument('--featureLength', type=int, default=4096, 
                        help='Length of input signal segment (心音信号片段长度)')
    parser.add_argument('--target_sr', type=int, default=1000, 
                        help='Target sampling rate of PCG signals (心音信号目标采样率)')
    parser.add_argument('--lowpass', type=str, default="240", 
                        help='Low-pass filter cutoff frequency (低通滤波器截止频率)')
    parser.add_argument('--toler', type=int, default=40, 
                        help='Time tolerance (ms) for event alignment (事件对齐的时间容忍度)')
    parser.add_argument('--year', type=int, default=2016, 
                        help='Dataset year (2016/2022, 训练数据集年份)')


    parser.add_argument('--infer', action='store_true',
                        help='Enable inference mode (skip training, 仅推理不训练)')
    parser.add_argument('--infer_2022', action='store_true', 
                        help='Infer on 2022 external dataset (仅在2022外部数据集推理)')
    parser.add_argument('--nofolder', action='store_true', 
                        help='Skip creating new result folder (不新建结果文件夹)')


    parser.add_argument('--multi', action='store_true', 
                        help='Enable multi-scale convolution blocks (启用多尺度卷积块)')
    parser.add_argument('--conv_', action='store_true', 
                        help='Use standard convolution (overrides other structures, 使用标准卷积)')
    parser.add_argument('--fft', action='store_true', 
                        help='Enable Fourier transform module (启用傅里叶变换模块)')

    parser.add_argument('--use_sa', action='store_true', 
                        help='Enable 8-head Self-Attention (启用8头自注意力)')
    parser.add_argument('--use_ca', action='store_true', 
                        help='Enable 8-head Cross-Attention (启用8头交叉注意力)')


    parser.add_argument('--mlp_expansion', type=float, default=4.0, 
                        help='MLP channel expansion ratio (default: 4.0, 建议范围2.0-6.0)')
    parser.add_argument('--mlp_dropout', type=float, default=0.1, 
                        help='MLP dropout rate (default: 0.1, 建议范围0.0-0.3)')


    parser.add_argument('--batch', type=int, default=32,
                        help='Training batch size (训练批次大小，多尺度+注意力建议≤32)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility (复现用随机种子)')


    parser.add_argument('--swa_start_epoch_ratio', type=float, default=0.2, 
                        help='Ratio of total epochs to start SWA (e.g., 0.5 = start at 50% epochs, SWA启动epoch比例)')
    parser.add_argument('--swa_batch_size', type=int, default=32, 
                        help='Batch size for updating BN statistics in SWA (SWA更新BN统计量的批次大小)')


    parser.add_argument('--not_2016', action='store_true', 
                        help='Skip 2016 internal test (跳过2016内部测试集)')
    parser.add_argument('--not_2022', action='store_true', 
                        help='Skip 2022 external test (跳过2022外部测试集)')
    parser.add_argument('--not_amc', action='store_true', 
                        help='Skip AMC external test (跳过AMC外部测试集)')

    return parser.parse_args()
