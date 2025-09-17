import monai
import numpy as np
import torch 

# Local imports
from utils import augment_neurokit, zscore

class PCGDataset():
    def __init__(self, args, data, phase='test'):
        self.data = data
        self.phase = phase
        self.target_sr = args.target_sr
        self.featureLength = args.featureLength
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        audio = self.data[idx]['wav']
        seg = self.data[idx]['seg']
        fname = self.data[idx]['fname']
        fname =self.data[idx]['fname']

        valid_indices = np.where(seg != 0)[0]
        start_idx, end_idx = valid_indices[0], valid_indices[-1]
        audio = audio[start_idx:end_idx]
        seg = seg[start_idx:end_idx]

        if self.phase == 'train':
            audio = augment_neurokit(audio, self.target_sr)

            total_len = audio.shape[-1]
            delta = (total_len - self.featureLength) // 2 #2016은 featureLength=8192
            random_shift = int(np.random.rand() * delta)
            start = random_shift
            end = start + self.featureLength

            audio_segment = audio[start:end]
            seg_segment = seg[start:end]

            # Random amplitude scaling
            p = 0.2
            if np.random.rand() <= p:
                scaling_factor = (np.random.rand() - 0.5) * 2
                audio_segment = audio_segment * scaling_factor
        else:
            audio_segment = audio
            seg_segment = seg

        audio_tensor = torch.from_numpy(audio_segment).unsqueeze(0).float()
        audio_tensor = zscore(audio_tensor)
        
        audio_2ch = torch.cat([audio_tensor, torch.sqrt(audio_tensor**2)], dim=0) # x_ = torch.concat([x_, torch.sqrt(x_**2)],dim=0) # original and amplitude as input

        try:
            seg_tensor_4class = torch.from_numpy(seg_segment)
        except ValueError:
            seg_tensor_4class = torch.zeros(audio_2ch.shape[-1])
        seg_tensor_4class = seg_tensor_4class.unsqueeze(0).long() 

        seg_tensor_4class[seg_tensor_4class == 2] = 2
        seg_tensor_4class[seg_tensor_4class == 4] = 4
        seg_tensor_4class[seg_tensor_4class == 1] = 1
        seg_tensor_4class[seg_tensor_4class == 3] = 3

        seg_tensor_2class = seg_tensor_4class.clone()
        seg_tensor_2class[seg_tensor_4class == 2] = 1
        seg_tensor_2class[seg_tensor_4class == 4] = 1
        seg_tensor_2class[seg_tensor_4class == 1] = 0
        seg_tensor_2class[seg_tensor_4class == 3] = 0

        # Build one-hot (5 classes => final slice is used for 2-class separation)
        seg_fourclass_zero_based = seg_tensor_4class - 1
        seg_one_hot = monai.networks.utils.one_hot(seg_fourclass_zero_based, num_classes=5, dim=0)

        seg_one_hot[-1] = seg_tensor_2class[0]

        seg_one_hot[0], seg_one_hot[-1] = seg_one_hot[-1].clone(), seg_one_hot[0].clone()
        seg_one_hot = seg_one_hot[1:]

        return {
            'x': audio_2ch,            # 2-channel waveform
            'two_class_label': seg_tensor_2class,
            'four_class_label': seg_tensor_4class,
            'y_onehot': seg_one_hot,
            'fname': fname
        }
  