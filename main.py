import os
import shutil
import warnings
import multiprocessing
import numpy as np
import sklearn
import sklearn.metrics
import monai
from monai.config import print_config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    TQDMProgressBar
)
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import dotenv

from config import get_args
from model import SEGNET, BasicUNet
from dataset import PCGDataset
from utils import set_seed


dotenv.load_dotenv()



def create_or_reset_folder(folder_path: str):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)


def load_data_2016(load_path: str):
    data2016 = np.load(load_path, allow_pickle=True)
    data_test2016 = data2016[:336]
    data_train2016 = data2016[336:]
    data_train2016, data_valid2016 = sklearn.model_selection.train_test_split(
        data_train2016, test_size=0.2, random_state=42
    )
    return data_train2016, data_valid2016, data_test2016



class OptimizedProgressBar(TQDMProgressBar):
    def __init__(self, total_steps, refresh_rate=1):
        super().__init__(refresh_rate)
        self.total_steps = total_steps
        self.current_step = 0
        self.epoch_progress = None
        self.current_epoch = -1
        self.first_epoch = True
        self.separator_printed = False
        
    def on_train_start(self, trainer, pl_module):

        pass
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch = trainer.current_epoch
        

        if self.first_epoch:
            self.first_epoch = False

        else:
            if not self.separator_printed:
                print("\n" + "-" * 80)
                self.separator_printed = True
        

        if self.epoch_progress:
            self.epoch_progress.close()
            

        self.epoch_progress = tqdm(
            desc=f"Epoch {self.current_epoch}",
            total=len(trainer.train_dataloader),
            position=1,
            leave=True,
            disable=not self.is_enabled
        )
        

        if not hasattr(self, 'total_progress'):
            self.total_progress = tqdm(
                desc="Training",
                total=self.total_steps,
                position=0,
                leave=True,
                disable=not self.is_enabled
            )
        
        self.separator_printed = False
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if self.epoch_progress:
            self.epoch_progress.update(1)
            self.epoch_progress.set_postfix(self.get_metrics(trainer, pl_module))
        

        self.current_step += 1
        if hasattr(self, 'total_progress'):
            self.total_progress.n = self.current_step
            self.total_progress.refresh()
            
    def on_validation_epoch_end(self, trainer, pl_module):

        if self.epoch_progress:
            self.epoch_progress.set_postfix(self.get_metrics(trainer, pl_module))
        
    def on_train_end(self, trainer, pl_module):

        if self.epoch_progress:
            self.epoch_progress.close()
        if hasattr(self, 'total_progress'):
            self.total_progress.close()


def build_trainer(model, max_epochs=100, total_steps=-1):

    progress_bar = OptimizedProgressBar(total_steps) if total_steps else TQDMProgressBar()
    
    trainer = pl.Trainer(
        log_every_n_steps=1,
        gradient_clip_algorithm='norm',
        accumulate_grad_batches=4,
        sync_batchnorm=True,
        benchmark=True,
        accelerator='gpu',
        devices=-1,
        max_epochs=max_epochs,
        max_steps=total_steps,
        strategy='ddp_find_unused_parameters_true',
        check_val_every_n_epoch=1,
        callbacks=[
            model.checkpoint_callback,
            LearningRateMonitor(),
            EarlyStopping('val_loss', patience=20),
            progress_bar
        ],
    )
    return trainer



def main():
    args = get_args()


    os.environ["HTTP_PROXY"] = os.getenv("HTTP_PROXY", "")
    os.environ["HTTPS_PROXY"] = os.getenv("HTTPS_PROXY", "")
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    warnings.filterwarnings(action='ignore')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = os.cpu_count()
    print("Number of workers:", NUM_WORKERS)
    print("cuda.is_available:", torch.cuda.is_available())
    print("Using device:", device)
    print_config()

    IN_CHANNELS = 2
    OUT_CHANNELS = 4
    MINSIZE = 50
    THR = 0.5
    VERSION = 2
    MAX_EPOCHS = 250
    LEARNING_RATE = 2e-4

    MLP_EXPANSION = args.mlp_expansion
    MLP_DROPOUT = args.mlp_dropout

    USE_SA = args.use_sa
    USE_CA = args.use_ca
    USE_MULTI_SCALE = args.multi


    comment = (
        f"ver{VERSION}_d{args.target_sr}_v{args.ver}_low{args.lowpass}"
        f"_multi_scale_{USE_MULTI_SCALE}_fft_{args.fft}"  
        f"_sa_{USE_SA}_ca_{USE_CA}_8heads"  
        f"_mlp_exp_{MLP_EXPANSION}_mlp_drop_{MLP_DROPOUT}"
    )
    load_path = f"/home/xcy/zby/pcgfs/data/PhysioNet{args.year}_{args.target_sr}Hz_{args.lowpass}_fe_{args.featureLength}.npy"
    infer_pth = f"/home/xcy/zby/pcgfs/result/lightning_logs/version_{args.ver}/checkpoints/"

    set_seed(args.seed)
    print(f"pytorch_lightning version: {pl.__version__}")


    net = BasicUNet(
        spatial_dims=1,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,

        features=(64, 64, 128, 256, 512, 512, 64),
        norm='instance',
        upsample='pixelshuffle',
        act='gelu',
        fft=args.fft,
        multi=USE_MULTI_SCALE,
        sa=USE_SA,
        ca=USE_CA,

        mlp_expansion=MLP_EXPANSION,
        mlp_dropout=MLP_DROPOUT
    )


    model = SEGNET(
        net=net,
        featureLength=args.featureLength,
        learning_rate=LEARNING_RATE,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        minsize=MINSIZE,
        thr=THR,
        device=device,

        path=f'/home/xcy/zby/pcgfs/result/{args.year}_toler{args.toler}_{comment}/',
        infer_path=infer_pth,
        toler=args.toler,
        swa_start_epoch_ratio=args.swa_start_epoch_ratio,
        swa_update_batch_size=args.swa_batch_size
    )


    if not args.infer:
        path = f"/home/xcy/zby/pcgfs/result/{args.year}_toler{args.toler}_{comment}/"
        create_or_reset_folder(path)
        data_train2016, data_valid2016, data_test2016 = load_data_2016(load_path)
        print(f"训练集大小: {len(data_train2016)}, 单epoch步数: {len(data_train2016)//args.batch}")

        print(f"\n===== 模型配置明细 =====")
        print(f"特征提取模块: 多尺度卷积（替换原双卷积）")
        print(f"多尺度卷积启用状态: {USE_MULTI_SCALE}")
        print(f"注意力配置: 自注意力(SA)={USE_SA}, 交叉注意力(CA)={USE_CA}, 注意力头数=8（固定）")
        print(f"MLP配置: 通道扩展系数={MLP_EXPANSION}, Dropout比例={MLP_DROPOUT}")
        print(f"频率增强(FFT)启用: {args.fft}")
        print(f"训练批次大小: {args.batch}, 总训练轮次: {MAX_EPOCHS}")
        print(f"=======================\n")


        train_ds2016 = PCGDataset(args, data_train2016, 'train')
        train_loader = DataLoader(
            train_ds2016,
            shuffle=True,
            batch_size=args.batch,
            drop_last=True,
            num_workers=1,
            pin_memory=True
        )


        steps_per_epoch = len(train_loader)
        total_train_steps = MAX_EPOCHS * steps_per_epoch
        print(f"总训练步数: {total_train_steps}")


        valid_ds2016 = PCGDataset(args, data_valid2016)
        valid_loader = DataLoader(
            valid_ds2016,
            batch_size=1,
            collate_fn=monai.data.utils.default_collate,
            num_workers=1,
        )


        trainer = build_trainer(model, max_epochs=MAX_EPOCHS, total_steps=total_train_steps)
        trainer.fit(model, train_loader, valid_loader)
    else:
        print("Inference mode requested. Skipping training.")

        print(f"\n===== 推理模型配置明细 =====")
        print(f"特征提取模块: 多尺度卷积（替换原双卷积）")
        print(f"多尺度卷积启用状态: {USE_MULTI_SCALE}")
        print(f"注意力配置: 自注意力(SA)={USE_SA}, 交叉注意力(CA)={USE_CA}, 注意力头数=8（固定）")
        print(f"MLP配置: 通道扩展系数={MLP_EXPANSION}, Dropout比例={MLP_DROPOUT}")
        print(f"==========================\n")


    if not args.not_2016:
        year = 2016
        test_path = f"/home/xcy/zby/pcgfs/result/{year}_toler{args.toler}_{comment}/"
        if not args.nofolder:
            create_or_reset_folder(test_path)
        data2016 = np.load(load_path, allow_pickle=True)
        data_test2016 = data2016[:336]
        test_ds2016 = PCGDataset(args, data_test2016)
        test_loader_2016 = DataLoader(
            test_ds2016,
            batch_size=1,
            collate_fn=monai.data.utils.default_collate
        )
        print("\n############# Toler 40 Internal 2016 start #############\n")
        checkpoint_file = os.path.join(infer_pth, 'best.ckpt')
        trainer = build_trainer(model, max_epochs=MAX_EPOCHS)
        trainer.test(model, test_loader_2016, ckpt_path=checkpoint_file)

    if not args.not_2022:
        print("\nToler 40 External 2022 start\n")
        year = 2022
        test_path = f"/home/xcy/zby/pcgfs/result/{year}_toler{args.toler}_{comment}/"
        if not args.nofolder:
            create_or_reset_folder(test_path)
        new_load_path = f"/home/xcy/zby/pcgfs/PCG_FTSeg-main/preprocessing/PhysioNet{year}_{args.target_sr}Hz_{args.lowpass}_fe_{args.featureLength}.npy"
        data2022 = np.load(new_load_path, allow_pickle=True)
        test_ds2022 = PCGDataset(args, data2022)
        test_loader_2022 = DataLoader(
            test_ds2022,
            batch_size=1,
            collate_fn=monai.data.utils.default_collate
        )
        trainer = build_trainer(model, max_epochs=MAX_EPOCHS)
        trainer.test(model, test_loader_2022, ckpt_path=os.path.join(infer_pth, 'best.ckpt'))

    if not args.not_amc:
        print("\nToler 40 External amc start\n")
        year = "amc"
        test_path = f"/home/xcy/zby/pcgfs/result/{year}_toler{args.toler}_{comment}/"
        if not args.nofolder:
            create_or_reset_folder(test_path)
        amc_load_path = f"/home/xcy/zby/pcgfs/data/{year}_{args.target_sr}Hz_{args.lowpass}_fe_{args.featureLength}.npy"
        data_amc = np.load(amc_load_path, allow_pickle=True)
        test_ds_amc = PCGDataset(args, data_amc)
        test_loader_amc = DataLoader(
            test_ds_amc,
            batch_size=1,
            collate_fn=monai.data.utils.default_collate
        )
        trainer = build_trainer(model, max_epochs=MAX_EPOCHS)
        trainer.test(model, test_loader_amc, ckpt_path=os.path.join(infer_pth, 'best.ckpt'))


if __name__ == "__main__":
    main()