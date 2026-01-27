from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import  TensorDataset, DataLoader, Dataset, random_split,Subset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import os


class PUCENGDataModule(LightningDataModule):

    def __init__(
        self, 
        data_dir: str = "data/puceng/",
        repreprocess: bool = False,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.20, 0.10),#划分训练集测试集和验证集
        batch_size: int = 256,
        points: int = -1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.batch_size_per_device = batch_size

    @property
    def num_points(self) -> int:
        """Get the f of points.

        :return: The number of frequency of curve.
        """ 
        return np.linspace(0.2,1.2,1049)

    def process(self):
        """get the data from the excel file and save it as a pt file""" 
        folder = self.hparams.data_dir
        exts = ['xlsx']
        paths_top = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        y = []
        x = []

        for i in tqdm.tqdm(paths_top):
            writer_1 = pd.ExcelFile(i)
            for i in writer_1.sheet_names: 
                data_frame = writer_1.parse(i,header=None)
                x.append((data_frame[0][:20]).astype('int')) # 直接读取1，2，3 
                y.append(data_frame[2])                      # 直接读取预测值 读取cst仿真的结果，34层铺层结构+1-18GHz的S11曲线插值点，共1001个点

        y_arr = np.array(y)
        x_arr = np.array(x)
        enc = OneHotEncoder(sparse_output=False)
        enc.fit(x_arr)
        x_arr = enc.transform(x_arr)
        # 保存预处理编码器
        import joblib
        joblib.dump(enc, self.hparams.data_dir + '/one_hot_encoder_20.pkl')  # 在加载20层数据时，这里的joblib对应one_hot_encoder_20，同时也需要改
        dataset = TensorDataset(torch.from_numpy(x_arr).float(), torch.from_numpy(y_arr).float())
        if os.path.exists(self.hparams.data_dir  + 'puceng.pt'):
            os.remove(self.hparams.data_dir  + 'puceng.pt')
        torch.save(dataset, self.hparams.data_dir  + 'puceng.pt')

        # print('dataset',len(dataset))
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
       
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if self.hparams.repreprocess:
                dataset = self.process()
                sub_dateset = dataset
            else:
                ori_dataset = torch.load(self.hparams.data_dir  + 'puceng.pt')
                if self.hparams.points == -1:
                    self.hparams.points = len(ori_dataset)
                sub_dateset = Subset(ori_dataset, list( range(self.hparams.points)))

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=sub_dateset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = PUCENGDataModule(repreprocess=True)
    # _ = PUCENGDataModule(repreprocess=False)
    _.setup()
    _.train_dataloader()
    