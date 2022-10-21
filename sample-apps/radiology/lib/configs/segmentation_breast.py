# @Time : 2022/10/20 9:27 PM 
# @Author : zyc
# @File : segmentation_breast.py 
# @Title :
# @Description :
import os
from distutils.util import strtobool
from typing import Optional, Union, Dict, Any

import lib.infers
import lib.trainers
from monai.networks.nets import UNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file


class SegmentationBreast(TaskConfig):
    def __init__(self):
        super().__init__()
        # 启用基于认知的主动学习策略
        self.epistemic_enabled = None
        # 限制样本数量进行认知评分
        self.epistemic_samples = None
        # 启用基于TTA（测试时间增广）的主动学习策略
        self.tta_enabled = None
        # 限制样本数量进行TTA评分
        self.tta_samples = None

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        # self.labels = ["mass"]
        self.labels = {
            "mass": 1,
            "background": 0,
        }

        # Number of input channels - 4 for BRATS and 1 for spleen
        self.number_intensity_ch = 1

        network = self.conf.get("network", "dynunet")

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{self.name}_{network}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{self.name}_{network}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}/deepedit_{network}_singlelabel.pt"
            download_file(url, self.path[0])

        self.target_spacing = (1.0, 1.0, 1.0)  # target space for image
        self.spatial_size = (128, 128, 48)  # train input size
        self.roi_size = (128, 128, 48)  # sliding window size for infer

        # Network
        self.network = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=len(self.labels.keys()) + 1,  # All labels plus background
            channels=[16, 32, 64, 128, 256],
            strides=[2, 2, 2, 2],
            num_res_units=2,
            norm="batch",
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.SegmentationBreast(
            path=self.path,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            labels=self.labels,
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        task: TrainTask = lib.trainers.SegmentationBreast(
            model_dir=output_dir,
            network=self.network,
            spatial_size=self.spatial_size,
            target_spacing=self.target_spacing,
            load_path=self.path[0],
            publish_path=self.path[1],
            description="Train Breast Segmentation Model",
            dimension=3,
            labels=self.labels,
        )
        return task