import logging
import os
from distutils.util import strtobool
from typing import Optional, Union, Dict, Any

import lib.infers
import lib.trainers
from monai.networks.nets import UNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.sum import Sum
from monailabel.utils.others.generic import download_file

logger = logging.getLogger(__name__)

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
        self.labels = {
            "mass": 1,
        }

        network = self.conf.get("network", "unet")

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "false")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}/segmentation_{network}_spleen.pt"
            download_file(url, self.path[0])

        # Network
        self.network = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=[16, 32, 64, 128, 256],
            strides=[2, 2, 2, 2],
            num_res_units=2,
            norm="batch",
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        return {
            self.name: lib.infers.SegmentationBreast(
                path=self.path,
                network=self.network,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                enviroment="dev"
            ),
            f"{self.name}_prod": lib.infers.SegmentationBreast(
                path=self.path,
                network=self.network,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                enviroment="prod"
            ),
        }
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)

        task: TrainTask = lib.trainers.SegmentationBreast(
            model_dir=output_dir,
            network=self.network,
            description="Train Breast Segmentation Model",
            load_path=self.path[0],
            publish_path=self.path[1],
            labels=self.labels,
        )
        return task

    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        methods: Dict[str, ScoringMethod] = {
            "dice": Dice(),
            "sum": Sum(),
        }
        return methods