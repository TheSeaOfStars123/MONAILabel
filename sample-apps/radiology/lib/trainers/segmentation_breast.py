import logging

import torch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.transforms import LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityd, RandFlipd, RandRotate90d, \
    RandShiftIntensityd, EnsureTyped, SelectItemsd, ToTensord, Activationsd, AsDiscreted, RandCropByPosNegLabeld

from lib.transforms.transforms import SpatialCropByRoiD
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)

class SegmentationBreast(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        description="Train Segmentation model for breast",
        **kwargs,
    ):
        self._network = network
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), lr=0.0001)

    def loss_function(self, context: Context):
        return DiceCELoss(to_onehot_y=True, softmax=True)

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            ScaleIntensityd(keys="image"),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            EnsureTyped(keys=("image", "label"), device=context.device),
            SpatialCropByRoiD(keys=["image", "label"]),
            SelectItemsd(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            ToTensord(keys=("pred", "label")),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=2,
            ),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys=("image", "label"), device=context.device),
            SpatialCropByRoiD(keys=["image", "label"]),
            SelectItemsd(keys=("image", "label")),
        ]

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(roi_size=(128, 128, 64), sw_batch_size=1, overlap=0.25)