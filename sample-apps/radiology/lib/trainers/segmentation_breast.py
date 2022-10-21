# @Time : 2022/10/20 10:02 PM 
# @Author : zyc
# @File : segmentation_breast.py 
# @Title :
# @Description :
import torch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.transforms import LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityd, RandFlipd, RandRotate90d, \
    RandShiftIntensityd, EnsureTyped, SelectItemsd, ToTensord, Activationsd, AsDiscreted

from monailabel.tasks.train.basic_train import BasicTrainTask, Context


class SegmentationBreast(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        description="Train Segmentation model for breast",
        spatial_size=(128, 128, 64),
        target_spacing=(1.0, 1.0, 1.0),
        number_intensity_ch=1,
        **kwargs,
    ):
        self._network = network
        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.number_intensity_ch = number_intensity_ch

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
            Spacingd(
                keys=("image", "label"),
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityd(keys="image"),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            #
            EnsureTyped(keys=("image", "label"), device=context.device),
            SelectItemsd(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            ToTensord(keys=("pred", "label")),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=True,
                n_classes=2,
            ),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Spacingd(
                keys=("image", "label"),
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys=("image", "label"), device=context.device),
        ]

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(roi_size=(128, 128, 64), sw_batch_size=1, overlap=0.25)