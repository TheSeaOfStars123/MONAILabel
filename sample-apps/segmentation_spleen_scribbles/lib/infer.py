# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Callable

from monai.inferers import SlidingWindowInferer, Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    CopyItemsd,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    ToNumpyd,
    ToTensord, EnsureTyped, EnsureChannelFirstd,
)

from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.scribbles.transforms import WriteLogits
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import BoundingBoxd, Restored


class SegmentationWithWriteLogits(InferTask):
    """
    Inference Engine for pre-trained Spleen segmentation (UNet) model for MSD Dataset. It additionally provides
    appropriate transforms to save logits that are needed for post processing stage.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels="spleen",
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation of the spleen from CT image",
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )

    def pre_transforms(self):
        return [
            LoadImaged(keys="image"),
            AddChanneld(keys="image"),
            Spacingd(keys="image", pixdim=[1.0, 1.0, 1.0]),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys="image"),
        ]

    def inferer(self):
        return SlidingWindowInferer(roi_size=[160, 160, 160])

    def post_transforms(self):
        return [
            CopyItemsd(keys="pred", times=1, names="logits"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys=["pred", "logits"]),
            Restored(keys=["pred", "logits"], ref_image="image"),
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
            WriteLogits(key="logits", result="result"),
        ]

class SegmentationBreastWithWriteLogits(BasicInferTask):
    """
    This provides Inference Engine for pre-trained spleen segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels="mass",
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation of the spleen from CT image",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels="mass",
            dimension=dimension,
            description=description,
            **kwargs,
        )

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
        ]

    def inferer(self, data=None) -> Inferer:
        # return SimpleInferer()
        return SlidingWindowInferer(roi_size=(128, 128, 48), sw_batch_size=1, overlap=0.25)

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            CopyItemsd(keys="pred", times=1, names="logits"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys=["pred", "logits"]),
            Restored(keys=["pred", "logits"], ref_image="image"), # pred and logits(2, 128, 128, 48)
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
            WriteLogits(key="logits", result="result"),
        ]
