# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Callable, Dict, Sequence

import numpy as np
from lib.transforms import LoadImagePatchd, PostFilterLabeld
from monai.transforms import (
    Activationsd,
    AsChannelFirstd,
    AsDiscreted,
    EnsureTyped,
    ScaleIntensityRangeD,
    SqueezeDimd,
    ToNumpyd,
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import FindContoursd
from monailabel.transform.writer import PolygonWriter

logger = logging.getLogger(__name__)


class SegmentationNuclei(BasicInferTask):
    """
    This provides Inference Engine for pre-trained segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        roi_size=(256, 256),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=2,
        description="A pre-trained semantic segmentation model for Pathology",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            roi_size=roi_size,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["pathology"] = True
        return d

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImagePatchd(keys="image", mode="RGB", dtype=np.uint8, padding=False),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            AsChannelFirstd(keys="image"),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
        ]

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys=("image", "pred")),
            PostFilterLabeld(keys="pred", image="image"),
            FindContoursd(keys="pred", labels=self.labels, max_poly_area=128 * 128),
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)
