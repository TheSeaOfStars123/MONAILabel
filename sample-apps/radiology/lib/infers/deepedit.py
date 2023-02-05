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

from typing import Callable, Sequence, Union

from monai.apps.deepedit.transforms import (
    AddGuidanceFromPointsDeepEditd,
    AddGuidanceSignalDeepEditd,
    DiscardAddGuidanced,
    ResizeGuidanceMultipleLabelDeepEditd,
    # LoadGuidanceFromJsonFiled
)
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    SqueezeDimd,
    ToNumpyd, ScaleIntensityd,
)

from lib.transforms.transforms import SpatialCropByRoiD, WriteCrop
from monailabel.deepedit.transforms import AddGeodisTKSignald, SplitPredsOtherd
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored


class DeepEdit(BasicInferTask):
    """
    This provides Inference Engine for pre-trained model over Multi Atlas Labeling Beyond The Cranial Vault (BTCV)
    dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPEDIT,
        labels=None,
        dimension=3,
        spatial_size=(128, 128, 64),
        target_spacing=(1.0, 1.0, 1.0),
        number_intensity_ch=1,
        description="A DeepEdit model for volumetric (3D) segmentation over 3D Images",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="image",
            output_label_key="pred",
            output_json_key="result",
            **kwargs,
        )

        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.number_intensity_ch = number_intensity_ch

    def pre_transforms(self, data=None):
        t = [
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            EnsureChannelFirstd(keys=("image", "label")),
            # Orientationd(keys="image", axcodes="RAS"),
            ScaleIntensityd(keys="image"),
            SpatialCropByRoiD(keys=["image", "label"]),
            # ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        ]

        self.add_cache_transform(t, data)

        if self.type == InferType.DEEPEDIT:
            t.extend(
                [
                    # LoadGuidanceFromJsonFiled(guidance=("firstpoint_guidances", "label_guidances", "random_guidances"), file_path="/Users/zyc/Desktop/DESKTOP/MONAILabel0.4/sample-apps/radiology/aaa.json"),
                    AddGeodisTKSignald(keys="image", guidance="firstpoint", lamb=0.05, iter=4, number_intensity_ch=1),
                    AddGuidanceFromPointsDeepEditd(ref_image="image", guidance="guidance", label_names=self.labels),
                    # Resized(keys="image", spatial_size=self.spatial_size, mode="area"),
                    # ResizeGuidanceMultipleLabelDeepEditd(guidance="guidance", ref_image="image"),
                    AddGuidanceSignalDeepEditd(
                        keys="image", guidance="guidance", number_intensity_ch=self.number_intensity_ch
                    ),
                ]
            )
        else:
            t.extend(
                [
                    Resized(keys="image", spatial_size=self.spatial_size, mode="area"),
                    DiscardAddGuidanced(
                        keys="image", label_names=self.labels, number_intensity_ch=self.number_intensity_ch
                    ),
                ]
            )

        t.append(EnsureTyped(keys="image", device=data.get("device") if data else None))
        return t

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        if self.type == InferType.DEEPEDIT:
            t = [
                SplitPredsOtherd(keys="pred", other_name="resultfirst"),
                EnsureTyped(keys="pred", device=data.get("device") if data else None),
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=(2, 2)),  # label:(2, 128, 128, 48)
                EnsureChannelFirstd(keys=("label", "pred")),  # pred:(1, 2, 128, 128, 48)  label:(1, 2, 128, 128, 48)
                ToNumpyd(keys="pred"),
                # writer前必须要使用restore
                Restored(keys=["image", "label", "pred"], ref_image="image"),
                WriteCrop(keys=["image", "label", "pred"],
                          location_tag=["images_crop", "labels_crop_monai", "test_labels_monai"]),
                SqueezeDimd(keys=("pred", "label"), dim=0),  # pred:(2, 128, 128, 48) label:(2, 128, 128, 48)
                AsDiscreted(keys=("pred", "label"), argmax=True),  # pred:(1, 128, 128, 48) label:(1, 128, 128, 48)
                SqueezeDimd(keys=("pred", "label"), dim=0),  # pred:(128, 128, 48) label:(128, 128, 48)
                WriteCrop(keys=["label"], location_tag=["labels_crop"]),
            ]
        else:  # InferType.SEGMENTATION
            t = [
                EnsureTyped(keys="pred", device=data.get("device") if data else None),  # image: (3, 128, 128, 48) label: (1, 128, 128, 48) pred: (2, 128, 128, 48)
                Activationsd(keys="pred", softmax=True),  # image: (3, 128, 128, 48) label: (1, 128, 128, 48) pred: (2, 128, 128, 48)
                AsDiscreted(keys="pred", argmax=True),  # image: (3, 128, 128, 48) label: (1, 128, 128, 48) pred: (1, 128, 128, 48)
                SqueezeDimd(keys="pred", dim=0),  # image: (3, 128, 128, 48) label: (1, 128, 128, 48) pred: (128, 128, 48)
                ToNumpyd(keys="pred"),
                Restored(keys="pred", ref_image="image"),
            ]
        return t


