# @Time : 2022/10/20 10:01 PM 
# @Author : zyc
# @File : segmentation_breast.py 
# @Title :
# @Description :
from typing import Sequence, Callable

from monai.transforms import LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ScaleIntensityd, EnsureTyped, \
    Activationsd, AsDiscreted, ToNumpyd, Spacingd

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored, BoundingBoxd


class SegmentationBreast(InferTask):
    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        spatial_size=(128, 128, 48),
        target_spacing=(1.0, 1.0, 1.0),
        number_intensity_ch=1,
        description="A pre-trained model for volumetric (3D) breast segmentation over 3D Images",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )

        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.number_intensity_ch = number_intensity_ch

    def pre_transforms(self, data=None) -> Sequence[Callable]:
         return [
            LoadImaged(keys="image", reader="ITKReader"),
            EnsureChannelFirstd(keys="image"),
            # Spacingd(keys="image", pixdim=self.target_spacing),
            # Orientationd(keys="image", axcodes="RAS"),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys="image"),
        ]

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
        ]