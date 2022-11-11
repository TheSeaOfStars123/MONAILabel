from typing import Sequence, Callable

from monai.transforms import LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ScaleIntensityd, EnsureTyped, \
    Activationsd, AsDiscreted, ToNumpyd, Spacingd, SelectItemsd, ToTensord, SqueezeDimd

from lib.transforms.transforms import SpatialCropByRoiD, WriteCrop
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
             # LoadImaged(keys="image", reader="ITKReader"),
             # EnsureChannelFirstd(keys="image"),
             # # Spacingd(keys="image", pixdim=self.target_spacing),
             # # Orientationd(keys="image", axcodes="RAS"),
             # ScaleIntensityd(keys="image"),
             # EnsureTyped(keys="image"),
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys="image"),
            SpatialCropByRoiD(keys=["image", "label"]),
        ]

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            # ApplyGraphCutOptimisationd(unary="pred",
            #                            pairwise="image",
            #                            post_proc_label="pred",
            #                            lamda=8.0,
            #                            sigma=0.1),
            EnsureTyped(keys="pred", device=data.get("device") if data else None),  # pred:(2, 128, 128, 48)
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),  # pred:(1, 128, 128, 48)
            # SqueezeDimd(keys="pred", dim=0),  # pred:(128, 128, 48)
            ToNumpyd(keys="pred"),
            # writer前必须要使用restore
            Restored(keys=["image", "label", "pred"], ref_image="image"),
            WriteCrop(keys=["image", "label"], location_tag=["images_crop", "labels_crop"])
            # BoundingBoxd(keys="pred", result="result", bbox="bbox"),
        ]