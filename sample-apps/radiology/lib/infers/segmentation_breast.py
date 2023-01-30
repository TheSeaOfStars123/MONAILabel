from typing import Sequence, Callable

from monai.inferers import SlidingWindowInferer, SimpleInferer
from monai.transforms import LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ScaleIntensityd, EnsureTyped, \
    Activationsd, AsDiscreted, ToNumpyd, Spacingd, SelectItemsd, ToTensord, SqueezeDimd

from lib.transforms.transforms import SpatialCropByRoiD, WriteCrop
from monailabel.interfaces.tasks.infer import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored, BoundingBoxd


class SegmentationBreast(BasicInferTask):
    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        enviroment="dev",
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

        self.enviroment = enviroment
        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.number_intensity_ch = number_intensity_ch

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        if self.enviroment == "dev":
            t = [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=("image", "label")),
                ScaleIntensityd(keys="image"),
                EnsureTyped(keys="image"),
                SpatialCropByRoiD(keys=["image", "label"]),
            ]
        elif self.enviroment == "prod":
            t = [
                LoadImaged(keys="image", reader="ITKReader"),
                EnsureChannelFirstd(keys="image"),
                # Spacingd(keys="image", pixdim=self.target_spacing),
                # Orientationd(keys="image", axcodes="RAS"),
                ScaleIntensityd(keys="image"),
                EnsureTyped(keys="image"),
            ]
        return t

    def inferer(self, data=None):
        if self.enviroment == "dev":
            return SimpleInferer()
        elif self.enviroment == "prod":
            return SlidingWindowInferer(roi_size=(128, 128, 48), sw_batch_size=1, overlap=0.25)

    def post_transforms(self, data=None) -> Sequence[Callable]:
        if self.enviroment == "dev":
            t = [
                EnsureTyped(keys="pred", device=data.get("device") if data else None),  # pred:(2, 128, 128, 48)  label:(1, 128, 128, 48)
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=(2, 2)),  # label:(2, 128, 128, 48)
                EnsureChannelFirstd(keys=("label", "pred")),  # pred:(1, 2, 128, 128, 48)  label:(1, 2, 128, 128, 48)
                ToNumpyd(keys="pred"),
                # writer前必须要使用restore
                Restored(keys=["image", "label", "pred"], ref_image="image"),
                WriteCrop(keys=["image", "label", "pred"], location_tag=["images_crop", "labels_crop_monai", "test_labels_monai"]),
                SqueezeDimd(keys=("pred", "label"), dim=0),  # pred:(2, 128, 128, 48) label:(2, 128, 128, 48)
                AsDiscreted(keys=("pred", "label"), argmax=True),  # pred:(1, 128, 128, 48) label:(1, 128, 128, 48)
                SqueezeDimd(keys=("pred", "label"), dim=0),  # pred:(128, 128, 48) label:(128, 128, 48)
                WriteCrop(keys=["label"], location_tag=["labels_crop"]),
                # BoundingBoxd(keys="pred", result="result", bbox="bbox"),
            ]
        elif self.enviroment == "prod":
            t = [
                EnsureTyped(keys="pred", device=data.get("device") if data else None), # pred:(2, 512, 512, 112)
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys="pred", argmax=True),
                ToNumpyd(keys="pred"),
                Restored(keys="pred", ref_image="image"),
                BoundingBoxd(keys="pred", result="result", bbox="bbox"),
            ]
        return t