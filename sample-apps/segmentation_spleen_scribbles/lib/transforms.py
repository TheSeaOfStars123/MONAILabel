import logging

import numpy as np
from monai.config import KeysCollection
from monai.transforms import SpatialCrop, MapTransform
from scipy import ndimage

from monailabel.scribbles.transforms import InteractiveSegmentationTransform

from .utils import make_mideepnd_unary

logger = logging.getLogger(__name__)

class MakeMIDeepEGDUnaryd(InteractiveSegmentationTransform):
    def __init__(
        self,
        image: str,
        logits: str,
        scribbles: str,
        meta_key_postfix: str = "meta_dict",
        unary: str = "unary",
        tau: float = 1.0,
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
    ) -> None:
        super(MakeMIDeepEGDUnaryd, self).__init__(meta_key_postfix)
        self.image = image
        self.logits = logits
        self.scribbles = scribbles
        self.unary = unary
        self.tau = tau
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label

    def _get_spacing(self, d, key):
        spacing = None
        src_key = "_".join([key, self.meta_key_postfix])
        if src_key in d.keys() and "affine" in d[src_key]:
            spacing = (np.sqrt(np.sum(np.square(d[src_key]["affine"]), 0))[
                       :-1]).astype(np.float32)

        return spacing

    def __call__(self, data):
        d = dict(data)

        # copy affine meta data from image input
        d = self._copy_affine(d, src=self.image, dst=self.unary)

        # read relevant terms from data
        image = self._fetch_data(d, self.image)
        logits = self._fetch_data(d, self.logits)
        scribbles = self._fetch_data(d, self.scribbles)

        # check if input logits are compatible with MIDeepSeg opt
        if logits.shape[0] > 2:
            raise ValueError(
                "MIDeepSeg can only be applied to binary probabilities for now, received {}".format(
                    logits.shape[0])
            )

        # attempt to unfold probability term
        logits = self._normalise_logits(logits, axis=0)
        spacing = self._get_spacing(d, self.image)

        unary_term = make_mideepnd_unary(
            image=image,
            prob=logits,
            scribbles=scribbles,
            scribbles_fg_label=self.scribbles_fg_label,
            scribbles_bg_label=self.scribbles_bg_label,
            spacing=spacing,
            tau=self.tau,
        )
        d[self.unary] = unary_term

        return d


class SpatialCropByRoiD(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        ORI_SHAPE = d["label"].squeeze().shape
        CM = list(map(int, ndimage.measurements.center_of_mass(d['label'].squeeze())))
        # 定义偏移量
        offsetX = 64
        offsetY = 64
        offsetZ = 24
        if CM[0] - offsetX < 0:
            delta = 0 - (CM[0] - offsetX)
            CM[0] += delta
        elif CM[0] + offsetX > ORI_SHAPE[0]:
            delta = CM[0] + offsetX - ORI_SHAPE[0]
            CM[0] -= delta
        if CM[1] - offsetY < 0:
            delta = 0 - (CM[1] - offsetY)
            CM[1] += delta
        elif CM[1] + offsetY > ORI_SHAPE[1]:
            delta = CM[1] + offsetY - ORI_SHAPE[1]
            CM[1] -= delta
        if CM[2] - offsetZ < 0:
            delta = 0 - (CM[2] - offsetZ)
            CM[2] += delta
        elif CM[2] + offsetZ > ORI_SHAPE[2]:
            delta = CM[2] + offsetZ - ORI_SHAPE[2]
            CM[2] -= delta
        # CM = (448, 235, 64)  # Breast_Traning_271
        for key in self.keys:
            img = d[key]
            crop = SpatialCrop(roi_center=CM, roi_size=(128, 128, 48))
            d[key] = crop(img)
            logger.info("Processing label: " + d["label_meta_dict"]["filename_or_obj"] + "->" + str(d["label"].shape))
        return d