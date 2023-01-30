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
import GeodisTK
import json
import logging
import os
import time
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Union

import numpy as np
import torch
from monai.config import IndexSelection, KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, Randomizable, Resize, SpatialCrop, generate_spatial_bounding_box, is_positive
from monai.utils import InterpolateMode, PostFix, ensure_tuple_rep
from scipy.ndimage import distance_transform_cdt, gaussian_filter, center_of_mass
from skimage import measure


logger = logging.getLogger(__name__)


class AddClickGuidanced(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, guidance="guidance"):
        super().__init__(keys, allow_missing_keys)
        self.guidance = guidance

    def __call__(self, data):
        d = dict(data)
        guidance = []
        for key in self.keys:
            g = d.get(key)
            g = np.array(g).astype(int).tolist() if g else []
            guidance.append(g)

        d[self.guidance] = guidance
        return d


class AddInitialSeedPointd(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, label="label", connected_regions=1):
        super().__init__(keys, allow_missing_keys)

        self.label = label
        self.connected_regions = connected_regions

    def _apply(self, label):
        default_guidance = [-1] * len(label.shape)

        if self.connected_regions > 1:
            blobs_labels = measure.label(label, background=0)
            u, count = np.unique(blobs_labels, return_counts=True)
            count_sort_ind = np.argsort(-count)
            connected_regions = u[count_sort_ind].astype(int).tolist()

            connected_regions = [r for r in connected_regions if r]
            connected_regions = connected_regions[: self.connected_regions]
        else:
            blobs_labels = None
            connected_regions = [1]

        pos_guidance = []
        for region in connected_regions:
            label = label if blobs_labels is None else (blobs_labels == region).astype(int)
            if np.sum(label) == 0:
                continue

            distance = distance_transform_cdt(label).flatten()
            probability = np.exp(distance) - 1.0

            idx = np.where(label.flatten() > 0)[0]
            seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = distance[seed]

            g = np.asarray(np.unravel_index(seed, label.shape)).transpose().tolist()[0]
            g[0] = dst[0]  # for debug
            pos_guidance.append(g)

        return np.asarray([pos_guidance, [default_guidance] * len(pos_guidance)]).astype(int, copy=False).tolist()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = json.dumps(self._apply(d[self.label]))
        return d


class AddGuidanceSignald(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        guidance: str = "guidance",
        sigma: int = 2,
        number_intensity_ch=3,
    ):
        super().__init__(keys, allow_missing_keys)

        self.guidance = guidance
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch

    def signal(self, shape, points):
        signal = np.zeros(shape, dtype=np.float32)
        flag = False
        for p in points:
            if np.any(np.asarray(p) < 0):
                continue
            if len(shape) == 3:
                signal[int(p[-3]), int(p[-2]), int(p[-1])] = 1.0
            else:
                signal[int(p[-2]), int(p[-1])] = 1.0
            flag = True

        if flag:
            signal = gaussian_filter(signal, sigma=self.sigma)
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        return torch.Tensor(signal)[None]

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]

            guidance = d[self.guidance]
            guidance = json.loads(guidance) if isinstance(guidance, str) else guidance
            if guidance and (guidance[0] or guidance[1]):
                img = img[0 : 0 + self.number_intensity_ch, ...]

                shape = img.shape[-2:] if len(img.shape) == 3 else img.shape[-3:]
                device = img.device if isinstance(img, torch.Tensor) else None
                pos = self.signal(shape, guidance[0]).to(device=device)
                neg = self.signal(shape, guidance[1]).to(device=device)
                result = torch.concat([img if isinstance(img, torch.Tensor) else torch.Tensor(img), pos, neg])
            else:
                s = torch.zeros_like(img[0])[None]
                result = torch.concat([img, s, s])

            d[key] = result
        return d

class AddInitialCenterSeedPointd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "firstpoint",
        connected_regions: int = 5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance_key = guidance
        self.connected_regions = connected_regions
        self.guidance: Dict[str, List[List[int]]] = {}

    def _apply(self, label):
        default_guidance = [-1] * 4
        label = label[0]
        # measure.label: Label connected regions of an integer array
        blobs_labels = measure.label(label.astype(int), background=0)
        firstpoint_guidance = []
        if np.max(blobs_labels) <= 0:
            firstpoint_guidance.append(default_guidance)
        else:
            for ridx in range(1, self.connected_regions + 1):
                label = (blobs_labels == ridx).astype(np.float32)
                if np.sum(label) == 0:
                    firstpoint_guidance.append(default_guidance)
                    continue
                distance = distance_transform_cdt(label).flatten()
                CM = np.round(center_of_mass(label)).astype(np.int32) # 向上取整
                seed = np.ravel_multi_index(CM, label.shape)
                dst = distance[seed]

                # g = np.asarray(np.unravel_index(seed, label.shape))
                # g[0] = dst[0]  # for debug
                firstpoint_guidance.append([dst, CM[-3], CM[-2], CM[-1]])
                logger.info(f"Number of simulated first point click: {CM}")

        return np.asarray(firstpoint_guidance)

    def __call__(self, data):
        d = dict(data)
        firstpoint_guidance = {}
        for key in self.keys:
            if key == "label":
                for key_label in d["label_names"].keys():
                    if key_label != "background":
                        tmp_label = np.copy(d["label"])
                        tmp_label[tmp_label != d["label_names"][key_label]] = 0
                        tmp_label = (tmp_label > 0.5).astype(np.float32)
                        firstpoint_guidance[key_label] = self._apply(tmp_label)

            else:
                print("This transform only applies to label key")
        d[self.guidance_key] = firstpoint_guidance.copy()
        if os.path.getsize("/Users/zyc/Desktop/DESKTOP/MONAILabel0.4/sample-apps/radiology/aaa.json") > 0:
            with open("/Users/zyc/Desktop/DESKTOP/MONAILabel0.4/sample-apps/radiology/aaa.json", 'r') as load_f:
                load_dict_list = json.load(load_f)
        else:
            load_dict_list = []
        basebame = os.path.split(d['image_meta_dict']['filename_or_obj'])[1].split('.')[0]
        for key_label in firstpoint_guidance.keys():
            firstpoint_guidance[key_label] = firstpoint_guidance[key_label].tolist()
        json_backup = {}
        json_backup[basebame+"_firstpoint_guidances"] = firstpoint_guidance
        load_dict_list.append(json_backup)
        with open("/Users/zyc/Desktop/DESKTOP/MONAILabel0.4/sample-apps/radiology/aaa.json", "w") as f:
            json.dump(load_dict_list, f)
        return d

class AddGeodisTKSignald(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "firstpoint",
        lamb: int = 0.05,
        iter: int = 4,
        number_intensity_ch: int = 1,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance = guidance
        self.lamb = lamb
        self.iter = iter
        self.number_intensity_ch = number_intensity_ch

    def _get_spacing(self, meta_dict):
        spacing = (np.sqrt(np.sum(np.square(meta_dict["affine"]).numpy(), 0))[
                   :-1]).astype(np.float32)
        return spacing

    def _get_signal(self, image, guidance, meta_info):
        dimensions = 3
        guidance = guidance.tolist()

        if len(guidance):
            if dimensions == 3:
                # Assume channel is first and depth is last CHWD
                signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)

            sshape = signal.shape
            for point in guidance:
                if np.any(np.asarray(point) < 0):
                    continue
                if dimensions == 3:
                    # Making sure points fall inside the image dimension
                    p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                    p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2, p3] = 1.0
            # Apply a GeodisTK to the signal
            if np.max(signal[0]) > 0:
                t1 = time.time()
                spacing = self._get_spacing(meta_info)
                S = signal[0].copy().astype(np.uint8)
                D2 = self.geodesic_distance_3d(image[0], S, spacing, 0.05, 4)
                dt2 = time.time() - t1
                print("runtime(s) raster scan   {0:}".format(dt2))
                signal[0] = D2
            return signal

    def geodesic_distance_3d(self, I, S, spacing, lamb, iter):
        '''
        Get 3D geodesic disntance by raser scanning.
        I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
           Type should be np.float32.
        S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
           Type should be np.uint8.
        spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
        lamb: weighting betwween 0.0 and 1.0
              if lamb==0.0, return spatial euclidean distance without considering gradient
              if lamb==1.0, the distance is based on gradient only without using spatial distance
        iter: number of iteration for raster scanning.
        '''
        return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamb, iter)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key == "image":
                image = d[key]
                logging.info(f"Run AddGeodisTKSignald function, image shape is {image.shape}")
                tmp_image = image[0: 0 + self.number_intensity_ch, ...]

                guidance = d[self.guidance]
                for key_label in guidance.keys():
                    signal = self._get_signal(image, guidance[key_label], d['image_meta_dict'])
                    tmp_image = np.concatenate([tmp_image, signal], axis=0)
                    logging.info(f"Run AddGeodisTKSignald function, tmp_image shape is {tmp_image.shape}")
                    if isinstance(d[key], MetaTensor):
                        d[key].array = tmp_image
                    else:
                        d[key] = tmp_image
                return d
            else:
                print("This transform only applies to image key")
        return d


class SpatialCropForegroundd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        spatial_size: Union[Sequence[int], np.ndarray],
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: int = 0,
        allow_smaller: bool = True,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        original_shape_key: str = "foreground_original_shape",
        cropped_shape_key: str = "foreground_cropped_shape",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.source_key = source_key
        self.spatial_size = list(spatial_size)
        self.select_fn = select_fn
        self.channel_indices = channel_indices
        self.margin = margin
        self.allow_smaller = allow_smaller

        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.original_shape_key = original_shape_key
        self.cropped_shape_key = cropped_shape_key

    def __call__(self, data):
        d = dict(data)
        box_start, box_end = generate_spatial_bounding_box(
            d[self.source_key], self.select_fn, self.channel_indices, self.margin, self.allow_smaller
        )

        center = list(np.mean([box_start, box_end], axis=0).astype(int, copy=False))
        current_size = list(np.subtract(box_end, box_start).astype(int, copy=False))

        if np.all(np.less(current_size, self.spatial_size)):
            cropper = SpatialCrop(roi_center=center, roi_size=self.spatial_size)
            box_start = np.array([s.start for s in cropper.slices])
            box_end = np.array([s.stop for s in cropper.slices])
        else:
            cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)

        for key in self.keys:
            image = d[key]
            meta = image.meta
            meta[self.start_coord_key] = box_start
            meta[self.end_coord_key] = box_end
            meta[self.original_shape_key] = d[key].shape

            result = cropper(image)
            meta[self.cropped_shape_key] = result.shape
            d[key] = result
        return d


class RestoreLabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        ref_image: str,
        mode: Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str] = InterpolateMode.NEAREST,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        meta_key_postfix: str = PostFix.meta(),
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        original_shape_key: str = "foreground_original_shape",
        cropped_shape_key: str = "foreground_cropped_shape",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.ref_image = ref_image
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))

        self.meta_key_postfix = meta_key_postfix
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.original_shape_key = original_shape_key
        self.cropped_shape_key = cropped_shape_key

    def __call__(self, data):
        d = dict(data)
        meta_dict = (
            d[self.ref_image].meta
            if isinstance(d[self.ref_image], MetaTensor)
            else d[f"{self.ref_image}_{self.meta_key_postfix}"]
        )

        for key, mode, align_corners in self.key_iterator(d, self.mode, self.align_corners):
            image = d[key]

            # Undo Resize
            current_shape = image.shape
            cropped_shape = meta_dict[self.cropped_shape_key]
            if np.any(np.not_equal(current_shape, cropped_shape)):
                resizer = Resize(spatial_size=cropped_shape[1:], mode=mode)
                image = resizer(image, mode=mode, align_corners=align_corners)

            # Undo Crop
            original_shape = meta_dict[self.original_shape_key][1:]
            result = np.zeros(original_shape, dtype=np.float32)
            box_start = meta_dict[self.start_coord_key]
            box_end = meta_dict[self.end_coord_key]

            spatial_dims = min(len(box_start), len(image.shape[1:]))
            slices = [slice(s, e) for s, e in zip(box_start[:spatial_dims], box_end[:spatial_dims])]
            slices = tuple(slices)
            result[slices] = image.array if isinstance(image, MetaTensor) else image

            d[key] = result
        return d


class SpatialCropGuidanced(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        guidance: str,
        spatial_size,
        margin=20,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        original_shape_key: str = "foreground_original_shape",
        cropped_shape_key: str = "foreground_cropped_shape",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.guidance = guidance
        self.spatial_size = list(spatial_size)
        self.margin = margin

        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.original_shape_key = original_shape_key
        self.cropped_shape_key = cropped_shape_key

    def bounding_box(self, points, img_shape):
        ndim = len(img_shape)
        margin = ensure_tuple_rep(self.margin, ndim)
        for m in margin:
            if m < 0:
                raise ValueError("margin value should not be negative number.")

        box_start = [0] * ndim
        box_end = [0] * ndim

        for di in range(ndim):
            dt = points[..., di]
            min_d = max(min(dt - margin[di]), 0)
            max_d = min(img_shape[di], max(dt + margin[di] + 1))
            box_start[di], box_end[di] = min_d, max_d
        return box_start, box_end

    def __call__(self, data):
        d: Dict = dict(data)
        first_key: Union[Hashable, List] = self.first_key(d)
        if not first_key:
            return d

        guidance = d[self.guidance]
        original_spatial_shape = d[first_key].shape[1:]
        box_start, box_end = self.bounding_box(np.array(guidance[0] + guidance[1]), original_spatial_shape)
        center = list(np.mean([box_start, box_end], axis=0).astype(int, copy=False))
        spatial_size = self.spatial_size

        box_size = list(np.subtract(box_end, box_start).astype(int, copy=False))
        spatial_size = spatial_size[-len(box_size) :]

        if np.all(np.less(box_size, spatial_size)):
            cropper = SpatialCrop(roi_center=center, roi_size=spatial_size)
        else:
            cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)

        # update bounding box in case it was corrected by the SpatialCrop constructor
        box_start = np.array([s.start for s in cropper.slices])
        box_end = np.array([s.stop for s in cropper.slices])

        for key in self.keys:
            image = d[key]
            meta = image.meta
            meta[self.start_coord_key] = box_start
            meta[self.end_coord_key] = box_end
            meta[self.original_shape_key] = d[key].shape

            result = cropper(image)
            result.meta[self.cropped_shape_key] = result.shape
            d[key] = result

        pos_clicks, neg_clicks = guidance[0], guidance[1]
        pos = np.subtract(pos_clicks, box_start).tolist() if len(pos_clicks) else []
        neg = np.subtract(neg_clicks, box_start).tolist() if len(neg_clicks) else []

        d[self.guidance] = [pos, neg]
        return d


class ResizeGuidanced(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        ref_image: str,
        cropped_shape_key: str = "foreground_cropped_shape",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.ref_image = ref_image
        self.cropped_shape_key = cropped_shape_key

    def __call__(self, data):
        d = dict(data)
        current_shape = d[self.ref_image].shape[1:]

        meta = d[self.ref_image].meta
        if self.cropped_shape_key and meta.get(self.cropped_shape_key):
            cropped_shape = meta[self.cropped_shape_key][1:]
        else:
            cropped_shape = meta.get("spatial_shape", current_shape)
        factor = np.divide(current_shape, cropped_shape)

        for key in self.keys:
            guidance = d[key]
            pos_clicks, neg_clicks = guidance[0], guidance[1]
            pos = np.multiply(pos_clicks, factor).astype(int, copy=False).tolist() if len(pos_clicks) else []
            neg = np.multiply(neg_clicks, factor).astype(int, copy=False).tolist() if len(neg_clicks) else []

            d[key] = [pos, neg]
        return d

class SplitPredsOtherd(MapTransform):
    """
    Split preds and others for individual evaluation
    """
    def __init__(
        self,
        keys: KeysCollection,
        other_name: str,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.other_name = other_name

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key == "pred":
                d[self.other_name] = d[key][1]
                d[key] = d[key][0]
        return d
