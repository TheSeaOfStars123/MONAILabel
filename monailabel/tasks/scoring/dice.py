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

import numpy as np
import torch
from monai.metrics import compute_dice, DiceMetric
from monai.transforms import LoadImage

from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.interfaces.tasks.scoring import ScoringMethod

logger = logging.getLogger(__name__)


class Dice(ScoringMethod):
    """
    Compute dice between final vs original tags
    """

    def __init__(self):
        super().__init__("Compute Dice for predicated label vs submitted")

    def __call__(self, request, datastore: Datastore):
        loader = LoadImage(image_only=True)

        tag_y = request.get("y", DefaultLabelTag.FINAL)
        tag_y_pred = request.get("y_pred", DefaultLabelTag.ORIGINAL)

        result = {}
        sum = 0
        i = 0
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        for image_id in datastore.list_images():
            y_i = datastore.get_label_by_image_id(image_id, tag_y) if tag_y else None
            y_pred_i = datastore.get_label_by_image_id(image_id, tag_y_pred) if tag_y_pred else None

            if y_i and y_pred_i:
                y = loader(datastore.get_label_uri(y_i, tag_y))
                y_pred = loader(datastore.get_label_uri(y_pred_i, tag_y_pred))
                # compute metric for current iteration
                dice_metric(y_pred=y_pred, y=y)
                y = y.flatten()
                if isinstance(y, torch.Tensor):
                    y = y.numpy()
                y_pred = y_pred.flatten()
                if isinstance(y_pred, torch.Tensor):
                    y_pred = y_pred.numpy()
                union = np.sum(y) + np.sum(y_pred)
                dice = 2.0 * np.sum(y * y_pred) / union if union != 0 else 1
                logger.info(f"Dice Score for {image_id} is {dice}")
                sum += dice
                i+=1
                datastore.update_image_info(image_id, {"dice": dice})
                result[image_id] = dice
        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()
        logger.info(f"Avg Dice Score is {sum / i}")
        logger.info(f"Avg DiceMonai Score is {metric}")
        return result
