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

import json
import logging
import os
from typing import Dict

import pandas as pd

import lib.configs
from lib.activelearning import Last
from lib.infers.deepgrow_pipeline import InferDeepgrowPipeline
from lib.infers.vertebra_pipeline import InferVertebraPipeline

import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.scribbles.infer import GMMBasedGraphCut, HistogramBasedGraphCut
from monailabel.tasks.activelearning.first import First
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.train.basic_train import Context
from monailabel.utils.others.class_utils import get_class_names
from monailabel.utils.others.generic import strtobool
from monailabel.utils.others.planner import HeuristicPlanner

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")

        configs = {}
        for c in get_class_names(lib.configs, "TaskConfig"):
            name = c.split(".")[-2].lower()
            configs[name] = c

        configs = {k: v for k, v in sorted(configs.items())}

        models = conf.get("models")
        if not models:
            print("")
            print("---------------------------------------------------------------------------------------")
            print("Provide --conf models <name>")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        models = models.split(",")
        models = [m.strip() for m in models]
        invalid = [m for m in models if m != "all" and not configs.get(m)]
        if invalid:
            print("")
            print("---------------------------------------------------------------------------------------")
            print(f"Invalid Model(s) are provided: {invalid}")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        # Use Heuristic Planner to determine target spacing and spatial size based on dataset+gpu
        spatial_size = json.loads(conf.get("spatial_size", "[48, 48, 32]"))
        target_spacing = json.loads(conf.get("target_spacing", "[1.0, 1.0, 1.0]"))
        self.heuristic_planner = strtobool(conf.get("heuristic_planner", "false"))
        self.planner = HeuristicPlanner(spatial_size=spatial_size, target_spacing=target_spacing)

        self.models: Dict[str, TaskConfig] = {}
        for n in models:
            for k, v in configs.items():
                if self.models.get(k):
                    continue
                if n == k or n == "all":
                    logger.info(f"+++ Adding Model: {k} => {v}")
                    self.models[k] = eval(f"{v}()")
                    self.models[k].init(k, self.model_dir, conf, self.planner)

        logger.info(f"+++ Using Models: {list(self.models.keys())}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - Radiology ({monailabel.__version__})",
            description="DeepLearning models for radiology",
            version=monailabel.__version__,
        )

    def init_datastore(self) -> Datastore:
        datastore = super().init_datastore()
        if self.heuristic_planner:
            self.planner.run(datastore)
        return datastore

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}
        #################################################
        # Models
        #################################################
        for n, task_config in self.models.items():
            c = task_config.infer()
            c = c if isinstance(c, dict) else {n: c}
            for k, v in c.items():
                logger.info(f"+++ Adding Inferer:: {k} => {v}")
                infers[k] = v

        #################################################
        # Scribbles
        #################################################
        infers.update(
            {
                "Histogram+GraphCut": HistogramBasedGraphCut(
                    intensity_range=(-300, 200, 0.0, 1.0, True),
                    pix_dim=(2.5, 2.5, 5.0),
                    lamda=1.0,
                    sigma=0.1,
                    num_bins=64,
                    labels=task_config.labels,
                ),
                "GMM+GraphCut": GMMBasedGraphCut(
                    intensity_range=(-300, 200, 0.0, 1.0, True),
                    pix_dim=(2.5, 2.5, 5.0),
                    lamda=5.0,
                    sigma=0.5,
                    num_mixtures=20,
                    labels=task_config.labels,
                ),
            }
        )

        #################################################
        # Pipeline based on existing infers
        #################################################
        if infers.get("deepgrow_2d") and infers.get("deepgrow_3d"):
            infers["deepgrow_pipeline"] = InferDeepgrowPipeline(
                path=self.models["deepgrow_2d"].path,
                network=self.models["deepgrow_2d"].network,
                model_3d=infers["deepgrow_3d"],
                description="Combines Clara Deepgrow 2D and 3D models",
            )

        #################################################
        # Pipeline based on existing infers for vertebra segmentation
        # Stages:
        # 1/ localization spine
        # 2/ localization vertebra
        # 3/ segmentation vertebra
        #################################################
        if (
            infers.get("localization_spine")
            and infers.get("localization_vertebra")
            and infers.get("segmentation_vertebra")
        ):
            infers["vertebra_pipeline"] = InferVertebraPipeline(
                task_loc_spine=infers["localization_spine"],  # first stage
                task_loc_vertebra=infers["localization_vertebra"],  # second stage
                task_seg_vertebra=infers["segmentation_vertebra"],  # third stage
                description="Combines three stage for vertebra segmentation",
            )
        logger.info(infers)
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        trainers: Dict[str, TrainTask] = {}
        if strtobool(self.conf.get("skip_trainers", "false")):
            return trainers

        for n, task_config in self.models.items():
            t = task_config.trainer()
            if not t:
                continue

            logger.info(f"+++ Adding Trainer:: {n} => {t}")
            trainers[n] = t
        return trainers

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {
            "random": Random(),
            "first": First(),
            "last": Last(),
        }

        if strtobool(self.conf.get("skip_strategies", "true")):
            return strategies

        for n, task_config in self.models.items():
            s = task_config.strategy()
            if not s:
                continue
            s = s if isinstance(s, dict) else {n: s}
            for k, v in s.items():
                logger.info(f"+++ Adding Strategy:: {k} => {v}")
                strategies[k] = v

        logger.info(f"Active Learning Strategies:: {list(strategies.keys())}")
        return strategies

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        methods: Dict[str, ScoringMethod] = {}
        if strtobool(self.conf.get("skip_scoring", "true")):
            return methods

        for n, task_config in self.models.items():
            s = task_config.scoring_method()
            if not s:
                continue
            s = s if isinstance(s, dict) else {n: s}
            for k, v in s.items():
                logger.info(f"+++ Adding Scoring Method:: {k} => {v}")
                methods[k] = v

        logger.info(f"Active Learning Scoring Methods:: {list(methods.keys())}")
        return methods


    def breast_partition_datalist(self):
        # default_prefix = 'D:/Desktop/BREAST/BREAST/'
        # default_prefix = '/Users/zyc/Desktop'
        # name_mapping_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping.csv'
        name_mapping_path = os.path.join(os.path.dirname(__file__), 'breast_name_mapping.csv')
        val_datalist = []
        train_datalist = []
        name_mapping_df = pd.read_csv(name_mapping_path, encoding='unicode_escape')
        for idx, data in name_mapping_df.iterrows():
            if data['Exclude'] != 1.0:
                image_id = data['Breast_subject_ID'] + '_ph3'
                file = {}
                file['image'] = self._datastore.get_image_uri(image_id)
                file['label'] = self._datastore.get_label_uri(image_id, "final")
                if file['image'] != "" and file['label'] != "":
                    if data['val_datalist'] == 1.0:
                        val_datalist.append(file)
                    else:
                        train_datalist.append(file)
        return train_datalist, val_datalist

"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    import argparse
    import shutil
    from pathlib import Path

    from monailabel.utils.others.generic import device_list, file_ext

    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    home = str(Path.home())
    studies = f"{home}/Dataset/Radiology"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default=studies)
    parser.add_argument("-m", "--model", default="segmentation_breast")
    parser.add_argument("-t", "--test", default="infer", choices=("train", "infer", "scoring"))
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies
    conf = {
        "models": args.model,
        "preload": "false",
        "skip_scoring": "false",
        "network": "unet"
    }

    app = MyApp(app_dir, studies, conf)

    # Infer
    if args.test == "infer":
        for image_id in app._datastore.list_images():
            # sample = app.next_sample(request={"strategy": "first"})
            # image_id = sample["id"]
            image_path = app._datastore.get_image_uri(image_id)

            # Run on all devices
            for device in device_list():
                res = app.infer(
                    request={
                        "model": args.model,
                        "image": image_id,
                        "device": device
                    }
                )
                # res = app.infer(
                #     request={"model": "vertebra_pipeline", "image": image_id, "device": device, "slicer": False}
                # )
                label = res["file"]
                label_json = res["params"]
                test_dir = os.path.join(args.studies, "test_labels")
                os.makedirs(test_dir, exist_ok=True)

                label_file = os.path.join(test_dir, image_id + file_ext(image_path))
                shutil.move(label, label_file)

                print(label_json)
                print(f"++++ Image File: {image_path}")
                print(f"++++ Label File: {label_file}")
        return

    # Train
    if args.test == "train":
        # 数据集划分
        train_ds, val_ds = app.breast_partition_datalist()
        train_ds_json = os.path.join(studies, 'breast_train_ds.json')
        val_ds_json = os.path.join(studies, 'breast_val_ds.json')
        with open(train_ds_json, "w") as fp:
            json.dump(train_ds, fp, indent=2)
        with open(val_ds_json, "w") as fp:
            json.dump(val_ds, fp, indent=2)

        app.train(
            request={
                "model": args.model,
                "max_epochs": 20,
                "dataset": "Dataset",  # PersistentDataset, CacheDataset
                "train_batch_size": 4,
                "val_batch_size": 1,
                "multi_gpu": True,
                "val_split": 0.1,
                "train_ds": train_ds_json,
                "val_ds": val_ds_json,
            },
        )
        return

    # Validation
    if args.test == "scoring":
        res = app.scoring(
            request={
                "method": "dice",
                "y": "labels_crop_human",
                "y_pred": "test_labels_human",
            }
        )
        print(res)
        logger.info("All Done!")
        return


if __name__ == "__main__":
    # export PYTHONPATH=D:\Desktop\MONAILabel0.4
    # python main.py
    main()
