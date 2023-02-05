'''
  @ Date: 2023/2/4 22:07
  @ Author: Zhao YaChen
'''
import logging

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.strategy import Strategy

logger = logging.getLogger(__name__)


class Search(Strategy):
    """
    Consider implementing a search strategy for active learning
    """

    def __init__(self):
        super().__init__("Get Search Sample")

    def __call__(self, request, datastore: Datastore):
        images = datastore.get_images()
        if not len(images):
            return None

        patient_name = request["patient_name"]
        image = [i for i in images if i == patient_name][0]

        logger.info(f"First: Selected Image: {image}")
        return {"id": image}
