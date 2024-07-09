from .builder import DATASETS
from .custom import CustomDataset
from collections import OrderedDict


@DATASETS.register_module()
class YAMAHADataset(CustomDataset):
    """RELLIS dataset.
- 0: void
  1: dirt
  3: grass
  4: tree
  5: pole
  6: water
  7: sky
  8: vehicle
  9: object
  10: asphalt
  12: building
  15: log
  17: person
  18: fence
  19: bush
  23: concrete
  27: barrier
  31: puddle
  33: mud
  34: rubble
    """


    LABEL_TO_COLOR = OrderedDict({
        "background": [255, 255, 255],
        "high_vegetation": [40, 80, 0],
        "traversable_grass": [128, 255, 0],
        "smooth_trail": [178, 176, 153],
        "obstacle": [255, 0, 0],
        "sky": [1, 88, 255],
        "rough_trial": [156, 76, 30],
        "puddle": [255, 0, 128],
        "non_traversable_low_vegetation": [0, 160, 0]
    })

    PALETTE = list(LABEL_TO_COLOR.values())
    CLASSES = list(LABEL_TO_COLOR.keys())


    # CLASSES = ("void", "dirt", "grass", "tree", "pole", "water", "sky", "vehicle",
    #         "object", "asphalt", "building", "log", "person", "fence", "bush",
    #         "concrete", "barrier", "puddle", "mud", "rubble")

    # PALETTE = [[0, 0, 0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153],
    #         [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64],
    #         [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204],
    #         [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]

    def __init__(self, **kwargs):
        super(YAMAHADataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.jpg',
            **kwargs)
        self.CLASSES = ("background",
                        "high_vegetation",
                        "traversable_grass",
                        "smooth_trail",
                        "obstacle",
                        "sky",
                        "rough_trial",
                        "puddle",
                        "non_traversable_low_vegetation")
        self.PALETTE = [[255, 255, 255],
                            [40, 80, 0],
                            [128, 255, 0],
                            [178, 176, 153],
                            [255, 0, 0],
                            [1, 88, 255],
                            [156, 76, 30],
                            [255, 0, 128],
                            [0, 160, 0]]
