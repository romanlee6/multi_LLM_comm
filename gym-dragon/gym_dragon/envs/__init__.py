from ..core import Region
from ..dragon import DragonBaseEnv, MiniDragonBaseEnv
import numpy as np


class DragonEnv(DragonBaseEnv):
    """
    Full Dragon environment.
    """
    pass


class ForestEnv(DragonBaseEnv):
    """
    Subset of Dragon environment over the "forest" region.
    """
    def __init__(self, **kwargs):
        super().__init__(valid_regions=[Region.forest], **kwargs)


class VillageEnv(DragonBaseEnv):
    """
    Subset of Dragon environment over the "village" region.
    """
    def __init__(self, **kwargs):
        super().__init__(valid_regions=[Region.village], **kwargs)


class DesertEnv(DragonBaseEnv):
    """
    Subset of Dragon environment over the "desert" region.
    """
    def __init__(self, **kwargs):
        super().__init__(valid_regions=[Region.desert], **kwargs)

class MiniDragonEnv(MiniDragonBaseEnv):
    """
    Subset of Dragon environment over the "desert" region.
    """
    def __init__(self, **kwargs):
        super().__init__(valid_regions=[Region.village], **kwargs)

class MiniDragonRandomEnv(MiniDragonBaseEnv):
    """
    Subset of Dragon environment over the "desert" region.
    """
    def __init__(self, **kwargs):
        super().__init__(valid_regions=[Region.village], **kwargs)
        self.seed(np.random.default_rng())
