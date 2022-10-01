

from typing import List
from utils.data.load import Data
import numpy as np


class Species():

    def __init__(self) -> None:
        self.name : str
        self.identifier : str
        self.data : Data


class Extrinsics(Species):

    def __init__(self) -> None:
        pass


class Reaction():

    def __init__(self) -> None:
        
        self.input : List[Species]
        self.output : List[Species]
        self.base_rate : np.ndarray
        self.environmental: List[Extrinsics]

