
from typing import List, Tuple
from utils.data.load import Data
import numpy as np


class Species():
    def __init__(self) -> None:
        self.name : str
        self.identifier : str
        self.data : Data


class Extrinsics():
    def __init__(self) -> None:
        pass

class Reaction():
    def __init__(self) -> None:
        self.input : List[Species]
        self.output : List[Species]
        self.base_rate : float
        #self.environmental: List[Tuple[int,Extrinsics]]

