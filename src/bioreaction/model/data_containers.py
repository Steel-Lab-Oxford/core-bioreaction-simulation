from typing import List, Tuple
#from utils.data.load import Data
import numpy as np
import chex


class Species():
    """
    A chemical, protein, or other item which we desire to model
    the amount of over time.
    """
    def __init__(self, name: str) -> None:
        self.name : str = name
        #self.identifier : str
        #self.data : Data


class Reaction():
    """
    Some process which converts species into other 
    species.
    """
    def __init__(self) -> None:
        self.input : List[Species]
        self.output : List[Species]
        self.base_rate : float
        #self.environmental: List[Tuple[int,Extrinsics]]


@chex.dataclass
class Reactions():

    def __init__(self) -> None:
        # Inputs, one hot, each row being a different reaction
        #  each column is a species
        inputs : chex.ArrayDevice
        # Outputs * rates. Each row is a different reaction
        #  each column is a species
        output_rates : chex.ArrayDevice

class Extrinsics():
    """
    Other factors which we desire to model, which are relevent to our
    process.
    """
    def __init__(self) -> None:
        pass


class BasicModel():
    """
    A class representing a collection of species, reactions, and other facts.
    This should represent the abstract notion of some mathematic model of a system.
    """
    def __init__(self) -> None:

        self.species : List[Species]
        self.reactions : List[Reaction]
        