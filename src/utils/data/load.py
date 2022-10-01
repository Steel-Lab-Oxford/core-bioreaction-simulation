

from typing import Any


class Data():

    def __init__(self) -> None:
        
        self.origin : str
        self.data : Any
        self.config: dict


class Loader():

    def __init__(self) -> None:
        pass

    def load(self, filename) -> Data:
        pass

    def preprocess(self):
        pass