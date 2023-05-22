import abc
from typing import Self


class Img2TxtModule(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_config(cls, config) -> Self:
        pass

    @abc.abstractmethod
    def generate_txt(self, *args, **kwargs):
        pass
