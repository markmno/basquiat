import abc
from typing import Self


class Txt2ImgModule(abc.ABC):
    @abc.abstractmethod
    def create_img(self, *args, **kwargs):
        pass
