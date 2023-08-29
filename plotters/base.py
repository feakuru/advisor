from typing import Type
import plotly.graph_objects as go

from databuilders.base import BaseDatabuilder


class BasePlotter:
    def __init__(
        self,
        databuilder: BaseDatabuilder | None = None,
        databuilder_class: Type[BaseDatabuilder] = BaseDatabuilder,
    ):
        self.databuilder_class = databuilder_class
        if databuilder is not None:
            if not isinstance(databuilder, self.databuilder_class):
                raise TypeError(
                    f'Expected {self.databuilder_class}, '
                    f'got {type(databuilder)}'
                )
            self.databuilder = databuilder
        else:
            self.databuilder = self.databuilder_class()

    def get_plot(self, **kwargs) -> go.Figure:
        raise NotImplementedError
