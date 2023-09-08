import typing as t
from dataclasses import dataclass

import tensorflow as tf

from advisors.base import BaseAdvisor, BaseAdvisorParams
from common.utils import get_logger


logger = get_logger('advisor-lstm')


@dataclass
class LSTMAdvisorParams(BaseAdvisorParams):
    layers: int = 1


class LSTMAdvisor(BaseAdvisor):
    params: LSTMAdvisorParams
    params_class = LSTMAdvisorParams

    def get_model_layers(self) -> t.List[tf.keras.layers.Layer]:
        layers = []

        for _ in range(self.params.layers):
            layers.append(tf.keras.layers.LSTM(32, return_sequences=True))
        layers.append(tf.keras.layers.Dense(1))

        return layers
