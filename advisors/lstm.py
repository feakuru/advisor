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

    def get_preprocessing_layers(self):
        return []

    def get_model_layers(self) -> t.List[tf.keras.layers.Layer]:
        layers = []

        lstm_layer_kwargs = [
            {'input_shape': (self.train_features.shape[1], 1)}
        ]
        for _ in range(self.params.layers - 1):
            lstm_layer_kwargs[-1]['return_sequences'] = True
            lstm_layer_kwargs.append({})
        for kwargs in lstm_layer_kwargs:
            layers.append(tf.keras.layers.LSTM(24, **kwargs))
        layers.append(tf.keras.layers.Dense(1))

        return layers
