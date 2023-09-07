from dataclasses import dataclass
import numpy as np
import tensorflow as tf

from advisors.base import BaseAdvisor, BaseAdvisorParams
from common.utils import get_logger


logger = get_logger('advisor-dnn')


@dataclass
class DNNAdvisorParams(BaseAdvisorParams):
    dense_layer_density: int = 24
    dense_layer_quantity: int = 2


class DNNAdvisor(BaseAdvisor):
    params: DNNAdvisorParams
    params_class = DNNAdvisorParams

    def get_model(self) -> tf.keras.Model:
        normalizer = tf.keras.layers.Normalization()
        normalizer.adapt(np.array(self.train_features))

        layers = [normalizer]
        for _ in range(self.params.dense_layer_quantity):
            layers.append(tf.keras.layers.Dense(
                self.params.dense_layer_density,
                activation='relu',
            ))
        layers.append(tf.keras.layers.Dense(1))
        model = tf.keras.Sequential(layers)

        model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
            loss='mean_absolute_error',
        )

        return model
