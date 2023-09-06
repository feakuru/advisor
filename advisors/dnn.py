import numpy as np
import tensorflow as tf

from advisors.base import BaseAdvisor
from common.utils import get_logger


logger = get_logger('advisor-regr')


class DNNAdvisor(BaseAdvisor):
    def get_model(self) -> tf.keras.Model:
        normalizer = tf.keras.layers.Normalization()
        normalizer.adapt(np.array(self.train_features))

        layers = [normalizer]
        for _ in range(2):
            layers.append(tf.keras.layers.Dense(24, activation='relu'))
        layers.append(tf.keras.layers.Dense(1))
        model = tf.keras.Sequential(layers)

        model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
            loss='mean_absolute_error',
        )

        return model
