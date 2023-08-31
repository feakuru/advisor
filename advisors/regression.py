import numpy as np
import tensorflow as tf

from advisors.base import BaseAdvisor
from common.utils import get_logger


logger = get_logger('advisor-regr')


class RegressionAdvisor(BaseAdvisor):
    def get_model(self) -> tf.keras.Model:
        normalizer = tf.keras.layers.Normalization()
        normalizer.adapt(np.array(self.train_features))

        model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(units=1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss='mean_absolute_error',
        )

        return model
