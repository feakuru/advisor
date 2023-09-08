from io import StringIO
import typing as t
from dataclasses import dataclass
import numpy as np

import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go

from common.utils import get_logger


logger = get_logger('advisor-base')


@dataclass
class BaseAdvisorParams:
    prepared_model: tf.keras.Model | None = None


class BaseAdvisor:
    loss: tf.keras.losses.Loss
    optimizer: tf.keras.optimizers.Optimizer
    metrics: t.List[tf.keras.metrics.Metric]
    train_dataset: pd.DataFrame
    test_dataset: pd.DataFrame
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    params_class: t.Type[BaseAdvisorParams] = BaseAdvisorParams

    def __init__(self, params: BaseAdvisorParams):
        if not isinstance(params, self.params_class):
            raise TypeError(
                '`params` must be an instance of %s'
                % self.params_class.__name__,
            )
        self.params = params
        self._model = self.params.prepared_model
        self.loss = tf.keras.losses.MeanAbsolutePercentageError()
        self.optimizer = tf.keras.optimizers.Adamax()
        self.metrics = [
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanAbsolutePercentageError(),
            tf.keras.metrics.KLDivergence(),
        ]

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            self._model = self.get_model()
        return self._model

    def get_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            *self.get_preprocessing_layers(),
            *self.get_model_layers()
        ])
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics,
        )
        return model

    def get_preprocessing_layers(self):
        normalizer = tf.keras.layers.Normalization()
        normalizer.adapt(np.array(self.train_features))
        return normalizer

    def get_model_layers(self):
        raise NotImplementedError

    def plot_history(self, history):
        figure = go.Figure()
        for metric in self.metrics:
            figure.add_trace(go.Scatter(
                y=history.history[metric.name],
                name=' '.join(
                    name.capitalize()
                    for name in metric.name.split('_')
                )
            ))
        figure.update_layout(
            title_text=f'{self.__class__.__name__} training history',
            showlegend=True,
            legend=dict(
                title='Historical metrics',
                yref='container',
                y=0.9,
                xref='container',
                x=0.8,
            )
        )
        figure.show()

    def get_model_summary(self) -> str:
        summary_buffer = StringIO('')

        def _print_to_buffer(arg: str):
            summary_buffer.write(f'{arg}\n')

        self.model.summary(print_fn=_print_to_buffer)

        summary_buffer.seek(0)
        return summary_buffer.read()

    def fit_model(self, epochs: int = 100):
        logger.info('Fitting model:\n%s', self.get_model_summary())
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
        )

        history = self.model.fit(
            self.train_features,
            self.train_labels,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[early_stopping],
        )
        self.plot_history(history)
        logger.info('Model is trained and loss is plotted.')

    def set_dataset(self, dataset: pd.DataFrame):
        self.dataset = dataset.dropna()

        for unneeded_key in [
            'open_time',
            'close_time',
            'open',
            'high',
            'low',
            'close',
            'volume',
            'quote_asset_volume',
            'number_of_trades',
            'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume',
            'ignored_field',
        ]:
            self.dataset.pop(unneeded_key)

        self.train_dataset = self.dataset.sample(frac=0.8, random_state=0)
        self.test_dataset = self.dataset.drop(self.train_dataset.index)

        self.train_features = self.train_dataset.copy()
        self.test_features = self.test_dataset.copy()

        self.train_labels = self.train_features.pop('target')
        self.test_labels = self.test_features.pop('target')

    def train(self, epochs: int = 100):
        self.fit_model(epochs=epochs)
        logger.info(
            'Evaluation result: %s',
            dict(zip(
                (metric.name for metric in self.metrics),
                self.model.evaluate(
                    self.test_features,
                    self.test_labels,
                ),
            ))
        )
