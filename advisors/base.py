import datetime
import typing as t
from dataclasses import dataclass
from io import StringIO

import numpy as np
from numpy import typing as np_typing

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
        self.loss = tf.keras.losses.MeanAbsoluteError()
        self.optimizer = tf.keras.optimizers.Adamax()

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
        model.build(self.train_features.shape)
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
        )
        return model

    def get_preprocessing_layers(self):
        normalizer = tf.keras.layers.Normalization()
        normalizer.adapt(np.array(self.train_features))
        return [normalizer]

    def get_model_layers(self):
        raise NotImplementedError

    def plot_history(
        self,
        history: tf.keras.callbacks.History,
        plot_label_prefix: str = '',
    ):
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            y=history.history['loss'],
            name='Training loss',
        ))
        figure.add_trace(go.Scatter(
            y=history.history['val_loss'],
            name='Validation loss',
        ))
        figure.update_layout(
            title=dict(
                text=f'{plot_label_prefix}: training history',
                x=0.5,
            ),
            showlegend=True,
            legend=dict(
                title='Historical metrics',
                orientation='h',
                yanchor='bottom',
                xanchor='left',
                x=0,
                y=1,
            )
        )
        figure.show()

    def plot_predictions(
        self,
        predictions: np_typing.NDArray[t.Any],
        actual_values: np_typing.NDArray[t.Any],
        plot_label_prefix: str = '',
    ):
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            y=actual_values,
            name='Actual values',
            mode='lines+markers',
            marker=dict(
                color='green',
                symbol='circle',
                size=12,
            ),
        ))
        figure.add_trace(go.Scatter(
            y=predictions.flatten(),
            name='Predictions',
            mode='markers',
            marker=dict(
                color='black',
                symbol='x',
                size=12,
            ),
        ))
        figure.update_layout(
            title=dict(
                text=f'{plot_label_prefix}: predictions',
                x=0.5,
            ),
            showlegend=True,
            legend=dict(
                title='Data',
                orientation='h',
                yanchor='bottom',
                xanchor='left',
                x=0,
                y=1,
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

    def fit_model(self, epochs: int = 100, add_early_stopping: bool = True):
        logger.info('Fitting model:\n%s', self.get_model_summary())
        callbacks = []
        if add_early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=12,
                mode='min',
            ))

        history = self.model.fit(
            self.train_features,
            self.train_labels,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
        )
        logger.info('Model is fitted.')

        return history

    def set_dataset(self, dataset: pd.DataFrame, sequence_length: int = 26):
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

        logger.info(
            'Loaded dataset.\n'
            'Data sample:\n%s\n'
            'Data stats:\n%s\n',
            self.dataset[:7],
            self.dataset.describe().transpose(),
        )

        logger.info('Windowing dataset. This might take a minute...')

        result_list = []
        for idx, _ in enumerate(self.dataset.values[sequence_length:]):
            windowed_values = self.dataset.values[idx - sequence_length:idx]
            windowed_values = [
                {
                    (f'{k}_{idx}' if k != 'target' else k): v
                    for k, v in zip(self.dataset.keys(), windowed_value)
                    if k != 'target' or idx == (len(windowed_values) - 1)
                }
                for idx, windowed_value in enumerate(windowed_values)
            ]
            result = {}
            for value in windowed_values:
                result.update(value)
            result_list.append(result)
        self.dataset = pd.DataFrame.from_dict(result_list)
        self.dataset = self.dataset.dropna()

        logger.info(
            'Windowed dataset.\n'
            'Data sample:\n%s\n'
            'Data stats:\n%s\n',
            self.dataset[:7],
            self.dataset.describe().transpose(),
        )

        self.train_dataset = self.dataset.sample(frac=0.8, random_state=0)
        self.test_dataset = self.dataset.drop(self.train_dataset.index)

        self.train_features = self.train_dataset.copy()
        self.test_features = self.test_dataset.copy()

        self.train_labels = self.train_features.pop('target')
        self.test_labels = self.test_features.pop('target')
        logger.info(
            'Loaded data.\n'
            'Training data sample:\n%s\n'
            'Training data stats:\n%s\n'
            'Test data sample:\n%s\n'
            'Test data stats:\n%s\n',
            self.train_dataset[:7],
            self.train_dataset.describe().transpose(),
            self.test_dataset[:7],
            self.test_dataset.describe().transpose(),
        )

    def train(self, epochs: int = 100, add_early_stopping: bool = True):
        history = self.fit_model(
            epochs=epochs,
            add_early_stopping=add_early_stopping,
        )
        plot_label_prefix = (
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: '
            f'{self.__class__.__name__}'
        )
        self.plot_history(
            history=history,
            plot_label_prefix=plot_label_prefix,
        )
        self.plot_predictions(
            predictions=self.model.predict(self.test_features[:100]),
            actual_values=self.test_labels[:100],
            plot_label_prefix=plot_label_prefix,
        )
        logger.info(
            'Loss on test data: %s',
            self.model.evaluate(
                self.test_features,
                self.test_labels,
            ),
        )
