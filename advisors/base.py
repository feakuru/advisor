import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go

from common.utils import get_logger


logger = get_logger('advisor-base')


class BaseAdvisor:
    train_dataset: pd.DataFrame
    test_dataset: pd.DataFrame
    train_features: pd.DataFrame
    test_features: pd.DataFrame

    def __init__(self, model: tf.keras.Model | None = None):
        self._model = model

    def get_model(self):
        raise NotImplementedError

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            self._model = self.get_model()
        return self._model

    def plot_loss(self, history):
        go.Figure(
            data=[
                go.Scatter(
                    y=history.history['loss'],
                    name='Loss',
                ),
                go.Scatter(
                    y=history.history['val_loss'],
                    name='Validation loss',
                ),
            ]
        ).show()

    def fit_model(self, epochs: int = 100):
        logger.info('Model is created:\n%s', self.model.summary())
        history = self.model.fit(
            self.train_features,
            self.train_labels,
            epochs=epochs,
            validation_split=0.2,
        )
        self.plot_loss(history)
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
            'Evaluation result: %f',
            self.model.evaluate(self.test_features, self.test_labels),
        )
