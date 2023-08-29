import numpy as np
import tensorflow as tf
import plotly.graph_objects as go

from advisors.base import BaseAdvisor


class RegressionAdvisor(BaseAdvisor):
    def ask_continue(self):
        if input('Continue? [y/n] ').lower() != 'y':
            exit(0)

    def plot_loss(self, history):
        go.Figure(
            data=[
                go.Scatter(
                    y=history.history['loss'],
                    name='Loss',
                ),
                go.Scatter(
                    y=history.history['val_loss'],
                    name='Loss value',
                ),
            ]
        ).show()

    def train(self):
        train_dataset = self.dataset.sample(frac=0.8, random_state=0)
        test_dataset = self.dataset.drop(train_dataset.index)
        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop('target')
        test_labels = test_features.pop('target')

        normalizer = tf.keras.layers.Normalization()
        normalizer.adapt(np.array(train_features))

        prediction_model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(units=1)
        ])

        print(prediction_model.summary())
        self.ask_continue()

        prediction_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss='mean_absolute_error',
        )

        history = prediction_model.fit(
            train_features,
            train_labels,
            epochs=100,
            validation_split=0.2,
        )
        self.plot_loss(history)
        self.ask_continue()

        print(prediction_model.evaluate(test_features, test_labels, verbose=0))

    def plot_predictions(self, x, data, predictions):
        data = [
            go.Scatter(
                y=data,
                name='Data',
                mode='markers',
            ),
            go.Scatter(
                y=[elt[0] for elt in predictions],
                name='Predictions',
            ),
        ]
        go.Figure(data=data).show()
