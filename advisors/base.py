import pandas as pd


class BaseAdvisor:
    def set_dataset(self, dataset: pd.DataFrame):
        self.dataset = dataset.dropna()
        self.dataset.pop('open_time')
        self.dataset.pop('close_time')
        self.dataset.pop('open')
        self.dataset.pop('high')
        self.dataset.pop('low')
        self.dataset.pop('close')
        self.dataset.pop('volume')
        self.dataset.pop('quote_asset_volume')
        self.dataset.pop('number_of_trades')
        self.dataset.pop('taker_buy_base_asset_volume')
        self.dataset.pop('taker_buy_quote_asset_volume')
        self.dataset.pop('ignored_field')

    def train(self):
        raise NotImplementedError
