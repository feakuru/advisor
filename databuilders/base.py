import pandas as pd


class BaseDatabuilder:
    def get_data(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError
