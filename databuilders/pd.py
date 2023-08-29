import datetime
import json
import pandas as pd

from databuilders.base import BaseDatabuilder
from memory_bank.client import MemoryBankClient


class PandasMemoryDatabuilder(BaseDatabuilder):
    def __init__(self, memory_bank: MemoryBankClient | None = None):
        self.memory_bank = memory_bank or MemoryBankClient()
        self.__memory = None

    @property
    def memory(self) -> str:
        if self.__memory is None:
            self.__memory = json.dumps([
                {
                    k: (
                        v
                        if not isinstance(v, datetime.datetime)
                        else v.isoformat()
                    )
                    for k, v in kline.items()
                    if k not in ['_id']
                }
                for kline in self.memory_bank.iterate_over_klines()
            ])
        return self.__memory

    def get_data(self) -> pd.DataFrame:
        return pd.read_json(self.memory)
