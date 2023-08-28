import typing as t

from config import config
from common.utils import get_kline_json
from memory_bank.db import get_memory_collection


class MemoryBankClient:
    def __init__(self, symbol: str | None = None) -> None:
        self.memory_collection = get_memory_collection(symbol or config.SYMBOL)

    def is_in_memory(self, kline: t.List) -> bool:
        return bool(self.memory_collection.find_one({
            'open_time': get_kline_json(kline)['open_time'],
        }))

    def remember_kline(self, kline: t.List):
        return self.memory_collection.insert_one(get_kline_json(kline))

    def iterate_over_klines(self):
        for kline in self.memory_collection.find().sort('open_time'):
            yield kline

    def update_kline(self, kline_filter: t.Dict, update_fields: t.Dict):
        return self.memory_collection.update_one(
            kline_filter,
            {'$set': update_fields},
        )

    def clear_memory(self):
        return self.memory_collection.drop()
