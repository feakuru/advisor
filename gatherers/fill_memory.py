from binance import Client

from common.utils import get_logger
from config import config
from memory_bank.client import MemoryBankClient


logger = get_logger('gatherer-fill')


def get_history(
    api_client: Client,
    memory_bank: MemoryBankClient,
    time: str = '2 days ago UTC',
):
    for kline in api_client.get_historical_klines(
        config.SYMBOL,
        Client.KLINE_INTERVAL_15MINUTE,
        time,
    ):
        if not memory_bank.is_in_memory(kline):
            memory_bank.remember_kline(kline)
        else:
            logger.warning('Kline %s already in memory.', kline[0])
