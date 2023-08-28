'''General settings.'''
from dataclasses import dataclass
from dotenv import dotenv_values


@dataclass
class Config:
    '''General settings.'''
    BINANCE_API_KEY: str
    BINANCE_API_SECRET: str
    MEMORY_BANK_MONGO_DBURL: str
    MEMORY_BANK_MONGO_DBNAME: str
    MEMORY_BANK_MONGO_COLLECTION: str
    SYMBOL: str = 'BTCUSDT'


config = Config(**dotenv_values(".env"))
