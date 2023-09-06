'''General settings.'''
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')


@dataclass
class Config:
    '''General settings.'''
    BINANCE_API_KEY: str
    BINANCE_API_SECRET: str
    MEMORY_BANK_MONGO_DBURL: str
    MEMORY_BANK_MONGO_DBNAME: str
    MEMORY_BANK_MONGO_COLLECTION: str
    SYMBOL: str = 'BTCUSDT'


config = Config(**{
    env_key: env_value
    for env_key, env_value in
    os.environ.items()
    if env_key in Config.__dataclass_fields__.keys()
})
