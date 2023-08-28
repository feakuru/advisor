from pymongo import MongoClient
from config import config


def get_database():
    client = MongoClient(config.MEMORY_BANK_MONGO_DBURL)
    return client[config.MEMORY_BANK_MONGO_DBNAME]


def get_memory_collection(symbol: str):
    return get_database()[f'{config.MEMORY_BANK_MONGO_COLLECTION}_{symbol}']
