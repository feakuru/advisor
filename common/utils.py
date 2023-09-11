import logging
import typing as t
from datetime import datetime


def get_kline_json(kline: t.List) -> t.Dict[str, t.Any]:
    if len(kline) != 12:
        raise ValueError('Kline must have 12 elements')
    return {
        'open_time': datetime.utcfromtimestamp(kline[0] / 1e3),
        'open': float(kline[1]),
        'high': float(kline[2]),
        'low': float(kline[3]),
        'close': float(kline[4]),
        'volume': float(kline[5]),
        'close_time': datetime.utcfromtimestamp(kline[6] / 1e3),
        'quote_asset_volume': float(kline[7]),
        'number_of_trades': int(kline[8]),
        'taker_buy_base_asset_volume': float(kline[9]),
        'taker_buy_quote_asset_volume': float(kline[10]),
        'ignored_field': kline[11],
    }


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s [%(levelname)6s:%(name)-13s]: %(message)s',
        ),
    )
    logger.addHandler(handler)
    return logger
