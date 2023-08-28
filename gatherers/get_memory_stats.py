from collections import defaultdict
from common.utils import get_logger

from memory_bank.client import MemoryBankClient


logger = get_logger(__name__)


def log_memory_stats(
    log_full_stats: bool = False,
    memory_bank: MemoryBankClient = MemoryBankClient(),
):
    stats = defaultdict(int)
    for kline in memory_bank.iterate_over_klines():
        stats[kline['open_time'].strftime("%d.%m.%y %H:%M:%S")] += 1
    if log_full_stats:
        logger.info('Stats:')
        for time, stat in stats.items():
            logger.info('%s: %d', time, stat)
        logger.info('=' * 32)
    logger.info('Unique open times: %d', len(stats))
    logger.info('Repeated open times: %d', sum(stats.values()) - len(stats))
