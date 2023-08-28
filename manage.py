import sys
import logging

from binance import Client

from gatherers.enrich_memory import enrich_memory, check_targets
from gatherers.fill_memory import get_history
from gatherers.get_memory_stats import log_memory_stats
from memory_bank.client import MemoryBankClient


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.error('Usage: manage.py [enrich|history|check]')
        sys.exit(1)

    cmd, args = sys.argv[1], sys.argv[2:]
    api_client = Client()
    memory_bank = MemoryBankClient()

    match cmd:
        case 'enrich':
            enrich_memory()

        case 'check':
            log_memory_stats(log_full_stats='--full' in args)
            check_targets(log_full_stats='--full' in args)

        case 'history':
            get_history(
                api_client=api_client,
                memory_bank=memory_bank,
                time=' '.join(args) if args else '30 days ago UTC',
            )
