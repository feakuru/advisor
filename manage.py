import enum
import sys

from binance import Client

from common.utils import get_logger
from gatherers.enrich_memory import enrich_memory, check_targets
from gatherers.fill_memory import get_history
from gatherers.get_memory_stats import log_memory_stats
from plotters.memory import MemoryPlotter
from memory_bank.client import MemoryBankClient


logger = get_logger('advisor-main')


class ManagementCommands(enum.Enum):
    ENRICH = 'enrich'
    CHECK = 'check'
    HISTORY = 'history'
    PLOT = 'plot'


if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.error(
            'Usage: manage.py [%s]',
            '|'.join([c.value for c in ManagementCommands]),
        )
        sys.exit(1)

    cmd, args = sys.argv[1], sys.argv[2:]
    api_client = Client()
    memory_bank = MemoryBankClient()

    match cmd:
        case ManagementCommands.ENRICH.value:
            enrich_memory()

        case ManagementCommands.CHECK.value:
            log_memory_stats(log_full_stats='--full' in args)
            check_targets(log_full_stats='--full' in args)

        case ManagementCommands.HISTORY.value:
            get_history(
                api_client=api_client,
                memory_bank=memory_bank,
                time=' '.join(args) if args else '30 days ago UTC',
            )

        case ManagementCommands.PLOT.value:
            MemoryPlotter().get_plot().show()
