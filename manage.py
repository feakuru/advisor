import enum
import sys

from common.utils import get_logger


logger = get_logger('advisor-main')


class ManagementCommands(enum.Enum):
    ENRICH = 'enrich'
    CHECK = 'check'
    HISTORY = 'history'
    PLOT = 'plot'
    TRAIN = 'train'


if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.error(
            'Usage: manage.py [%s]',
            '|'.join([c.value for c in ManagementCommands]),
        )
        sys.exit(1)

    cmd, args = sys.argv[1], sys.argv[2:]
    kwargs = {
        (kv_pair := arg.split('='))[0][2:]: (
            True if kv_pair[1].lower() == 'true' else
            False if kv_pair[1].lower() == 'false' else
            int(kv_pair[1]) if kv_pair[1].isdigit() else
            kv_pair[1]
        ) if '=' in arg else True
        for arg in args
        if arg.startswith('--')
    }
    args = [
        arg for arg in args
        if not arg.startswith('--')
    ]

    match cmd:
        case ManagementCommands.ENRICH.value:
            from gatherers.enrich_memory import enrich_memory

            enrich_memory()

        case ManagementCommands.CHECK.value:
            from gatherers.get_memory_stats import log_memory_stats
            from gatherers.enrich_memory import enrich_memory, check_targets
            log_memory_stats(log_full_stats=kwargs.get('full', False))
            check_targets(log_full_stats=kwargs.get('full', False))

        case ManagementCommands.HISTORY.value:
            from binance import Client
            from gatherers.fill_memory import get_history
            from memory_bank.client import MemoryBankClient

            memory_bank = MemoryBankClient()
            api_client = Client()
            get_history(
                api_client=api_client,
                memory_bank=memory_bank,
                time=' '.join(args) if args else '30 days ago UTC',
            )

        case ManagementCommands.PLOT.value:
            from plotters.memory import MemoryPlotter

            MemoryPlotter().get_plot().show()

        case ManagementCommands.TRAIN.value:
            from memory_bank.client import MemoryBankClient
            from databuilders.pd import PandasMemoryDatabuilder

            memory_bank = MemoryBankClient()
            databuilder = PandasMemoryDatabuilder(memory_bank)

            match kwargs.get('advisor', 'lstm'):
                case 'dnn':
                    from advisors.dnn import (
                        DNNAdvisor,
                        DNNAdvisorParams,
                    )
                    advisor = DNNAdvisor(DNNAdvisorParams(
                        dense_layer_quantity=kwargs.get('layers', 2),
                        dense_layer_density=kwargs.get('density', 24),
                    ))
                case 'lstm':
                    from advisors.lstm import (
                        LSTMAdvisor,
                        LSTMAdvisorParams,
                    )
                    advisor = LSTMAdvisor(LSTMAdvisorParams(
                        layers=kwargs.get('layers', 1),
                    ))
            advisor.set_dataset(databuilder.get_data())
            advisor.train(
                epochs=int(args[0]) if args else 100,
                add_early_stopping=kwargs.get('early_stopping', True),
            )
