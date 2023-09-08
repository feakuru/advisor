from collections import defaultdict
import typing as t

from common.utils import get_logger
from memory_bank.client import MemoryBankClient


logger = get_logger('gatherer-enrich')


UpdatesType = t.Dict[str, t.Dict[str, t.Any]]


def enrich_memory_with_targets(
    memory_bank: MemoryBankClient = MemoryBankClient(),
) -> UpdatesType:
    prev_kline = None
    updates = {}
    for kline in memory_bank.iterate_over_klines():
        if prev_kline:
            target = 1000 * (
                kline['close'] - prev_kline['close']
            ) / prev_kline['close']
            updates[prev_kline['open_time']] = {'target': target}
        prev_kline = kline
    return updates


def check_targets(
    log_full_stats: bool = False,
    memory_bank: MemoryBankClient = MemoryBankClient(),
):
    prev_kline = None

    we_were_right = 0
    we_were_wrong = 0
    we_didnt_know = 0

    for kline in memory_bank.iterate_over_klines():
        if prev_kline:
            # TODO scale targets by some constant factor
            # that should not ever change for a symbol
            expected_target = 1000 * (
                kline['close'] - prev_kline['close']
            ) / prev_kline['close']

            if prev_kline.get('target') == expected_target:
                we_are = 'RIGHT'
                we_were_right += 1
            else:
                we_are = 'WRONG'
                we_were_wrong += 1

            if log_full_stats:
                logger.info(
                    'We are %s (%6.2f vs %6.2f, from %9.2f to %9.2f).',
                    we_are,
                    prev_kline.get('target'),
                    expected_target,
                    prev_kline['close'],
                    kline['close'],
                )

        else:
            we_are = 'N/A'
            we_didnt_know += 1

        prev_kline = kline

    logger.info(
        'We were right %d times, wrong %d times, didn\'t know %d times.',
        we_were_right,
        we_were_wrong,
        we_didnt_know,
    )


def enrich_memory_with_sma(
    period: int,
    memory_bank: MemoryBankClient = MemoryBankClient(),
) -> UpdatesType:
    data = list(enumerate(memory_bank.iterate_over_klines()))
    updates = {}
    for idx, kline in data:
        if idx >= period:
            sma = sum(
                prev_kline['close']
                for _, prev_kline in data[idx - period:idx]
            ) / period
            updates[kline['open_time']] = {f'sma_{period}': sma}
    return updates


def enrich_memory_with_ema(
    period: int,
    memory_bank: MemoryBankClient = MemoryBankClient(),
) -> UpdatesType:
    updates = {}
    prev_ema = None
    k = 2 / (period + 1)
    for kline in memory_bank.iterate_over_klines():
        ema = (
            (kline['close'] * k + prev_ema * (1 - k))
            if prev_ema
            else kline['close']
        )
        updates[kline['open_time']] = {f'ema_{period}': ema}
        prev_ema = ema
    return updates


def enrich_memory_with_stochastic_oscillators(
    fast_period: int = 14,
    slow_period: int = 3,
    memory_bank: MemoryBankClient = MemoryBankClient(),
) -> UpdatesType:
    data = list(enumerate(memory_bank.iterate_over_klines()))
    updates = {}
    for idx, kline in data:
        updates[kline['open_time']] = {}
        if idx >= fast_period:
            prev_klines = data[idx - fast_period:idx]
            lowest_previous: float = min(
                prev_kline['low'] for _, prev_kline in prev_klines
            )
            highest_previous: float = max(
                prev_kline['high'] for _, prev_kline in prev_klines
            )
            fast_stochastic_oscillator = (
                (kline['close'] - lowest_previous)
                / (highest_previous - lowest_previous)
            )
            updates[
                kline['open_time']
            ][f'fso_{fast_period}'] = fast_stochastic_oscillator
            data[idx][1]['fso'] = fast_stochastic_oscillator
        if idx >= fast_period + slow_period:
            slow_stochastic_oscillator = (
                sum(
                    kline['fso']
                    for _, kline in data[idx - slow_period:idx]
                ) / slow_period
            )
            updates[
                kline['open_time']
            ][f'sso_{fast_period}_{slow_period}'] = (
                slow_stochastic_oscillator
            )
    return updates


def enrich_memory_with_rsi(
    period: int = 14,
    memory_bank: MemoryBankClient = MemoryBankClient(),
) -> UpdatesType:
    def get_gain(kline):
        return kline['close'] - data[idx - 1][1]['close']

    def get_loss(kline):
        return data[idx - 1][1]['close'] - kline['close']

    data = list(enumerate(memory_bank.iterate_over_klines()))
    updates = {}

    for idx, kline in data:
        if idx > 0:
            gains = [
                gain
                for _, prev_kline in data[max(idx - period, 0):idx]
                if (gain := get_gain(prev_kline)) > 0
            ]
            avg_gain = sum(gains) / len(gains) if gains else 0
            losses = [
                loss
                for _, prev_kline in data[max(idx - period, 0):idx]
                if (loss := get_loss(prev_kline)) > 0
            ]
            avg_loss = sum(losses) / len(losses) if losses else 0
            rs = avg_gain / avg_loss if avg_loss else 0
            rsi = 100 - 100 / (1 + rs)
            updates[kline['open_time']] = {f'rsi_{period}': rsi}
    return updates


def enrich_memory_with_macd(
    memory_bank: MemoryBankClient = MemoryBankClient(),
) -> UpdatesType:
    done, not_done = 0, 0
    updates = {}
    for kline in memory_bank.iterate_over_klines():
        if 'ema_9' in kline and 'ema_12' in kline and 'ema_26' in kline:
            macd = kline['ema_12'] - kline['ema_26']
            macd_hist = macd - kline['ema_9']
            updates[kline['open_time']] = {
                'macd': macd,
                'macd_hist': macd_hist,
            }
            done += 1
        else:
            not_done += 1
    logger.info('MACDs done: %d, not done: %d', done, not_done)
    return updates


def enrich_memory_with_obv(
    memory_bank: MemoryBankClient = MemoryBankClient(),
) -> UpdatesType:
    obv = 0
    updates = {}
    for kline in memory_bank.iterate_over_klines():
        if kline['close'] != kline['open']:
            obv += (
                kline['volume']
                if kline['close'] > kline['open']
                else -kline['volume']
            )
        updates[kline['open_time']] = {'obv': obv}
    return updates


def gather_enrichment_updates(
    memory_bank: MemoryBankClient = MemoryBankClient(),
) -> UpdatesType:
    enrichment_updates = [
        enrich_memory_with_targets(memory_bank=memory_bank),
        enrich_memory_with_sma(5, memory_bank=memory_bank),
        enrich_memory_with_sma(9, memory_bank=memory_bank),
        enrich_memory_with_sma(12, memory_bank=memory_bank),
        enrich_memory_with_sma(26, memory_bank=memory_bank),
        enrich_memory_with_ema(5, memory_bank=memory_bank),
        enrich_memory_with_ema(9, memory_bank=memory_bank),
        enrich_memory_with_ema(12, memory_bank=memory_bank),
        enrich_memory_with_ema(26, memory_bank=memory_bank),
        enrich_memory_with_stochastic_oscillators(memory_bank=memory_bank),
        enrich_memory_with_rsi(memory_bank=memory_bank),
        enrich_memory_with_macd(memory_bank=memory_bank),
        enrich_memory_with_obv(memory_bank=memory_bank),
    ]
    all_keys = set().union(*enrichment_updates)
    updates = defaultdict(dict)
    for key in all_keys:
        for update in enrichment_updates:
            if key in update:
                updates[key].update(update[key])
    return updates


def enrich_memory(memory_bank: MemoryBankClient = MemoryBankClient()):
    updates = gather_enrichment_updates()
    for open_time, update in updates.items():
        logger.info('Updating %s:', open_time)
        for key, value in update.items():
            logger.info('  %s: %s', key.rjust(10), value)
        logger.info('-' * 32)
        memory_bank.update_kline({'open_time': open_time}, update)
