# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Practical CPU-temperature settling criterion on physical-time samples."""
import math


def _slope(rows):
    origin = rows[0]['time_s']
    xs = [r['time_s'] - origin for r in rows]
    ys = [r['CPU_C'] for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    denominator = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator


def evaluate_temperature(history, rate_limit=0.01, hold_s=10.0, rate_window_s=1.0):
    """Require every 1 s fitted rate to stay below the limit for 10 s.

    A single flat sample, a constant positive heating rate, and oscillations
    with a small overall fitted slope must not be mistaken for settling.
    This is a CPU-temperature criterion, not full-field steady convergence.
    """
    if min(rate_limit, hold_s, rate_window_s) <= 0:
        raise ValueError('Rate and physical-time windows must be positive')
    result = dict(temperature_settled=False, rate_limit_C_per_s=rate_limit,
                  hold_s=hold_s, rate_window_s=rate_window_s,
                  CPU_rate_C_per_s=None, max_abs_CPU_rate_C_per_s=None,
                  assessed_duration_s=0.0)
    if len(history) < 2:
        return result
    end = float(history[-1]['time_s'])
    # Keep one predecessor so irregular sampling still spans each window.
    start = 0
    for i, row in enumerate(history):
        if row['time_s'] <= end - hold_s - rate_window_s:
            start = i
        else:
            break
    rows = history[start:]
    if any(not math.isfinite(r[k]) for r in rows for k in ('time_s', 'CPU_C')):
        return result
    if any(b['time_s'] <= a['time_s'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Physical-time samples must be strictly increasing')
    rates = []
    for i, row in enumerate(rows):
        target = row['time_s'] - rate_window_s
        if row['time_s'] < end - hold_s - 1e-9 or target < rows[0]['time_s'] - 1e-9:
            continue
        j = i - 1
        while j > 0 and rows[j]['time_s'] > target + 1e-9:
            j -= 1
        window = rows[j:i + 1]
        # Missing output cannot establish a sustained temperature trend.
        if any(b['time_s'] - a['time_s'] > rate_window_s + 1e-9
               for a, b in zip(window, window[1:])):
            continue
        rates.append((row['time_s'], _slope(window)))
    if not rates:
        return result
    result.update(CPU_rate_C_per_s=rates[-1][1],
                  max_abs_CPU_rate_C_per_s=max(abs(rate) for _, rate in rates),
                  assessed_duration_s=rates[-1][0] - rates[0][0])
    continuous = all(b[0] - a[0] <= rate_window_s + 1e-9
                     for a, b in zip(rates, rates[1:]))
    result['temperature_settled'] = (
        continuous and rates[-1][0] >= end - 1e-9
        and result['assessed_duration_s'] >= hold_s - 1e-9
        and result['max_abs_CPU_rate_C_per_s'] < rate_limit
    )
    return result
