"""
Reference (ground-truth) solution - the exact predict() function.
"""

import math


def predict(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16):
    score = (
        35.0 * math.sin(0.5 * x3)
        + 28.0 * math.cos(0.4 * x9)
        + 20.0 * math.sin(0.3 * x6 - 0.2 * x14)
        + (5.0 if x4 > 25 else -5.0)
        + (-4.0 if x7 + x16 > 50 else 4.0)
        + 2.0 * math.floor(x11 / 7.0)
        + 50
    )
    return round(score, 2)
