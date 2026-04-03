import math
def predict(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16):
    # Work in integer cents to match the dataset's exact 2-decimal formatting
    s = 10795

    s += -4527 * (1 if x3 > 45 else 0)
    s += +3605 * (1 if x12 > 20 else 0)
    s += -2927 * (1 if x5 > 5 else 0)
    s += -2872 * (1 if x4 > 45 else 0)
    s += -2683 * (1 if x3 > 5 else 0)
    s += -1846 * (1 if x16 > 40 else 0)
    s += +1644 * (1 if x3 > 25 else 0)

    s += +440 * (x3 % 5)
    s += -429 * (x6 % 8)
    s += -403 * (x11 % 10)
    s += -353 * (x4 % 6)
    s += +348 * (x2 % 6)

    s += +2 * (x5 * x6)
    s += -1 * (x10 * x14)

    return round(s / 100.0, 2)
