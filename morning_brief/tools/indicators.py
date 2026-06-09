def calc_rsi(series, period=14):
    if len(series) < period + 1:
        return None
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = float(gain.iloc[:period].mean())
    avg_loss = float(loss.iloc[:period].mean())
    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + float(gain.iloc[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(loss.iloc[i])) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def calc_rsi_series(series, period=14):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    if len(gain) < period:
        return []
    avg_gain = float(gain.iloc[:period].mean())
    avg_loss = float(loss.iloc[:period].mean())
    result = []
    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + float(gain.iloc[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(loss.iloc[i])) / period
        if avg_loss == 0:
            result.append(100.0)
        else:
            result.append(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)))
    return result
