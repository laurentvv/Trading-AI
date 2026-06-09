from smolagents import Tool


def _compute_volume_ratio(hist) -> float | None:
    if "Volume" not in hist.columns or len(hist) < 20:
        return None
    vol = hist["Volume"]
    avg_20 = float(vol.iloc[-20:].mean())
    if avg_20 <= 0:
        return None
    return round(float(vol.iloc[-1]) / avg_20, 2)


def _format_summary(latest: float, change: float, rsi, macd_data, volume_ratio, wti_corr, divergence) -> str:
    if rsi is None or wti_corr is None:
        return f"NDX {latest:.0f} ({change:+.1f}%)"
    return (
        f"NDX {latest:.0f} ({change:+.1f}%) | "
        f"RSI={rsi:.0f} | MACD hist={macd_data['histogram']:.0f} | "
        f"VolRatio={volume_ratio} | WTI_corr={wti_corr:.2f} | {divergence}"
    )


class AnalyzeNasdaqTool(Tool):
    name = "analyze_nasdaq"
    description = (
        "Nasdaq 100 (^NDX) analysis: RSI, MACD, volume ratio, "
        "20-day correlation with WTI, and price/RSI divergence. "
        "Returns a compact summary string. Full data saved to output/tools/."
    )
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        import yfinance as yf
        from morning_brief.tools import save_tool_result

        try:
            ndx = yf.Ticker("^NDX")
            hist = ndx.history(period="6mo")
            if hist.empty:
                save_tool_result("nasdaq", {"error": "No Nasdaq data"})
                return "ERROR: No Nasdaq data"

            close = hist["Close"]
            latest = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) >= 2 else latest
            change = (latest - prev) / prev * 100

            rsi = self._calc_rsi(close, 14)
            macd_data = self._calc_macd(close)
            volume_ratio = _compute_volume_ratio(hist)
            wti_corr = self._calc_wti_correlation(close)
            divergence = self._detect_divergence(close, 5)

            full = {
                "price": round(latest, 2),
                "change_pct": round(change, 2),
                "rsi": round(rsi, 1) if rsi else None,
                "macd": macd_data,
                "volume_ratio": volume_ratio,
                "wti_correlation_20d": round(wti_corr, 3) if wti_corr else None,
                "divergence_signal": divergence,
            }
            save_tool_result("nasdaq", full)

            return _format_summary(latest, change, rsi, macd_data, volume_ratio, wti_corr, divergence)
        except Exception as e:
            save_tool_result("nasdaq", {"error": str(e)})
            return f"ERROR: {e}"

    @staticmethod
    def _calc_rsi(series, period=14):
        from morning_brief.tools.indicators import calc_rsi
        return calc_rsi(series, period)

    @staticmethod
    def _calc_macd(close):
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return {
            "macd": round(float(macd_line.iloc[-1]), 2),
            "signal": round(float(signal_line.iloc[-1]), 2),
            "histogram": round(float((macd_line - signal_line).iloc[-1]), 2),
        }

    @staticmethod
    def _calc_wti_correlation(ndx_close):
        import yfinance as yf
        try:
            wti_hist = yf.Ticker("CL=F").history(period="6mo")
            if wti_hist.empty:
                return None
            wti_close = wti_hist["Close"]
            merged = ndx_close.to_frame("NDX").join(wti_close.to_frame("WTI"), how="inner")
            ndx_ret = merged["NDX"].pct_change().dropna()
            wti_ret = merged["WTI"].pct_change().dropna()
            common = ndx_ret.index.intersection(wti_ret.index)
            ndx_r = ndx_ret.loc[common].tail(20)
            wti_r = wti_ret.loc[common].tail(20)
            if len(ndx_r) < 10:
                return None
            return float(ndx_r.corr(wti_r))
        except Exception:
            return None

    @staticmethod
    def _detect_divergence(close, lookback=5):
        from morning_brief.tools.indicators import calc_rsi_series
        if len(close) < lookback + 15:
            return "INSUFFICIENT_DATA"
        tail = close.tail(lookback)
        price_trend = float(tail.iloc[-1]) - float(tail.iloc[0])
        rsi_series = calc_rsi_series(close, 14)
        if len(rsi_series) < lookback:
            return "INSUFFICIENT_DATA"
        rsi_tail = rsi_series[-lookback:]
        rsi_trend = rsi_tail[-1] - rsi_tail[0]
        if price_trend > 0 and rsi_trend < -5:
            return "BEARISH_DIVERGENCE"
        elif price_trend < 0 and rsi_trend > 5:
            return "BULLISH_DIVERGENCE"
        return "NO_DIVERGENCE"
