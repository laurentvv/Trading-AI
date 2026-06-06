import numpy as np
import pandas as pd
import logging
from scipy.linalg import eigh
from typing import Dict, Any

from src.enhanced_decision_engine import BaseModel, ModelResult

logger = logging.getLogger(__name__)


class GrebenkovTrendModel(BaseModel):
    """
    Implémentation du modèle de suivi de tendance EMA et Agnostic Risk Parity (ARP)
    basé sur Grebenkov et Serror (2014) et "Breaking the Trend" (Valeyre, 2026).
    """

    def __init__(
        self,
        eta: float = 1 / 112,
        rho: float = 1 / 20,
        vol_window: int = 40,
        corr_window: int = 750,
        atr_lookback: int = 14,
    ):
        self.eta = eta
        self.rho = rho
        self.vol_window = vol_window
        self.corr_window = corr_window
        self.target_volatility = 0.15
        self.atr_lookback = atr_lookback
        self._position_type = "FLAT"
        self._last_ticker = None

    def reset(self):
        """Reset internal state (position, last ticker). Called automatically on ticker change."""
        self._position_type = "FLAT"
        self._last_ticker = None

    def _bun_rie_filter(self, corr_matrix: np.ndarray, num_obs: int) -> np.ndarray:
        """
        Filtre RIE (Rotational Invariant Estimator) de Bun et al. (2016) simplifié.
        Nettoie la matrice de corrélation pour éviter le bruit.
        Dans une matrice 2x2, l'impact est mineur, mais on garde la structure.
        """
        N = corr_matrix.shape[0]
        # Décomposition en valeurs propres
        vals, vecs = eigh(corr_matrix)

        # Pour une matrice très petite (2x2), on applique un shrinkage basique vers l'identité
        # car la limite de Marchenko-Pastur n'a pas de sens asymptotique ici.
        shrinkage = 0.5 if num_obs < 100 else 0.1
        clean_matrix = (1 - shrinkage) * corr_matrix + shrinkage * np.eye(N)
        return clean_matrix

    def _inverse_sqrt_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Calcule M^{-0.5} d'une matrice symétrique définie positive."""
        vals, vecs = eigh(matrix)
        # Éviter les valeurs propres négatives dues aux erreurs d'arrondi
        vals = np.maximum(vals, 1e-8)
        inv_sqrt_vals = np.diag(1.0 / np.sqrt(vals))
        return vecs @ inv_sqrt_vals @ vecs.T

    def predict(self, data: Dict[str, Any]) -> ModelResult:
        """Generate a trading signal using EMA trend-following and Agnostic Risk Parity.

        Automatically resets internal state when the ticker changes between calls.
        """
        try:
            hist_data = data.get("hist_data")
            wti_data = data.get("wti_data")
            nasdaq_data = data.get("nasdaq_data")
            ticker = data.get("ticker", "Unknown")

            if ticker != self._last_ticker:
                self.reset()
                self._last_ticker = ticker

            if hist_data is None or hist_data.empty:
                return ModelResult("HOLD", 0.0, "Missing hist_data for Grebenkov model")



            if wti_data is None or nasdaq_data is None or wti_data.empty or nasdaq_data.empty:
                logger.warning(
                    "GrebenkovModel: Données WTI ou NASDAQ manquantes. Mode dégradé sans Agnostic Risk Parity."
                )
                result = self._predict_single_asset(hist_data, ticker)
            else:
                df_returns = pd.concat(
                    [nasdaq_data["Close"].pct_change().fillna(0), wti_data["Close"].pct_change().fillna(0)],
                    axis=1,
                    join="inner",
                ).tail(self.corr_window)
                df_returns.columns = ["NASDAQ", "WTI"]

                if len(df_returns) < 50:
                    return ModelResult("HOLD", 0.0, "Not enough common data points for Grebenkov")

                corr_matrix = df_returns.corr().values
                C_cleaned = self._bun_rie_filter(corr_matrix, len(df_returns))
                C_inv_sqrt = self._inverse_sqrt_matrix(C_cleaned)

                vol_ndx = df_returns["NASDAQ"].ewm(span=self.vol_window).std().iloc[-1]
                vol_wti = df_returns["WTI"].ewm(span=self.vol_window).std().iloc[-1]
                vol_ndx = max(vol_ndx, 1e-4)
                vol_wti = max(vol_wti, 1e-4)

                inv_Sigma = np.diag([1.0 / vol_ndx, 1.0 / vol_wti])

                returns = hist_data["Close"].pct_change().fillna(0)
                sigma2 = returns.pow(2).ewm(alpha=self.eta, adjust=False).mean()
                sigma_t = np.sqrt(sigma2).shift(1).bfill()
                norm_returns = returns / np.maximum(sigma_t, 1e-6)

                phi = 0.0
                sqrt_eta = np.sqrt(self.eta)
                norm_ret_arr = norm_returns.values
                for r in norm_ret_arr[-100:]:
                    phi = (1 - self.eta) * phi + sqrt_eta * r

                is_wti = "CRUD" in ticker.upper() or "CL=F" in ticker.upper()
                idx = 1 if is_wti else 0

                phi_vec = np.zeros(2)
                phi_vec[idx] = phi

                target_position_vec = inv_Sigma @ C_inv_sqrt @ phi_vec
                target_weight = target_position_vec[idx]

                result = self._weight_to_signal(target_weight, phi, hist_data)

            if result.signal == "BUY":
                self._position_type = "LONG"
            elif result.signal == "SELL":
                self._position_type = "FLAT"

            return result

        except Exception as e:
            logger.error(f"Erreur GrebenkovTrendModel: {e}", exc_info=True)
            return ModelResult("HOLD", 0.0, f"Error: {str(e)}")

    def _predict_single_asset(self, hist_data: pd.DataFrame, ticker: str) -> ModelResult:
        returns = hist_data["Close"].pct_change().fillna(0)
        sigma2 = returns.pow(2).ewm(alpha=self.eta, adjust=False).mean()
        sigma_t = np.sqrt(sigma2).shift(1).bfill()
        norm_returns = returns / np.maximum(sigma_t, 1e-6)

        phi = 0.0
        sqrt_eta = np.sqrt(self.eta)
        for r in norm_returns.values[-150:]:
            phi = (1 - self.eta) * phi + sqrt_eta * r

        return self._weight_to_signal(phi, phi, hist_data)

    def _true_range(self, hist_data: pd.DataFrame) -> np.ndarray:
        close = hist_data["Close"].values
        if "High" in hist_data.columns and "Low" in hist_data.columns:
            high = hist_data["High"].values
            low = hist_data["Low"].values
            tr = np.abs(high[1:] - low[1:])
            tr = np.maximum(tr, np.abs(high[1:] - close[:-1]))
            tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
        else:
            tr = np.abs(np.diff(close))
        return tr

    def _compute_atr(self, hist_data: pd.DataFrame) -> float:
        tr = self._true_range(hist_data)
        if len(tr) < self.atr_lookback:
            return float(np.mean(tr)) if len(tr) > 0 else 1.0
        return float(np.mean(tr[-self.atr_lookback :]))

    def _compute_atr_median(self, hist_data: pd.DataFrame) -> float:
        tr = self._true_range(hist_data)
        if len(tr) < self.atr_lookback:
            return float(np.median(tr)) if len(tr) > 0 else 1.0
        chunks = tr[: len(tr) - len(tr) % self.atr_lookback].reshape(-1, self.atr_lookback)
        atr_vals = chunks.mean(axis=1)
        return float(np.median(atr_vals)) if len(atr_vals) > 0 else 1.0

    def _weight_to_signal(self, weight: float, raw_phi: float, hist_data: pd.DataFrame = None) -> ModelResult:
        """Convert a continuous target weight into a discrete BUY/SELL/HOLD signal.

        Uses ATR-adaptive thresholding: higher volatility widens the neutral band.
        """
        abs_weight = abs(weight)
        confidence = min(max(abs_weight * 0.4, 0.0), 1.0)

        base_threshold = 0.15
        if hist_data is not None and len(hist_data) > self.atr_lookback + 1:
            current_atr = self._compute_atr(hist_data)
            median_atr = self._compute_atr_median(hist_data)
            if median_atr > 0:
                base_threshold = 0.15 * (current_atr / median_atr)

        if weight > base_threshold:
            signal = "BUY"
        elif weight < -base_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        reasoning = (
            f"Grebenkov ARP: Target Weight={weight:.3f}, Raw Phi={raw_phi:.3f}, "
            f"Conf={confidence:.2f}, ATR_Threshold={base_threshold:.3f}"
        )

        return ModelResult(signal, confidence, reasoning)
