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

    def __init__(self, eta: float = 1 / 112, rho: float = 1 / 20, vol_window: int = 40, corr_window: int = 750):
        """
        Initialise les paramètres du modèle.

        :param eta: Paramètre de lissage de l'EMA (Optimal = 1/112 selon l'étude).
        :param rho: Paramètre de lissage du portefeuille pour réduire le turnover.
        :param vol_window: Fenêtre pour l'estimation de la volatilité quotidienne (Sigma).
        :param corr_window: Fenêtre pour l'estimation de la matrice de corrélation (C).
        """
        self.eta = eta
        self.rho = rho
        self.vol_window = vol_window
        self.corr_window = corr_window
        # Fallback target volatility if not enough assets to normalize perfectly
        self.target_volatility = 0.15

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
        """
        Génère une prédiction selon l'API BaseModel.

        Args:
            data: Dictionnaire contenant:
                - 'hist_data': pd.DataFrame du ticker analysé
                - 'wti_data': pd.DataFrame du ticker WTI (CRUDP.PA ou CL=F)
                - 'nasdaq_data': pd.DataFrame du ticker NASDAQ (SXRV.DE ou ^IXIC)
        """
        try:
            hist_data = data.get("hist_data")
            wti_data = data.get("wti_data")
            nasdaq_data = data.get("nasdaq_data")
            ticker = data.get("ticker", "Unknown")

            if hist_data is None or hist_data.empty:
                return ModelResult("HOLD", 0.0, "Missing hist_data for Grebenkov model")

            if wti_data is None or nasdaq_data is None or wti_data.empty or nasdaq_data.empty:
                logger.warning(
                    "GrebenkovModel: Données WTI ou NASDAQ manquantes. Mode dégradé sans Agnostic Risk Parity."
                )
                return self._predict_single_asset(hist_data, ticker)

            # --- Préparation des données communes (Alignement des index) ---
            # Conversion en rendements
            ret_wti = wti_data["Close"].pct_change().fillna(0)
            ret_ndx = nasdaq_data["Close"].pct_change().fillna(0)

            # Alignement
            df_returns = pd.concat([ret_ndx, ret_wti], axis=1, join="inner").tail(self.corr_window)
            df_returns.columns = ["NASDAQ", "WTI"]

            if len(df_returns) < 50:
                return ModelResult("HOLD", 0.0, "Not enough common data points for Grebenkov")

            # 1. Matrice de Corrélation C et Filtre
            corr_matrix = df_returns.corr().values
            C_cleaned = self._bun_rie_filter(corr_matrix, len(df_returns))
            C_inv_sqrt = self._inverse_sqrt_matrix(C_cleaned)

            # 2. Volatilité (EMA 40j) pour chaque actif
            vol_ndx = df_returns["NASDAQ"].ewm(span=self.vol_window).std().iloc[-1]
            vol_wti = df_returns["WTI"].ewm(span=self.vol_window).std().iloc[-1]

            vol_ndx = max(vol_ndx, 1e-4)
            vol_wti = max(vol_wti, 1e-4)

            inv_Sigma = np.diag([1.0 / vol_ndx, 1.0 / vol_wti])

            # 3. Calcul du Signal Normalisé phi_t pour le ticker actuel
            # On recrée l'EMA récursive
            returns = hist_data["Close"].pct_change().fillna(0)
            sigma2 = returns.pow(2).ewm(alpha=self.eta, adjust=False).mean()
            sigma_t = np.sqrt(sigma2).shift(1).bfill()
            norm_returns = returns / np.maximum(sigma_t, 1e-6)

            phi = 0.0
            sqrt_eta = np.sqrt(self.eta)
            norm_ret_arr = norm_returns.values
            for r in norm_ret_arr[-100:]:  # Warmup sur les 100 derniers jours
                phi = (1 - self.eta) * phi + sqrt_eta * r

            # Pour le vecteur complet (on suppose que si le ticker est WTI, sa position est [1], sinon [0])
            is_wti = "CRUD" in ticker.upper() or "CL=F" in ticker.upper()
            idx = 1 if is_wti else 0

            # On construit le vecteur de signaux (simplifié : on met 0 pour l'autre pour isoler notre position ARP)
            phi_vec = np.zeros(2)
            phi_vec[idx] = phi

            # 4. Agnostic Risk Parity Target Position
            target_position_vec = inv_Sigma @ C_inv_sqrt @ phi_vec
            target_weight = target_position_vec[idx]

            # Lissage rho appliqué en théorie sur la série, ici on approxime le scaling final
            # Pour l'intégration dans l'Engine existant, on traduit le poids en signal de classification
            return self._weight_to_signal(target_weight, phi)

        except Exception as e:
            logger.error(f"Erreur GrebenkovTrendModel: {e}", exc_info=True)
            return ModelResult("HOLD", 0.0, f"Error: {str(e)}")

    def _predict_single_asset(self, hist_data: pd.DataFrame, ticker: str) -> ModelResult:
        """Mode dégradé sans ARP, calcule uniquement l'EMA normalisée."""
        returns = hist_data["Close"].pct_change().fillna(0)
        sigma2 = returns.pow(2).ewm(alpha=self.eta, adjust=False).mean()
        sigma_t = np.sqrt(sigma2).shift(1).bfill()
        norm_returns = returns / np.maximum(sigma_t, 1e-6)

        phi = 0.0
        sqrt_eta = np.sqrt(self.eta)
        for r in norm_returns.values[-150:]:  # Warmup
            phi = (1 - self.eta) * phi + sqrt_eta * r

        return self._weight_to_signal(phi, phi)

    def _weight_to_signal(self, weight: float, raw_phi: float) -> ModelResult:
        """Convertit un poids continu en signal discret (BUY/SELL/HOLD) avec confiance."""
        # Un signal phi ~ 1.0 indique une tendance d'1 écart-type
        abs_weight = abs(weight)

        # Scaling heuristique pour la confiance (maxé à 1.0)
        # weight typiquement dans [-2, 2]
        confidence = min(max(abs_weight * 0.4, 0.0), 1.0)

        if weight > 0.15:
            signal = "BUY"
        elif weight < -0.15:
            signal = "SELL"
        else:
            signal = "HOLD"

        reasoning = f"Grebenkov ARP: Target Weight={weight:.3f}, Raw Phi={raw_phi:.3f}, Conf={confidence:.2f}"

        return ModelResult(signal, confidence, reasoning)
