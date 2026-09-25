import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List

from src.enhanced_decision_engine import BaseModel, ModelResult

logger = logging.getLogger(__name__)


def forward(
    obs_seq: List[int], N_states: int, start_p: np.ndarray, trans_p: np.ndarray, emit_p: np.ndarray
) -> Tuple[float, np.ndarray]:
    T = len(obs_seq)
    alpha = np.zeros((N_states, T))

    # Étape 1 : Initialisation
    for j in range(N_states):
        alpha[j, 0] = start_p[j] * emit_p[j, obs_seq[0]]

    # Étape 2 : Récursion
    for t in range(1, T):
        for j in range(N_states):
            alpha[j, t] = sum(alpha[i, t - 1] * trans_p[i, j] * emit_p[j, obs_seq[t]] for i in range(N_states))

    # Étape 3 : Terminaison
    forward_prob = sum(alpha[i, T - 1] for i in range(N_states))
    return forward_prob, alpha


def viterbi(
    obs_seq: List[int], N_states: int, start_p: np.ndarray, trans_p: np.ndarray, emit_p: np.ndarray
) -> Tuple[List[int], float]:
    T = len(obs_seq)
    v = np.zeros((N_states, T))
    backpointer = np.zeros((N_states, T), dtype=int)

    # Étape 1 : Initialisation
    for j in range(N_states):
        v[j, 0] = start_p[j] * emit_p[j, obs_seq[0]]
        backpointer[j, 0] = 0

    # Étape 2 : Récursion
    for t in range(1, T):
        for j in range(N_states):
            probs = [v[i, t - 1] * trans_p[i, j] * emit_p[j, obs_seq[t]] for i in range(N_states)]
            v[j, t] = max(probs)
            backpointer[j, t] = int(np.argmax(probs))

    # Étape 3 : Terminaison
    bestpathprob = max(v[:, T - 1])
    bestpathpointer = int(np.argmax(v[:, T - 1]))

    # Rétro-traçage (Viterbi backtrace)
    bestpath = [bestpathpointer]
    for t in range(T - 1, 0, -1):
        bestpath.insert(0, backpointer[bestpath[0], t])

    return bestpath, bestpathprob


def backward(obs_seq: List[int], N_states: int, trans_p: np.ndarray, emit_p: np.ndarray) -> np.ndarray:
    T = len(obs_seq)
    beta = np.zeros((N_states, T))

    for i in range(N_states):
        beta[i, T - 1] = 1

    for t in range(T - 2, -1, -1):
        for i in range(N_states):
            beta[i, t] = sum(trans_p[i, j] * emit_p[j, obs_seq[t + 1]] * beta[j, t + 1] for j in range(N_states))

    return beta


def baum_welch(
    obs_seq: List[int], N_states: int, V_vocab_size: int, iterations: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(obs_seq)

    # Initialisation aléatoire pour briser la symétrie
    np.random.seed(42) # Reproductibilité

    A = np.random.rand(N_states, N_states)
    A = A / A.sum(axis=1, keepdims=True)

    B = np.random.rand(N_states, V_vocab_size)
    B = B / B.sum(axis=1, keepdims=True)

    pi = np.random.rand(N_states)
    pi = pi / pi.sum()

    for _ in range(iterations):
        P_O, alpha = forward(obs_seq, N_states, pi, A, B)
        # Avoid division by zero if P_O is extremely small (underflow)
        if P_O == 0:
            P_O = 1e-100

        beta = backward(obs_seq, N_states, A, B)

        gamma = np.zeros((N_states, T))
        xi = np.zeros((N_states, N_states, T - 1))

        for t in range(T):
            for j in range(N_states):
                gamma[j, t] = (alpha[j, t] * beta[j, t]) / P_O

            if t < T - 1:
                for i in range(N_states):
                    for j in range(N_states):
                        xi[i, j, t] = (alpha[i, t] * A[i, j] * B[j, obs_seq[t + 1]] * beta[j, t + 1]) / P_O

        for i in range(N_states):
            denominator = np.sum(xi[i, :, :])
            if denominator > 0:
                for j in range(N_states):
                    A[i, j] = np.sum(xi[i, j, :]) / denominator

        for j in range(N_states):
            denominator = np.sum(gamma[j, :])
            if denominator > 0:
                for vk in range(V_vocab_size):
                    mask = np.array(obs_seq) == vk
                    B[j, vk] = np.sum(gamma[j, mask]) / denominator

    return pi, A, B


class HMMDecisionModel(BaseModel):
    """
    Modèle de décision basé sur les Modèles de Markov Cachés (HMM).
    Discrétise les rendements boursiers récents, entraîne dynamiquement les matrices
    via Baum-Welch et prédit le régime de marché actuel via Viterbi.
    """

    def __init__(self, n_states: int = 2, lookback: int = 252, baum_welch_iterations: int = 10):
        """
        Args:
            n_states: Nombre d'états cachés (ex: 2 = Bullish vs Bearish)
            lookback: Nombre de jours de données historiques à analyser (par défaut 252 jours de trading = 1 an)
            baum_welch_iterations: Nombre d'itérations pour l'apprentissage de l'HMM
        """
        self.n_states = n_states
        self.lookback = lookback
        self.iterations = baum_welch_iterations
        self.vocab_size = 3  # 0: Down, 1: Neutral, 2: Up

    def _discretize_returns(self, returns: pd.Series) -> List[int]:
        """
        Transforme des rendements continus en séquence d'observations discrètes.
        0: Forte baisse (< -0.5%)
        1: Neutre (-0.5% à 0.5%)
        2: Forte hausse (> 0.5%)
        """
        obs = []
        for r in returns:
            if r < -0.005:
                obs.append(0)
            elif r > 0.005:
                obs.append(2)
            else:
                obs.append(1)
        return obs

    def _identify_bullish_state(self, B: np.ndarray) -> int | None:
        """
        Identifie quel état caché correspond au marché "Haussier" (Bullish).
        Vérifie également que les états sont significativement différenciés.
        Retourne None si les états sont trop similaires.
        """
        bullish_state = int(np.argmax(B[:, 2]))
        bearish_state = int(np.argmin(B[:, 2]))

        # Check if the states actually learned distinct behaviors
        if abs(B[bullish_state, 2] - B[bearish_state, 2]) < 0.05:
            return None

        return bullish_state

    def predict(self, data: Dict[str, Any]) -> ModelResult:
        try:
            hist_data = data.get("hist_data")
            if hist_data is None or hist_data.empty:
                return ModelResult("HOLD", 0.0, "Missing historical data for HMM")

            # Extraire les prix de clôture et calculer les rendements
            closes = hist_data["Close"].dropna()
            if len(closes) < 10:
                return ModelResult("HOLD", 0.0, "Not enough data points")

            # Limiter au lookback défini
            closes = closes.tail(self.lookback + 1)
            returns = closes.pct_change().dropna()

            # 1. Préparer les observations
            obs_seq = self._discretize_returns(returns)

            if len(obs_seq) < 10:
                return ModelResult("HOLD", 0.0, "Not enough observations after formatting")

            # 2. Apprentissage des paramètres de l'HMM (Baum-Welch)
            pi, A, B = baum_welch(obs_seq, self.n_states, self.vocab_size, self.iterations)

            # 3. Décodage de la séquence d'états (Viterbi)
            bestpath, bestprob = viterbi(obs_seq, self.n_states, pi, A, B)

            # L'état actuel est le dernier état du chemin
            current_state = bestpath[-1]

            # 4. Identification du sens des états
            bullish_state = self._identify_bullish_state(B)

            if bullish_state is None:
                return ModelResult(
                    "HOLD", 0.0, "Les régimes HMM ne sont pas clairement séparables (différence trop faible)"
                )

            bearish_state = int(np.argmin(B[:, 2]))

            # 5. Décision
            # La confiance est basée sur la séparation des états : plus l'état haussier a une
            # probabilité d'émission "Up" nettement supérieure au pire état baissier, plus le régime est clair.
            state_separation = float(B[bullish_state, 2] - B[bearish_state, 2])

            # Normalisation : state_separation est généralement entre 0.05 et ~0.4.
            # On le mappe vers une confiance entre 0.2 et 0.8 de manière proportionnelle.
            confidence = min(max(state_separation * 2.0, 0.2), 0.8)

            if current_state == bullish_state:
                signal = "BUY"
                reason = "HMM a identifié le régime de marché actuel comme Haussier"
            else:
                signal = "SELL"
                reason = "HMM a identifié le régime de marché actuel comme Baissier"

            return ModelResult(
                signal=signal,
                confidence=confidence,
                reasoning=reason,
                metadata={"current_state": current_state, "bullish_state_id": bullish_state, "prob_path": bestprob},
            )

        except Exception as e:
            logger.error(f"HMM Model error: {e}")
            return ModelResult("HOLD", 0.0, f"Error in HMM prediction: {str(e)}")
