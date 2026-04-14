# Modèle de Décision Vincent Ganne (Point Bas Nasdaq)

## **Philosophie du Modèle**
Ce modèle est un indicateur **cross-asset** et **macro-géopolitique** dont l'unique but est de valider des opportunités d'achat (**BUY**) sur le **Nasdaq-100**. 

Il repose sur l'observation historique que les grands creux de marché sur les actions américaines coïncident souvent avec une stabilisation ou une baisse des prix de l'énergie et une détente sur le marché obligataire.

### **Règles Critiques d'Utilisation**
1.  **Exclusivité Nasdaq** : Ce modèle n'est utilisé QUE pour l'analyse du Nasdaq (`SXRV.DE`, `QQQ`, `^NDX`). Il est désactivé pour le trading du Pétrole ou d'autres actifs.
2.  **Signal Unidirectionnel (BUY Only)** : Le modèle ne génère **JAMAIS** de signal de vente (`SELL`). Son rôle est de dire "Oui, c'est un point bas" ou "Non, restez prudent". S'il n'est pas convaincu, il renvoie `HOLD`.
3.  **Verrou Géopolitique** : Si les prix du pétrole dépassent les seuils critiques (WTI > 94$), le modèle considère que la pression inflationniste/géopolitique est trop forte pour un point bas boursier durable.

---

## **Indicateurs et Niveaux Techniques**

Le score de confiance est calculé en distinguant le franchissement technique minimal et l'atteinte de la zone idéale.

| Actif / Indicateur | Niveau Minimum Technique (50% pts) | Niveau Idéal / Signal Fort (100% pts) |
| :--- | :--- | :--- |
| **1. Pétrole WTI** | < 94 $ | ≈ 80 $ |
| **2. Pétrole Brent** | < 95 $ | ≈ 83 $ |
| **3. Gaz Naturel (TTF)** | < 55 € | < 38 € |
| **4. Urée (Engrais)** | < 506 $ | — |
| **5. Taux US 2 ans** | Écart avec taux FED < 0.25% | — |
| **6. Dollar US (DXY)** | < 101 points | < 100 points |
| **7. Confirmations** | Prix > MA200j (S&P500, Nasdaq, Dow Jones, Tech) | — |

---

## **Enrichissement : Sentiment Décentralisé (Hyperliquid)**
Le système utilise les données de la blockchain **Hyperliquid** comme signal contrarien pour le Pétrole (qui sert d'indicateur macro pour le Nasdaq) :
- **Funding Rate (OIL)** : Un taux **très négatif** (excess de shorts) est un bonus de confiance pour le modèle, signalant une capitulation des vendeurs et un point bas potentiel.
- **Open Interest** : Utilisé pour valider la force de la conviction derrière le mouvement.

---

## **Logique de Fusion**
Dans le moteur `EnhancedDecisionEngine`, ce modèle apporte un poids de **20%** à la décision finale du Nasdaq. Il agit comme un filtre de sécurité macroéconomique venant confirmer ou infirmer les signaux techniques purs du modèle classique et du TimesFM.
