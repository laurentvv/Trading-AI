# Contexte Produit

## 1. Énoncé du Problème
Les traders individuels et semi-professionnels s'appuient souvent sur un mélange d'analyse technique et de jugement qualitatif pour prendre des décisions de trading. Ce processus peut être chronophage, sujet aux biais émotionnels et difficile à tester de manière systématique. Les outils existants peuvent être soit trop simplistes, incapables de capturer la complexité du marché, soit trop complexes, nécessitant une courbe d'apprentissage abrupte.

## 2. Vision
Ce projet vise à créer un système de support à la décision de trading puissant mais accessible. Il comble le fossé entre l'analyse quantitative et le raisonnement qualitatif, semblable à celui d'un humain, en exploitant un modèle d'IA hybride. Le système permettra à l'utilisateur de prendre des décisions de trading plus informées, basées sur les données et testées de manière systématique, sur les ETF du NASDAQ.

## 3. Fonctionnement Attendu
L'utilisateur exécutera un seul script depuis la ligne de commande. Le script :
1. Récupérera les dernières données de marché pour un ETF NASDAQ spécifié, en utilisant un cache local pour accélérer le processus.
2. Traitera les données pour calculer un large éventail d'indicateurs techniques.
3. Injectera ces informations dans deux modèles d'IA parallèles :
    - Un modèle `scikit-learn` entraîné pour la prédiction de signaux.
    - Un LLM (via Ollama) qui fournit un signal et une analyse narrative du marché.
4. Combinera les sorties des deux modèles en une recommandation de trading unique et actionnable (par exemple, "ACHAT FORT", "NEUTRE", "VENTE").
5. Affichera la décision, l'analyse du LLM et les principales métriques de performance d'un backtest dans la console.
6. Générera des graphiques visualisant la performance de la stratégie.

## 4. Objectifs d'Expérience Utilisateur
- **Simplicité** : Le système doit être facile à exécuter avec une seule commande.
- **Clarté** : La sortie doit être claire et fournir à la fois un signal direct et le raisonnement derrière celui-ci (via l'analyse du LLM).
- **Transparence** : Les résultats du backtesting et les métriques de performance doivent être transparents, permettant à l'utilisateur de comprendre la performance historique et les risques de la stratégie.
