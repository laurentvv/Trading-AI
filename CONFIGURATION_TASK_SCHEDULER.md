# 🕐 CONFIGURATION TASK SCHEDULER WINDOWS
## Automatisation de l'Exécution Quotidienne du Trading AI

---

## 🎯 **OBJECTIF**
Configurer Windows Task Scheduler pour exécuter automatiquement le système Trading AI tous les jours à 18h00 (après fermeture des marchés US).

---

## 📋 **ÉTAPES DE CONFIGURATION**

### **Étape 1 : Ouvrir Task Scheduler**
1. Appuyer sur `Windows + R`
2. Taper `taskschd.msc`
3. Appuyer sur `Entrée`

### **Étape 2 : Créer une Tâche de Base**
1. Dans le panneau de droite, cliquer sur **"Créer une tâche de base..."**
2. **Nom** : `Trading AI - Analyse Quotidienne`
3. **Description** : `Exécution automatique du système de trading AI avec apprentissage continu`
4. Cliquer sur **"Suivant"**

### **Étape 3 : Configurer le Déclencheur**
1. Sélectionner **"Quotidien"**
2. Cliquer sur **"Suivant"**
3. **Heure de début** : `18:00:00` (6h00 PM)
4. **Répéter chaque** : `1 jour(s)`
5. **Date de début** : `23/08/2025` (ou date du lendemain)
6. Cliquer sur **"Suivant"**

### **Étape 4 : Configurer l'Action**
1. Sélectionner **"Démarrer un programme"**
2. Cliquer sur **"Suivant"**
3. **Programme/script** : 
   ```
   C:\test\Trading-AI\run_daily.bat
   ```
4. **Commencer dans (optionnel)** :
   ```
   C:\test\Trading-AI
   ```
5. Cliquer sur **"Suivant"**

### **Étape 5 : Finaliser**
1. Vérifier le résumé
2. Cocher **"Ouvrir la boîte de dialogue Propriétés..."**
3. Cliquer sur **"Terminer"**

---

## ⚙️ **CONFIGURATION AVANCÉE**

### **Onglet "Général" :**
- ✅ **"Exécuter avec les autorisations maximales"**
- ✅ **"Exécuter même si l'utilisateur n'est pas connecté"**
- 🔍 **"Configurer pour"** : Windows 10/11

### **Onglet "Déclencheurs" :**
- Vérifier que l'heure est `18:00:00`
- **Paramètres avancés** :
  - ✅ **"Activé"**
  - **"Répéter la tâche toutes les"** : Laisser vide
  - **"Arrêter la tâche en cours après"** : `1 heure`

### **Onglet "Actions" :**
- Vérifier le chemin vers `run_daily.bat`
- **Arguments (optionnel)** : Laisser vide

### **Onglet "Conditions" :**
- ❌ Décocher **"Démarrer la tâche seulement si l'ordinateur est alimenté sur secteur"**
- ❌ Décocher **"Arrêter si l'ordinateur passe sur batterie"**
- ✅ Cocher **"Démarrer seulement si une connexion réseau est disponible"**

### **Onglet "Paramètres" :**
- ✅ **"Autoriser l'exécution de la tâche à la demande"**
- ✅ **"Exécuter la tâche dès que possible après un démarrage planifié manqué"**
- ✅ **"Si la tâche échoue, la redémarrer toutes les"** : `1 minute`
- **"Tenter un redémarrage jusqu'à"** : `3 fois`
- **"Arrêter la tâche si elle s'exécute plus de"** : `1 heure`

---

## 🧪 **TEST DE LA CONFIGURATION**

### **Test Manuel :**
1. Dans Task Scheduler, trouver votre tâche **"Trading AI - Analyse Quotidienne"**
2. Clic droit → **"Exécuter"**
3. Vérifier que le script s'exécute correctement
4. Contrôler les fichiers générés dans `C:\test\Trading-AI\`

### **Test Automatique :**
1. Attendre l'heure programmée (18h00)
2. Vérifier l'exécution dans l'historique des tâches
3. Contrôler le fichier `execution_log.txt`

---

## 📊 **MONITORING DE LA TÂCHE**

### **Vérifier l'Historique :**
1. Dans Task Scheduler, sélectionner votre tâche
2. Onglet **"Historique"** en bas
3. Vérifier les codes de retour :
   - **0x0** : Succès ✅
   - **0x1** : Erreur générale ❌
   - **0x2** : Fichier non trouvé ❌

### **Logs à Surveiller :**
- `execution_log.txt` : Log des exécutions
- Console Windows (si ouverte) : Détails de l'exécution
- Fichiers générés quotidiennement

---

## 🔧 **DÉPANNAGE**

### **Problème : La tâche ne s'exécute pas**
- ✅ Vérifier les autorisations utilisateur
- ✅ Vérifier le chemin vers `run_daily.bat`
- ✅ Tester l'exécution manuelle du `.bat`
- ✅ Vérifier que Python et le .venv sont accessibles

### **Problème : Erreur dans l'exécution**
- 🔍 Consulter `execution_log.txt`
- 🔍 Tester manuellement : `run_daily.bat`
- 🔍 Vérifier que Ollama est démarré
- 🔍 Vérifier la connexion internet

### **Problème : Tâche s'exécute mais ne génère rien**
- 🔍 Vérifier les permissions d'écriture dans le dossier
- 🔍 Contrôler que l'environnement virtuel est activé
- 🔍 Tester en mode manuel d'abord

---

## 📅 **CALENDRIER D'EXÉCUTION**

### **Jours d'Exécution :**
- **Lundi à Vendredi** : 18h00 (marchés US fermés)
- **Weekend** : Pas d'exécution (marchés fermés)
- **Jours fériés US** : Pas d'exécution (ajuster manuellement)

### **Modifications Saisonnières :**
- **Heure d'été/hiver** : Ajuster l'heure si nécessaire
- **Vacances** : Désactiver temporairement la tâche

---

## ✅ **CHECKLIST FINALE**

Avant de valider la configuration :

- [ ] Task Scheduler configuré avec tous les paramètres
- [ ] Test manuel réussi
- [ ] Chemin vers `run_daily.bat` correct
- [ ] Permissions d'écriture vérifiées
- [ ] Ollama démarré et fonctionnel
- [ ] Connexion internet stable
- [ ] Fichier `execution_log.txt` créé lors du test
- [ ] Graphiques générés lors du test

**🎉 CONFIGURATION TERMINÉE ! Votre système Trading AI s'exécutera automatiquement tous les jours à 18h00.**