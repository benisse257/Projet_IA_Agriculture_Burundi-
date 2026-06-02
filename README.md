# 🌾 Prédiction des Bonnes et Mauvaises Récoltes au Burundi

## 📋 Vue d'ensemble

Ce projet est un **système complet de Machine Learning** conçu pour prédire si une récolte sera **bonne (1)** ou **mauvaise (0)** au Burundi en fonction de données agricoles.

Le projet combine:
- ✅ Analyse exploratoire des données (EDA)
- ✅ Prétraitement et normalisation
- ✅ 3 modèles de classification (Decision Tree, Random Forest, Logistic Regression)
- ✅ Évaluation rigoureuse avec multiples métriques
- ✅ Application web interactive (Streamlit)
- ✅ Sauvegarde des modèles en production

---

## 🎯 Objectif du Projet

Construire un système de prédiction fiable et déployable pour:
- Anticiper les mauvaises récoltes
- Aider les agriculteurs et le Ministère de l'Agriculture du Burundi
- Optimiser les stratégies agricoles

---

## 📂 Structure du Projet

```
Projet_IA_Agriculture_Burundi/
│
├── agriculture_burundi.csv           # Dataset complet
├── TP_IA_Agriculture_Burundi.ipynb   # Notebook principal (EDA + ML)
├── requirements.txt                   # Dépendances Python
├── README.md                          # Ce fichier
│
├── decision_tree.pkl                 # Modèle Decision Tree
├── random_forest.pkl                 # Modèle Random Forest
├── logistic_regression.pkl           # Modèle Logistic Regression
├── scaler.pkl                        # Objet StandardScaler
├── encoder.pkl                       # Objet OneHotEncoder
│
└── app/
    ├── app.py                        # Application Streamlit
    └── README.md                     # Instructions Streamlit
```

---

## 🚀 Installation

### 1. Prérequis
- Python 3.8+
- pip ou conda

### 2. Cloner le projet
```bash
git clone <repository_url>
cd Projet_IA_Agriculture_Burundi
```

### 3. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate     # Sur Windows
```

### 4. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## 📊 Exécution du Projet

### Option 1: Jupyter Notebook (Analyse complète)
```bash
jupyter notebook TP_IA_Agriculture_Burundi.ipynb
```

Ce notebook contient:
- Chargement et exploration des données
- Visualisations complètes
- Prétraitement des données
- Entraînement des 3 modèles
- Évaluation détaillée
- Analyse de l'overfitting
- Validation croisée
- Sauvegarde des modèles

### Option 2: Application Streamlit (Prédictions interactives)
```bash
streamlit run app/app.py
```

---

## 🧠 Modèles Utilisés

### 1. Decision Tree (Arbre de Décision)
- **Paramètres**: `max_depth=4, criterion='gini'`
- **Avantages**: Facile à interpréter
- **Performance**: ~82% accuracy

### 2. Random Forest (Forêt Aléatoire)
- **Paramètres**: `n_estimators=100`
- **Avantages**: Bonne performance, résistant à l'overfitting
- **Performance**: ~88% accuracy ⭐ **MEILLEUR**

### 3. Logistic Regression
- **Paramètres**: `max_iter=1000`
- **Avantages**: Rapide, interprétable
- **Performance**: ~84% accuracy

---

## 📈 Variables du Dataset

### Variables Catégoriques (encodées)
- **province**: 12 provinces du Burundi
- **culture**: 8 types de cultures
- **saison**: 3 saisons agricoles

### Variables Numériques (normalisées)
- **altitude**: Altitude en mètres
- **pluviométrie**: Précipitations en mm
- **température**: Température moyenne en °C
- **superficie**: Superficie cultivée en hectares
- **engrais**: Quantité d'engrais en kg/ha
- **irrigation**: Heures d'irrigation par semaine
- **nombre_menages**: Nombre de ménages

### Variable Cible
- **bonne_recolte**: 1 (Bonne récolte) / 0 (Mauvaise récolte)

---

## 🔍 Étapes du Projet

### 1. Analyse Exploratoire (EDA)
- Chargement du dataset
- Vérification des types de données
- Analyse des valeurs manquantes
- Statistiques descriptives
- Corrélations entre variables

### 2. Visualisations
- Distribution des récoltes (0/1)
- Boxplot rendement par culture
- Heatmap de corrélation
- Production par année
- Impact de l'engrais sur les récoltes

### 3. Prétraitement
- **OneHotEncoding** pour variables catégoriques
- **StandardScaler** pour variables numériques
- Suppression du data leakage
- Split 80/20 avec stratification

### 4. Modélisation
- Entraînement de 3 modèles
- Prédictions sur ensemble de test
- Extraction des probabilités

### 5. Évaluation
Pour chaque modèle:
- **Accuracy**: Précision générale
- **Precision**: Vrais positifs / Prédictions positives
- **Recall**: Vrais positifs / Réels positifs
- **F1-Score**: Moyenne harmonique Precision/Recall
- **Matrice de confusion**: TP, TN, FP, FN
- **Courbe ROC**: Performance à différents seuils
- **AUC-ROC**: Surface sous la courbe ROC

### 6. Analyse de l'Overfitting
- Test avec max_depth de 1 à 20
- Courbes accuracy train vs test
- Identification du surapprentissage

### 7. Validation Croisée
- 5-fold cross-validation
- Moyenne et écart-type des scores
- Intervalle de confiance

### 8. Sauvegarde
- Modèles en `.pkl` avec joblib
- Objets de prétraitement sauvegardés
- Prêts pour production

---

## 🌐 Application Streamlit

### Fonctionnalités

#### Inputs Utilisateur (Sidebar)
- Province (selectbox)
- Culture (selectbox)
- Saison (selectbox)
- Altitude (slider)
- Pluviométrie (slider)
- Température (slider)
- Superficie (slider)
- Engrais (slider)
- Irrigation (slider)
- Nombre de ménages (slider)

#### Options
- Choix du modèle (Decision Tree, Random Forest, Logistic Regression)

#### Outputs
- Prédiction (Bonne/Mauvaise récolte)
- Confiance en pourcentage
- Probabilités estimées (graphique)
- Tableau de résumé des paramètres
- Informations sur les modèles
- Performances de chaque modèle

---

## 📊 Résultats Obtenus

| Métrique | Decision Tree | Random Forest | Logistic Regression |
|----------|---------------|---------------|---------------------|
| Accuracy | 82% | 88% | 84% |
| Precision | 85% | 90% | 86% |
| Recall | 78% | 85% | 82% |
| F1-Score | 81% | 87% | 84% |
| AUC-ROC | 0.80 | 0.88 | 0.84 |

**Recommandation**: Utiliser le **Random Forest** pour les prédictions en production.

---

## 💾 Fichiers Générés

### Modèles ML
- `decision_tree.pkl` - Arbre de décision entraîné
- `random_forest.pkl` - Forêt aléatoire entraînée
- `logistic_regression.pkl` - Régression logistique entraînée

### Objets de Prétraitement
- `scaler.pkl` - StandardScaler pour normalisation
- `encoder.pkl` - OneHotEncoder pour catégories

### Autres
- `TP_IA_Agriculture_Burundi.ipynb` - Notebook complet
- `requirements.txt` - Dépendances
- `README.md` - Documentation

---

## 🔧 Configuration

### Variables d'Environnement
Aucune requise pour le fonctionnement basique.

### Paramètres Modifiables
Vous pouvez modifier:
- Les hyperparamètres des modèles dans le notebook
- Le ratio train/test
- Les variables pour le preprocessing
- Les intervalles des sliders Streamlit

---

## 📝 Notes Importantes

1. **Data Leakage**: Les colonnes `rendement_t_ha` et `production_totale_t` ont été supprimées pour éviter le data leakage.

2. **Stratification**: Le split train/test utilise `stratify=y` pour maintenir la distribution des classes.

3. **Normalisation**: Les variables numériques sont normalisées avec StandardScaler après le split pour éviter le data leakage.

4. **Encodage**: OneHotEncoding est utilisé pour les variables catégoriques.

5. **Random State**: `random_state=42` pour reproductibilité.

---

## 🚨 Dépannage

### Les modèles ne se chargent pas
```bash
# Vérifier que les fichiers .pkl existent
ls *.pkl
# Réexécuter le notebook pour les régénérer
jupyter notebook TP_IA_Agriculture_Burundi.ipynb
```

### Erreur Streamlit
```bash
# Vérifier les dépendances
pip install -r requirements.txt

# Lancer avec plus de verbosité
streamlit run app/app.py --logger.level=debug
```

### Problème de chemin
Les fichiers `.pkl` doivent être au même niveau que `app.py` ou dans le répertoire parent.

---

## 📚 Ressources

- **Scikit-Learn**: https://scikit-learn.org/
- **Streamlit**: https://streamlit.io/
- **Pandas**: https://pandas.pydata.org/
- **Plotly**: https://plotly.com/

---

## 👨‍💻 Auteur

**Étudiant BAC 4 Génie Logiciel**
- Projet: Prédiction des Récoltes au Burundi
- Année académique: 2024-2025

---

## 📄 Licence

Ce projet est utilisé à titre éducatif.

---

## 🤝 Support

Pour toute question ou problème:
1. Consulter le notebook `TP_IA_Agriculture_Burundi.ipynb`
2. Vérifier la documentation Streamlit
3. Assurez-vous que toutes les dépendances sont installées

---

## ✅ Checklist

- ✅ EDA complète avec visualisations
- ✅ Prétraitement des données
- ✅ 3 modèles ML entraînés
- ✅ Évaluation complète
- ✅ Analyse de l'overfitting
- ✅ Validation croisée
- ✅ Sauvegarde des modèles
- ✅ Application Streamlit interactive
- ✅ Documentation complète
- ✅ Requirements.txt

---

**Dernière mise à jour**: 2024
