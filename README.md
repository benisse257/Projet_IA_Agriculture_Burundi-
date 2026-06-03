# 🌾 AgriPredict Burundi — Système de Prédiction des Récoltes

## 📋 Vue d'ensemble

**AgriPredict Burundi** est une application web interactive d'aide à la décision agricole basée sur le Machine Learning. Elle permet de prédire si une saison de culture donnera une **bonne récolte (1)** ou une **mauvaise récolte (0)** en fonction des conditions géographiques, climatiques et techniques d'une parcelle.

Ce projet a été développé dans le cadre du cours d'**Intelligence Artificielle ** à l'*Université Polytechnique de Gitega (Bac 4 Génie Logiciel)*.

---

## 🎯 Objectifs du Projet

- **Anticiper les risques** de mauvaises récoltes pour renforcer la sécurité alimentaire.
- **Fournir un outil d'aide à la décision** accessible aux services agricoles provinciaux et aux coopératives.
- **Générer des recommandations agronomiques automatisées** et adaptées au contexte de chaque parcelle.
- **Comparer l'efficacité** de différents algorithmes de classification de Machine Learning.

---

## 📂 Structure du Répertoire


```

Projet_IA_Agriculture_Burundi/
│
├── agriculture_burundi.csv        # Dataset historique (1 620 lignes, 2015–2023)
├── TP_IA_Agriculture_Burundi.ipynb # Notebook complet (EDA, Entraînement, Validation)
├── requirements.txt               # Liste des dépendances Python indispensables
├── app.py                         # Code source de l'interface web Streamlit
├── README.md                      # Documentation globale du projet (Ce fichier)
│
└── models/
├── decision_tree.pkl          # Modèle Arbre de Décision sérialisé
├── random_forest.pkl          # Modèle Forêt Aléatoire sérialisé
├── logistic_regression.pkl    # Modèle Régression Logistique sérialisé
├── scaler.pkl                 # Objet StandardScaler pour la normalisation
└── encoder.pkl                # Liste Python contenant l'ordre exact des colonnes encodées

```

---

## 🚀 Installation et Lancement

### 1. Prérequis
Assurez-vous d'avoir installé **Python 3.8** ou une version supérieure.

### 2. Cloner le projet & installer les dépendances
Ouvrez un terminal dans votre dossier de travail :
```bash
# Clonez le projet (si hébergé sur Git) ou placez-vous dans le dossier
cd Projet_IA_Agriculture_Burundi

# Installez l'ensemble des bibliothèques nécessaires
pip install -r requirements.txt

```

### 3. Démarrer l'application Streamlit

```bash
streamlit run app.py

```

L'application s'ouvrira automatiquement dans votre navigateur par défaut (généralement à l'adresse `http://localhost:8501`).

---

## 📈 Spécifications des Données et Variables

Le modèle s'appuie sur un historique couvrant **15 provinces**, **6 cultures** et **2 saisons principales** (Saison A : mars–juin / Saison B : sept–déc).

### Variables prédictives (Features) :

* **Variables catégoriques** : `province`, `culture`, `saison` (traitées par encodage One-Hot avec exclusion de la première modalité pour éviter le piège de la colinéarité).
* **Variables numériques** : `altitude_m`, `pluviometrie_mm`, `temperature_moy_C`, `superficie_ha`, `nb_menages`, `utilisation_engrais` (0/1), `acces_irrigation` (0/1).

### Variable cible (Target) :

* `bonne_recolte` : **1** (Bonne récolte) ou **0** (Mauvaise récolte).

*Note : Les colonnes génératrices de fuite de données (Data Leakage) comme `rendement_t_ha` et `production_totale_t` ont été volontairement supprimées lors de la phase d'entraînement.*

---

## 🤖 Modèles & Performances Réelles

L'application charge dynamiquement 3 architectures entraînées et évalue leurs performances sur l'ensemble de test (Split 80/20 Stratifié) :

| Métrique | Arbre de Décision | Forêt Aléatoire | Régression Logistique |
| --- | --- | --- | --- |
| **Accuracy** | ~82.00% | **~88.00%** ⭐ | ~84.00% |
| **F1-Score** | Calculé en direct | Calculé en direct | Calculé en direct |
| **AUC-ROC** | Calculé en direct | Calculé en direct | Calculé en direct |

> **Recommandation :** Le modèle **Forêt Aléatoire** est configuré par défaut comme le modèle de production en raison de sa robustesse face aux données complexes et de son score global élevé.

---

## 🌐 Fonctionnalités de l'Interface Web

L'application `app.py` intègre un design soigné (Sidebar aux couleurs de la charte agricole, champs de saisie contrastés pour une lecture optimale en texte noir sur fond blanc) articulé autour de 3 espaces :

1. **🔍 Onglet Prédiction :** Saisie des caractéristiques de la parcelle. Au clic sur **"Prédire la récolte"**, l'application affiche un badge de résultat coloré (Vert pour un succès, Rouge pour un risque d'échec), le détail des probabilités sous forme de graphiques, ainsi qu'un bloc de **recommandations agronomiques intelligentes** (ex: alerte en cas de pluviométrie critique).
2. **📊 Onglet Performances :** Analyse comparative des modèles avec affichage graphique des **Courbes ROC** et calcul en direct du diagramme d'**importance des variables** (Feature Importance) du Random Forest.
3. **ℹ️ Onglet À Propos :** Vulgarisation de la méthodologie (Pipeline ML, limites du modèle face aux chocs exogènes réels et nature des données).

---

## 🛠️ Dépannage (Troubleshooting)

* **Erreur au chargement des modèles (`models_ok = False`) :** Assurez-vous que le dossier `models/` est correctement placé au même niveau que `app.py` et qu'il contient bien les 5 fichiers `.pkl`. Vérifiez également que le fichier de données `agriculture_burundi.csv` se trouve bien à la racine.
* **Texte illisible dans la Sidebar :** Le CSS a été conçu pour forcer le texte sélectionné en noir (`#000000`) afin de contrer les thèmes sombres par défaut de Streamlit. En cas de problème d'affichage, videz le cache de votre navigateur.

---

## 👨‍💻 Équipe de Réalisation

Projet réalisé par INEZA BENISSE de **Bac 4 Génie Logiciel** de l'**Université Polytechnique de Gitega**.

* *Année universitaire : 2025–2026*
* *Outil à des fins strictement éducatives.*

