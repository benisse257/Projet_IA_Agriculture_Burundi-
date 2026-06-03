"""
generate_models.py
==================
Script de génération des modèles de Machine Learning
TP Intelligence Artificielle — Agriculture au Burundi
Université Polytechnique de Gitega — Bac 4 Génie Logiciel

Exécution :
    python generate_models.py

Fichiers produits dans le dossier models/ :
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── logistic_regression.pkl
    ├── scaler.pkl
    └── encoder.pkl   (liste des feature_cols)
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CHEMINS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "agriculture_burundi.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("  GÉNÉRATION DES MODÈLES — AgriPredict Burundi")
print("=" * 60)

# ─────────────────────────────────────────────
#  ÉTAPE 1 — CHARGEMENT
# ─────────────────────────────────────────────
print("\n[1/6] Chargement du dataset...")
df = pd.read_csv(CSV_PATH)
print(f"      ✓ {df.shape[0]} lignes × {df.shape[1]} colonnes chargées")
print(f"      ✓ Années : {df['annee'].min()} → {df['annee'].max()}")
print(f"      ✓ Provinces : {df['province'].nunique()}")
print(f"      ✓ Cultures  : {sorted(df['culture'].unique())}")

# ─────────────────────────────────────────────
#  ÉTAPE 2 — NETTOYAGE / VALEURS MANQUANTES
# ─────────────────────────────────────────────
print("\n[2/6] Traitement des valeurs manquantes...")

avant = df.isnull().sum().sum()

# pluviometrie_mm → médiane par province (robuste aux outliers)
df["pluviometrie_mm"] = df.groupby("province")["pluviometrie_mm"].transform(
    lambda x: x.fillna(x.median())
)

# utilisation_engrais → mode par culture (variable binaire)
df["utilisation_engrais"] = df.groupby("culture")["utilisation_engrais"].transform(
    lambda x: x.fillna(x.mode()[0])
)

# rendement et production → médiane par culture
df["rendement_t_ha"] = df.groupby("culture")["rendement_t_ha"].transform(
    lambda x: x.fillna(x.median())
)
df["production_totale_t"] = df.groupby("culture")["production_totale_t"].transform(
    lambda x: x.fillna(x.median())
)

# bonne_recolte (variable cible) → suppression des lignes
df = df.dropna(subset=["bonne_recolte"])
df["bonne_recolte"] = df["bonne_recolte"].astype(int)

apres = df.isnull().sum().sum()
print(f"      ✓ Valeurs manquantes : {avant} → {apres}")
print(f"      ✓ Dataset final : {df.shape[0]} lignes")

# ─────────────────────────────────────────────
#  ÉTAPE 3 — ENCODAGE & CONSTRUCTION DE X, y
# ─────────────────────────────────────────────
print("\n[3/6] Encodage des variables catégorielles...")

# One-Hot Encoding (drop_first=True évite la dummy variable trap)
# Référence supprimée : province=Bubanza, culture=Bananier, saison=A
df_encoded = pd.get_dummies(df, columns=["province", "culture", "saison"], drop_first=True)

# Colonnes à exclure de X
# - bonne_recolte  → variable cible
# - rendement_t_ha / production_totale_t → data leakage (définissent la cible)
# - annee          → identifiant temporel, pas une feature agronomique
EXCLUDE = ["bonne_recolte", "rendement_t_ha", "production_totale_t", "annee"]
feature_cols = [c for c in df_encoded.columns if c not in EXCLUDE]

X = df_encoded[feature_cols].copy()
y = df_encoded["bonne_recolte"].copy()

print(f"      ✓ {len(feature_cols)} features retenues")
print(f"      ✓ Distribution cible — Bonne: {y.mean()*100:.1f}%  Mauvaise: {(1-y.mean())*100:.1f}%")

# ─────────────────────────────────────────────
#  ÉTAPE 4 — NORMALISATION & SPLIT
# ─────────────────────────────────────────────
print("\n[4/6] Normalisation et division Train/Test...")

NUM_FEATURES = [
    "altitude_m", "pluviometrie_mm", "temperature_moy_C",
    "superficie_ha", "nb_menages", "utilisation_engrais", "acces_irrigation"
]
num_in_X = [c for c in NUM_FEATURES if c in feature_cols]

scaler = StandardScaler()
X[num_in_X] = scaler.fit_transform(X[num_in_X])

# Split 80/20 stratifié (préserve la proportion de la cible dans les 2 ensembles)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"      ✓ Train : {X_train.shape[0]} obs | Test : {X_test.shape[0]} obs")
print(f"      ✓ Proportion bonne_recolte — Train: {y_train.mean():.3f}  Test: {y_test.mean():.3f}")

# ─────────────────────────────────────────────
#  ÉTAPE 5 — ENTRAÎNEMENT DES 3 MODÈLES
# ─────────────────────────────────────────────
print("\n[5/6] Entraînement des modèles...")

# ── Arbre de Décision ──────────────────────────────────────────────────────
print("\n  → Arbre de Décision (max_depth=4, criterion='gini')...")
dt = DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=42)
dt.fit(X_train, y_train)

# ── Forêt Aléatoire ────────────────────────────────────────────────────────
print("  → Forêt Aléatoire (n_estimators=100)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ── Régression Logistique ──────────────────────────────────────────────────
print("  → Régression Logistique (max_iter=1000)...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# ── Évaluation ─────────────────────────────────────────────────────────────
print("\n  Résultats sur le jeu de test :")
print(f"  {'Modèle':<28} {'Accuracy':>10} {'F1-Score':>10} {'AUC':>8}")
print("  " + "-" * 58)

for model, name in [(dt, "Arbre de Décision"),
                    (rf, "Forêt Aléatoire"),
                    (lr, "Régression Logistique")]:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"  {name:<28} {acc*100:>9.2f}% {f1:>10.4f} {auc:>8.4f}")

# ─────────────────────────────────────────────
#  ÉTAPE 6 — SAUVEGARDE
# ─────────────────────────────────────────────
print("\n[6/6] Sauvegarde des fichiers .pkl...")

joblib.dump(dt,           os.path.join(MODELS_DIR, "decision_tree.pkl"))
joblib.dump(rf,           os.path.join(MODELS_DIR, "random_forest.pkl"))
joblib.dump(lr,           os.path.join(MODELS_DIR, "logistic_regression.pkl"))
joblib.dump(scaler,       os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, "encoder.pkl"))

for fname in ["decision_tree.pkl", "random_forest.pkl",
              "logistic_regression.pkl", "scaler.pkl", "encoder.pkl"]:
    path = os.path.join(MODELS_DIR, fname)
    size = os.path.getsize(path) / 1024
    print(f"      ✓ models/{fname:<35} ({size:.1f} Ko)")

print("\n" + "=" * 60)
print("  ✅ Tous les modèles ont été générés avec succès !")
print(f"  📁 Dossier : {MODELS_DIR}")
print("=" * 60)
print("\nPour lancer l'application :")
print("  cd app && streamlit run app.py")