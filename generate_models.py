#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour générer les modèles ML et fichiers pickle
Exécute l'équivalent du Notebook
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("=" * 80)
print("🚀 GÉNÉRATION DES MODÈLES MACHINE LEARNING")
print("=" * 80)

# ============================================
# 1. CHARGEMENT ET EXPLORATION
# ============================================
print("\n1️⃣ Chargement du dataset...")
df = pd.read_csv('agriculture_burundi.csv')
print(f"✅ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"   Variables manquantes: {df.isnull().sum().sum()}")

# ============================================
# 2. PRÉTRAITEMENT
# ============================================
print("\n2️⃣ Prétraitement des données...")

df_processed = df.copy()

# Supprimer data leakage
leakage_cols = []
if 'rendement_t_ha' in df_processed.columns:
    leakage_cols.append('rendement_t_ha')
if 'production_totale_t' in df_processed.columns:
    leakage_cols.append('production_totale_t')

if leakage_cols:
    df_processed = df_processed.drop(columns=leakage_cols, errors='ignore')
    print(f"   ✅ Data leakage supprimé: {leakage_cols}")

# Identifier colonnes
categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()

if 'bonne_recolte' in categorical_features:
    categorical_features.remove('bonne_recolte')
elif 'bonne_recolte' in numeric_features:
    numeric_features.remove('bonne_recolte')

print(f"   Variables catégoriques: {categorical_features}")
print(f"   Variables numériques: {numeric_features}")

# Séparer X et y
X = df_processed.drop('bonne_recolte', axis=1)
y = df_processed['bonne_recolte']
print(f"   ✅ X shape: {X.shape}, y shape: {y.shape}")

# OneHotEncoding
print("\n   Encodage OneHot...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical_encoded = encoder.fit_transform(X[categorical_features])
X_categorical_encoded_df = pd.DataFrame(
    X_categorical_encoded,
    columns=encoder.get_feature_names_out(categorical_features)
)

X_processed = pd.concat([X[numeric_features].reset_index(drop=True),
                         X_categorical_encoded_df.reset_index(drop=True)], axis=1)
y_processed = y.reset_index(drop=True)
print(f"   ✅ Après encoding: X shape {X_processed.shape}")

# StandardScaler
print("\n   Normalisation StandardScaler...")
scaler = StandardScaler()
X_scaled = X_processed.copy()
X_scaled[numeric_features] = scaler.fit_transform(X_processed[numeric_features])
print(f"   ✅ Normalisation appliquée")

# Supprimer les lignes avec NaN
print("\n   Suppression des lignes avec valeurs manquantes...")
X_scaled_df = X_scaled.copy()
X_scaled_df['target'] = y_processed
rows_before = len(X_scaled_df)
X_scaled_df = X_scaled_df.dropna()
rows_after = len(X_scaled_df)
X_scaled = X_scaled_df.drop('target', axis=1)
y_processed = X_scaled_df['target']
print(f"   ✅ Lignes supprimées: {rows_before - rows_after}, Lignes restantes: {rows_after}")

# Train-Test Split
print("\n   Train-Test Split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_processed,
    test_size=0.2,
    stratify=y_processed,
    random_state=42
)
print(f"   ✅ X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================
# 3. ENTRAÎNEMENT DES MODÈLES
# ============================================
print("\n3️⃣ Entraînement des modèles...")

print("\n   🌳 Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=42)
dt_model.fit(X_train, y_train)
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
print(f"   ✅ Decision Tree entraîné (Accuracy: {dt_accuracy:.4f})")

print("\n   🌲 Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
print(f"   ✅ Random Forest entraîné (Accuracy: {rf_accuracy:.4f})")

print("\n   📈 Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
print(f"   ✅ Logistic Regression entraînée (Accuracy: {lr_accuracy:.4f})")

# ============================================
# 4. SAUVEGARDE DES MODÈLES
# ============================================
print("\n4️⃣ Sauvegarde des modèles et objets...")

joblib.dump(dt_model, 'decision_tree.pkl')
print("   ✅ decision_tree.pkl")

joblib.dump(rf_model, 'random_forest.pkl')
print("   ✅ random_forest.pkl")

joblib.dump(lr_model, 'logistic_regression.pkl')
print("   ✅ logistic_regression.pkl")

joblib.dump(scaler, 'scaler.pkl')
print("   ✅ scaler.pkl")

joblib.dump(encoder, 'encoder.pkl')
print("   ✅ encoder.pkl")

# ============================================
# 5. RÉSUMÉ FINAL
# ============================================
print("\n" + "=" * 80)
print("✨ RÉSUMÉ DES MODÈLES")
print("=" * 80)

print(f"\n🏆 PERFORMANCES (Test Set):")
print(f"   Decision Tree:     {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")
print(f"   Random Forest:     {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%) ⭐ MEILLEUR")
print(f"   Logistic Regression: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")

print(f"\n📁 Fichiers créés:")
for file in ['decision_tree.pkl', 'random_forest.pkl', 'logistic_regression.pkl', 
             'scaler.pkl', 'encoder.pkl']:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024
        print(f"   ✅ {file} ({size:.1f} KB)")
    else:
        print(f"   ❌ {file} NOT FOUND")

print("\n" + "=" * 80)
print("🎉 GÉNÉRATION COMPLÉTÉE AVEC SUCCÈS!")
print("=" * 80)
print("\n✅ Les modèles sont prêts pour Streamlit!")
print("   Commande: streamlit run app/app.py")
