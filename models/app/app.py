import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AgriPredict Burundi",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CSS PERSONNALISÉ
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fond principal */
    .stApp { background-color: #f0f4f0; }

    /* Sidebar — fond */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a472a 0%, #2d6a4f 100%);
    }

    /* Textes généraux sidebar : labels, titres, markdown → blanc */
    [data-testid="stSidebar"] * { color: white !important; }

    /* ══ SELECTBOX — fond blanc + texte NOIR ══ */
    [data-testid="stSidebar"] .stSelectbox > div > div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1.5px solid #52b788 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div[data-baseweb="select"] > div > div,
    [data-testid="stSidebar"] .stSelectbox > div > div[data-baseweb="select"] > div > div > div,
    [data-testid="stSidebar"] .stSelectbox span {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stSelectbox svg { fill: #1a472a !important; }

    /* Liste déroulante ouverte → texte NOIR */
    [data-baseweb="popover"] ul[role="listbox"],
    [data-baseweb="menu"] ul {
        background-color: #ffffff !important;
    }
    [data-baseweb="popover"] ul[role="listbox"] li,
    [data-baseweb="menu"] ul li {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    [data-baseweb="popover"] ul[role="listbox"] li:hover,
    [data-baseweb="menu"] ul li:hover {
        background-color: #d4edda !important;
        color: #000000 !important;
    }

    /* ══ NUMBER INPUT — fond blanc + texte NOIR ══ */
    [data-testid="stSidebar"] .stNumberInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1.5px solid #52b788 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stSidebar"] .stNumberInput button {
        background-color: #d4edda !important;
        color: #000000 !important;
        border: none !important;
    }

    /* ══ SLIDER — valeur affichée en NOIR sur fond blanc ══ */
    [data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
    [data-testid="stSidebar"] .stSlider p {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 5px !important;
        padding: 1px 5px !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #d4edda !important;
        font-weight: 600 !important;
    }

    /* ══ RADIO & TOGGLE — labels restent blancs ══ */
    [data-testid="stSidebar"] .stRadio label p { color: white !important; }
    [data-testid="stSidebar"] .stToggle label p { color: white !important; }

    /* Carte résultat BONNE */
    .result-good {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 28px 32px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(39,174,96,0.35);
        margin: 12px 0;
    }
    /* Carte résultat MAUVAISE */
    .result-bad {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        padding: 28px 32px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(192,57,43,0.35);
        margin: 12px 0;
    }
    .result-title { font-size: 2rem; font-weight: 800; margin-bottom: 6px; }
    .result-prob  { font-size: 1.2rem; opacity: 0.92; }

    /* Carte métrique */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 4px solid #2d6a4f;
    }
    .metric-label { font-size: 0.82rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.7rem; font-weight: 700; color: #1a472a; }

    /* Section titre */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a472a;
        border-left: 4px solid #2d6a4f;
        padding-left: 10px;
        margin: 20px 0 10px 0;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 60%, #52b788 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
    }
    .main-header h1 { font-size: 2rem; margin: 0; font-weight: 800; }
    .main-header p  { margin: 6px 0 0 0; opacity: 0.85; font-size: 1rem; }

    /* Badge modèle */
    .model-badge {
        display: inline-block;
        background: #e8f5e9;
        color: #1a472a;
        border: 1px solid #2d6a4f;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CHARGEMENT DES MODÈLES ET DONNÉES
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Charge les modèles sauvegardés."""
    models = {
        "Arbre de Décision":      joblib.load("models/decision_tree.pkl"),
        "Forêt Aléatoire":        joblib.load("models/random_forest.pkl"),
        "Régression Logistique":  joblib.load("models/logistic_regression.pkl"),
    }
    scaler       = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/encoder.pkl")   # liste des colonnes
    return models, scaler, feature_cols

@st.cache_data
def load_and_prepare_data():
    """Charge et prépare les données pour calculer les métriques."""
    df = pd.read_csv("agriculture_burundi.csv")

    # Nettoyage (identique au notebook)
    df["pluviometrie_mm"]    = df.groupby("province")["pluviometrie_mm"].transform(
        lambda x: x.fillna(x.median()))
    df["utilisation_engrais"] = df.groupby("culture")["utilisation_engrais"].transform(
        lambda x: x.fillna(x.mode()[0]))
    df["rendement_t_ha"]      = df.groupby("culture")["rendement_t_ha"].transform(
        lambda x: x.fillna(x.median()))
    df["production_totale_t"] = df.groupby("culture")["production_totale_t"].transform(
        lambda x: x.fillna(x.median()))
    df = df.dropna(subset=["bonne_recolte"])
    df["bonne_recolte"] = df["bonne_recolte"].astype(int)

    df_enc = pd.get_dummies(df, columns=["province", "culture", "saison"], drop_first=True)
    exclude = ["bonne_recolte", "rendement_t_ha", "production_totale_t", "annee"]
    feature_cols = [c for c in df_enc.columns if c not in exclude]

    X = df_enc[feature_cols].copy()
    y = df_enc["bonne_recolte"].copy()

    num_features = ["altitude_m","pluviometrie_mm","temperature_moy_C",
                    "superficie_ha","nb_menages","utilisation_engrais","acces_irrigation"]

    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    return X_test, y_test, feature_cols

@st.cache_data
def compute_metrics(_models, X_test, y_test):
    """Calcule accuracy, F1, AUC pour chaque modèle."""
    metrics = {}
    for name, model in _models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred),
            "auc":      roc_auc_score(y_test, y_proba),
            "fpr":      roc_curve(y_test, y_proba)[0],
            "tpr":      roc_curve(y_test, y_proba)[1],
        }
    return metrics

# ─────────────────────────────────────────────
#  FONCTION : CONSTRUIRE LE VECTEUR DE PRÉDICTION
# ─────────────────────────────────────────────
def build_input(province, culture, saison, altitude, pluie, temp,
                superficie, engrais, irrigation, nb_menages, feature_cols, scaler):
    """
    Reconstruit un DataFrame d'1 ligne avec exactement les colonnes
    attendues par le modèle, puis normalise les variables numériques.
    """
    # 1. Ligne vide avec toutes les colonnes à 0
    row = pd.DataFrame(columns=feature_cols)
    row.loc[0] = 0.0

    # 2. Variables numériques
    row["altitude_m"]           = float(altitude)
    row["pluviometrie_mm"]      = float(pluie)
    row["temperature_moy_C"]   = float(temp)
    row["superficie_ha"]        = float(superficie)
    row["utilisation_engrais"]  = float(engrais)
    row["acces_irrigation"]     = float(irrigation)
    row["nb_menages"]           = float(nb_menages)

    # 3. One-Hot province (drop_first a retiré 'Bubanza' comme référence)
    prov_col = f"province_{province}"
    if prov_col in feature_cols:
        row[prov_col] = 1.0
    # Si province == "Bubanza" (référence), toutes les colonnes province_ restent à 0 → correct

    # 4. One-Hot culture (drop_first a retiré 'Bananier' comme référence)
    cult_col = f"culture_{culture}"
    if cult_col in feature_cols:
        row[cult_col] = 1.0

    # 5. Saison (drop_first a retiré 'saison_A')
    if saison == "B" and "saison_B" in feature_cols:
        row["saison_B"] = 1.0

    # 6. Normalisation des colonnes numériques
    num_features = ["altitude_m","pluviometrie_mm","temperature_moy_C",
                    "superficie_ha","nb_menages","utilisation_engrais","acces_irrigation"]
    num_in = [c for c in num_features if c in feature_cols]
    row[num_in] = scaler.transform(row[num_in])

    return row

# ─────────────────────────────────────────────
#  CHARGEMENT
# ─────────────────────────────────────────────
try:
    models, scaler, feature_cols = load_models()
    X_test, y_test, _ = load_and_prepare_data()
    metrics = compute_metrics(models, X_test, y_test)
    models_ok = True
except Exception as e:
    models_ok = False
    load_error = str(e)

# ─────────────────────────────────────────────
#  SIDEBAR — FORMULAIRE DE SAISIE
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 AgriPredict Burundi")
    st.markdown("---")

    st.markdown("### 📍 Localisation & Culture")
    province = st.selectbox("Province", [
        "Bubanza","Bujumbura Rural","Bururi","Cankuzo","Cibitoke",
        "Gitega","Kayanza","Kirundo","Makamba","Muramvya",
        "Muyinga","Mwaro","Ngozi","Rutana","Ruyigi"
    ])
    culture = st.selectbox("Culture", [
        "Bananier","Haricot","Manioc","Maïs","Patate douce","Sorgho"
    ])
    saison = st.radio("Saison", ["A (mars–juin)", "B (sept–déc)"],
                      horizontal=True)
    saison_code = "A" if saison.startswith("A") else "B"

    st.markdown("---")
    st.markdown("### 🌦️ Conditions climatiques")
    altitude = st.slider("Altitude (m)", 500, 2500, 1500, step=10)
    pluie    = st.slider("Pluviométrie (mm)", 100, 1500, 700, step=10)
    temp     = st.slider("Température moyenne (°C)", 14.0, 30.0, 20.0, step=0.1)

    st.markdown("---")
    st.markdown("### 🌾 Informations de la parcelle")
    superficie  = st.number_input("Superficie cultivée (ha)", 0.5, 500.0, 10.0, step=0.5)
    nb_menages  = st.number_input("Nombre de ménages", 1, 5000, 150, step=10)
    engrais     = st.toggle("Utilisation d'engrais", value=False)
    irrigation  = st.toggle("Accès à l'irrigation", value=False)

    st.markdown("---")
    st.markdown("### 🤖 Modèle")
    model_choice = st.selectbox("Choisir le modèle", list(models.keys()) if models_ok else ["—"])

    st.markdown("---")
    predict_btn = st.button("🔍 Prédire la récolte", use_container_width=True)

# ─────────────────────────────────────────────
#  CONTENU PRINCIPAL
# ─────────────────────────────────────────────

# En-tête
st.markdown("""
<div class="main-header">
  <h1>🌿 Prédiction des Récoltes au Burundi</h1>
  <p>Outil d'aide à la décision agricole basé sur le Machine Learning — 15 provinces · 6 cultures · 2015–2023</p>
</div>
""", unsafe_allow_html=True)

if not models_ok:
    st.error(f"❌ Impossible de charger les modèles : {load_error}")
    st.info("Vérifiez que les fichiers `.pkl` sont dans le dossier `models/` et que `agriculture_burundi.csv` est présent.")
    st.stop()

# ─── ONGLETS ───
tab1, tab2, tab3 = st.tabs(["🔍 Prédiction", "📊 Performances des modèles", "ℹ️ À propos"])

# ══════════════════════════════════════════════
#  TAB 1 — PRÉDICTION
# ══════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-header">Récapitulatif de la parcelle</div>', unsafe_allow_html=True)

        # Tableau récapitulatif
        recap_data = {
            "Paramètre": [
                "Province", "Culture", "Saison", "Altitude", "Pluviométrie",
                "Température", "Superficie", "Ménages", "Engrais", "Irrigation"
            ],
            "Valeur": [
                province, culture, saison,
                f"{altitude} m", f"{pluie} mm", f"{temp} °C",
                f"{superficie} ha", str(nb_menages),
                "✅ Oui" if engrais else "❌ Non",
                "✅ Oui" if irrigation else "❌ Non"
            ]
        }
        st.dataframe(
            pd.DataFrame(recap_data),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            f'<span class="model-badge">🤖 Modèle : {model_choice}</span>',
            unsafe_allow_html=True
        )

    with col_result:
        st.markdown('<div class="section-header">Résultat de la prédiction</div>', unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("Analyse en cours..."):
                # Construction du vecteur
                input_df = build_input(
                    province, culture, saison_code,
                    altitude, pluie, temp, superficie,
                    int(engrais), int(irrigation), nb_menages,
                    feature_cols, scaler
                )

                model = models[model_choice]
                pred  = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]
                prob_good = proba[1] * 100
                prob_bad  = proba[0] * 100

            if pred == 1:
                st.markdown(f"""
                <div class="result-good">
                  <div class="result-title">✅ BONNE RÉCOLTE</div>
                  <div class="result-prob">Probabilité : <strong>{prob_good:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-bad">
                  <div class="result-title">⚠️ MAUVAISE RÉCOLTE</div>
                  <div class="result-prob">Probabilité : <strong>{prob_bad:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)

            # Jauge de probabilité
            st.markdown("**Probabilités détaillées :**")
            col_g, col_b = st.columns(2)
            col_g.metric("🟢 Bonne récolte",  f"{prob_good:.1f}%")
            col_b.metric("🔴 Mauvaise récolte", f"{prob_bad:.1f}%")

            # Graphique en barres horizontales
            fig, ax = plt.subplots(figsize=(6, 1.8))
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")
            ax.barh(["Mauvaise"], [prob_bad],  color="#e74c3c", height=0.4)
            ax.barh(["Bonne"],    [prob_good], color="#27ae60", height=0.4)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilité (%)")
            ax.axvline(50, color="gray", linestyle="--", linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            # Métriques du modèle sélectionné
            st.markdown("---")
            st.markdown(f"**Performances du modèle sélectionné :** `{model_choice}`")
            m = metrics[model_choice]
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""<div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{m['accuracy']*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">{m['f1']:.3f}</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class="metric-card">
                <div class="metric-label">AUC</div>
                <div class="metric-value">{m['auc']:.3f}</div>
            </div>""", unsafe_allow_html=True)

            # Recommandations agronomiques
            st.markdown("---")
            st.markdown("**💡 Recommandations agronomiques :**")
            recommandations = []
            if not engrais:
                recommandations.append("🌿 Utiliser des engrais organiques ou minéraux pour améliorer les rendements")
            if not irrigation and pluie < 500:
                recommandations.append("💧 Pluviométrie faible — envisager un système d'irrigation ou de collecte d'eau")
            if pluie < 400:
                recommandations.append("⚠️ Pluviométrie critique — préférer des cultures résistantes à la sécheresse (Sorgho, Manioc)")
            if temp > 27:
                recommandations.append("🌡️ Température élevée — choisir des variétés thermotolérantes")
            if pred == 0:
                recommandations.append("📋 Mauvaise récolte anticipée — contacter le service agricole provincial pour un appui")
            if not recommandations:
                recommandations.append("✅ Conditions favorables — maintenir les bonnes pratiques agricoles en place")
            for r in recommandations:
                st.markdown(f"- {r}")

        else:
            st.info("👈 Renseignez les paramètres dans le panneau gauche et cliquez sur **Prédire la récolte**.")

            # Afficher les 4 scénarios du TP en exemple
            st.markdown("**Exemples de scénarios (du TP) :**")
            exemples = pd.DataFrame({
                "Province": ["Kayanza", "Bubanza", "Gitega", "Cibitoke"],
                "Culture":  ["Maïs", "Manioc", "Haricot", "Patate douce"],
                "Pluie (mm)": [920, 550, 430, 810],
                "Engrais": ["Oui", "Non", "Non", "Oui"],
            })
            st.dataframe(exemples, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
#  TAB 2 — PERFORMANCES
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Comparaison des 3 modèles</div>', unsafe_allow_html=True)

    # Tableau de synthèse
    perf_data = []
    for name, m in metrics.items():
        perf_data.append({
            "Modèle":   name,
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "F1-Score": f"{m['f1']:.4f}",
            "AUC":      f"{m['auc']:.4f}",
        })
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    col_roc, col_bar = st.columns(2)

    with col_roc:
        st.markdown("**Courbes ROC**")
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = {"Arbre de Décision": "steelblue",
                  "Forêt Aléatoire":   "forestgreen",
                  "Régression Logistique": "darkorange"}
        for name, m in metrics.items():
            ax.plot(m["fpr"], m["tpr"], label=f"{name} (AUC={m['auc']:.3f})",
                    color=colors[name], linewidth=2)
        ax.plot([0,1],[0,1],"k--",linewidth=0.8,label="Aléatoire (AUC=0.5)")
        ax.set_xlabel("Taux Faux Positifs")
        ax.set_ylabel("Taux Vrais Positifs")
        ax.set_title("Courbes ROC — 3 modèles")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_bar:
        st.markdown("**Accuracy vs AUC**")
        names  = list(metrics.keys())
        short  = ["Arbre", "Forêt", "LogReg"]
        accs   = [metrics[n]["accuracy"] for n in names]
        aucs   = [metrics[n]["auc"]      for n in names]

        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(6, 5))
        w = 0.35
        b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#2d6a4f", alpha=0.85)
        b2 = ax.bar(x + w/2, aucs, w, label="AUC",      color="#52b788", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(short)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Accuracy & AUC par modèle")
        ax.legend()
        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Importance des variables — Forêt Aléatoire
    st.markdown("---")
    st.markdown('<div class="section-header">Importance des variables — Forêt Aléatoire</div>', unsafe_allow_html=True)
    rf_model = models["Forêt Aléatoire"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    importances.plot.barh(ax=ax, color="#2d6a4f", alpha=0.85, edgecolor="white")
    ax.set_title("Top 15 variables importantes (Forêt Aléatoire)")
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.3, axis="x")
    for spine in ["top","right"]: ax.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 3 — À PROPOS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    ## 🌿 AgriPredict Burundi

    Application développée dans le cadre du **TP Intelligence Artificielle appliquée à l'Agriculture**  
    *Université Polytechnique de Gitega — Bac 4 Génie Logiciel*

    ---

    ### 📦 Dataset
    - **1 620 observations** couvrant 9 années (2015–2023)
    - **15 provinces** du Burundi, **6 cultures**, **2 saisons** agricoles
    - Variable cible : `bonne_recolte` (1 = bonne, 0 = mauvaise)

    ### 🤖 Modèles entraînés
    | Modèle | Avantages | Inconvénients |
    |---|---|---|
    | Arbre de Décision | Interprétable, règles visibles | Surapprentissage facile |
    | Forêt Aléatoire | Robuste, haute accuracy | Moins interprétable |
    | Régression Logistique | Probabilités calibrées, rapide | Suppose la linéarité |

    ### ⚙️ Pipeline ML
    1. Nettoyage (imputation par médiane/mode par groupe)
    2. Encodage One-Hot (province, culture, saison)
    3. Normalisation StandardScaler
    4. Division Train/Test 80/20 stratifiée

    ### ⚠️ Limites
    - Basé sur des données simulées — à valider sur le terrain
    - Ne capture pas les chocs exogènes (conflits, prix des intrants)
    - Outil d'**aide à la décision**, pas un oracle
    """)

# Footer
st.markdown("---")
st.markdown(
    "<center><small>🌱 AgriPredict Burundi · Université Polytechnique de Gitega · Bac 4 Génie Logiciel</small></center>",
    unsafe_allow_html=True
)