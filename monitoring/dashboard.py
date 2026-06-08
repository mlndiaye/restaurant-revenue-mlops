"""
Monitoring dashboard — détection de drift en temps réel.
Lance avec : PYTHONPATH=src streamlit run monitoring/dashboard.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from restaurant_revenue.utils.drift import detect_drift

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REF_PATH = PROJECT_ROOT / "data" / "processed" / "train_processed.csv"
DEFAULT_CURRENT_PATH = PROJECT_ROOT / "data" / "processed" / "test_processed.csv"
THRESHOLD = 0.05

st.set_page_config(
    page_title="Monitoring — Restaurant Revenue",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Données de production")
use_default = st.sidebar.checkbox("Utiliser données test (démo)", value=True)

if not use_default:
    uploaded = st.sidebar.file_uploader("Charger un CSV de production", type="csv")
    if uploaded is None:
        st.info("Chargez un fichier CSV ou cochez 'Utiliser données test'.")
        st.stop()
    current_df = pd.read_csv(uploaded)
else:
    if not DEFAULT_CURRENT_PATH.exists():
        st.error(f"Fichier introuvable : {DEFAULT_CURRENT_PATH}")
        st.stop()
    current_df = pd.read_csv(DEFAULT_CURRENT_PATH)

if not REF_PATH.exists():
    st.error(f"Données de référence introuvables : {REF_PATH}")
    st.stop()

ref_df = pd.read_csv(REF_PATH)

# ── Calcul drift ──────────────────────────────────────────────────────────────
report = detect_drift(ref_df, current_df, threshold=THRESHOLD)
metrics = report["metrics"]
drifted_count = metrics["drifted_features_count"]
total_count = metrics["total_features_count"]
drift_ratio = metrics["drift_ratio"]

# ── En-tête ───────────────────────────────────────────────────────────────────
st.title("📊 Restaurant Revenue — Monitoring Dashboard")
st.caption(
    f"Référence : {metrics['reference_count']} obs. · "
    f"Production : {metrics['current_count']} obs. · "
    f"Seuil p-value : {THRESHOLD}"
)

if report["drift_detected"]:
    st.error(f"⚠️ DRIFT DÉTECTÉ — {drifted_count}/{total_count} features ({drift_ratio:.0%})")
else:
    st.success(f"✅ Pas de drift significatif — {drifted_count}/{total_count} feature(s) driftée(s) ({drift_ratio:.0%})")

# ── KPIs ──────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Observations référence", f"{metrics['reference_count']:,}")
col2.metric("Observations production", f"{metrics['current_count']:,}")
col3.metric("Features driftées", f"{drifted_count} / {total_count}")
col4.metric("Ratio drift", f"{drift_ratio:.1%}", delta_color="inverse")

st.divider()

# ── Tableau features ──────────────────────────────────────────────────────────
st.subheader("Détail par feature")

rows = []
for feat, info in report["features"].items():
    rows.append({
        "Feature": feat,
        "Type": info["type"],
        "Test": info["test"],
        "Statistique": round(info["statistic"], 4),
        "p-value": round(info["p_value"], 4),
        "Drift": "⚠️ Oui" if info["drift_detected"] else "✅ Non",
    })

df_table = pd.DataFrame(rows).sort_values("p-value")

def highlight_drift(row):
    if row["Drift"].startswith("⚠️"):
        return ["background-color: #fff3cd"] * len(row)
    return [""] * len(row)

st.dataframe(
    df_table.style.apply(highlight_drift, axis=1),
    use_container_width=True,
    height=400,
)

st.divider()

# ── Distributions — features driftées ────────────────────────────────────────
drifted_feats = [f for f, info in report["features"].items() if info["drift_detected"]]

if drifted_feats:
    st.subheader(f"Distributions — {len(drifted_feats)} feature(s) driftée(s)")
    cols = st.columns(2)
    for i, feat in enumerate(drifted_feats[:8]):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(5, 3))
            is_numeric = report["features"][feat]["type"] == "numerical"
            if is_numeric:
                ax.hist(ref_df[feat].dropna(), bins=20, alpha=0.6, label="Référence", color="steelblue", density=True)
                ax.hist(current_df[feat].dropna(), bins=20, alpha=0.6, label="Production", color="coral", density=True)
            else:
                ref_counts = ref_df[feat].value_counts(normalize=True).sort_index()
                cur_counts = current_df[feat].value_counts(normalize=True).sort_index()
                x = np.arange(len(ref_counts))
                ax.bar(x - 0.2, ref_counts.values, 0.4, label="Référence", color="steelblue", alpha=0.8)
                ax.bar(x + 0.2, cur_counts.reindex(ref_counts.index, fill_value=0).values, 0.4, label="Production", color="coral", alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(ref_counts.index, rotation=30, ha="right")
            p = report["features"][feat]["p_value"]
            ax.set_title(f"{feat}  (p={p:.4f})", fontsize=10)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
else:
    st.info("Aucune feature driftée à afficher.")

# ── Comparaison statistiques descriptives ────────────────────────────────────
st.divider()
st.subheader("Statistiques descriptives — Référence vs Production")

num_cols = [c for c in ref_df.columns if pd.api.types.is_numeric_dtype(ref_df[c])]
common = [c for c in num_cols if c in current_df.columns]

ref_stats = ref_df[common].agg(["mean", "std", "median"]).T.round(2)
cur_stats = current_df[common].agg(["mean", "std", "median"]).T.round(2)

ref_stats.columns = ["Moy. réf.", "Std réf.", "Méd. réf."]
cur_stats.columns = ["Moy. prod.", "Std prod.", "Méd. prod."]

st.dataframe(pd.concat([ref_stats, cur_stats], axis=1), use_container_width=True, height=350)
