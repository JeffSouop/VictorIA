"""
performance.py – Streamlit page for model evaluation metrics.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from training.pipeline import load_metrics, models_are_ready, train_models


def render_performance_page(force_retrain: bool = False):
    st.markdown("## 📈 Performance du modèle")

    metrics_preview = load_metrics()
    is_real = (
        metrics_preview
        and metrics_preview.get("data_source") == "football-data.org-real"
    )
    eval_desc = (
        "Métriques calculées sur une saison de test récente (split chronologique)."
        if is_real
        else "Métriques calculées sur un hold-out (données synthétiques ou anciennes)."
    )
    st.markdown(
        f"<p style='color:#6b7280'>{eval_desc}</p>",
        unsafe_allow_html=True,
    )

    if force_retrain:
        with st.spinner("Réentraînement sur données réelles (plusieurs minutes) …"):
            train_models(force=True, refresh_data=True)
        st.cache_resource.clear()
        st.success("Modèles réentraînés.")
        st.rerun()

    if not models_are_ready():
        st.warning(
            "Aucun modèle entraîné. Lancez `python train.py` ou utilisez le bouton ci-dessous."
        )
        if st.button("🚀 Entraîner les modèles maintenant", type="primary"):
            with st.spinner("Entraînement sur données réelles …"):
                train_models(force=True)
            st.cache_resource.clear()
            st.rerun()
        return

    metrics = load_metrics()
    if not metrics:
        st.info(
            "Modèles présents mais métriques absentes. Réentraînez avec "
            "`python train.py` pour générer le rapport."
        )
        if st.button("Générer les métriques"):
            with st.spinner("Évaluation …"):
                train_models(force=True)
            st.cache_resource.clear()
            st.rerun()
        return

    source = metrics.get("data_source", "—")
    source_label = (
        "Données réelles (football-data.org)"
        if source == "football-data.org-real"
        else ("Synthétiques" if source == "synthetic" else source)
    )
    st.info(f"Source d'entraînement : **{source_label}**")

    c1, c2, c3, c4 = st.columns(4)
    acc = metrics.get("accuracy")
    ll = metrics.get("log_loss")
    base = metrics.get("baseline_accuracy")
    gain = metrics.get("improvement_vs_baseline")

    c1.metric("Accuracy (test)", f"{acc:.1%}" if acc else "—")
    c2.metric("Log loss", f"{ll:.3f}" if ll else "—")
    c3.metric("Baseline (majorité)", f"{base:.1%}" if base else "—")
    c4.metric("Gain vs baseline", f"{gain:+.1%}" if gain is not None else "—")

    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.metric("Matchs entraînement", metrics.get("n_train", metrics.get("n_samples", "—")))
    with meta_cols[1]:
        st.metric("Matchs test", metrics.get("n_test", "—"))
    with meta_cols[2]:
        st.metric("Total échantillons", metrics.get("n_samples", "—"))

    st.caption(
        f"Dernier entraînement : {metrics.get('trained_at', '—')} · "
        f"Mode : {metrics.get('eval_mode', '—')}"
    )

    if metrics.get("competitions"):
        st.caption(
            f"Compétitions : {', '.join(metrics['competitions'])} · "
            f"Saisons : {metrics.get('seasons_used', metrics.get('seasons_requested', '—'))}"
        )

    if metrics.get("label_distribution"):
        dist = metrics["label_distribution"]
        st.caption(
            f"Répartition labels — Domicile: {dist.get('HomeWin', 0)} · "
            f"Nul: {dist.get('Draw', 0)} · Extérieur: {dist.get('AwayWin', 0)}"
        )

    st.markdown("### Matrice de confusion")
    cm = metrics.get("confusion_matrix")
    labels = metrics.get("class_labels", ["HomeWin", "Draw", "AwayWin"])
    if cm:
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Prédit {l}" for l in labels],
                y=[f"Réel {l}" for l in labels],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 14, "color": "white"},
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=360,
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Scores cross-validation (entraînement)")
    cv = metrics.get("cv_scores", {})
    active = metrics.get("active_models", [])
    rows = []
    for name, score in cv.items():
        if score is None and name not in active:
            continue
        rows.append(
            {
                "Modèle": name,
                "CV Accuracy": f"{score:.1%}" if score else "—",
                "Actif": "✅" if name in active else "⏸️",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if metrics.get("skipped_fetch"):
        with st.expander("Saisons/compétitions ignorées"):
            st.json(metrics["skipped_fetch"])

    if source == "football-data.org-real":
        st.markdown(
            """
            <div style="background:#1a2035;border:1px solid #374151;border-radius:12px;
                        padding:1rem;margin-top:1rem;color:#9ca3af;font-size:0.85rem">
            Ces métriques sont calculées sur de <b>vrais matchs historiques</b>.
            Elles restent indicatives : la performance passée ne garantit pas les résultats futurs.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background:#1a2035;border:1px solid #374151;border-radius:12px;
                        padding:1rem;margin-top:1rem;color:#9ca3af;font-size:0.85rem">
            ⚠️ Modèle entraîné sur données synthétiques. Lancez
            <code>python train.py</code> avec votre clé API pour un entraînement réel.
            </div>
            """,
            unsafe_allow_html=True,
        )
