import streamlit as st
import requests
import pandas as pd
import time
import json
import plotly.express as px
from datetime import datetime
import os

# ─── CONFIG PAGE ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jan Data Factory",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CHARGEMENT CONFIG ────────────────────────────────────────────────────────
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "data_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    # Le générateur est maintenant intégré au backend principal qui tourne sur le port 8000
    API_URL = "http://localhost:8000/api"
except Exception:
    st.error("⚠️ Fichier data_config.json introuvable ou invalide.")
    st.stop()


# ─── HELPERS API ──────────────────────────────────────────────────────────────
def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None

def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None

def api_put(path: str, payload: dict):
    try:
        r = requests.put(f"{API_URL}{path}", json=payload, timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None

def api_delete(path: str):
    try:
        r = requests.delete(f"{API_URL}{path}", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


# ─── CSS GLOBAL ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0b0c0f; color: #e2e4e9; }

[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1f2230; }
[data-testid="stSidebar"] * { color: #c9ccd6 !important; }

.stButton > button {
    background: linear-gradient(135deg, #00e5a0 0%, #00bcd4 100%) !important;
    color: #0b0c0f !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 6px !important;
}

[data-testid="metric-container"] {
    background: #111318;
    border: 1px solid #1f2230;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] { color: #00e5a0 !important; font-family: 'Space Mono', monospace; }

.stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background: #111318 !important;
    border: 1px solid #1f2230 !important;
    color: #e2e4e9 !important;
    font-family: 'Space Mono', monospace !important;
}

.edit-row {
    background: #1a1d26;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── HEADER ───────────────────────────────────────────────────────────────────
col_title, col_time = st.columns([5, 1])
with col_title:
    st.markdown("""
    <div style="display:flex;align-items:baseline;gap:14px;">
        <span style="font-size:2rem;font-weight:800;background:linear-gradient(135deg,#00e5a0,#00bcd4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            ⚙ JAN DATA FACTORY
        </span>
        <span style="font-size:.75rem;color:#4a4f66;letter-spacing:.15em;font-family:'Space Mono';">SQL DASHBOARD</span>
    </div>
    """, unsafe_allow_html=True)
with col_time:
    st.markdown(f"<p style='text-align:right;font-family:Space Mono;font-size:11px;color:#4a4f66;'>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)

st.markdown("<hr style='margin:10px 0 24px 0; border-color:#1f2230;'>", unsafe_allow_html=True)


# ─── SIDEBAR : GÉNÉRATION ─────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("NOUVELLE GÉNÉRATION")
    proj = st.text_input("Projet", "dataset_test")
    top  = st.text_input("Thématique", placeholder="ex: avis clients")
    lab  = st.text_input("Label", placeholder="ex: positif")
    qty  = st.number_input("Entrées", min_value=1, max_value=500, value=5)
    
    # Templates
    templates = api_get("/templates") or {"templates": ["default"]}
    template_options = templates.get("templates", ["default"])
    selected_template = st.selectbox("Template de prompt", template_options)

    if st.button("▶ DÉMARRER", use_container_width=True):
        if top and lab:
            res = api_post("/generate", {
                "project": proj, 
                "topic": top, 
                "label": lab, 
                "count": qty,
                "template": selected_template
            })
            if res and "task_id" in res:
                st.session_state.current_tid = res["task_id"]
                st.success(f"Tâche {res['task_id']} lancée")
                st.rerun()

# ─── MONITORING TÂCHE ─────────────────────────────────────────────────────────
if "current_tid" in st.session_state:
    tid = st.session_state.current_tid
    task_status = api_get(f"/task/{tid}")
    if task_status:
        prog = task_status.get("progress", 0)
        stat = task_status.get("status", "Inconnu")
        st.write(f"**Tâche {tid}** : {stat} ({prog}%)")
        st.progress(prog / 100)
        if stat == "Terminé":
            st.balloons()
            del st.session_state.current_tid
            time.sleep(1)
            st.rerun()
        elif stat == "Erreur":
            st.error("Erreur détectée.")
            del st.session_state.current_tid
        else:
            time.sleep(2)
            st.rerun()

# ─── STATS GLOBALES ───────────────────────────────────────────────────────────
projects_data = api_get("/projects") or []
total_entries = 0
if projects_data:
    for p in projects_data:
        pdata = api_get(f"/data/{p['name']}")
        if pdata: total_entries += len(pdata)

c1, c2, c3 = st.columns(3)
c1.metric("Projets", len(projects_data))
c2.metric("Total Entrées", total_entries)
c3.metric("Status API", "OK" if projects_data is not None else "OFF")

st.markdown("<br>", unsafe_allow_html=True)

# ─── EXPLORATION AVEC VISUALISATIONS ET ÉDITION ───────────────────────────────
st.markdown("### 🔍 EXPLORER ET ÉDITER UN DATASET")

project_names = [p["name"] for p in projects_data]
search_col, filter_col = st.columns([2, 3])

with search_col:
    selected_p = st.selectbox("Choisir un projet", ["—"] + project_names, label_visibility="collapsed")

if selected_p != "—":
    raw_data = api_get(f"/data/{selected_p}")
    if raw_data:
        df = pd.DataFrame(raw_data)
        
        # ─── VISUALISATIONS ─────────────────────────────────────────────────
        st.markdown("#### 📊 Statistiques")
        viz_cols = st.columns(3)
        
        with viz_cols[0]:
            # Distribution des labels
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                if len(label_counts) > 0:
                    fig = px.pie(values=label_counts.values, names=label_counts.index, 
                                title="Distribution des labels", 
                                color_discrete_sequence=px.colors.sequential.Emrld)
                    fig.update_layout(paper_bgcolor='#0b0c0f', font_color='#e2e4e9', 
                                    showlegend=True, legend_font_color='#e2e4e9')
                    st.plotly_chart(fig, use_container_width=True)
        
        with viz_cols[1]:
            # Évolution temporelle
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                daily_counts = df.groupby(df['date'].dt.date).size().reset_index(name='count')
                if len(daily_counts) > 0:
                    fig = px.line(daily_counts, x='date', y='count', title="Évolution temporelle",
                                 color_discrete_sequence=['#00e5a0'])
                    fig.update_layout(paper_bgcolor='#0b0c0f', plot_bgcolor='#111318',
                                    font_color='#e2e4e9', xaxis_gridcolor='#1f2230',
                                    yaxis_gridcolor='#1f2230')
                    st.plotly_chart(fig, use_container_width=True)
        
        with viz_cols[2]:
            # Top labels bar chart
            if 'label' in df.columns:
                label_counts = df['label'].value_counts().head(10)
                if len(label_counts) > 0:
                    fig = px.bar(x=label_counts.index, y=label_counts.values,
                                title="Top Labels", color_discrete_sequence=['#00bcd4'])
                    fig.update_layout(paper_bgcolor='#0b0c0f', plot_bgcolor='#111318',
                                    font_color='#e2e4e9', xaxis_gridcolor='#1f2230',
                                    yaxis_gridcolor='#1f2230')
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ─── FILTRES AVANCÉS ────────────────────────────────────────────────
        filt_cols = st.columns(4)
        
        with filt_cols[0]:
            query = st.text_input("Rechercher...", placeholder="Mot-clé dans le texte")
        
        with filt_cols[1]:
            if 'label' in df.columns:
                all_labels = ['Tous'] + list(df['label'].unique())
                label_filter = st.selectbox("Filtrer par label", all_labels)
        
        with filt_cols[2]:
            export_format = st.selectbox("Format export", ["CSV", "JSON"])
        
        with filt_cols[3]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 Télécharger export"):
                if export_format == "JSON":
                    json_data = json.dumps(raw_data, ensure_ascii=False, indent=2)
                    st.download_button("⬇ JSON", json_data.encode('utf-8'), 
                                     f"{selected_p}_export.json", "application/json")
                else:
                    st.download_button("⬇ CSV", df.to_csv(index=False).encode('utf-8'),
                                     f"{selected_p}_export.csv", "text/csv")
        
        # Application des filtres
        if query:
            df = df[df['text'].str.contains(query, case=False, na=False)]
        
        if 'label_filter' in locals() and label_filter != 'Tous':
            df = df[df['label'] == label_filter]
        
        st.markdown(f"**{len(df)}** résultats affichés")
        
        # ─── TABLEAU ÉDITABLE ─────────────────────────────────────────────
        st.markdown("#### 📝 Données (cliquez pour éditer/supprimer)")
        
        rows_per_page = 5
        total_pages = max(1, (len(df) // rows_per_page) + (1 if len(df) % rows_per_page > 0 else 0))
        
        col_pag1, col_pag2, _ = st.columns([1, 2, 4])
        with col_pag1:
            page = st.number_input(f"Page", min_value=1, max_value=total_pages, step=1)
        
        start_idx = (page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(df))
        page_df = df.iloc[start_idx:end_idx]
        
        # Affichage avec édition inline
        for idx, row in page_df.iterrows():
            with st.container():
                edit_cols = st.columns([6, 2, 2])
                
                with edit_cols[0]:
                    text_key = f"text_{row.get('id', idx)}"
                    new_text = st.text_area("", value=row.get('text', ''), 
                                          key=text_key, height=80, label_visibility="collapsed")
                
                with edit_cols[1]:
                    label_key = f"label_{row.get('id', idx)}"
                    new_label = st.text_input("", value=row.get('label', ''),
                                            key=label_key, label_visibility="collapsed")
                
                with edit_cols[2]:
                    btn_cols = st.columns(2)
                    with btn_cols[0]:
                        if st.button("💾", key=f"save_{row.get('id', idx)}", help="Sauvegarder"):
                            data_id = row.get('id')
                            if data_id:
                                res = api_put(f"/data/{data_id}", {
                                    "text": new_text,
                                    "label": new_label
                                })
                                if res:
                                    st.success("Mis à jour!")
                                    st.rerun()
                    
                    with btn_cols[1]:
                        if st.button("🗑️", key=f"del_{row.get('id', idx)}", help="Supprimer"):
                            data_id = row.get('id')
                            if data_id:
                                res = api_delete(f"/data/delete/{data_id}")
                                if res:
                                    st.success("Supprimé!")
                                    st.rerun()
                
                st.markdown("---")
        
        # ─── AUGMENTATION DE DONNÉES ───────────────────────────────────────
        st.markdown("#### 🔄 Augmentation de données")
        aug_cols = st.columns([3, 2, 2])
        
        with aug_cols[0]:
            aug_entry_id = st.number_input("ID entrée à augmenter", min_value=1, step=1)
        
        with aug_cols[1]:
            aug_count = st.number_input("Variations", min_value=1, max_value=10, value=3)
        
        with aug_cols[2]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Générer variations"):
                res = api_post(f"/data/augment/{int(aug_entry_id)}", {"variations": int(aug_count)})
                if res:
                    st.success(f"{res.get('message', 'Variations créées')}!")
                    st.rerun()
    else:
        st.info("Projet vide ou introuvable.")
