"""
Stock Agent - Eureka | Interface Streamlit

- Dados: upload do Excel (ou geração de dados sintéticos) e exploração.
- Treinar: treino PPO (com pré-treino por imitação) por perfil x cultura x tipo de agente; download do modelo ZIP.
- Simulação & Inferência: corre uma temporada com o agente e compara com FEFO + Procura e Venda Imediata;
  inspeção das decisões dia a dia com probabilidades e explicação; exemplo de pedido/resposta do Web Service.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_agent.schema import read_excel, profiles_and_cultures, to_excel_bytes, readme_frame, SHEETS, DataValidationError
from stock_agent.synthetic import generate, CULTURES
from stock_agent.simulator import SimParams
from stock_agent.training import TrainConfig, train_generator
from stock_agent.model_io import load_zip, save_to_storage, storage_path, DEFAULT_PROFILE
from stock_agent.inference import (simulate_season, comparison_table, report_excel_bytes, decide_snapshot,
                                   snapshot_from_excel, AGENT_NAME)

MODELS_ROOT = os.environ.get("STOCK_AGENT_MODELS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_storage"))
COLORS = {AGENT_NAME: "#ea580c", "FEFO + Procura": "#0ea5e9", "Venda Imediata": "#94a3b8"}

st.set_page_config(page_title="Stock Agent - Eureka", page_icon="📦", layout="wide", initial_sidebar_state="expanded")

lang = st.sidebar.selectbox("🌐 Language / Idioma", ["Português", "English"], index=0)


def T(pt, en):
    return en if lang == "English" else pt


st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #1e293b; }
    section[data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
    .custom-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03); }
    .main-title { background: linear-gradient(135deg, #ff8c00 0%, #ff4500 100%); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; font-weight: 700; font-size: 2.6rem; margin-bottom: 5px; }
    .subtitle { color: #64748b; font-size: 1.1rem; margin-bottom: 25px; }
    .console-header { background-color: #0f172a; border-radius: 8px 8px 0 0; border: 1px solid #1e293b; border-bottom: none;
        padding: 8px 15px; font-size: 0.85rem; color: #ff8c00; font-family: monospace; }
    .console-box { background-color: #020617; border: 1px solid #1e293b; border-radius: 0 0 8px 8px; padding: 15px;
        font-family: 'Courier New', Courier, monospace; font-size: 0.85rem; color: #38bdf8; height: 360px; overflow-y: auto;
        white-space: pre-wrap; box-shadow: inset 0 2px 10px rgba(0,0,0,0.5); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-size: 1.05rem; font-weight: 600; color: #64748b; }
    .stTabs [data-baseweb="tab"]:hover { color: #ff8c00; }
    .stTabs [aria-selected="true"] { color: #ff8c00 !important; border-bottom-color: #ff8c00 !important; }
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; text-align: center; }
    .metric-card.hero { background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%); border: 1.5px solid #fdba74; }
    .metric-val { font-size: 1.55rem; font-weight: 700; color: #ea580c; margin: 4px 0; }
    .metric-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; font-weight: 600; }
    .metric-sub { font-size: 0.8rem; color: #475569; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; }
    .badge-green { background-color: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
    .badge-amber { background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .badge-red { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Stock Agent - Eureka</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">{T("Agente de IA que decide que lotes ou armazéns colocar à venda, maximizando o lucro e minimizando o desperdício",
                                     "AI agent that decides which lots or warehouses to put on sale, maximizing profit and minimizing waste")}</p>',
            unsafe_allow_html=True)

for key, default in [("data", None), ("data_name", None), ("train_log", ""), ("trained_zip", None), ("trained_config", None),
                     ("trained_history", None), ("sim_results", None), ("sim_key", None), ("synthetic_bytes", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def metric_card(label, value, sub="", hero=False):
    return (f'<div class="metric-card{" hero" if hero else ""}"><div class="metric-label">{label}</div>'
            f'<div class="metric-val">{value}</div><div class="metric-sub">{sub}</div></div>')


def eur(x):
    return f"{x:,.0f} €".replace(",", " ")


def kg(x):
    return f"{x:,.0f} kg".replace(",", " ")


# =====================================================================================================
# SIDEBAR
# =====================================================================================================
with st.sidebar:
    st.markdown(f"### 📥 {T('Dados', 'Data')}")
    up = st.file_uploader(T("Excel com Armazéns, Lotes, Leituras e Mercado", "Excel with Armazens, Lotes, Leituras and Mercado sheets"),
                          type=["xlsx"])
    if up is not None and st.session_state.data_name != up.name:
        try:
            st.session_state.data = read_excel(up.getvalue())
            st.session_state.data_name = up.name
            st.session_state.sim_results = None
        except (DataValidationError, ValueError) as e:
            st.error(str(e))

    with st.expander(T("🧪 Gerar dados sintéticos", "🧪 Generate synthetic data"), expanded=st.session_state.data is None):
        g_perfil = st.text_input(T("Perfil (produtor)", "Profile (producer)"), "Produtor_Demo")
        g_frutas = st.multiselect(T("Culturas", "Crops"), list(CULTURES), default=["kiwi_hayward", "pear"],
                                  format_func=lambda k: CULTURES[k]["cultura"])
        c1, c2 = st.columns(2)
        g_wh = c1.number_input(T("Armazéns", "Warehouses"), 1, 20, 6)
        g_ca = c2.slider(T("% com AC", "% with CA"), 0, 100, 50, help=T("Atmosfera controlada", "Controlled atmosphere"))
        c1, c2 = st.columns(2)
        g_seasons = c1.number_input(T("Temporadas", "Seasons"), 2, 12, 5)
        g_lots = c2.number_input(T("Lotes/temporada", "Lots/season"), 4, 80, 24)
        g_seed = st.number_input("Seed", 0, 99999, 42)
        if st.button(T("Gerar", "Generate"), width="stretch", disabled=not g_frutas):
            raw = generate(perfil=g_perfil, fruit_keys=g_frutas, n_warehouses=int(g_wh), frac_controlled=g_ca / 100.0,
                           n_seasons=int(g_seasons), lots_per_season=int(g_lots), seed=int(g_seed))
            b = to_excel_bytes(raw, readme=readme_frame())
            st.session_state.synthetic_bytes = b
            st.session_state.data = read_excel(b)
            st.session_state.data_name = "dados_sinteticos.xlsx"
            st.session_state.sim_results = None
        if st.session_state.synthetic_bytes is not None:
            st.download_button(T("📥 Descarregar Excel gerado", "📥 Download generated Excel"), st.session_state.synthetic_bytes,
                               file_name="dados_sinteticos_stock_agent.xlsx", width="stretch")

    data = st.session_state.data
    if data is not None:
        st.success(T(f"Dados: {st.session_state.data_name}", f"Data: {st.session_state.data_name}"))
        pc = profiles_and_cultures(data)
        st.markdown("---")
        st.markdown(f"### 🎯 {T('Modelo', 'Model')}")
        perfil = st.selectbox(T("Perfil", "Profile"), sorted(pc["perfil"].unique()))
        cultura = st.selectbox(T("Cultura", "Crop"), sorted(pc.loc[pc["perfil"] == perfil, "cultura"].unique()))
        mode_label = st.radio(T("Tipo de agente", "Agent type"),
                              [T("Agente de Lotes (diário)", "Lot Agent (daily)"), T("Agente de Armazéns (semanal)", "Warehouse Agent (weekly)")])
        mode = "lot" if mode_label.startswith(T("Agente de Lotes", "Lot Agent")) else "warehouse"
        st.caption(T("Lotes: todos os dias escolhe 0, 1 ou vários lotes para colocar à venda.\n\nArmazéns: de 7 em 7 dias escolhe no máximo 1 armazém para abrir (todo o stock vai para venda).",
                     "Lots: every day picks 0, 1 or several lots to put on sale.\n\nWarehouses: every 7 days opens at most 1 warehouse (all its stock goes on sale)."))

    st.markdown("---")
    with st.expander(T("ℹ️ Estrutura do Excel", "ℹ️ Excel structure")):
        for sheet, cols in SHEETS.items():
            st.markdown(f"**{sheet}**: " + ", ".join(f"`{c}`" for c in cols))

if data is None:
    st.info(T("👋 Carregue um Excel na barra lateral ou gere dados sintéticos para começar.",
              "👋 Upload an Excel file in the sidebar or generate synthetic data to start."))
    st.stop()

pc_row = pc[(pc["perfil"] == perfil) & (pc["cultura"] == cultura)].iloc[0]
all_seasons = [int(x) for x in pc_row["temporadas"]]

tab_data, tab_train, tab_test = st.tabs([T("📊 Dados", "📊 Data"), T("🏋️ Treinar Modelo", "🏋️ Train Model"),
                                         T("🧪 Simulação & Inferência", "🧪 Simulation & Inference")])

# =====================================================================================================
# TAB DADOS
# =====================================================================================================
with tab_data:
    lotes = data["Lotes"][(data["Lotes"]["perfil"] == perfil) & (data["Lotes"]["cultura"] == cultura)]
    arm = data["Armazens"][data["Armazens"]["perfil"] == perfil]
    c = st.columns(5)
    c[0].markdown(metric_card(T("Armazéns", "Warehouses"), len(arm), T(f"{int(arm['controlado'].sum())} com AC", f"{int(arm['controlado'].sum())} with CA")), unsafe_allow_html=True)
    c[1].markdown(metric_card(T("Lotes", "Lots"), len(lotes), cultura), unsafe_allow_html=True)
    c[2].markdown(metric_card(T("Temporadas", "Seasons"), len(all_seasons), f"{min(all_seasons)}–{max(all_seasons)}"), unsafe_allow_html=True)
    c[3].markdown(metric_card(T("Stock total", "Total stock"), f"{lotes['quantidade_kg'].sum() / 1000:,.0f} t", T("todas as temporadas", "all seasons")), unsafe_allow_html=True)
    c[4].markdown(metric_card(T("Fruta (Lifecycle)", "Fruit (Lifecycle)"), lotes["tipo_fruta"].iloc[0], ""), unsafe_allow_html=True)
    st.write("")

    merc = data["Mercado"][(data["Mercado"]["perfil"] == perfil) & (data["Mercado"]["cultura"] == cultura)].sort_values("data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merc["data"], y=merc["vendas_kg"].rolling(7, min_periods=1).mean(), name=T("Procura real (média 7d)", "Actual demand (7d avg)"),
                             line=dict(color="#0ea5e9")))
    fig.add_trace(go.Scatter(x=merc["data"], y=merc["procura_prevista_kg"].rolling(7, min_periods=1).mean(), name=T("Procura prevista (média 7d)", "Forecast demand (7d avg)"),
                             line=dict(color="#0ea5e9", dash="dot")))
    fig.add_trace(go.Scatter(x=merc["data"], y=merc["preco_kg"], name=T("Preço (€/kg)", "Price (€/kg)"), yaxis="y2", line=dict(color="#ea580c")))
    fig.update_layout(title=T("Mercado", "Market"), height=360, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15),
                      yaxis=dict(title="kg/dia"), yaxis2=dict(title="€/kg", overlaying="y", side="right"), plot_bgcolor="#ffffff")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns([1, 1])
    with c1:
        by_wh = lotes.groupby(["temporada", "armazem_id"])["quantidade_kg"].sum().reset_index()
        fig = go.Figure()
        for w in sorted(by_wh["armazem_id"].unique()):
            d = by_wh[by_wh["armazem_id"] == w]
            ca = bool(arm.set_index("armazem_id").loc[w, "controlado"]) if w in set(arm["armazem_id"]) else False
            fig.add_trace(go.Bar(x=d["temporada"].astype(str), y=d["quantidade_kg"], name=f"{w} ({'AC' if ca else T('Frio', 'Cold')})"))
        fig.update_layout(barmode="stack", title=T("Stock por armazém e temporada (kg)", "Stock per warehouse and season (kg)"),
                          height=340, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="#ffffff")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown(f"**{T('Armazéns', 'Warehouses')}**")
        st.dataframe(arm.drop(columns=["controlado"]), hide_index=True, width="stretch", height=300)
    with st.expander(T("Ver lotes", "Show lots")):
        st.dataframe(lotes, hide_index=True, width="stretch")

# =====================================================================================================
# TAB TREINO
# =====================================================================================================
with tab_train:
    st.markdown(f"#### ⚙️ {T('Configuração do treino', 'Training configuration')} — {perfil} / {cultura} / "
                f"{T('Lotes', 'Lots') if mode == 'lot' else T('Armazéns', 'Warehouses')}")
    c1, c2 = st.columns(2)
    default_val = all_seasons[-1:] if len(all_seasons) > 1 else []
    val_seasons = c2.multiselect(T("Temporadas de teste/validação", "Test/validation seasons"), all_seasons, default=default_val,
                                 help=T("Usadas para escolher o melhor modelo e, depois, na simulação.", "Used to pick the best model and later in simulation."))
    train_seasons = c1.multiselect(T("Temporadas de treino", "Training seasons"), all_seasons,
                                   default=[s for s in all_seasons if s not in default_val])

    with st.expander(T("🧠 Hiperparâmetros PPO", "🧠 PPO hyperparameters"), expanded=True):
        c = st.columns(4)
        episodes = c[0].number_input(T("Episódios", "Episodes"), 10, 20000, 400 if mode == "lot" else 600, step=50,
                                     help=T("1 episódio = 1 temporada simulada (com perturbações aleatórias)", "1 episode = 1 simulated season (randomly perturbed)"))
        ep_upd = c[1].number_input(T("Episódios por atualização", "Episodes per update"), 1, 64, 4)
        lr = c[2].number_input("Learning rate", 1e-5, 1e-2, 3e-4, format="%.5f")
        gamma = c[3].number_input("Gamma", 0.80, 0.999, 0.99, format="%.3f")
        c = st.columns(4)
        clip = c[0].number_input("PPO clip", 0.05, 0.5, 0.2)
        ppo_epochs = c[1].number_input(T("Épocas PPO", "PPO epochs"), 1, 20, 4)
        ent = c[2].number_input(T("Coef. entropia", "Entropy coef."), 0.0, 0.1, 0.01, format="%.3f")
        hidden = c[3].selectbox(T("Neurónios", "Hidden units"), [64, 128, 256], index=1)
        c = st.columns(4)
        bc_eps = c[0].number_input(T("Episódios imitação FEFO", "FEFO imitation episodes"), 0, 500, 30,
                                   help=T("Pré-treino por imitação da heurística FEFO (0 = desligado)", "Behavior cloning of the FEFO heuristic (0 = off)"))
        bc_epochs = c[1].number_input(T("Épocas imitação", "Imitation epochs"), 0, 100, 15)
        eval_every = c[2].number_input(T("Validar a cada N atualizações", "Validate every N updates"), 1, 100, 5)
        seed = c[3].number_input("Seed", 0, 99999, 42, key="train_seed")

    with st.expander(T("💶 Recompensa e simulação", "💶 Reward and simulation")):
        c = st.columns(4)
        waste_w = c[0].number_input(T("Peso do desperdício (× preço)", "Waste weight (× price)"), 0.0, 10.0, 2.0, 0.25,
                                    help=T("Cada kg desperdiçado custa N × preço médio. Mais alto = mais aversão ao desperdício.",
                                           "Each wasted kg costs N × average price. Higher = more waste-averse."))
        q_floor = c[1].number_input(T("Qualidade mínima comercial", "Minimum commercial quality"), 5.0, 90.0, 30.0, 5.0)
        hc_ca = c[2].number_input(T("Custo armazenagem AC (€/kg/dia)", "CA storage cost (€/kg/day)"), 0.0, 0.05, 0.0012, format="%.4f")
        hc_cold = c[3].number_input(T("Custo armazenagem frio (€/kg/dia)", "Cold storage cost (€/kg/day)"), 0.0, 0.05, 0.0008, format="%.4f")
        c = st.columns(4)
        horizon = c[0].number_input(T("Horizonte após última entrada (dias)", "Horizon after last arrival (days)"), 30, 400, 240)
        ca_pkg = c[1].number_input(T("Fator respiração em AC", "CA respiration factor"), 0.01, 1.0, 0.10,
                                   help=T("Redução da respiração em AC (o LC usa 0.10 para MAP)", "Respiration reduction in CA (LC uses 0.10 for MAP)"))
        min_pf = c[2].number_input(T("Fator de preço com qualidade 0", "Price factor at quality 0"), 0.0, 1.0, 0.5)
        interval = c[3].number_input(T("Intervalo agente armazéns (dias)", "Warehouse agent interval (days)"), 1, 30, 7)

    btn_train = st.button(T("🚀 Iniciar Treino do Agente", "🚀 Start Agent Training"), width="stretch", type="primary",
                          disabled=not train_seasons)
    st.markdown(f'<div class="console-header">📁 {T("Terminal do Stock Agent - Logs de Treino", "Stock Agent Terminal - Training Logs")}</div>',
                unsafe_allow_html=True)
    console = st.empty()
    console.markdown(f'<div class="console-box">{st.session_state.train_log or T("Agente à espera de comando...", "Agent waiting for command...")}</div>',
                     unsafe_allow_html=True)
    progress_box = st.empty()

    if btn_train:
        sim_params = SimParams(horizon_days=int(horizon), quality_floor=q_floor, waste_weight=waste_w, holding_cost_ca=hc_ca,
                               holding_cost_cold=hc_cold, ca_pkg_factor=ca_pkg, min_price_factor=min_pf, warehouse_interval=int(interval))
        cfg = TrainConfig(episodes=int(episodes), episodes_per_update=int(ep_upd), lr=float(lr), gamma=float(gamma), clip=float(clip),
                          ppo_epochs=int(ppo_epochs), ent_coef=float(ent), hidden=int(hidden), bc_episodes=int(bc_eps),
                          bc_epochs=int(bc_epochs), eval_every=int(eval_every), seed=int(seed))
        st.session_state.train_log = ""
        bar = progress_box.progress(0.0)
        try:
            for ev in train_generator(data, perfil, cultura, mode, train_seasons, val_seasons, sim_params, cfg):
                if ev["type"] == "log":
                    st.session_state.train_log += ev["msg"] + "\n"
                    tail = "\n".join(st.session_state.train_log.splitlines()[-200:])
                    console.markdown(f'<div class="console-box">{tail}</div>', unsafe_allow_html=True)
                elif ev["type"] == "progress":
                    bar.progress(min(1.0, ev["episode"] / ev["total"]))
                elif ev["type"] == "done":
                    st.session_state.trained_zip = ev["zip_bytes"]
                    st.session_state.trained_config = ev["config"]
                    st.session_state.trained_history = ev["history"]
                    st.session_state.sim_results = None
            bar.progress(1.0)
            st.success(T("🎉 Treino concluído!", "🎉 Training completed!"))
        except Exception as e:
            st.error(f"{T('Erro no treino', 'Training error')}: {e}")

    cfg_t = st.session_state.trained_config
    if cfg_t is not None:
        m = cfg_t["metrics"]
        if not m.get("melhor_que_fefo", True):
            st.warning(T("⚠️ Este modelo ainda não supera a heurística FEFO na validação. Treine mais episódios antes de o usar.",
                         "⚠️ This model does not beat the FEFO heuristic on validation yet. Train more episodes before using it."))
        st.markdown(f"#### 📈 {T('Resultado do treino', 'Training result')} — {cfg_t['perfil']} / {cfg_t['cultura']} / {cfg_t['mode']}")
        c = st.columns(4)
        d_profit = m["validacao_lucro_eur"] - m["fefo_lucro_eur"]
        c[0].markdown(metric_card(T("Lucro validação (RL)", "Validation profit (RL)"), eur(m["validacao_lucro_eur"]),
                                  T(f"{d_profit:+,.0f} € vs FEFO", f"{d_profit:+,.0f} € vs FEFO").replace(",", " "), hero=True), unsafe_allow_html=True)
        c[1].markdown(metric_card(T("Desperdício validação (RL)", "Validation waste (RL)"), f"{m['validacao_desperdicio_pct']:.1f}%",
                                  f"FEFO: {m['fefo_desperdicio_pct']:.1f}%", hero=True), unsafe_allow_html=True)
        c[2].markdown(metric_card(T("Lucro FEFO", "FEFO profit"), eur(m["fefo_lucro_eur"]), T("referência", "baseline")), unsafe_allow_html=True)
        c[3].markdown(metric_card(T("Tempo de treino", "Training time"), f"{cfg_t['training_seconds']:.0f} s",
                                  T(f"temporadas {cfg_t['train_seasons']}", f"seasons {cfg_t['train_seasons']}")), unsafe_allow_html=True)
        hist = st.session_state.trained_history
        if hist is not None and len(hist):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["episodio"], y=hist["lucro_treino"], name=T("Lucro treino", "Train profit"),
                                     line=dict(color="#fdba74")))
            if "lucro_validacao" in hist:
                h = hist.dropna(subset=["lucro_validacao"])
                fig.add_trace(go.Scatter(x=h["episodio"], y=h["lucro_validacao"], name=T("Lucro validação", "Validation profit"),
                                         mode="lines+markers", line=dict(color="#ea580c")))
            fig.add_hline(y=m["fefo_lucro_eur"], line_dash="dash", line_color="#0ea5e9", annotation_text="FEFO")
            fig.update_layout(title=T("Evolução do lucro (€)", "Profit evolution (€)"), height=320, margin=dict(l=10, r=10, t=40, b=10),
                              plot_bgcolor="#ffffff", legend=dict(orientation="h", y=-0.2))
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=hist["episodio"], y=hist["desperdicio_treino_pct"], name=T("Desperdício treino", "Train waste"),
                                      line=dict(color="#fca5a5")))
            if "desperdicio_validacao_pct" in hist:
                h = hist.dropna(subset=["desperdicio_validacao_pct"])
                fig2.add_trace(go.Scatter(x=h["episodio"], y=h["desperdicio_validacao_pct"], name=T("Desperdício validação", "Validation waste"),
                                          mode="lines+markers", line=dict(color="#dc2626")))
            fig2.add_hline(y=m["fefo_desperdicio_pct"], line_dash="dash", line_color="#0ea5e9", annotation_text="FEFO")
            fig2.update_layout(title=T("Evolução do desperdício (%)", "Waste evolution (%)"), height=320, margin=dict(l=10, r=10, t=40, b=10),
                               plot_bgcolor="#ffffff", legend=dict(orientation="h", y=-0.2))
            c1, c2 = st.columns(2)
            c1.plotly_chart(fig, width="stretch")
            c2.plotly_chart(fig2, width="stretch")
        c1, c2 = st.columns(2)
        fname = f"stock_agent_{cfg_t['mode']}_{cfg_t['perfil']}_{cfg_t['cultura']}.zip".replace(" ", "_")
        c1.download_button(T("📥 Descarregar Modelo (.ZIP)", "📥 Download Model (.ZIP)"), st.session_state.trained_zip,
                           file_name=fname, mime="application/zip", width="stretch")
        as_default = c2.checkbox(
            T(f"Publicar como modelo por defeito da fruta ({cfg_t['fruit_key']})", f"Publish as default model for fruit ({cfg_t['fruit_key']})"),
            help=T("Usado pela plataforma para todos os perfis que ainda não têm modelo próprio desta fruta.",
                   "Used by the platform for every profile that has no model of its own for this fruit."))
        if c2.button(T("💾 Publicar no repositório do Web Service", "💾 Publish to Web Service repository"), width="stretch"):
            if as_default:
                p = save_to_storage(MODELS_ROOT, st.session_state.trained_zip, DEFAULT_PROFILE, cfg_t["fruit_key"], cfg_t["mode"])
            else:
                p = save_to_storage(MODELS_ROOT, st.session_state.trained_zip, cfg_t["perfil"], cfg_t["cultura"], cfg_t["mode"])
            st.success(T(f"Modelo guardado em {p}", f"Model saved to {p}"))
        st.caption(T(f"Repositório: {MODELS_ROOT} (defina a variável STOCK_AGENT_MODELS para apontar para Web Services/stock_agent_service/models_storage).",
                     f"Repository: {MODELS_ROOT} (set STOCK_AGENT_MODELS to point to Web Services/stock_agent_service/models_storage)."))

# =====================================================================================================
# TAB SIMULAÇÃO & INFERÊNCIA
# =====================================================================================================
with tab_test:
    st.markdown(f"#### 🤖 {T('Modelo', 'Model')}")
    sources = [T("Modelo treinado nesta sessão", "Model trained in this session"), T("Upload do modelo (.ZIP)", "Upload model (.ZIP)"),
               T("Repositório do Web Service", "Web Service repository")]
    src = st.radio(T("Origem do modelo", "Model source"), sources, horizontal=True,
                   index=0 if st.session_state.trained_zip is not None else 1)
    model, cfg_m = None, None
    try:
        if src == sources[0] and st.session_state.trained_zip is not None:
            model, cfg_m, _ = load_zip(st.session_state.trained_zip)
        elif src == sources[1]:
            zf = st.file_uploader(T("Modelo ZIP", "Model ZIP"), type=["zip"])
            if zf is not None:
                model, cfg_m, _ = load_zip(zf.getvalue())
        elif src == sources[2]:
            path = storage_path(MODELS_ROOT, perfil, cultura, mode)
            if os.path.exists(path):
                model, cfg_m, _ = load_zip(path)
            else:
                st.warning(T(f"Não existe modelo publicado para {perfil} / {cultura} / {mode}.", f"No published model for {perfil} / {cultura} / {mode}."))
    except Exception as e:
        st.error(f"{T('Erro a carregar o modelo', 'Error loading model')}: {e}")

    if model is None:
        st.info(T("Treine um modelo na tab anterior ou carregue um ZIP.", "Train a model in the previous tab or upload a ZIP."))
        st.stop()

    st.markdown(
        f'<div class="custom-card"><b>{T("Agente de Lotes (diário)", "Lot Agent (daily)") if cfg_m["mode"] == "lot" else T("Agente de Armazéns (semanal)", "Warehouse Agent (weekly)")}</b>'
        f' &nbsp;|&nbsp; {T("Perfil", "Profile")}: <b>{cfg_m["perfil"]}</b> &nbsp;|&nbsp; {T("Cultura", "Crop")}: <b>{cfg_m["cultura"]}</b>'
        f' ({cfg_m["fruit_key"]}) &nbsp;|&nbsp; {T("Treinado em", "Trained on")}: {cfg_m["train_seasons"]} &nbsp;|&nbsp; {cfg_m["trained_at"]}</div>',
        unsafe_allow_html=True)

    pc_m = pc[(pc["perfil"] == cfg_m["perfil"]) & (pc["cultura"] == cfg_m["cultura"])]
    if pc_m.empty:
        st.error(T(f"Os dados carregados não têm lotes de {cfg_m['perfil']} / {cfg_m['cultura']}. Carregue um Excel com esse perfil e cultura.",
                   f"Loaded data has no lots for {cfg_m['perfil']} / {cfg_m['cultura']}. Upload an Excel with that profile and crop."))
        st.stop()
    seasons_m = [int(x) for x in pc_m.iloc[0]["temporadas"]]
    default_season = [s for s in cfg_m.get("val_seasons", []) if s in seasons_m] or seasons_m[-1:]
    c1, c2 = st.columns([1, 3])
    temporada = c1.selectbox(T("Temporada a simular", "Season to simulate"), seasons_m, index=seasons_m.index(default_season[0]))
    trained_note = T("⚠️ Esta temporada foi usada no treino.", "⚠️ This season was used for training.") if temporada in cfg_m["train_seasons"] else \
        T("✅ Temporada não vista no treino.", "✅ Season not seen during training.")
    c2.markdown(f"<br>{trained_note}", unsafe_allow_html=True)

    if st.button(T("🏁 Simular temporada (Agente vs FEFO vs Venda Imediata)", "🏁 Simulate season (Agent vs FEFO vs Immediate Sale)"),
                 width="stretch", type="primary"):
        with st.spinner(T("A simular...", "Simulating...")):
            st.session_state.sim_results = simulate_season(model, cfg_m, data, temporada)
            st.session_state.sim_key = (cfg_m["trained_at"], cfg_m["mode"], temporada)

    res = st.session_state.sim_results
    if res is not None and st.session_state.sim_key == (cfg_m["trained_at"], cfg_m["mode"], temporada):
        mode_m = cfg_m["mode"]
        rl = res[AGENT_NAME]["kpis"]
        fefo = res["FEFO + Procura"]["kpis"]
        imed = res["Venda Imediata"]["kpis"]

        st.markdown(f"#### 🏆 {T('Resultados da temporada', 'Season results')} {temporada}")
        c = st.columns(5)
        dp = rl["lucro_eur"] - fefo["lucro_eur"]
        c[0].markdown(metric_card(T("Lucro Agente", "Agent profit"), eur(rl["lucro_eur"]), f"{dp:+,.0f} € vs FEFO".replace(",", " "), hero=True), unsafe_allow_html=True)
        badge = "badge-green" if rl["pct_desperdicio"] <= fefo["pct_desperdicio"] + 1e-9 else "badge-amber"
        c[1].markdown(metric_card(T("Desperdício Agente", "Agent waste"), f"{rl['pct_desperdicio']:.1f}%",
                                  f'<span class="badge {badge}">FEFO {fefo["pct_desperdicio"]:.1f}% · {T("Imediata", "Immediate")} {imed["pct_desperdicio"]:.1f}%</span>', hero=True),
                      unsafe_allow_html=True)
        c[2].markdown(metric_card(T("Kg desperdiçados", "Wasted kg"), kg(rl["kg_desperdicio"]), f"FEFO: {kg(fefo['kg_desperdicio'])}"), unsafe_allow_html=True)
        c[3].markdown(metric_card(T("Preço médio de venda", "Avg. selling price"), f"{rl['preco_medio_venda']:.3f} €/kg",
                                  f"FEFO: {fefo['preco_medio_venda']:.3f} €/kg"), unsafe_allow_html=True)
        c[4].markdown(metric_card(T("Receita Agente", "Agent revenue"), eur(rl["receita_eur"]),
                                  T(f"armazenagem: {eur(rl['custo_armazenagem_eur'])}", f"storage: {eur(rl['custo_armazenagem_eur'])}")), unsafe_allow_html=True)
        st.write("")
        comp = comparison_table(res)
        st.dataframe(comp.style.format({c: "{:,.0f}" for c in comp.columns if c not in ("Política", "% Desperdício", "Preço médio venda (EUR/kg)")})
                     .format({"% Desperdício": "{:.1f}%", "Preço médio venda (EUR/kg)": "{:.3f}"}), hide_index=True, width="stretch")

        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig2 = go.Figure()
        for name in [AGENT_NAME, "FEFO + Procura", "Venda Imediata"]:
            d = res[name]["daily"]
            fig.add_trace(go.Scatter(x=d["data"], y=d["lucro_acumulado"], name=name, line=dict(color=COLORS[name], width=3 if name == AGENT_NAME else 2)))
            fig2.add_trace(go.Scatter(x=d["data"], y=d["desperdicio_acumulado_kg"], name=name, line=dict(color=COLORS[name], width=3 if name == AGENT_NAME else 2)))
        fig.update_layout(title=T("Lucro acumulado (€)", "Cumulative profit (€)"), height=340, margin=dict(l=10, r=10, t=40, b=10),
                          plot_bgcolor="#ffffff", legend=dict(orientation="h", y=-0.2))
        fig2.update_layout(title=T("Desperdício acumulado (kg)", "Cumulative waste (kg)"), height=340, margin=dict(l=10, r=10, t=40, b=10),
                           plot_bgcolor="#ffffff", legend=dict(orientation="h", y=-0.2))
        c1.plotly_chart(fig, width="stretch")
        c2.plotly_chart(fig2, width="stretch")

        # Stock e preço com marcas de colocação à venda
        d = res[AGENT_NAME]["daily"]
        lots_rl = res[AGENT_NAME]["lots"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["data"], y=d["stock_armazenado_kg"], name=T("Stock armazenado", "Stored stock"), stackgroup="s",
                                 line=dict(color="#cbd5e1")))
        fig.add_trace(go.Scatter(x=d["data"], y=d["stock_a_venda_kg"], name=T("Stock à venda", "Stock on sale"), stackgroup="s",
                                 line=dict(color="#fdba74")))
        fig.add_trace(go.Scatter(x=d["data"], y=d["preco_kg"], name=T("Preço (€/kg)", "Price (€/kg)"), yaxis="y2", line=dict(color="#ea580c")))
        listed = lots_rl.dropna(subset=["data_colocado_venda"])
        if len(listed):
            ev = listed.groupby("data_colocado_venda").agg(kg=("kg_inicial", "sum"), lotes=("lote_id", lambda x: ", ".join(x))).reset_index()
            ev["data_colocado_venda"] = pd.to_datetime(ev["data_colocado_venda"])
            px = d.set_index("data")["preco_kg"].reindex(ev["data_colocado_venda"]).values
            fig.add_trace(go.Scatter(x=ev["data_colocado_venda"], y=px, mode="markers", yaxis="y2",
                                     marker=dict(symbol="triangle-down", size=11, color="#dc2626"),
                                     name=T("Colocado à venda", "Put on sale"), text=ev["lotes"] + "<br>" + ev["kg"].round(0).astype(str) + " kg",
                                     hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>"))
        fig.update_layout(title=T("Agente: stock, preço e momentos de venda", "Agent: stock, price and sale moments"), height=380,
                          margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="#ffffff", legend=dict(orientation="h", y=-0.15),
                          yaxis=dict(title="kg"), yaxis2=dict(title="€/kg", overlaying="y", side="right"))
        st.plotly_chart(fig, width="stretch")

        # Linha temporal por lote
        if len(lots_rl):
            fig = go.Figure()
            season = res["_season"]
            end_date = season.days[-1]
            lr_sorted = lots_rl.sort_values(["armazem", "data_entrada"])
            ylab = (lr_sorted["lote_id"] + " · " + lr_sorted["armazem"] + np.where(lr_sorted["atmosfera_controlada"], " (AC)", "")).tolist()
            for (_, r), y in zip(lr_sorted.iterrows(), ylab):
                start = pd.Timestamp(r["data_entrada"])
                lst = pd.Timestamp(r["data_colocado_venda"]) if pd.notna(r["data_colocado_venda"]) else None
                stop_store = lst or end_date
                fig.add_trace(go.Bar(y=[y], x=[(stop_store - start).days * 86400000], base=[start], orientation="h",
                                     marker_color="#94a3b8" if r["atmosfera_controlada"] else "#cbd5e1", showlegend=False,
                                     hovertemplate=f"{r['lote_id']}: {T('armazenado', 'stored')} {r['kg_inicial']:,.0f} kg<extra></extra>"))
                if lst is not None:
                    color = "#dc2626" if r["kg_desperdicio"] > 0 else "#ea580c"
                    dur = max(1, int(r.get("dias_ate_venda") or 0))
                    sold_days = max(1, int(np.ceil(r["kg_inicial"] / max(1.0, d["procura_kg"].mean()))))
                    fig.add_trace(go.Bar(y=[y], x=[sold_days * 86400000], base=[lst], orientation="h", marker_color=color, showlegend=False,
                                         hovertemplate=(f"{r['lote_id']}: {T('à venda desde', 'on sale since')} {lst.date()}<br>"
                                                        f"Q={r['qualidade_na_venda']} RSL={r['rsl_na_venda']}d<br>"
                                                        f"{T('vendido', 'sold')} {r['kg_vendidos']:,.0f} kg · {T('desperdício', 'waste')} {r['kg_desperdicio']:,.0f} kg<extra></extra>")))
            fig.update_layout(barmode="overlay", title=T("Linha temporal dos lotes (cinza = armazenado, laranja = à venda, vermelho = com desperdício)",
                                                         "Lot timeline (grey = stored, orange = on sale, red = with waste)"),
                              height=max(320, 18 * len(lr_sorted) + 80), margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="#ffffff",
                              xaxis=dict(type="date"))
            st.plotly_chart(fig, width="stretch")

        # Inspeção das decisões
        st.markdown(f"#### 🔎 {T('Decisões do agente, dia a dia', 'Agent decisions, day by day')}")
        dec = res[AGENT_NAME]["decisions"]
        if len(dec):
            days_opts = sorted(dec["data"].unique())
            if mode_m == "lot":
                action_days = sorted(dec.loc[dec["decisao"] == "COLOCAR À VENDA", "data"].unique())
            else:
                action_days = sorted(dec.loc[dec["decisao"] == "ABRIR", "data"].unique())
            c1, c2 = st.columns([1, 2])
            only_actions = c1.toggle(T("Só dias com ação", "Only days with action"), value=True)
            opts = action_days if (only_actions and action_days) else days_opts
            day = c2.select_slider(T("Dia de decisão", "Decision day"), options=opts, value=opts[0])
            view = dec[dec["data"] == day].drop(columns=["data"])
            sort_col = "prob_venda_pct" if mode_m == "lot" else "prob_pct"
            view = view.sort_values(sort_col, ascending=False)
            st.dataframe(view, hide_index=True, width="stretch",
                         column_config={sort_col: st.column_config.ProgressColumn(T("Probabilidade (%)", "Probability (%)"), min_value=0, max_value=100, format="%.1f"),
                                        "explicacao": st.column_config.TextColumn(T("Explicação", "Explanation"), width="large")})
        with st.expander(T("📋 Relatório por lote (Agente)", "📋 Per-lot report (Agent)")):
            st.dataframe(lots_rl, hide_index=True, width="stretch")
        st.download_button(T("📥 Descarregar relatório Excel", "📥 Download Excel report"), report_excel_bytes(res, cfg_m),
                           file_name=f"relatorio_stock_agent_{mode_m}_{temporada}.xlsx", width="stretch")

    # Web service: pedido / resposta
    st.markdown("---")
    with st.expander(T("🌐 Decisão para uma data (formato do Web Service)", "🌐 Decision for a date (Web Service format)")):
        st.caption(T("Constrói o pedido que a plataforma Django enviaria ao Web Service (lotes em armazém nessa data, leituras dos sensores, "
                     "mercado e previsão) e mostra a resposta do agente. Assume que nenhum lote foi colocado à venda antes dessa data.",
                     "Builds the request the Django platform would send to the Web Service (lots stored on that date, sensor readings, "
                     "market and forecast) and shows the agent's response. Assumes no lot was put on sale before that date."))
        lot_s = data["Lotes"][(data["Lotes"]["perfil"] == cfg_m["perfil"]) & (data["Lotes"]["cultura"] == cfg_m["cultura"]) &
                              (data["Lotes"]["temporada"] == temporada)]
        dmin, dmax = lot_s["data_entrada"].min().date(), (lot_s["data_entrada"].max() + pd.Timedelta(days=30)).date()
        ddate = st.date_input(T("Data", "Date"), value=lot_s["data_entrada"].max().date(), min_value=dmin, max_value=dmax)
        if st.button(T("Pedir decisão", "Request decision")):
            try:
                payload = snapshot_from_excel(data, cfg_m, ddate, temporada)
                resp = decide_snapshot(model, cfg_m, payload)
                c1, c2 = st.columns(2)
                preview = dict(payload, leituras=payload["leituras"][:3] + ["..."], mercado=payload["mercado"][:3] + ["..."])
                c1.markdown("**POST /decide** (`service/main.py`, porta 8005)")
                c1.code(json.dumps(preview, indent=2, ensure_ascii=False)[:6000], language="json")
                c2.markdown(f"**{T('Resposta', 'Response')}**")
                c2.code(json.dumps(resp, indent=2, ensure_ascii=False)[:8000], language="json")
            except Exception as e:
                st.error(str(e))
