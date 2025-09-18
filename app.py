import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# =========================
# CONFIG & TEMA
# =========================
st.set_page_config(page_title="Análise frota CTCP 2016 × 2025 - Pelotas (RS)", layout="wide")
st.title("Análise frota CTCP 2016 × 2025 - Pelotas (RS)")

# =========================
# DADOS (2 fontes)
# =========================
st.sidebar.header("Entrada de dados")
col_in1, col_in2 = st.sidebar.columns(2)
with col_in1:
    csv_2016_path = st.text_input(
        "CSV 2016 (caminho opcional)",
        "https://github.com/israel9531/analise_ctcp_2016_2025/tree/main/data/dados_ctcp_2016.csv"
    )
with col_in2:
    csv_2025_path = st.text_input(
        "CSV 2025 (caminho opcional)",
        "https://github.com/israel9531/analise_ctcp_2016_2025/tree/main/data/dados_ctcp_2025.csv"
    )

up2016 = st.sidebar.file_uploader("Ou faça upload do CSV 2016", type=["csv"], key="up2016")
up2025 = st.sidebar.file_uploader("Ou faça upload do CSV 2025", type=["csv"], key="up2025")

@st.cache_data(show_spinner=False)
def load_csv(obj):
    if hasattr(obj, "read"):  # UploadedFile
        return pd.read_csv(obj)
    return pd.read_csv(obj)

# Carrega 2016
df16 = None
err = None
try:
    if up2016 is not None:
        df16 = load_csv(up2016)
    elif csv_2016_path.strip():
        df16 = load_csv(csv_2016_path)
except Exception as e:
    err = f"2016: {e}"

# Carrega 2025
df25 = None
try:
    if up2025 is not None:
        df25 = load_csv(up2025)
    elif csv_2025_path.strip():
        df25 = load_csv(csv_2025_path)
except Exception as e:
    err = (err + " | " if err else "") + f"2025: {e}"

if err:
    st.error(f"Erro ao ler CSV(s): {err}")
if df16 is None or df25 is None:
    st.info("Informe caminhos ou faça upload dos dois CSVs (2016 e 2025) para começar.")
    st.stop()

# =========================
# PREPARAÇÃO
# =========================
# Normaliza colunas essenciais que você usa no app original:
# Esperado: 'Ano', 'Empresa', 'Carroceria', 'Chassi', 'Letreiro', 'Elevador', 'Ar', 'Prefixo'
for df in (df16, df25):
    # Tenta alinhar capitalização (caso o CSV venha diferente)
    ren = {c: c.strip().title() for c in df.columns}
    df.rename(columns=ren, inplace=True)

# Idade: calcule com referência do ano do dataset
if "Ano" not in df16.columns or "Ano" not in df25.columns:
    st.error("Os CSVs precisam ter a coluna 'Ano'.")
    st.stop()

df16["idade"] = 2016 - df16["Ano"]
df25["idade"] = 2025 - df25["Ano"]

# =========================
# SIDEBAR — FILTROS (aplicados aos dois anos)
# =========================
st.sidebar.header("Filtros")

# Faixa de Ano baseada no MIN/MAX combinados
min_ano = int(min(df16["Ano"].min(), df25["Ano"].min()))
max_ano = int(max(df16["Ano"].max(), df25["Ano"].max()))
f_ano = st.sidebar.slider("Ano (fabr.)", min_ano, max_ano, (min_ano, max_ano), step=1)

def _opts(col):
    a = pd.Series(dtype=object)
    if col in df16.columns:
        a = pd.concat([a, df16[col].dropna().astype(str)])
    if col in df25.columns:
        a = pd.concat([a, df25[col].dropna().astype(str)])
    return sorted(a.unique().tolist())

opts_emp = _opts("Empresa")
f_emp = st.sidebar.multiselect("Empresas", opts_emp, default=opts_emp)

opts_car = _opts("Carroceria")
f_car = st.sidebar.multiselect("Carrocerias", opts_car, default=opts_car)

opts_ch = _opts("Chassi")
f_ch = st.sidebar.multiselect("Chassis", opts_ch, default=opts_ch)

opts_let = _opts("Letreiro")
f_let = st.sidebar.multiselect("Letreiros", opts_let, default=opts_let)

opts_elev = _opts("Elevador")
f_elev = st.sidebar.multiselect("Elevador", opts_elev, default=opts_elev)

opts_ar = _opts("Ar")
f_ar = st.sidebar.multiselect("Ar-condicionado", opts_ar, default=opts_ar)

TOP_N = st.sidebar.slider("Top-N (Carroceria/Chassi/Letreiro)", 5, 30, 30, step=1)

# Aplica filtros em ambos
def _apply_filters(df):
    # Converte para string para filtros de lista
    for c in ["Empresa","Carroceria","Chassi","Letreiro","Elevador","Ar"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    mask = (
        df["Ano"].between(f_ano[0], f_ano[1]) &
        df["Empresa"].isin(f_emp) &
        df["Carroceria"].isin(f_car) &
        df["Chassi"].isin(f_ch) &
        df["Letreiro"].isin(f_let) &
        df["Elevador"].isin(f_elev) &
        df["Ar"].isin(f_ar)
    )
    return df.loc[mask].copy()

df16_f = _apply_filters(df16)
df25_f = _apply_filters(df25)

# =========================
# KPIs
# =========================
frota16 = int(df16_f.shape[0] if "Prefixo" not in df16_f.columns else df16_f["Prefixo"].count())
frota25 = int(df25_f.shape[0] if "Prefixo" not in df25_f.columns else df25_f["Prefixo"].count())
idade16 = float(df16_f["idade"].mean()) if len(df16_f) else 0.0
idade25 = float(df25_f["idade"].mean()) if len(df25_f) else 0.0

st.markdown("""
<style>
.kpi-wrap {display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 14px; margin: 4px 0 8px 0;}
.kpi-card {
  background: #0f172a; color: #e5e7eb; border: 1px solid #1f2937;
  border-radius: 14px; padding: 16px 18px; box-shadow: 0 6px 18px rgba(0,0,0,.12);
}
.kpi-title {font-size: 0.95rem; letter-spacing: .02em; color: #cbd5e1; margin: 0 0 8px 0; font-weight: 600; text-transform: uppercase;}
.kpi-value {font-size: 2.2rem; line-height: 1.1; font-weight: 800; color: #f8fafc; margin: 0 0 6px 0;}
.kpi-sub {font-size: .95rem; color: #94a3b8; margin: 0;}
.kpi-accent { color: #60a5fa; }
.kpi-accent-2 { color: #34d399; }
.kpi-accent-3 { color: #f59e0b; }
</style>

<div class="kpi-wrap">
  <div class="kpi-card">
    <div class="kpi-title">FROTA 2016</div>
    <div class="kpi-value kpi-accent">""" + f"{frota16:,}".replace(",", ".") + """</div>
    <p class="kpi-sub">Veículos no recorte selecionado</p>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">FROTA 2025</div>
    <div class="kpi-value kpi-accent-2">""" + f"{frota25:,}".replace(",", ".") + """</div>
    <p class="kpi-sub">Veículos no recorte selecionado</p>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Δ FROTA (2025 − 2016)</div>
    <div class="kpi-value kpi-accent-3">""" + f"{(frota25 - frota16):,}".replace(",", ".") + """</div>
    <p class="kpi-sub">Variação absoluta no recorte</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="kpi-wrap" style="grid-template-columns: repeat(2, minmax(260px, 1fr));">
  <div class="kpi-card">
    <div class="kpi-title">IDADE MÉDIA — 2016</div>
    <div class="kpi-value kpi-accent">""" + f"{idade16:.2f}".replace(".", ",") + """ <span style="font-size:1.2rem;font-weight:700;">anos</span></div>
    <p class="kpi-sub">Média de idade (2016−Ano)</p>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">IDADE MÉDIA — 2025</div>
    <div class="kpi-value kpi-accent-2">""" + f"{idade25:.2f}".replace(".", ",") + """ <span style="font-size:1.2rem;font-weight:700;">anos</span></div>
    <p class="kpi-sub">Média de idade (2025−Ano)</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# HELPERS VISUAIS — CONFIG
# =========================
FACE = "#FFFFFF"
CHART_CFG = {
    "mean_age_emp": {"figsize": (10, 10), "font": {"family":"sans-serif","name":"DejaVu Sans","size":{"label":10,"tick":13,"title":14}}},
    "frota_por_empresa": {"figsize": (10, 10), "font": {"family":"sans-serif","name":"DejaVu Sans","size":{"label":10,"tick":13,"title":16}}},
    "elevador": {"figsize": (3, 3), "font": {"family":"sans-serif","name":"DejaVu Sans","size":{"label":7,"tick":7,"title":6}}},
    "ar_condicionado": {"figsize": (3, 3), "font": {"family":"sans-serif","name":"DejaVu Sans","size":{"label":7,"tick":7,"title":6}}},
    "carrocerias_topN": {"figsize": (10, 10), "font": {"family":"sans-serif","name":"DejaVu Sans","size":{"label":10,"tick":13,"title":14}}},
    "chassis_topN": {"figsize": (10, 10), "font": {"family":"sans-serif","name":"DejaVu Sans","size":{"label":10,"tick":13,"title":14}}},
    "letreiros_topN": {"figsize": (18, 7.5), "font": {"family":"sans-serif","name":"DejaVu Sans","size":{"label":10,"tick":12,"title":14}}},
}

def _rc_from_cfg(cfg):
    fam = cfg["font"]["family"]; name = cfg["font"]["name"]
    return {"font.family": fam, "font.sans-serif":[name], "font.serif":[name], "font.monospace":[name]}

def hbar(series, title, cfg, xlabel="Quantidade", annotate_fmt="{:,.0f}"):
    if series.empty:
        st.info("Sem dados para os filtros selecionados."); return
    series = series.sort_values(ascending=True)
    with plt.rc_context(_rc_from_cfg(cfg)):
        fig, ax = plt.subplots(figsize=cfg["figsize"]); fig.set_facecolor(FACE)
        ax.barh(series.index, series.values, color=plt.cm.tab10.colors[:len(series)])
        for i, v in enumerate(series.values):
            ax.text(v, i, "  " + annotate_fmt.format(v), va="center",
                    fontsize=cfg["font"]["size"]["label"], weight="bold")
        ax.set_xlabel(xlabel, fontsize=cfg["font"]["size"]["label"]); ax.set_ylabel("")
        ax.tick_params(axis='both', labelsize=cfg["font"]["size"]["tick"])
        ax.set_title(title, pad=10, weight="bold", fontsize=cfg["font"]["size"]["title"])
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

def bar_mean(series, title, cfg, ylabel="Idade média (anos)"):
    if series.empty:
        st.info("Sem dados para os filtros selecionados."); return
    series = series.sort_values(ascending=False)
    with plt.rc_context(_rc_from_cfg(cfg)):
        fig, ax = plt.subplots(figsize=cfg["figsize"]); fig.set_facecolor(FACE)
        bars = ax.bar(series.index, series.values, color=plt.cm.tab10.colors[:len(series)])
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.06, f"{b.get_height():.2f}",
                    ha="center", va="bottom",
                    fontsize=cfg["font"]["size"]["label"], weight="bold")
        ax.set_ylabel(ylabel, fontsize=cfg["font"]["size"]["label"]); ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right", fontsize=cfg["font"]["size"]["tick"])
        ax.tick_params(axis='y', labelsize=cfg["font"]["size"]["tick"])
        ax.set_title(title, pad=10, weight="bold", fontsize=cfg["font"]["size"]["title"])
        st.pyplot(fig, use_container_width=True)

def pie_chart(series, title, cfg):
    if series.empty:
        st.info("Sem dados para os filtros selecionados."); return
    vals = series.values; labs = series.index.tolist()
    cores = list(plt.cm.tab10.colors[:len(vals)])
    with plt.rc_context(_rc_from_cfg(cfg)):
        fig, ax = plt.subplots(figsize=cfg["figsize"]); fig.set_facecolor(FACE)
        ax.pie(vals, labels=labs, autopct='%1.1f%%', startangle=90, colors=cores,
               wedgeprops=dict(edgecolor='white', linewidth=2),
               textprops=dict(color='black', weight='bold',
                              fontsize=cfg["font"]["size"]["label"],
                              family=cfg["font"]["family"], fontname=cfg["font"]["name"]))
        ax.set_aspect('equal'); plt.tight_layout()
        ax.set_title(title, pad=10, weight="bold", fontsize=cfg["font"]["size"]["title"])
        st.pyplot(fig, use_container_width=True)

# =========================
# CHAVE DE COMPARAÇÃO (entraram/saíram/permaneceram)
# =========================
comuns = sorted(list(set(df16.columns).intersection(df25.columns)))
if not comuns:
    st.warning("Não há colunas em comum entre 2016 e 2025."); st.stop()

# Chave fixa (sem mostrar seletor)
chave = "Prefixo" if "Prefixo" in comuns else comuns[0]


for d in (df16_f, df25_f):
    d[chave] = d[chave].astype(str).str.strip()

set16, set25 = set(df16_f[chave]), set(df25_f[chave])
permaneceram = sorted(set16 & set25)
entraram = sorted(set25 - set16)
sairam = sorted(set16 - set25)

#st.subheader("Dinâmica da frota (com base na chave selecionada)")
#c_dyn1, c_dyn2, c_dyn3 = st.columns(3)
#with c_dyn1: st.metric("Permaneceram", len(permaneceram))
#with c_dyn2: st.metric("Entraram (novos em 2025)", len(entraram))
#with c_dyn3: st.metric("Saíram (presentes só em 2016)", len(sairam))

#with st.expander("Ver listas"):
#    colL1, colL2, colL3 = st.columns(3)
#    with colL1: st.caption("Permaneceram"); st.dataframe(pd.DataFrame({chave: permaneceram}))
#    with colL2: st.caption("Entraram (2025)"); st.dataframe(pd.DataFrame({chave: entraram}))
#    with colL3: st.caption("Saíram (2016)"); st.dataframe(pd.DataFrame({chave: sairam}))

#st.divider()

# =========================
# GRÁFICOS — COMPARATIVOS POR ANO
# =========================

# Linha 1 — 2 gráficos grandes (por empresa)
c1, c2 = st.columns(2, gap="large")
with c1:
    mean_age_emp_2016 = df16_f.groupby("Empresa")["idade"].mean() if not df16_f.empty else pd.Series(dtype=float)
    bar_mean(mean_age_emp_2016, "Idade média por empresa — 2016", cfg=CHART_CFG["mean_age_emp"])
with c2:
    mean_age_emp_2025 = df25_f.groupby("Empresa")["idade"].mean() if not df25_f.empty else pd.Series(dtype=float)
    bar_mean(mean_age_emp_2025, "Idade média por empresa — 2025", cfg=CHART_CFG["mean_age_emp"])

# Linha 2 — Frota por empresa (2016/2025)
c3, c4 = st.columns(2, gap="large")
with c3:
    emp_counts_2016 = df16_f["Empresa"].value_counts() if "Empresa" in df16_f.columns else pd.Series(dtype=int)
    hbar(emp_counts_2016, "Frota por empresa — 2016", cfg=CHART_CFG["frota_por_empresa"])
with c4:
    emp_counts_2025 = df25_f["Empresa"].value_counts() if "Empresa" in df25_f.columns else pd.Series(dtype=int)
    hbar(emp_counts_2025, "Frota por empresa — 2025", cfg=CHART_CFG["frota_por_empresa"])

# Linha 3 — Pizzas (Elevador/Ar) 2016 vs 2025
c5, c6 = st.columns(2, gap="large")
with c5:
    elev_counts_2016 = df16_f["Elevador"].value_counts() if "Elevador" in df16_f.columns else pd.Series(dtype=int)
    pie_chart(elev_counts_2016, "Elevador — 2016", cfg=CHART_CFG["elevador"])
    ar_counts_2016 = df16_f["Ar"].value_counts() if "Ar" in df16_f.columns else pd.Series(dtype=int)
    pie_chart(ar_counts_2016, "Ar condicionado — 2016", cfg=CHART_CFG["ar_condicionado"])
with c6:
    elev_counts_2025 = df25_f["Elevador"].value_counts() if "Elevador" in df25_f.columns else pd.Series(dtype=int)
    pie_chart(elev_counts_2025, "Elevador — 2025", cfg=CHART_CFG["elevador"])
    ar_counts_2025 = df25_f["Ar"].value_counts() if "Ar" in df25_f.columns else pd.Series(dtype=int)
    pie_chart(ar_counts_2025, "Ar condicionado — 2025", cfg=CHART_CFG["ar_condicionado"])

# Linha 4 — Top-N Carrocerias/Chassis (2016 × 2025)
c7, c8 = st.columns(2, gap="large")
with c7:
    car_counts_2016 = df16_f["Carroceria"].value_counts().head(TOP_N) if "Carroceria" in df16_f.columns else pd.Series(dtype=int)
    hbar(car_counts_2016, f"Top-{TOP_N} Carrocerias — 2016", cfg=CHART_CFG["carrocerias_topN"])
with c8:
    car_counts_2025 = df25_f["Carroceria"].value_counts().head(TOP_N) if "Carroceria" in df25_f.columns else pd.Series(dtype=int)
    hbar(car_counts_2025, f"Top-{TOP_N} Carrocerias — 2025", cfg=CHART_CFG["carrocerias_topN"])

c9, c10 = st.columns(2, gap="large")
with c9:
    ch_counts_2016 = df16_f["Chassi"].value_counts().head(TOP_N) if "Chassi" in df16_f.columns else pd.Series(dtype=int)
    hbar(ch_counts_2016, f"Top-{TOP_N} Chassis — 2016", cfg=CHART_CFG["chassis_topN"])
with c10:
    ch_counts_2025 = df25_f["Chassi"].value_counts().head(TOP_N) if "Chassi" in df25_f.columns else pd.Series(dtype=int)
    hbar(ch_counts_2025, f"Top-{TOP_N} Chassis — 2025", cfg=CHART_CFG["chassis_topN"])

# Linha 5 — Letreiros (full-width)
st.subheader(f"Top-{TOP_N} Letreiros — 2016 e 2025")
let16 = df16_f["Letreiro"].value_counts().head(TOP_N) if "Letreiro" in df16_f.columns else pd.Series(dtype=int)
let25 = df25_f["Letreiro"].value_counts().head(TOP_N) if "Letreiro" in df25_f.columns else pd.Series(dtype=int)

c11, c12 = st.columns(2, gap="large")
with c11:
    hbar(let16, f"Letreiros — 2016 (Top-{TOP_N})", cfg=CHART_CFG["letreiros_topN"])
with c12:
    hbar(let25, f"Letreiros — 2025 (Top-{TOP_N})", cfg=CHART_CFG["letreiros_topN"])

st.divider()

# =========================
# COMPARAÇÃO TABULAR (categorias lado a lado)
# =========================
st.subheader("Tabelas comparativas por categoria")

def comp_table(col):
    if col not in df16_f.columns or col not in df25_f.columns:
        return None
    s16 = df16_f[col].astype(str).str.strip().value_counts()
    s25 = df25_f[col].astype(str).str.strip().value_counts()
    comp = pd.DataFrame({"2016": s16, "2025": s25}).fillna(0).astype(int)
    comp["Variação"] = comp["2025"] - comp["2016"]
    return comp.sort_values("Variação", ascending=False)

# Descrições específicas para cada categoria
descriptions = {
    "Empresa": "📌 Mostra a variação de veículos por empresa entre 2016 e 2025.",
    "Carroceria": "📌 Quantidade de veículos por tipo de carroceria nos dois anos.",
    "Chassi": "📌 Comparação da distribuição de chassis por modelo.",
    "Letreiro": "📌 Alterações nos tipos de letreiros identificados.",
    "Elevador": "📌 Quantidade de veículos com/sem elevador em 2016 vs 2025.",
    "Ar": "📌 Veículos com e sem ar-condicionado nos dois períodos."
}

for colcat in ["Empresa", "Carroceria", "Chassi", "Letreiro", "Elevador", "Ar"]:
    comp = comp_table(colcat)
    if comp is not None and not comp.empty:
        with st.expander(f"Comparação por {colcat}"):
            st.caption(descriptions.get(colcat, "📊 Comparação entre 2016 e 2025."))
            st.dataframe(comp)


# =========================
# TABELAS FILTRADAS (2016 e 2025)
# =========================
st.subheader("Tabelas filtradas — 2016 e 2025")
c_tab1, c_tab2 = st.columns(2)
with c_tab1:
    st.caption("2016 (após filtros)")
    st.dataframe(df16_f, use_container_width=True)
with c_tab2:
    st.caption("2025 (após filtros)")
    st.dataframe(df25_f, use_container_width=True)


