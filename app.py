import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pathlib import Path

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

DATA_DIR = Path(__file__).parent / "data"
csv_2016_path = DATA_DIR / "dados_ctcp_2016.csv"
csv_2025_path = DATA_DIR / "dados_ctcp_2025.csv"

df = pd.read_csv(csv_2016_path, sep=";", encoding="utf-8")
print(df.head())

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
    elif csv_2016_path:
        df16 = load_csv(csv_2016_path)
except Exception as e:
    err = f"2016: {e}"

# Carrega 2025
df25 = None
try:
    if up2025 is not None:
        df25 = load_csv(up2025)
    elif csv_2025_path:
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
for df in (df16, df25):
    ren = {c: c.strip().title() for c in df.columns}
    df.rename(columns=ren, inplace=True)

if "Ano" not in df16.columns or "Ano" not in df25.columns:
    st.error("Os CSVs precisam ter a coluna 'Ano'.")
    st.stop()

df16["idade"] = 2016 - df16["Ano"]
df25["idade"] = 2025 - df25["Ano"]

# =========================
# SIDEBAR — FILTROS (aplicados aos dois anos)
# =========================
st.sidebar.header("Filtros")

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

TOP_N = st.sidebar.slider("Limite de itens (Carroceria/Chassi/Letreiro)", 5, 30, 30, step=1)

def _apply_filters(df):
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

st.markdown(""" ... KPIs HTML ... """, unsafe_allow_html=True)

st.divider()

# =========================
# HELPERS VISUAIS — CONFIG
# =========================
FACE = "#FFFFFF"
CHART_CFG = { ... }

def _rc_from_cfg(cfg): ...
def hbar(series, title, cfg, xlabel="Quantidade", annotate_fmt="{:,.0f}"): ...
def bar_mean(series, title, cfg, ylabel="Idade média (anos)"): ...
def pie_chart(series, title, cfg): ...

# =========================
# CHAVE DE COMPARAÇÃO (entraram/saíram/permaneceram)
# =========================
comuns = sorted(list(set(df16.columns).intersection(df25.columns)))
if not comuns:
    st.warning("Não há colunas em comum entre 2016 e 2025."); st.stop()

chave = "Prefixo" if "Prefixo" in comuns else comuns[0]

for d in (df16_f, df25_f):
    d[chave] = d[chave].astype(str).str.strip()

set16, set25 = set(df16_f[chave]), set(df25_f[chave])
permaneceram = sorted(set16 & set25)
entraram = sorted(set25 - set16)
sairam = sorted(set16 - set25)

# =========================
# GRÁFICOS — COMPARATIVOS POR ANO
# =========================
c1, c2 = st.columns(2, gap="large")
with c1: ...
with c2: ...

c3, c4 = st.columns(2, gap="large")
with c3: ...
with c4: ...

c5, c6 = st.columns(2, gap="large")
with c5: ...
with c6: ...

c7, c8 = st.columns(2, gap="large")
with c7: ...
with c8: ...

c9, c10 = st.columns(2, gap="large")
with c9: ...
with c10: ...

st.subheader("Letreiros — 2016 e 2025")
c11, c12 = st.columns(2, gap="large")
with c11: ...
with c12: ...

st.divider()

# =========================
# COMPARAÇÃO TABULAR (categorias lado a lado)
# =========================
st.subheader("Tabelas comparativas por categoria")

def comp_table(col): ...

descriptions = { ... }

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
