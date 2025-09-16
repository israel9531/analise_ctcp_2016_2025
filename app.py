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

up2016 = st.sidebar.file_uploader("Ou faça upload do CSV 2016", type=["csv"], key="up2016")
up2025 = st.sidebar.file_uploader("Ou faça upload do CSV 2025", type=["csv"], key="up2025")

@st.cache_data(show_spinner=False)
def load_csv(obj):
    if hasattr(obj, "read"):  # UploadedFile
        return pd.read_csv(obj, sep=";", encoding="utf-8")
    return pd.read_csv(obj, sep=";", encoding="utf-8")

# Carrega 2016
df16 = None
err = None
try:
    if up2016 is not None:
        df16 = load_csv(up2016)
    elif csv_2016_path.exists():
        df16 = load_csv(csv_2016_path)
except Exception as e:
    err = f"2016: {e}"

# Carrega 2025
df25 = None
try:
    if up2025 is not None:
        df25 = load_csv(up2025)
    elif csv_2025_path.exists():
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
# SIDEBAR — FILTROS
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

st.metric("Frota 2016", frota16)
st.metric("Frota 2025", frota25)
st.metric("Idade média 2016", round(idade16, 1))
st.metric("Idade média 2025", round(idade25, 1))

st.divider()

# =========================
# FUNÇÕES DE GRÁFICOS
# =========================
def hbar(series, title, xlabel="Quantidade"):
    fig, ax = plt.subplots(figsize=(5, 3))
    series.sort_values().plot(kind="barh", ax=ax, color="skyblue")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    st.pyplot(fig)

def bar_mean(series, title, ylabel="Idade média (anos)"):
    fig, ax = plt.subplots(figsize=(5, 3))
    series.sort_values().plot(kind="bar", ax=ax, color="orange")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

def pie_chart(series, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    series.value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%")
    ax.set_ylabel("")
    ax.set_title(title)
    st.pyplot(fig)

# =========================
# GRÁFICOS COMPARATIVOS
# =========================
c1, c2 = st.columns(2)
with c1:
    hbar(df16_f["Carroceria"].value_counts().head(TOP_N), "Top Carrocerias 2016")
with c2:
    hbar(df25_f["Carroceria"].value_counts().head(TOP_N), "Top Carrocerias 2025")

c3, c4 = st.columns(2)
with c3:
    pie_chart(df16_f["Elevador"], "Elevador 2016")
with c4:
    pie_chart(df25_f["Elevador"], "Elevador 2025")

c5, c6 = st.columns(2)
with c5:
    hbar(df16_f["Chassi"].value_counts().head(TOP_N), "Chassis 2016")
with c6:
    hbar(df25_f["Chassi"].value_counts().head(TOP_N), "Chassis 2025")

# =========================
# TABELAS COMPARATIVAS
# =========================
st.subheader("Tabelas filtradas — 2016 e 2025")
c_tab1, c_tab2 = st.columns(2)
with c_tab1:
    st.caption("2016 (após filtros)")
    st.dataframe(df16_f, use_container_width=True)
with c_tab2:
    st.caption("2025 (após filtros)")
    st.dataframe(df25_f, use_container_width=True)
