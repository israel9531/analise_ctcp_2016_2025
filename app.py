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

TOP_N = st.sidebar.slider("Limite de itens (Carroceria/Chassi/Letreiro)", 5, 30, 30, step=1)

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
""", unsafe_allow_html=True)

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

# =========================
# GRÁFICOS — COMPARATIVOS POR ANO
# =========================

# Carrocerias
c7, c8 = st.columns(2, gap="large")
with c7:
    car_counts_2016 = df16_f["Carroceria"].value_counts().head(TOP_N) if "Carroceria" in df16_f.columns else pd.Series(dtype=int)
    hbar(car_counts_2016, "Carrocerias — 2016", cfg=CHART_CFG["carrocerias_topN"])
with c8:
    car_counts_2025 = df25_f["Carroceria"].value_counts().head(TOP_N) if "Carroceria" in df25_f.columns else pd.Series(dtype=int)
    hbar(car_counts_2025, "Carrocerias — 2025", cfg=CHART_CFG["carrocerias_topN"])

# Chassis
c9, c10 = st.columns(2, gap="large")
with c9:
    ch_counts_2016 = df16_f["Chassi"].value_counts().head(TOP_N) if "Chassi" in df16_f.columns else pd.Series(dtype=int)
    hbar(ch_counts_2016, "Chassis — 2016", cfg=CHART_CFG["chassis_topN"])
with c10:
    ch_counts_2025 = df25_f["Chassi"].value_counts().head(TOP_N) if "Chassi" in df25_f.columns else pd.Series(dtype=int)
    hbar(ch_counts_2025, "Chassis — 2025", cfg=CHART_CFG["chassis_topN"])

# Letreiros
st.subheader("Letreiros — 2016 e 2025")
let16 = df16_f["Letreiro"].value_counts().head(TOP_N) if "Letreiro" in df16_f.columns else pd.Series(dtype=int)
let25 = df25_f["Letreiro"].value_counts().head(TOP_N) if "Letreiro" in df25_f.columns else pd.Series(dtype=int)

c11, c12 = st.columns(2, gap="large")
with c11:
    hbar(let16, "Letreiros — 2016", cfg=CHART_CFG["letreiros_topN"])
with c12:
    hbar(let25, "Letreiros — 2025", cfg=CHART_CFG["letreiros_topN"])
