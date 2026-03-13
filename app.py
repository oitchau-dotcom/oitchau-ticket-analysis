import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Oitchau | Análise de Chamados",
    page_icon="📊",
    layout="wide",
)


# =========================
# Estilo
# =========================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #5f6368;
            margin-bottom: 1rem;
        }
        .insight-box {
            background: #f7f8fa;
            border: 1px solid #e6e8eb;
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Configuração esperada de colunas
# =========================
ALIASES = {
    "ticket_id": ["id", "ticket id", "ticket_id"],
    "subject": ["subject", "assunto", "ticket subject"],
    "status": ["status"],
    "type": ["type", "ticket type", "tipo"],
    "channel": ["channel", "via", "canal"],
    "category": ["category", "categoria", "group category", "problem category"],
    "created_at": ["created at", "created_at", "data de criação", "created"],
    "updated_at": ["updated at", "updated_at", "data de atualização", "updated"],
    "solved_at": ["solved at", "solved_at", "resolved at", "data de resolução", "resolution date"],
    "closed_at": ["closed at", "closed_at", "data de fechamento"],
    "requester": ["requester", "solicitante", "requester name"],
    "assignee": ["assignee", "responsável", "assigned to", "owner"],
    "csm": ["csm", "customer success manager", "gerente da conta"],
    "satisfaction": ["satisfaction", "csat", "score", "ticket satisfaction"],
}


@dataclass
class ColumnMap:
    ticket_id: Optional[str] = None
    subject: Optional[str] = None
    status: Optional[str] = None
    type: Optional[str] = None
    channel: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    solved_at: Optional[str] = None
    closed_at: Optional[str] = None
    requester: Optional[str] = None
    assignee: Optional[str] = None
    csm: Optional[str] = None
    satisfaction: Optional[str] = None


# =========================
# Helpers
# =========================
def normalize_col(col: str) -> str:
    return (
        str(col)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("  ", " ")
    )


def detect_columns(df: pd.DataFrame) -> ColumnMap:
    normalized = {normalize_col(c): c for c in df.columns}
    result = {}
    for key, aliases in ALIASES.items():
        original = None
        for alias in aliases:
            if alias in normalized:
                original = normalized[alias]
                break
        result[key] = original
    return ColumnMap(**result)


def read_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin1")
    return pd.read_excel(uploaded_file)


def prepare_dataframe(df: pd.DataFrame, colmap: ColumnMap) -> pd.DataFrame:
    out = df.copy()

    for col_name in [colmap.created_at, colmap.updated_at, colmap.solved_at, colmap.closed_at]:
        if col_name and col_name in out.columns:
            out[col_name] = pd.to_datetime(out[col_name], errors="coerce")

    for col_name in [colmap.status, colmap.type, colmap.channel, colmap.category, colmap.requester, colmap.assignee, colmap.csm]:
        if col_name and col_name in out.columns:
            out[col_name] = out[col_name].astype(str).str.strip()
            out.loc[out[col_name].isin(["nan", "None", "NaT"]), col_name] = np.nan

    if colmap.created_at and colmap.created_at in out.columns:
        out["mes_referencia"] = out[colmap.created_at].dt.to_period("M").astype(str)
        out["data_abertura"] = out[colmap.created_at].dt.date
        out["dia_semana"] = out[colmap.created_at].dt.day_name()
        out["hora_abertura"] = out[colmap.created_at].dt.hour

    out["foi_resolvido"] = False
    if colmap.status and colmap.status in out.columns:
        status_norm = out[colmap.status].astype(str).str.lower()
        out["foi_resolvido"] = status_norm.isin(["solved", "closed", "resolved"])

    if colmap.created_at and colmap.solved_at and colmap.created_at in out.columns and colmap.solved_at in out.columns:
        delta = (out[colmap.solved_at] - out[colmap.created_at]).dt.total_seconds() / 86400
        out["tempo_resolucao_dias"] = delta.round(2)
    else:
        out["tempo_resolucao_dias"] = np.nan

    out["aging_bucket"] = pd.cut(
        out["tempo_resolucao_dias"],
        bins=[-0.001, 1, 3, 7, 10000],
        labels=["0–1 dia", "2–3 dias", "4–7 dias", "8+ dias"],
    )

    if colmap.subject and colmap.subject in out.columns:
        out["tema_normalizado"] = (
            out[colmap.subject]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
        )
    else:
        out["tema_normalizado"] = np.nan

    return out


def generate_insights(df: pd.DataFrame, colmap: ColumnMap) -> list[str]:
    insights = []
    total = len(df)

    if total == 0:
        return ["Nenhum ticket encontrado no período ou nos filtros aplicados."]

    if colmap.category and colmap.category in df.columns:
        cat = df[colmap.category].fillna("Sem categoria").value_counts()
        if len(cat) > 0:
            insights.append(
                f"A categoria com maior volume de chamados é **{cat.index[0]}**, com **{cat.iloc[0]} tickets**."
            )

    resolved = int(df["foi_resolvido"].sum())
    insights.append(
        f"A taxa de resolução é **{(resolved/total)*100:.1f}%**, com **{resolved} tickets concluídos**."
    )

    tempo_valid = df["tempo_resolucao_dias"].dropna()
    if len(tempo_valid) > 0:
        insights.append(
            f"O tempo médio de resolução é **{tempo_valid.mean():.1f} dias**."
        )

    return insights


def generate_executive_summary(df: pd.DataFrame, colmap: ColumnMap) -> str:
    total = len(df)
    resolved = int(df["foi_resolvido"].sum())
    backlog = total - resolved

    media = df["tempo_resolucao_dias"].dropna().mean()
    media_txt = f"{media:.1f} dias" if pd.notna(media) else "não disponível"

    return (
        f"Foram registrados **{total} chamados**, com **{resolved} resolvidos** "
        f"e **{backlog} pendentes**. O tempo médio de resolução foi **{media_txt}**."
    )


# =========================
# Header
# =========================
st.markdown('<div class="main-title">Oitchau | Análise de Chamados Zendesk</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ferramenta interna para diagnóstico operacional e apresentação a clientes.</div>',
    unsafe_allow_html=True,
)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Configurações")
    cliente = st.text_input("Nome do cliente", value="C.Vale")
    uploaded_file = st.file_uploader("Upload do arquivo Zendesk", type=["xlsx", "csv"])


if not uploaded_file:
    st.info("Faça upload de um arquivo para iniciar a análise.")
    st.stop()


raw_df = read_file(uploaded_file)
colmap = detect_columns(raw_df)
df = prepare_dataframe(raw_df, colmap)


# =========================
# Overview
# =========================
st.markdown("### Visão geral")

total = len(df)
resolved = int(df["foi_resolvido"].sum())
backlog = total - resolved

c1, c2, c3 = st.columns(3)

c1.metric("Total de chamados", total)
c2.metric("Resolvidos", resolved)
c3.metric("Backlog", backlog)


# =========================
# Gráfico categorias
# =========================
if colmap.category and colmap.category in df.columns:

    st.markdown("### Chamados por categoria")

    cat_counts = df[colmap.category].fillna("Sem categoria").value_counts().head(10)

    fig, ax = plt.subplots()
    cat_counts.plot(kind="barh", ax=ax)

    st.pyplot(fig)


# =========================
# Insights
# =========================
st.markdown("### Insights automáticos")

for insight in generate_insights(df, colmap):
    st.markdown(f"- {insight}")


# =========================
# Resumo executivo
# =========================
st.markdown("### Resumo executivo")

summary = generate_executive_summary(df, colmap)

st.text_area("Texto pronto para apresentação", summary, height=150)


# =========================
# Base exploratória
# =========================
st.markdown("### Base exploratória")

st.dataframe(df)


# =========================
# Download
# =========================
buffer = io.BytesIO()

with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df.to_excel(writer, index=False)

st.download_button(
    "Baixar base tratada",
    data=buffer.getvalue(),
    file_name="analise_chamados.xlsx"
)
