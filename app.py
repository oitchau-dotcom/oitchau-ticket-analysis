import io
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Oitchau | Análise de Chamados V2",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
        .stApp h1, .main-title {
            line-height: 1.2;
            word-break: break-word;
        }
        .help-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border-radius: 999px;
            border: 1px solid #cfd4dc;
            color: #4b5563;
            font-size: 12px;
            font-weight: 700;
            margin-left: 6px;
            cursor: help;
            background: #ffffff;
        }
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 1.5rem;
            max-width: 1450px;
        }
        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
            line-height: 1.15;
            word-break: break-word;
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
        .small-note {
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: -0.15rem;
            margin-bottom: 0.65rem;
        }
        .hero-card {
            background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 0.8rem;
        }
        .hero-kicker {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 0.25rem;
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 800;
            color: #111827;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }
        .hero-meta {
            color: #4b5563;
            font-size: 0.96rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


ALIASES = {
    "ticket_id": ["ticket id", "id", "ticket_id"],
    "organization": ["ticket organization name", "organization", "org", "cliente", "client"],
    "experience": ["experience", "xp"],
    "ticket_channel": ["ticket channel", "via"],
    "status": ["ticket status", "status"],
    "subject": ["ticket subject", "subject", "assunto"],
    "category": ["category", "categoria"],
    "type": ["type", "ticket type", "tipo"],
    "reason": ["ticket reason", "reason", "motivo"],
    "channel": ["channel", "canal"],
    "requester": ["requester name", "requester", "solicitante"],
    "assignee": ["assignee name", "assignee", "responsável", "owner"],
    "csm": ["csm", "customer success manager", "gerente da conta"],
    "created_at": ["ticket created - date", "created at", "created_at", "created"],
    "updated_at": ["ticket updated - date", "updated at", "updated_at", "updated"],
    "solved_at": ["ticket solved - date", "solved at", "solved_at", "resolved at", "resolution date"],
    "due_at": ["ticket due - date", "due date", "due_at"],
    "satisfaction": ["ticket satisfaction rating", "ticket satisfaction", "satisfaction", "csat"],
    "satisfaction_score": ["% satisfaction score", "satisfaction score", "csat %"],
}


@dataclass
class ColumnMap:
    ticket_id: Optional[str] = None
    organization: Optional[str] = None
    experience: Optional[str] = None
    ticket_channel: Optional[str] = None
    status: Optional[str] = None
    subject: Optional[str] = None
    category: Optional[str] = None
    type: Optional[str] = None
    reason: Optional[str] = None
    channel: Optional[str] = None
    requester: Optional[str] = None
    assignee: Optional[str] = None
    csm: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    solved_at: Optional[str] = None
    due_at: Optional[str] = None
    satisfaction: Optional[str] = None
    satisfaction_score: Optional[str] = None


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
        found = None
        for alias in aliases:
            alias_norm = normalize_col(alias)
            if alias_norm in normalized:
                found = normalized[alias_norm]
                break
        result[key] = found
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


def clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
    )
    cleaned = cleaned.replace({"nan": np.nan, "None": np.nan, "NaT": np.nan, "": np.nan})
    return cleaned


def normalize_status_value(value: str) -> str:
    if pd.isna(value):
        return np.nan
    txt = str(value).strip().lower()
    mapping = {
        "new": "New",
        "open": "Open",
        "pending": "Pending",
        "hold": "Hold",
        "solved": "Solved",
        "closed": "Closed",
        "resolved": "Solved",
    }
    return mapping.get(txt, str(value).strip())


def prepare_dataframe(df: pd.DataFrame, colmap: ColumnMap) -> pd.DataFrame:
    out = df.copy()

    datetime_cols = [colmap.created_at, colmap.updated_at, colmap.solved_at, colmap.due_at]
    for col_name in datetime_cols:
        if col_name and col_name in out.columns:
            out[col_name] = out[col_name].replace({"\xa0": np.nan, "": np.nan, " ": np.nan})
            out[col_name] = pd.to_datetime(out[col_name], errors="coerce")

    text_cols = [
        colmap.organization,
        colmap.experience,
        colmap.ticket_channel,
        colmap.status,
        colmap.subject,
        colmap.category,
        colmap.type,
        colmap.reason,
        colmap.channel,
        colmap.requester,
        colmap.assignee,
        colmap.csm,
        colmap.satisfaction,
    ]
    for col_name in text_cols:
        if col_name and col_name in out.columns:
            out[col_name] = clean_text_series(out[col_name])

    if colmap.status and colmap.status in out.columns:
        out["status_padronizado"] = out[colmap.status].apply(normalize_status_value)
    else:
        out["status_padronizado"] = np.nan

    if colmap.created_at and colmap.created_at in out.columns:
        out["mes_referencia"] = out[colmap.created_at].dt.to_period("M").astype(str)
        out["data_abertura"] = out[colmap.created_at].dt.date
        out["dia_semana"] = out[colmap.created_at].dt.day_name()
        out["hora_abertura"] = out[colmap.created_at].dt.hour

    out["foi_resolvido"] = out["status_padronizado"].isin(["Solved", "Closed"])
    out["esta_aberto"] = out["status_padronizado"].isin(["New", "Open", "Pending", "Hold"])
    out["em_hold"] = out["status_padronizado"].eq("Hold")

    if colmap.created_at and colmap.solved_at and colmap.created_at in out.columns and colmap.solved_at in out.columns:
        delta_res = (out[colmap.solved_at] - out[colmap.created_at]).dt.total_seconds() / 86400
        out["tempo_resolucao_dias"] = delta_res.round(2)
    else:
        out["tempo_resolucao_dias"] = np.nan

    if colmap.created_at and colmap.updated_at and colmap.created_at in out.columns and colmap.updated_at in out.columns:
        delta_age = (out[colmap.updated_at] - out[colmap.created_at]).dt.total_seconds() / 86400
        out["idade_ticket_dias"] = delta_age.round(2)
    else:
        out["idade_ticket_dias"] = np.nan

    if colmap.created_at and colmap.due_at and colmap.created_at in out.columns and colmap.due_at in out.columns:
        delta_sla = (out[colmap.due_at] - out[colmap.created_at]).dt.total_seconds() / 3600
        out["sla_previsto_horas"] = delta_sla.round(2)
    else:
        out["sla_previsto_horas"] = np.nan

    if colmap.due_at and colmap.due_at in out.columns:
        out["sla_estourado"] = np.where(
            out["foi_resolvido"] & out[colmap.solved_at].notna() & out[colmap.due_at].notna(),
            out[colmap.solved_at] > out[colmap.due_at],
            np.where(
                out["esta_aberto"] & out[colmap.due_at].notna(),
                pd.Timestamp.now() > out[colmap.due_at],
                False,
            ),
        )
    else:
        out["sla_estourado"] = False

    out["aging_bucket"] = pd.cut(
        out["tempo_resolucao_dias"],
        bins=[-0.001, 1, 3, 7, 100000],
        labels=["0–1 dia", "2–3 dias", "4–7 dias", "8+ dias"],
    )

    if colmap.type and colmap.type in out.columns:
        type_norm = out[colmap.type].astype(str).str.strip().str.lower()
        out["is_bug"] = type_norm.eq("open bug")
        out["is_question"] = type_norm.eq("hr questions")
        out["is_request"] = type_norm.isin(["open request", "manual action"])
    else:
        out["is_bug"] = False
        out["is_question"] = False
        out["is_request"] = False

    if colmap.category and colmap.category in out.columns:
        cat_norm = out[colmap.category].astype(str).str.strip().str.lower()
        out["is_integration"] = cat_norm.eq("api/integration")
        out["is_performance"] = cat_norm.eq("performance")
    else:
        out["is_integration"] = False
        out["is_performance"] = False

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

    if colmap.satisfaction_score and colmap.satisfaction_score in out.columns:
        out[colmap.satisfaction_score] = pd.to_numeric(out[colmap.satisfaction_score], errors="coerce")

    return out


def safe_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col and col in df.columns:
        return df[col]
    return pd.Series(dtype="object")


def metric_card(label: str, value: str):
    st.metric(label, value)


def generate_insights(df: pd.DataFrame, colmap: ColumnMap) -> list[str]:
    insights = []
    total = len(df)
    if total == 0:
        return ["Nenhum ticket encontrado no período ou nos filtros aplicados."]

    if colmap.category and colmap.category in df.columns:
        cat_counts = df[colmap.category].fillna("Sem categoria").value_counts()
        if len(cat_counts) > 0:
            top_cat = cat_counts.index[0]
            top_cat_n = int(cat_counts.iloc[0])
            insights.append(
                f"A categoria com maior volume é **{top_cat}**, com **{top_cat_n} tickets** ({(top_cat_n / total) * 100:.1f}% do total)."
            )

    if colmap.type and colmap.type in df.columns:
        type_counts = df[colmap.type].fillna("Sem tipo").value_counts()
        if len(type_counts) > 0:
            top_type = type_counts.index[0]
            top_type_n = int(type_counts.iloc[0])
            insights.append(
                f"O tipo mais recorrente é **{top_type}**, com **{top_type_n} ocorrências**, ajudando a separar incidentes técnicos de dúvidas operacionais."
            )

    resolved = int(df["foi_resolvido"].sum())
    insights.append(
        f"A taxa de resolução/encerramento no recorte é de **{(resolved / total) * 100:.1f}%**, com **{resolved} tickets concluídos** de **{total}**."
    )

    mean_resolution = df["tempo_resolucao_dias"].dropna()
    if len(mean_resolution) > 0:
        insights.append(
            f"O SLA médio é de **{mean_resolution.mean():.1f} dias**, com **{int((mean_resolution > 7).sum())} tickets** acima de 7 dias."
        )

    sla_overdue = int(pd.Series(df["sla_estourado"]).fillna(False).sum())
    if sla_overdue > 0:
        insights.append(
            f"Foram identificados **{sla_overdue} tickets fora do SLA**, ponto importante para discussões de priorização e fluxo operacional."
        )

    if colmap.subject and colmap.subject in df.columns:
        recurrent = df["tema_normalizado"].dropna().value_counts()
        recurrent = recurrent[recurrent >= 2]
        if len(recurrent) > 0:
            insights.append(
                f"Há sinais de recorrência: o tema normalizado **“{recurrent.index[0]}”** apareceu **{int(recurrent.iloc[0])} vezes**."
            )

    return insights


def generate_executive_summary(df: pd.DataFrame, colmap: ColumnMap) -> str:
    total = len(df)
    resolved = int(df["foi_resolvido"].sum())
    backlog = total - resolved
    avg_resolution = df["tempo_resolucao_dias"].dropna().mean()
    avg_txt = f"{avg_resolution:.1f} dias" if pd.notna(avg_resolution) else "não disponível"

    top_category_txt = "não identificada"
    if colmap.category and colmap.category in df.columns and df[colmap.category].notna().any():
        top_cat = df[colmap.category].fillna("Sem categoria").value_counts().head(1)
        top_category_txt = f"{top_cat.index[0]} ({int(top_cat.iloc[0])} chamados)"

    top_type_txt = "não identificado"
    if colmap.type and colmap.type in df.columns and df[colmap.type].notna().any():
        top_type = df[colmap.type].fillna("Sem tipo").value_counts().head(1)
        top_type_txt = f"{top_type.index[0]} ({int(top_type.iloc[0])} chamados)"

    org_txt = "o cliente"
    if colmap.organization and colmap.organization in df.columns and df[colmap.organization].notna().any():
        org_txt = str(df[colmap.organization].dropna().iloc[0])

    return (
        f"No período analisado para **{org_txt}**, foram registrados **{total} chamados**, com **{resolved} tickets concluídos** "
        f"e **{backlog} ainda pendentes ou em acompanhamento**. O **SLA médio** foi de **{avg_txt}**. "
        f"A principal frente observada foi **{top_category_txt}**, enquanto o tipo de demanda mais recorrente foi **{top_type_txt}**. "
        f"Esse recorte permite direcionar discussões de causa raiz, treinamento operacional, revisão de integrações e aderência ao SLA."
    )


st.markdown('<div class="main-title">Oitchau | Análise de Chamados Zendesk</div>', unsafe_allow_html=True)
# subtítulo removido a pedido

with st.sidebar:
    st.header("Configurações")
    cliente = st.text_input("Nome do cliente", value="C.Vale")
    periodo_analisado = st.text_input("Período analisado", value="Fevereiro/2026")
    uploaded_file = st.file_uploader("Upload do arquivo Zendesk", type=["xlsx", "csv"])
    mostrar_base = st.checkbox("Mostrar base exploratória", value=True)
    top_n = st.slider("Top N para rankings", min_value=5, max_value=15, value=10)
    st.divider()
    st.markdown("**Filtros rápidos**")
    usar_periodo_automatico = st.checkbox("Usar período automático do arquivo", value=True)
    mostrar_diagnostico = st.checkbox("Mostrar diagnóstico de colunas", value=False)

if not uploaded_file:
    st.info("Faça upload de um arquivo para iniciar a análise.")
    st.stop()

try:
    raw_df = read_file(uploaded_file)
except Exception as e:
    st.error(f"Não foi possível ler o arquivo: {e}")
    st.stop()

colmap = detect_columns(raw_df)
df = prepare_dataframe(raw_df, colmap)

if usar_periodo_automatico and colmap.created_at and colmap.created_at in df.columns and df[colmap.created_at].notna().any():
    data_min = df[colmap.created_at].min()
    data_max = df[colmap.created_at].max()
    if pd.notna(data_min) and pd.notna(data_max):
        periodo_exibicao = f"{data_min.strftime('%d/%m/%Y')} a {data_max.strftime('%d/%m/%Y')}"
    else:
        periodo_exibicao = periodo_analisado
else:
    periodo_exibicao = periodo_analisado

st.markdown(
    f'''
    <div class="hero-card">
        <div class="hero-kicker">Relatório executivo</div>
        <div class="hero-title">{cliente}</div>
        <div class="hero-meta"><strong>Período analisado:</strong> {periodo_exibicao}</div>
    </div>
    ''',
    unsafe_allow_html=True,
)

if mostrar_diagnostico:
    with st.expander("Diagnóstico do mapeamento de colunas", expanded=False):
        mapping_df = pd.DataFrame(
            {
                "campo_logico": list(colmap.__dict__.keys()),
                "coluna_detectada": list(colmap.__dict__.values()),
            }
        )
        st.dataframe(mapping_df, use_container_width=True)

with st.expander("Filtros", expanded=True):
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        status_options = sorted(df["status_padronizado"].dropna().unique().tolist())
        status_sel = st.multiselect("Status", status_options, default=status_options)
    with f2:
        type_options = sorted(safe_series(df, colmap.type).dropna().unique().tolist())
        type_sel = st.multiselect("Tipo", type_options, default=type_options)
    with f3:
        category_options = sorted(safe_series(df, colmap.category).dropna().unique().tolist())
        category_sel = st.multiselect("Categoria", category_options, default=category_options)
    with f4:
        assignee_options = sorted(safe_series(df, colmap.assignee).dropna().unique().tolist())
        assignee_sel = st.multiselect("Responsável", assignee_options, default=assignee_options)
    with f5:
        if colmap.created_at and colmap.created_at in df.columns and df[colmap.created_at].notna().any():
            data_min_filtro = df[colmap.created_at].min().date()
            data_max_filtro = df[colmap.created_at].max().date()
            intervalo_datas = st.date_input(
                "Período do arquivo",
                value=(data_min_filtro, data_max_filtro),
                min_value=data_min_filtro,
                max_value=data_max_filtro,
            )
        else:
            intervalo_datas = None

filtered = df.copy()
if status_sel:
    filtered = filtered[filtered["status_padronizado"].isin(status_sel)]
if colmap.type and type_sel:
    filtered = filtered[filtered[colmap.type].isin(type_sel)]
if colmap.category and category_sel:
    filtered = filtered[filtered[colmap.category].isin(category_sel)]
if colmap.assignee and assignee_sel:
    filtered = filtered[filtered[colmap.assignee].isin(assignee_sel)]
if intervalo_datas and colmap.created_at and colmap.created_at in filtered.columns:
    if isinstance(intervalo_datas, tuple) and len(intervalo_datas) == 2:
        data_inicio, data_fim = intervalo_datas
        filtered = filtered[
            filtered[colmap.created_at].dt.date.between(data_inicio, data_fim)
        ]

st.markdown('<div class="section-title">Visão geral</div>', unsafe_allow_html=True)

total = len(filtered)
resolved = int(filtered["foi_resolvido"].sum())
backlog = int(filtered["esta_aberto"].sum())
hold_count = int(filtered["em_hold"].sum())
mean_resolution = filtered["tempo_resolucao_dias"].dropna().mean()
mean_resolution_txt = f"{mean_resolution:.1f} dias" if pd.notna(mean_resolution) else "N/D"
bug_pct = f"{(filtered['is_bug'].sum() / total) * 100:.1f}%" if total else "N/D"
integration_pct = f"{(filtered['is_integration'].sum() / total) * 100:.1f}%" if total else "N/D"
question_pct = f"{(filtered['is_question'].sum() / total) * 100:.1f}%" if total else "N/D"
sla_breached = int(pd.Series(filtered["sla_estourado"]).fillna(False).sum())

m1, m2, m3, m4, m5 = st.columns(5, gap="small")
with m1:
    metric_card("Total de chamados", str(total))
with m2:
    metric_card("Resolvidos/encerrados", str(resolved))
with m3:
    metric_card("Backlog aberto", str(backlog))
with m4:
    metric_card("% Bugs", bug_pct)
with m5:
    metric_card("% Integração", integration_pct)

st.markdown('<div class="section-title">KPIs executivos</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4, gap="small")
with k1:
    metric_card("% Dúvidas operacionais", question_pct)
with k2:
    metric_card("Taxa de resolução", f"{(resolved / total) * 100:.1f}%" if total else "N/D")
with k3:
    metric_card("Tickets > 7 dias", str(int((filtered["tempo_resolucao_dias"].fillna(-1) > 7).sum())))
with k4:
    metric_card("Fora do SLA", str(sla_breached))

left, right = st.columns(2)
with left:
    st.markdown('<div class="section-title">Chamados por status <span class="help-chip" title="Mostra a distribuição dos tickets por status padronizado, facilitando a leitura entre backlog, hold e tickets concluídos.">?</span></div>', unsafe_allow_html=True)
    status_counts = filtered["status_padronizado"].fillna("Sem status").value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    status_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Quantidade")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)

with right:
    st.markdown('<div class="section-title">Chamados por categoria <span class="help-chip" title="Mostra quais categorias concentram mais chamados no período, ajudando a identificar causas raiz e frentes prioritárias de atuação.">?</span></div>', unsafe_allow_html=True)
    if colmap.category and colmap.category in filtered.columns:
        cat_counts = filtered[colmap.category].fillna("Sem categoria").value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        cat_counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Quantidade")
        ax.set_ylabel("")
        st.pyplot(fig)
    else:
        st.warning("Coluna de categoria não encontrada.")

left2, right2 = st.columns(2)
with left2:
    st.markdown('<div class="section-title">Chamados por tipo <span class="help-chip" title="Separa os tickets entre bugs, dúvidas operacionais e solicitações, ajudando a entender o perfil da demanda do cliente.">?</span></div>', unsafe_allow_html=True)
    if colmap.type and colmap.type in filtered.columns:
        type_counts = filtered[colmap.type].fillna("Sem tipo").value_counts()
        fig, ax = plt.subplots(figsize=(8, 4.2))
        type_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Quantidade")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
    else:
        st.warning("Coluna de tipo não encontrada.")

with right2:
    st.markdown('<div class="section-title">Aging de resolução <span class="help-chip" title="Agrupa os tickets resolvidos por faixa de tempo de resolução, evidenciando chamados rápidos e casos com maior demora.">?</span></div>', unsafe_allow_html=True)
    aging = filtered["aging_bucket"].value_counts().reindex(["0–1 dia", "2–3 dias", "4–7 dias", "8+ dias"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    aging.plot(kind="bar", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Quantidade")
    st.pyplot(fig)

left3, right3 = st.columns(2)
with left3:
    st.markdown('<div class="section-title">Tickets por responsável <span class="help-chip" title="Mostra a concentração de tickets por responsável, apoiando a leitura de distribuição operacional do atendimento.">?</span></div>', unsafe_allow_html=True)
    if colmap.assignee and colmap.assignee in filtered.columns:
        assignee_counts = filtered[colmap.assignee].fillna("Sem responsável").value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        assignee_counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Quantidade")
        ax.set_ylabel("")
        st.pyplot(fig)
    else:
        st.warning("Coluna de responsável não encontrada.")

with right3:
    st.markdown('<div class="section-title">Principais assuntos <span class="help-chip" title="Exibe os assuntos mais frequentes do período, útil para detectar temas recorrentes e oportunidades de ação preventiva.">?</span></div>', unsafe_allow_html=True)
    if colmap.subject and colmap.subject in filtered.columns:
        subject_counts = filtered[colmap.subject].fillna("Sem assunto").value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        subject_counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Quantidade")
        ax.set_ylabel("")
        st.pyplot(fig)
    else:
        st.warning("Coluna de assunto não encontrada.")

st.markdown('<div class="section-title">Evolução diária de abertura <span class="help-chip" title="Apresenta a variação diária de tickets abertos no período, ajudando a identificar picos operacionais e eventos concentrados.">?</span></div>', unsafe_allow_html=True)
if colmap.created_at and colmap.created_at in filtered.columns:
    timeline = (
        filtered.dropna(subset=[colmap.created_at])
        .groupby(filtered[colmap.created_at].dt.date)
        .size()
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    timeline.plot(ax=ax)
    ax.set_xlabel("Data")
    ax.set_ylabel("Chamados")
    st.pyplot(fig)
else:
    st.warning("Coluna de data de criação não encontrada.")


st.markdown('<div class="section-title">Insights <span class="help-chip" title="Síntese textual gerada automaticamente com os principais achados do período, pronta para apoiar apresentações executivas.">?</span></div>', unsafe_allow_html=True)
for insight in generate_insights(filtered, colmap):
    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Resumo executivo <span class="help-chip" title="Texto consolidado para usar em relatórios e apresentações ao cliente, resumindo volume, perfil dos chamados e pontos de atenção.">?</span></div>', unsafe_allow_html=True)
summary_text = generate_executive_summary(filtered, colmap)
st.text_area("Texto pronto para apresentação", value=summary_text, height=180)

if mostrar_base:
    st.markdown('<div class="section-title">Base exploratória <span class="help-chip" title="Tabela detalhada dos tickets após tratamento e filtros aplicados, útil para validação e análises mais profundas.">?</span></div>', unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, height=420)

export_df = filtered.copy()
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    export_df.to_excel(writer, index=False, sheet_name="analise")

st.download_button(
    label="Baixar base tratada (.xlsx)",
    data=buffer.getvalue(),
    file_name=f"analise_chamados_{cliente.lower().replace(' ', '_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
