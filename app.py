import io
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages


st.set_page_config(
    page_title="Oitchau | Análise de Chamados V3",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
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
        .logo-wrap {
            margin-bottom: 0.75rem;
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
    cleaned = series.astype(str).str.replace("\xa0", "", regex=False).str.strip()
    cleaned = cleaned.replace({"nan": np.nan, "None": np.nan, "NaT": np.nan, "": np.nan})
    return cleaned


def normalize_status_value(value: str):
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

    for col_name in [colmap.created_at, colmap.updated_at, colmap.solved_at, colmap.due_at]:
        if col_name and col_name in out.columns:
            out[col_name] = out[col_name].replace({"\xa0": np.nan, "": np.nan, " ": np.nan})
            out[col_name] = pd.to_datetime(out[col_name], errors="coerce")

    for col_name in [
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
    ]:
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
        out["tempo_resolucao_dias"] = ((out[colmap.solved_at] - out[colmap.created_at]).dt.total_seconds() / 86400).round(2)
    else:
        out["tempo_resolucao_dias"] = np.nan

    if colmap.created_at and colmap.updated_at and colmap.created_at in out.columns and colmap.updated_at in out.columns:
        out["idade_ticket_dias"] = ((out[colmap.updated_at] - out[colmap.created_at]).dt.total_seconds() / 86400).round(2)
    else:
        out["idade_ticket_dias"] = np.nan

    if colmap.created_at and colmap.due_at and colmap.created_at in out.columns and colmap.due_at in out.columns:
        out["sla_medio_horas"] = ((out[colmap.due_at] - out[colmap.created_at]).dt.total_seconds() / 3600).round(2)
    else:
        out["sla_medio_horas"] = np.nan

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
    else:
        out["is_integration"] = False

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
            insights.append(
                f"A categoria com maior volume é {cat_counts.index[0]}, com {int(cat_counts.iloc[0])} tickets ({(cat_counts.iloc[0] / total) * 100:.1f}% do total)."
            )

    if colmap.type and colmap.type in df.columns:
        type_counts = df[colmap.type].fillna("Sem tipo").value_counts()
        if len(type_counts) > 0:
            insights.append(
                f"O tipo mais recorrente é {type_counts.index[0]}, com {int(type_counts.iloc[0])} ocorrências, ajudando a separar incidentes técnicos de dúvidas operacionais."
            )

    resolved = int(df["foi_resolvido"].sum())
    insights.append(
        f"A taxa de resolução ou encerramento no recorte é de {(resolved / total) * 100:.1f}%, com {resolved} tickets concluídos de {total}."
    )

    mean_resolution = df["tempo_resolucao_dias"].dropna()
    if len(mean_resolution) > 0:
        insights.append(
            f"O tempo médio de resolução é de {mean_resolution.mean():.1f} dias, com {int((mean_resolution > 7).sum())} tickets acima de 7 dias."
        )

    sla_mean = df["sla_medio_horas"].dropna()
    if len(sla_mean) > 0:
        insights.append(
            f"O SLA médio calculado pela diferença entre criação e due date dos tickets é de {sla_mean.mean():.1f} horas."
        )

    if colmap.subject and colmap.subject in df.columns:
        recurrent = df["tema_normalizado"].dropna().value_counts()
        recurrent = recurrent[recurrent >= 2]
        if len(recurrent) > 0:
            insights.append(
                f"Há sinais de recorrência: o tema normalizado “{recurrent.index[0]}” apareceu {int(recurrent.iloc[0])} vezes."
            )

    return insights


def generate_executive_summary(df: pd.DataFrame, colmap: ColumnMap) -> str:
    total = len(df)
    resolved = int(df["foi_resolvido"].sum())
    backlog = total - resolved
    avg_resolution = df["tempo_resolucao_dias"].dropna().mean()
    avg_txt = f"{avg_resolution:.1f} dias" if pd.notna(avg_resolution) else "não disponível"
    avg_sla = df["sla_medio_horas"].dropna().mean()
    avg_sla_txt = f"{avg_sla:.1f} horas" if pd.notna(avg_sla) else "não disponível"

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
        f"No período analisado para {org_txt}, foram registrados {total} chamados, com {resolved} tickets concluídos e {backlog} ainda pendentes ou em acompanhamento. "
        f"O tempo médio de resolução foi de {avg_txt}. A principal frente observada foi {top_category_txt}, enquanto o tipo de demanda mais recorrente foi {top_type_txt}. "
        f"O SLA médio calculado para o período foi de {avg_sla_txt}. Esse recorte permite direcionar discussões de causa raiz, treinamento operacional e revisão de integrações."
    )


def build_onepage_pdf(cliente: str, periodo: str, resumo: str, plano_acao: str, logo_file=None) -> bytes:
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")

        if logo_file is not None:
            try:
                logo_file.seek(0)
                img = plt.imread(logo_file)
                ax_img = fig.add_axes([0.08, 0.90, 0.18, 0.07])
                ax_img.imshow(img)
                ax_img.axis("off")
            except Exception:
                pass

        fig.text(0.08, 0.95, "One-page executivo | Análise de Chamados", fontsize=18, fontweight="bold")
        fig.text(0.08, 0.92, f"Cliente: {cliente}", fontsize=11)
        fig.text(0.08, 0.90, f"Período analisado: {periodo}", fontsize=11)
        fig.text(0.08, 0.84, "Resumo executivo", fontsize=14, fontweight="bold")
        fig.text(0.08, 0.82, resumo, fontsize=10.5, va="top", wrap=True)
        fig.text(0.08, 0.52, "Plano de ação", fontsize=14, fontweight="bold")
        fig.text(0.08, 0.50, plano_acao.strip() if plano_acao.strip() else "Plano de ação não preenchido.", fontsize=10.5, va="top", wrap=True)
        fig.text(0.08, 0.06, "Gerado automaticamente pela ferramenta interna de análise de chamados.", fontsize=9)
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


st.markdown('<div class="main-title">Oitchau | Análise de Chamados Zendesk</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Configurações")
    cliente = st.text_input("Nome do cliente", value="C.Vale")
    periodo_analisado = st.text_input("Período analisado", value="Fevereiro/2026")
    uploaded_file = st.file_uploader("Upload do arquivo Zendesk", type=["xlsx", "csv"])
    logo_cliente = st.file_uploader("Upload da logo do cliente", type=["png", "jpg", "jpeg"])
    mostrar_base = st.checkbox("Mostrar base exploratória", value=True)
    top_n = st.slider("Top N para rankings", min_value=5, max_value=15, value=10)
    st.divider()
    st.markdown("**Acesso**")
    senha_digitada = st.text_input("Senha do projeto", type="password")
    senha_correta = st.secrets.get("APP_PASSWORD", "oitchau123")
    if senha_digitada != senha_correta:
        st.warning("Digite a senha correta para acessar o projeto.")
        st.stop()
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

if logo_cliente is not None:
    st.markdown('<div class="logo-wrap"></div>', unsafe_allow_html=True)
    st.image(logo_cliente, width=180)

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
    with f5:
        if colmap.channel and colmap.channel in df.columns:
            channel_options = sorted(safe_series(df, colmap.channel).dropna().unique().tolist())
            channel_sel = st.multiselect("Canal", channel_options, default=channel_options)
        else:
            channel_sel = []

filtered = df.copy()
if status_sel:
    filtered = filtered[filtered["status_padronizado"].isin(status_sel)]
if colmap.type and type_sel:
    filtered = filtered[filtered[colmap.type].isin(type_sel)]
if colmap.category and category_sel:
    filtered = filtered[filtered[colmap.category].isin(category_sel)]
if colmap.channel and channel_sel:
    filtered = filtered[filtered[colmap.channel].isin(channel_sel)]
if intervalo_datas and colmap.created_at and colmap.created_at in filtered.columns and isinstance(intervalo_datas, tuple) and len(intervalo_datas) == 2:
    data_inicio, data_fim = intervalo_datas
    filtered = filtered[filtered[colmap.created_at].dt.date.between(data_inicio, data_fim)]

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
avg_sla = filtered["sla_medio_horas"].dropna().mean()

m1, m2, m3, m4, m5, m6, m7 = st.columns(7, gap="small")
with m1:
    metric_card("Total de chamados", str(total))
with m2:
    metric_card("Resolvidos/encerrados", str(resolved))
with m3:
    metric_card("Backlog aberto", str(backlog))
with m4:
    metric_card("Em Hold", str(hold_count))
with m5:
    metric_card("Tempo médio", mean_resolution_txt)
with m6:
    metric_card("% Bugs", bug_pct)
with m7:
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
    metric_card("SLA médio", f"{avg_sla:.1f} h" if pd.notna(avg_sla) else "N/D")

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

st.markdown('<div class="section-title">Evolução diária de abertura <span class="help-chip" title="Apresenta a variação diária de tickets abertos no período, ajudando a identificar picos operacionais e eventos concentrados.">?</span></div>', unsafe_allow_html=True)
if colmap.created_at and colmap.created_at in filtered.columns:
    timeline = filtered.dropna(subset=[colmap.created_at]).groupby(filtered[colmap.created_at].dt.date).size()
    fig, ax = plt.subplots(figsize=(12, 4))
    timeline.plot(ax=ax)
    ax.set_xlabel("Data")
    ax.set_ylabel("Chamados")
    st.pyplot(fig)
else:
    st.warning("Coluna de data de criação não encontrada.")

st.markdown('<div class="section-title">SLA <span class="help-chip" title="Apresenta o SLA médio calculado a partir da diferença entre criação e due date dos tickets no período analisado.">?</span></div>', unsafe_allow_html=True)
s1, s2 = st.columns(2)
with s1:
    metric_card("SLA médio", f"{avg_sla:.1f} h" if pd.notna(avg_sla) else "N/D")
with s2:
    metric_card("Tickets com due date", str(int(filtered["sla_medio_horas"].notna().sum())))

st.markdown('<div class="section-title">Insights <span class="help-chip" title="Síntese textual gerada automaticamente com os principais achados do período, pronta para apoiar apresentações executivas.">?</span></div>', unsafe_allow_html=True)
for insight in generate_insights(filtered, colmap):
    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Resumo executivo <span class="help-chip" title="Texto consolidado para usar em relatórios e apresentações ao cliente, resumindo volume, perfil dos chamados e pontos de atenção.">?</span></div>', unsafe_allow_html=True)
summary_text = generate_executive_summary(filtered, colmap)
st.text_area("Texto pronto para apresentação", value=summary_text, height=180)

st.markdown('<div class="section-title">Plano de ação <span class="help-chip" title="Campo livre para registrar encaminhamentos, responsáveis e próximos passos combinados para o cliente.">?</span></div>', unsafe_allow_html=True)
plano_acao = st.text_area("Plano de ação do período", height=180, placeholder="Ex.: Revisar integrações críticas, alinhar treinamento com usuários-chave, acompanhar tickets de lentidão...")

pdf_bytes = build_onepage_pdf(cliente, periodo_exibicao, summary_text, plano_acao, logo_cliente)
st.download_button(
    label="Baixar one-page em PDF",
    data=pdf_bytes,
    file_name=f"onepage_chamados_{cliente.lower().replace(' ', '_')}.pdf",
    mime="application/pdf",
)

if mostrar_base:
    st.markdown('<div class="section-title">Base exploratória <span class="help-chip" title="Tabela detalhada dos tickets após tratamento e filtros aplicados, útil para validação e análises mais profundas.">?</span></div>', unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, height=420)

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    filtered.to_excel(writer, index=False, sheet_name="analise")

st.download_button(
    label="Baixar base tratada (.xlsx)",
    data=buffer.getvalue(),
    file_name=f"analise_chamados_{cliente.lower().replace(' ', '_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
