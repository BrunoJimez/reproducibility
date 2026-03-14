"""
app.py — Interface Streamlit do sistema Reprodutibilidade
─────────────────────────────────────────────────────────
Execute com:  streamlit run app.py
"""

import io
import json
import streamlit as st
import pandas as pd
import numpy as np

from reproducibility_core import (
    Variable, Hypothesis,
    OpenTranslator, RuleParser,
    DataFetcher,
    run_pipeline,
    generate_dashboard,
    ReproducibilityEngine,
    EmpiricalTester,
    HypothesisComparator,
    TemporalAnalyser,
)

# ─────────────────────────────────────────────────────────────────────
#  CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Reprodutibilidade",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title  { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
    .sub-title   { font-size: 1rem; color: #888; margin-bottom: 1.5rem; }
    .status-ok   { color: #76ff03; font-weight: 600; }
    .status-warn { color: #ffea00; font-weight: 600; }
    .metric-box  { background: #1a1a1a; border-radius: 8px; padding: 1rem;
                   border: 1px solid #2a2a2a; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  ESTADO DA SESSÃO
# ─────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "hipoteses":    [],
        "df":           None,
        "target_column":  None,
        "mappings":  [],
        "resultados":   None,
        "log_nlp":      [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 Reprodutibilidade")
    st.markdown("Teste hipóteses científicas contra dados reais.")
    st.divider()

    # ── Configuração do Tradutor ─────────────────────────────────────
    st.markdown("### Módulo 1 — Tradutor de Hipóteses")
    with st.expander("Configurar LLM (opcional)", expanded=False):
        groq_key = st.text_input(
            "Chave Groq (gratuita)",
            type="password",
            help="Cadastro gratuito em console.groq.com — sem cartão de crédito",
            placeholder="gsk_...",
        )
        groq_modelo = st.selectbox(
            "Modelo Groq",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
             "mixtral-8x7b-32768", "gemma2-9b-it"],
        )
        ollama_model = st.text_input(
            "Modelo Ollama (local)",
            value="llama3.2",
            help="Instale em ollama.ai e execute: ollama pull llama3.2",
        )

    tradutor = OpenTranslator(
        groq_key=groq_key or None,
        groq_modelo=groq_modelo,
        ollama_model=ollama_model,
    )
    status = tradutor.status()
    for camada, s in status.items():
        cor = "status-ok" if "✅" in s else "status-warn"
        st.markdown(f'<span class="{cor}">{s}</span><br><small style="color:#666">{camada}</small>',
                    unsafe_allow_html=True)

    st.divider()

    # ── Parâmetros ───────────────────────────────────────────────────
    st.markdown("### ⚙️ Parâmetros")
    n_trials    = st.slider("Trials de simulação", 20, 200, 60, 10)
    sample_size = st.slider("Amostras por trial",  50, 500, 200, 50)
    n_windows   = st.slider("Janelas temporais",    3,  10,   5,  1)
    seed        = st.number_input("Seed aleatório", value=42, step=1)

    st.divider()
    st.markdown("""
    <small style="color:#555">
    Ciência Aberta — sem APIs pagas.<br>
    <a href="https://console.groq.com" target="_blank">Groq (gratuito)</a> ·
    <a href="https://ollama.ai" target="_blank">Ollama (local)</a>
    </small>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  CONTEÚDO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🔬 Reprodutibilidade</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Se os padrões da realidade são consistentes, '
    'uma hipótese verdadeira deve ser reproduzível.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["📝 Hipóteses", "📊 Dados", "🚀 Executar & Resultados"])


# ════════════════════════════════════════════════════════════════════
#  TAB 1 — HIPÓTESES
# ════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Defina suas hipóteses em linguagem natural")
    st.markdown(
        "Escreva como um cientista escreveria num caderno de pesquisa. "
        "O sistema converte automaticamente para função matemática."
    )

    col_input, col_info = st.columns([2, 1])

    with col_input:
        texto_hipotese = st.text_area(
            "Hipótese",
            placeholder=(
                "Exemplos:\n"
                "• A renda cresce linearmente com os anos de estudo e experiência\n"
                "• A população cresce exponencialmente com a taxa de reprodução\n"
                "• A temperatura de um gás cresce proporcionalmente à pressão\n"
                "• O PIB per capita aumenta com o acesso à educação"
            ),
            height=120,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            btn_traduzir = st.button("🔄 Traduzir hipótese", type="primary", use_container_width=True)
        with col_b:
            btn_limpar   = st.button("🗑 Limpar lista", use_container_width=True)

        if btn_limpar:
            st.session_state.hipoteses = []
            st.session_state.mappings = []
            st.rerun()

        if btn_traduzir and texto_hipotese.strip():
            log = []
            with st.spinner("Traduzindo..."):
                h = tradutor.traduzir(texto_hipotese.strip(), log)
            for msg in log:
                st.info(msg)
            st.session_state.hipoteses.append(h)
            st.session_state.mappings.append({})
            st.success(f"✅ Hipótese adicionada: **{h.nome}**")

    with col_info:
        st.markdown("**Como funciona:**")
        st.markdown("""
        1. Escreve em português ou inglês
        2. O tradutor identifica variáveis e relação matemática
        3. Gera código Python automaticamente
        4. Você pode ajustar manualmente se quiser
        """)
        st.markdown("**Relações suportadas:**")
        st.markdown("Linear · Inversa · Quadrática · Cúbica · Raiz · Exponencial · Logarítmica")

    # Lista de hipóteses adicionadas
    if st.session_state.hipoteses:
        st.divider()
        st.markdown(f"### Hipóteses ({len(st.session_state.hipoteses)})")

        for idx, h in enumerate(st.session_state.hipoteses):
            with st.expander(f"**{idx+1}. {h.nome}**  `{h.origem}`", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"*{h.descricao}*")
                    if h.predicao:
                        st.markdown(f"**Predição:** {h.predicao}")
                    st.markdown("**Variáveis:**")
                    for v in h.variaveis:
                        st.markdown(f"- `{v.nome}` ∈ [{v.minimo}, {v.maximo}] — {v.descricao}")
                    if h.codigo_gerado:
                        st.code(h.codigo_gerado, language="python")
                with col2:
                    if st.button("🗑 Remover", key=f"rm_{idx}"):
                        st.session_state.hipoteses.pop(idx)
                        st.session_state.mappings.pop(idx)
                        st.rerun()


# ════════════════════════════════════════════════════════════════════
#  TAB 2 — DADOS
# ════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Fonte de dados")

    fonte = st.radio(
        "Escolha a fonte:",
        ["📂 Upload CSV/Excel", "🌐 World Bank (API aberta)",
         "🛰 NASA POWER (clima)", "🇧🇷 IBGE (PIB Brasil)", "🎲 Dataset sintético"],
        horizontal=False,
    )

    buscador = DataFetcher()

    # ── Upload ──────────────────────────────────────────────────────
    if fonte == "📂 Upload CSV/Excel":
        arquivo = st.file_uploader("Selecione o arquivo", type=["csv", "xlsx", "xls"])
        if arquivo:
            try:
                if arquivo.name.endswith(".csv"):
                    sep = st.selectbox("Separador", [",", ";", "\t", "|"], index=0)
                    df = pd.read_csv(arquivo, sep=sep)
                else:
                    df = pd.read_excel(arquivo)
                st.session_state.df = df
                st.success(f"✅ {len(df)} linhas × {len(df.columns)} colunas carregadas")
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")

    # ── World Bank ──────────────────────────────────────────────────
    elif fonte == "🌐 World Bank (API aberta)":
        st.markdown("Dados abertos do Banco Mundial — sem chave de API.")
        col1, col2, col3 = st.columns(3)
        with col1:
            indicadores_disp = list(buscador.listar_indicadores_worldbank().keys())
            ind_y = st.selectbox("Variável dependente (y)", indicadores_disp, index=0)
            ind_x = st.selectbox("Variável independente (x)", indicadores_disp, index=1)
        with col2:
            pais_opcoes = {"Brasil": "BR", "EUA": "US", "Alemanha": "DE",
                          "China": "CN", "Índia": "IN", "Todos": "all"}
            pais_label  = st.selectbox("País", list(pais_opcoes.keys()), index=0)
            pais_codigo = pais_opcoes[pais_label]
        with col3:
            ano_ini = st.number_input("Ano início", 1990, 2022, 2000)
            ano_fim = st.number_input("Ano fim",    1990, 2023, 2022)

        if st.button("🌐 Buscar dados", type="primary"):
            with st.spinner("Consultando World Bank API..."):
                try:
                    if pais_label == "Todos":
                        df = buscador.buscar_worldbank_multiplos_paises(
                            [ind_y, ind_x], ano_inicio=int(ano_ini), ano_fim=int(ano_fim))
                        df = df.rename(columns={ind_y: "y", ind_x: "x"})
                    else:
                        df_y = buscador.buscar_worldbank(ind_y, pais_codigo, int(ano_ini), int(ano_fim))
                        df_x = buscador.buscar_worldbank(ind_x, pais_codigo, int(ano_ini), int(ano_fim))
                        df = pd.merge(df_y.rename(columns={"valor": "y"})[["ano", "y"]],
                                      df_x.rename(columns={"valor": "x"})[["ano", "x"]],
                                      on="ano").dropna()
                    st.session_state.df = df
                    st.success(f"✅ {len(df)} observações carregadas — World Bank")
                except Exception as e:
                    st.error(f"Erro ao buscar dados: {e}\n\nVerifique sua conexão com a internet.")

    # ── NASA POWER ──────────────────────────────────────────────────
    elif fonte == "🛰 NASA POWER (clima)":
        st.markdown("Dados climáticos da NASA — sem chave de API.")
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude",  value=-15.78, step=0.1)
            lon = st.number_input("Longitude", value=-47.93, step=0.1)
            st.caption("Ex: Brasília = -15.78, -47.93 | São Paulo = -23.55, -46.63")
        with col2:
            params_disp = {"Temperatura (°C)": "T2M", "Precipitação (mm/dia)": "PRECTOTCORR",
                           "Radiação solar (W/m²)": "ALLSKY_SFC_SW_DWN",
                           "Umidade relativa (%)": "RH2M"}
            param_y = st.selectbox("Variável dependente", list(params_disp.keys()), 0)
            param_x = st.selectbox("Variável independente", list(params_disp.keys()), 1)
            ano_ini_n = st.number_input("Ano início", 1990, 2022, 2000, key="nasa_ini")
            ano_fim_n = st.number_input("Ano fim",    1990, 2022, 2020, key="nasa_fim")

        if st.button("🛰 Buscar dados climáticos", type="primary"):
            with st.spinner("Consultando NASA POWER..."):
                try:
                    parametros = list(set([params_disp[param_y], params_disp[param_x]]))
                    df = buscador.buscar_nasa_clima(lat, lon, parametros,
                                                   int(ano_ini_n), int(ano_fim_n))
                    df = df.rename(columns={params_disp[param_y]: "y",
                                            params_disp[param_x]: "x"})
                    st.session_state.df = df
                    st.success(f"✅ {len(df)} observações — NASA POWER")
                except Exception as e:
                    st.error(f"Erro: {e}")

    # ── IBGE ────────────────────────────────────────────────────────
    elif fonte == "🇧🇷 IBGE (PIB Brasil)":
        st.markdown("PIB brasileiro anual do IBGE SIDRA — sem chave de API.")
        if st.button("🇧🇷 Buscar PIB Brasil", type="primary"):
            with st.spinner("Consultando IBGE SIDRA..."):
                try:
                    df = buscador.buscar_ibge_pib()
                    df["ano_idx"] = range(len(df))
                    st.session_state.df = df
                    st.success(f"✅ {len(df)} anos de PIB carregados — IBGE")
                except Exception as e:
                    st.error(f"Erro: {e}")

    # ── Sintético ───────────────────────────────────────────────────
    else:
        st.markdown("Dataset sintético realista (offline, sem internet).")
        col1, col2 = st.columns(2)
        with col1:
            n_sint = st.slider("Observações", 100, 1000, 300, 50)
            seed_s = st.number_input("Seed", value=42, step=1, key="seed_sint")
        with col2:
            st.markdown("""
            **Variáveis geradas:**
            - `ano` (2000–2023)
            - `estudo` (anos de educação)
            - `experiencia` (anos de trabalho)
            - `renda` (com componente aleatório realista)
            """)
        if st.button("🎲 Gerar dataset", type="primary"):
            df = buscador.gerar_sintetico(n_sint, int(seed_s))
            st.session_state.df = df
            st.success(f"✅ {len(df)} observações geradas")

    # ── Preview dos dados ───────────────────────────────────────────
    if st.session_state.df is not None:
        df = st.session_state.df
        st.divider()
        st.markdown("### Pré-visualização")
        col1, col2, col3 = st.columns(3)
        col1.metric("Linhas",   len(df))
        col2.metric("Colunas",  len(df.columns))
        col3.metric("Fonte",    df.attrs.get("fonte", "—"))
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("**Estatísticas descritivas:**")
        st.dataframe(df.describe().round(3), use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  TAB 3 — EXECUTAR
# ════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Pipeline completo")

    if not st.session_state.hipoteses:
        st.warning("⚠️ Adicione pelo menos uma hipótese na aba **Hipóteses**.")
    elif st.session_state.df is None:
        st.warning("⚠️ Carregue um dataset na aba **Dados**.")
    else:
        df = st.session_state.df

        st.markdown("#### Configurar mapping variáveis → colunas")
        st.markdown(
            "Para cada hipótese, informe qual coluna do dataset corresponde a cada variável."
        )

        colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        mappings = []
        target_column = st.selectbox("Coluna alvo (variável dependente y)", colunas_num)

        colunas_tempo_disp = ["(nenhuma)"] + [c for c in df.columns
                                              if "ano" in c.lower() or "year" in c.lower()
                                              or "data" in c.lower() or "date" in c.lower()]
        time_column = st.selectbox("Coluna de tempo (para análise temporal)", colunas_tempo_disp)
        time_column = None if time_column == "(nenhuma)" else time_column

        valido = True
        for idx, h in enumerate(st.session_state.hipoteses):
            with st.expander(f"**{h.nome}** — mapping", expanded=True):
                mapa = {}
                for v in h.variaveis:
                    col_escolhida = st.selectbox(
                        f"`{v.nome}` ({v.descricao})",
                        colunas_num,
                        key=f"map_{idx}_{v.nome}",
                    )
                    mapa[v.nome] = col_escolhida
                mappings.append(mapa)

        st.divider()
        btn_executar = st.button("🚀 Executar análise completa", type="primary",
                                 use_container_width=True, disabled=not valido)

        if btn_executar:
            progress = st.progress(0, text="Iniciando...")
            status_txt = st.empty()

            try:
                status_txt.text("Módulo 2 — Simulação de reprodutibilidade...")
                progress.progress(20)

                status_txt.text("Módulo 3 — Teste com dados empíricos...")
                progress.progress(40)

                status_txt.text("Módulo 4 — Comparando hipóteses...")
                progress.progress(60)

                status_txt.text("Módulo 5 — Análise temporal...")
                progress.progress(80)

                resultados = run_pipeline(
                    hipoteses    = st.session_state.hipoteses,
                    df           = df,
                    target_column  = target_column,
                    mappings  = mappings,
                    time_column = time_column,
                    n_trials     = n_trials,
                    sample_size  = sample_size,
                    n_windows    = n_windows,
                    seed         = int(seed),
                )
                st.session_state.resultados = resultados

                status_txt.text("Gerando dashboard...")
                progress.progress(95)
                progress.progress(100, text="Concluído!")

            except Exception as e:
                st.error(f"Erro durante a análise: {e}")
                st.exception(e)

        # ── Exibição dos resultados ──────────────────────────────────
        if st.session_state.resultados:
            res = st.session_state.resultados
            st.divider()

            # Métricas rápidas
            vencedor = res["comparacao"].vencedor
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🏆 Melhor hipótese", vencedor["nome"].split()[0])
            col2.metric("R² vencedor",  f"{vencedor['r2']:.3f}")
            col3.metric("Peso Akaike",  f"{vencedor['peso_akaike']:.1%}")
            col4.metric("Hipóteses testadas", len(res["simulacao"]))

            st.divider()

            # Dashboard matplotlib
            st.markdown("### Dashboard completo")
            st.pyplot(res["figura"], use_container_width=True)

            # Download do dashboard
            buf = io.BytesIO()
            res["figura"].savefig(buf, format="png", dpi=150,
                                  bbox_inches="tight", facecolor="#0d0d0d")
            st.download_button(
                "⬇️ Baixar dashboard (PNG)",
                data=buf.getvalue(),
                file_name="reprodutibilidade_dashboard.png",
                mime="image/png",
            )

            st.divider()

            # Tabela de ranking
            st.markdown("### Ranking completo")
            ranking_df = pd.DataFrame([{
                "Hipótese":       r["nome"],
                "R²":             round(r["r2"], 4),
                "RMSE":           round(r["rmse"], 2),
                "AIC":            round(r["aic"], 1),
                "Δ AIC":          round(r["delta_aic"], 1),
                "Peso Akaike":    f"{r['peso_akaike']:.1%}",
                "Score simulação":f"{r['score_simulado']:.1f}" if r["score_simulado"] else "—",
            } for r in res["comparacao"].ranking])
            st.dataframe(ranking_df, use_container_width=True)

            # Download CSV
            csv = ranking_df.to_csv(index=False)
            st.download_button(
                "⬇️ Baixar ranking (CSV)",
                data=csv,
                file_name="reprodutibilidade_ranking.csv",
                mime="text/csv",
            )

            # Detalhes por hipótese
            st.divider()
            st.markdown("### Detalhes por hipótese")
            for rs, re_ in zip(res["simulacao"], res["empirico"]):
                with st.expander(f"**{rs.hypothesis_name}**", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Score simulação", f"{rs.reproducibility_score:.1f}/100")
                    c2.metric("CV",    f"{rs.coefficient_of_variation:.1f}%")
                    c3.metric("R²",    f"{re_.r_squared:.4f}")
                    c4.metric("RMSE",  f"{re_.rmse:.2f}")
                    st.markdown(f"**Classificação:** {rs.classification}")
                    st.markdown(f"**Resíduos normais (p-valor):** {re_.p_valor_residuals:.4f} "
                                + ("✅" if re_.p_valor_residuals > 0.05 else "⚠️"))

            # Análise temporal
            if res["temporal"]:
                st.divider()
                st.markdown("### Análise temporal")
                for rt in res["temporal"]:
                    cols = st.columns(len(rt.windows))
                    for col, jan, score in zip(cols, rt.windows, rt.scores_per_window):
                        col.metric(jan, f"{score:.1f}")
                    status = "⚠️ Deriva detectada" if rt.drift_detected else "✅ Estável no tempo"
                    st.markdown(f"**{rt.hypothesis_name}:** {status} "
                                f"(tendência: {rt.trend:+.2f} pontos/janela)")
