"""
app.py — Streamlit web interface for the Reproducibility system
───────────────────────────────────────────────────────────────
Run with:  streamlit run app.py
"""

import io
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
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Reproducibility",
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
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "hypotheses": [],
        "df":          None,
        "target_column": None,
        "mappings":    [],
        "results":     None,
        "nlp_log":     [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 Reproducibility")
    st.markdown("Test scientific hypotheses against real data.")
    st.divider()

    # ── Translator configuration ─────────────────────────────────────
    st.markdown("### Module 1 — Hypothesis Translator")
    with st.expander("Configure LLM (optional)", expanded=False):
        groq_key = st.text_input(
            "Groq API key (free)",
            type="password",
            help="Free account at console.groq.com — no credit card required",
            placeholder="gsk_...",
        )
        groq_modelo = st.selectbox(
            "Groq model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
             "mixtral-8x7b-32768", "gemma2-9b-it"],
        )
        ollama_model = st.text_input(
            "Ollama model (local)",
            value="llama3.2",
            help="Install at ollama.ai then run: ollama pull llama3.2",
        )

    translator = OpenTranslator(
        groq_key=groq_key or None,
        groq_modelo=groq_modelo,
        ollama_model=ollama_model,
    )
    status = translator.status()
    for layer, s in status.items():
        css_class = "status-ok" if "✅" in s else "status-warn"
        st.markdown(
            f'<span class="{css_class}">{s}</span>'
            f'<br><small style="color:#666">{layer}</small>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Simulation parameters ────────────────────────────────────────
    st.markdown("### ⚙️ Parameters")
    n_trials    = st.slider("Simulation trials",  20, 200, 60,  10)
    sample_size = st.slider("Samples per trial",  50, 500, 200, 50)
    n_windows   = st.slider("Temporal windows",    3,  10,   5,  1)
    seed        = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.markdown("""
    <small style="color:#555">
    Open Science — no paid APIs.<br>
    <a href="https://console.groq.com" target="_blank">Groq (free)</a> ·
    <a href="https://ollama.ai" target="_blank">Ollama (local)</a>
    </small>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🔬 Reproducibility</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'If the patterns of reality are consistent, a true hypothesis must be reproducible.'
    '</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["📝 Hypotheses", "📊 Data", "🚀 Run & Results"])


# ════════════════════════════════════════════════════════════════════
#  TAB 1 — HYPOTHESES
# ════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Define your hypotheses in natural language")
    st.markdown(
        "Write as a scientist would in a research notebook. "
        "The system automatically converts the text into a mathematical function."
    )

    col_input, col_info = st.columns([2, 1])

    with col_input:
        hypothesis_text = st.text_area(
            "Hypothesis",
            placeholder=(
                "Examples:\n"
                "• Income grows linearly with years of education and experience\n"
                "• Population grows exponentially with the reproduction rate\n"
                "• Temperature of a gas increases proportionally with pressure\n"
                "• GDP per capita increases with access to education"
            ),
            height=120,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            btn_translate = st.button("🔄 Translate hypothesis", type="primary",
                                      use_container_width=True)
        with col_b:
            btn_clear = st.button("🗑 Clear list", use_container_width=True)

        if btn_clear:
            st.session_state.hypotheses = []
            st.session_state.mappings   = []
            st.rerun()

        if btn_translate and hypothesis_text.strip():
            log = []
            with st.spinner("Translating..."):
                h = translator.translate(hypothesis_text.strip(), log)
            for msg in log:
                st.info(msg)
            st.session_state.hypotheses.append(h)
            st.session_state.mappings.append({})
            st.success(f"✅ Hypothesis added: **{h.name}**")

    with col_info:
        st.markdown("**How it works:**")
        st.markdown("""
        1. Write in English or Portuguese
        2. The translator identifies variables and the mathematical relationship
        3. Generates Python code automatically
        4. You can adjust the mapping manually if needed
        """)
        st.markdown("**Supported relationships:**")
        st.markdown("Linear · Inverse · Quadratic · Cubic · Square root · "
                    "Exponential · Logarithmic")

    # ── List of added hypotheses ─────────────────────────────────────
    if st.session_state.hypotheses:
        st.divider()
        st.markdown(f"### Hypotheses ({len(st.session_state.hypotheses)})")

        for idx, h in enumerate(st.session_state.hypotheses):
            with st.expander(f"**{idx+1}. {h.name}**  `{h.origin}`", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"*{h.description}*")
                    if h.prediction:
                        st.markdown(f"**Prediction:** {h.prediction}")
                    st.markdown("**Variables:**")
                    for v in h.variables:
                        st.markdown(
                            f"- `{v.name}` ∈ [{v.minimum}, {v.maximum}] — {v.description}"
                        )
                    if h.generated_code:
                        st.code(h.generated_code, language="python")
                with col2:
                    if st.button("🗑 Remove", key=f"rm_{idx}"):
                        st.session_state.hypotheses.pop(idx)
                        st.session_state.mappings.pop(idx)
                        st.rerun()


# ════════════════════════════════════════════════════════════════════
#  TAB 2 — DATA
# ════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Data source")

    source = st.radio(
        "Choose source:",
        ["📂 Upload CSV/Excel", "🌐 World Bank (open API)",
         "🛰 NASA POWER (climate)", "🇧🇷 IBGE (Brazilian GDP)", "🎲 Synthetic dataset"],
        horizontal=False,
    )

    fetcher = DataFetcher()

    # ── Upload ───────────────────────────────────────────────────────
    if source == "📂 Upload CSV/Excel":
        uploaded = st.file_uploader("Select file", type=["csv", "xlsx", "xls"])
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    sep = st.selectbox("Separator", [",", ";", "\t", "|"], index=0)
                    df  = pd.read_csv(uploaded, sep=sep)
                else:
                    df = pd.read_excel(uploaded)
                st.session_state.df = df
                st.success(f"✅ {len(df)} rows × {len(df.columns)} columns loaded")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # ── World Bank ───────────────────────────────────────────────────
    elif source == "🌐 World Bank (open API)":
        st.markdown("World Bank Open Data — no API key required.")
        col1, col2, col3 = st.columns(3)
        with col1:
            available_indicators = list(fetcher.list_worldbank_indicators().keys())
            ind_y = st.selectbox("Dependent variable (y)",   available_indicators, index=0)
            ind_x = st.selectbox("Independent variable (x)", available_indicators, index=1)
        with col2:
            country_options = {
                "Brazil": "BR", "USA": "US", "Germany": "DE",
                "China": "CN", "India": "IN", "All": "all",
            }
            country_label = st.selectbox("Country", list(country_options.keys()), index=0)
            country_code  = country_options[country_label]
        with col3:
            year_start = st.number_input("Start year", 1990, 2022, 2000)
            year_end   = st.number_input("End year",   1990, 2023, 2022)

        if st.button("🌐 Fetch data", type="primary"):
            with st.spinner("Querying World Bank API..."):
                try:
                    if country_label == "All":
                        df = fetcher.fetch_worldbank_multiple_countries(
                            [ind_y, ind_x],
                            year_start=int(year_start),
                            year_end=int(year_end),
                        )
                        df = df.rename(columns={ind_y: "y", ind_x: "x"})
                    else:
                        df_y = fetcher.fetch_worldbank(ind_y, country_code,
                                                       int(year_start), int(year_end))
                        df_x = fetcher.fetch_worldbank(ind_x, country_code,
                                                       int(year_start), int(year_end))
                        df = pd.merge(
                            df_y.rename(columns={"value": "y"})[["year", "y"]],
                            df_x.rename(columns={"value": "x"})[["year", "x"]],
                            on="year",
                        ).dropna()
                    st.session_state.df = df
                    st.success(f"✅ {len(df)} observations loaded — World Bank")
                except Exception as e:
                    st.error(f"Error fetching data: {e}\n\nCheck your internet connection.")

    # ── NASA POWER ───────────────────────────────────────────────────
    elif source == "🛰 NASA POWER (climate)":
        st.markdown("NASA climate data — no API key required.")
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude",  value=-15.78, step=0.1)
            lon = st.number_input("Longitude", value=-47.93, step=0.1)
            st.caption("e.g. Brasília = -15.78, -47.93 | São Paulo = -23.55, -46.63")
        with col2:
            params_available = {
                "Temperature (°C)":        "T2M",
                "Precipitation (mm/day)":  "PRECTOTCORR",
                "Solar radiation (W/m²)":  "ALLSKY_SFC_SW_DWN",
                "Relative humidity (%)":   "RH2M",
            }
            param_y      = st.selectbox("Dependent variable",   list(params_available.keys()), 0)
            param_x      = st.selectbox("Independent variable", list(params_available.keys()), 1)
            year_start_n = st.number_input("Start year", 1990, 2022, 2000, key="nasa_start")
            year_end_n   = st.number_input("End year",   1990, 2022, 2020, key="nasa_end")

        if st.button("🛰 Fetch climate data", type="primary"):
            with st.spinner("Querying NASA POWER..."):
                try:
                    parameters = list(set([params_available[param_y],
                                           params_available[param_x]]))
                    df = fetcher.fetch_nasa_climate(lat, lon, parameters,
                                                    int(year_start_n), int(year_end_n))
                    df = df.rename(columns={params_available[param_y]: "y",
                                            params_available[param_x]: "x"})
                    st.session_state.df = df
                    st.success(f"✅ {len(df)} observations — NASA POWER")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── IBGE ─────────────────────────────────────────────────────────
    elif source == "🇧🇷 IBGE (Brazilian GDP)":
        st.markdown("Annual Brazilian GDP from IBGE SIDRA — no API key required.")
        if st.button("🇧🇷 Fetch Brazilian GDP", type="primary"):
            with st.spinner("Querying IBGE SIDRA..."):
                try:
                    df = fetcher.fetch_ibge_gdp()
                    df["year_index"] = range(len(df))
                    st.session_state.df = df
                    st.success(f"✅ {len(df)} years of GDP loaded — IBGE")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Synthetic ────────────────────────────────────────────────────
    else:
        st.markdown("Realistic synthetic dataset (offline, no internet required).")
        col1, col2 = st.columns(2)
        with col1:
            n_synth  = st.slider("Observations", 100, 1000, 300, 50)
            seed_syn = st.number_input("Seed", value=42, step=1, key="seed_synth")
        with col2:
            st.markdown("""
            **Generated variables:**
            - `ano` — year (2000–2023)
            - `estudo` — years of education
            - `experiencia` — years of work experience
            - `renda` — income (with realistic random component)
            """)
        if st.button("🎲 Generate dataset", type="primary"):
            df = fetcher.generate_synthetic(n_synth, int(seed_syn))
            st.session_state.df = df
            st.success(f"✅ {len(df)} observations generated")

    # ── Data preview ─────────────────────────────────────────────────
    if st.session_state.df is not None:
        df = st.session_state.df
        st.divider()
        st.markdown("### Preview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows",    len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Source",  df.attrs.get("source", "—"))
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("**Descriptive statistics:**")
        st.dataframe(df.describe().round(3), use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  TAB 3 — RUN & RESULTS
# ════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Full pipeline")

    if not st.session_state.hypotheses:
        st.warning("⚠️ Add at least one hypothesis in the **Hypotheses** tab.")
    elif st.session_state.df is None:
        st.warning("⚠️ Load a dataset in the **Data** tab.")
    else:
        df = st.session_state.df

        st.markdown("#### Configure variable → column mapping")
        st.markdown(
            "For each hypothesis, specify which dataset column corresponds to each variable."
        )

        numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        mappings        = []
        target_column   = st.selectbox("Target column (dependent variable y)",
                                        numeric_columns)

        time_options = ["(none)"] + [
            c for c in df.columns
            if any(kw in c.lower() for kw in ["ano", "year", "data", "date"])
        ]
        time_column = st.selectbox("Time column (for temporal analysis)", time_options)
        time_column = None if time_column == "(none)" else time_column

        valid = True
        for idx, h in enumerate(st.session_state.hypotheses):
            with st.expander(f"**{h.name}** — mapping", expanded=True):
                mapa = {}
                for v in h.variables:
                    chosen_col = st.selectbox(
                        f"`{v.name}` ({v.description})",
                        numeric_columns,
                        key=f"map_{idx}_{v.name}",
                    )
                    mapa[v.name] = chosen_col
                mappings.append(mapa)

        st.divider()
        btn_run = st.button("🚀 Run full analysis", type="primary",
                            use_container_width=True, disabled=not valid)

        if btn_run:
            progress   = st.progress(0, text="Starting...")
            status_txt = st.empty()

            try:
                status_txt.text("Module 2 — Monte Carlo reproducibility simulation...")
                progress.progress(20)

                status_txt.text("Module 3 — Empirical data test...")
                progress.progress(40)

                status_txt.text("Module 4 — Comparing hypotheses...")
                progress.progress(60)

                status_txt.text("Module 5 — Temporal analysis...")
                progress.progress(80)

                results = run_pipeline(
                    hypotheses    = st.session_state.hypotheses,
                    df            = df,
                    target_column = target_column,
                    mappings      = mappings,
                    time_column   = time_column,
                    n_trials      = n_trials,
                    sample_size   = sample_size,
                    n_windows     = n_windows,
                    seed          = int(seed),
                )
                st.session_state.results = results

                status_txt.text("Generating dashboard...")
                progress.progress(95)
                progress.progress(100, text="Done!")

            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.exception(e)

        # ── Results display ──────────────────────────────────────────
        if st.session_state.results:
            res = st.session_state.results
            st.divider()

            # Quick metrics
            winner = res["comparison"].winner
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🏆 Best hypothesis", winner["name"].split()[0])
            col2.metric("Winner R²",          f"{winner['r2']:.3f}")
            col3.metric("Akaike weight",       f"{winner['akaike_weight']:.1%}")
            col4.metric("Hypotheses tested",   len(res["simulation"]))

            st.divider()

            # Matplotlib dashboard
            st.markdown("### Complete dashboard")
            st.pyplot(res["figure"], use_container_width=True)

            # Dashboard download
            buf = io.BytesIO()
            res["figure"].savefig(buf, format="png", dpi=150,
                                  bbox_inches="tight", facecolor="#0d0d0d")
            st.download_button(
                "⬇️ Download dashboard (PNG)",
                data=buf.getvalue(),
                file_name="reproducibility_dashboard.png",
                mime="image/png",
            )

            st.divider()

            # Full ranking table
            st.markdown("### Full ranking")
            ranking_df = pd.DataFrame([{
                "Hypothesis":    r["name"],
                "R²":            round(r["r2"], 4),
                "RMSE":          round(r["rmse"], 2),
                "AIC":           round(r["aic"], 1),
                "Δ AIC":         round(r["delta_aic"], 1),
                "Akaike weight": f"{r['akaike_weight']:.1%}",
                "Sim. score":    f"{r['simulated_score']:.1f}" if r["simulated_score"] else "—",
            } for r in res["comparison"].ranking])
            st.dataframe(ranking_df, use_container_width=True)

            csv = ranking_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download ranking (CSV)",
                data=csv,
                file_name="reproducibility_ranking.csv",
                mime="text/csv",
            )

            # Per-hypothesis details
            st.divider()
            st.markdown("### Details per hypothesis")
            for rs, re_ in zip(res["simulation"], res["empirical"]):
                with st.expander(f"**{rs.hypothesis_name}**", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Sim. score", f"{rs.reproducibility_score:.1f}/100")
                    c2.metric("CV",   f"{rs.coefficient_of_variation:.1f}%")
                    c3.metric("R²",   f"{re_.r_squared:.4f}")
                    c4.metric("RMSE", f"{re_.rmse:.2f}")
                    st.markdown(f"**Classification:** {rs.classification}")
                    st.markdown(
                        f"**Residuals normal (p-value):** {re_.residuals_p_value:.4f} "
                        + ("✅" if re_.residuals_p_value > 0.05 else "⚠️")
                    )

            # Temporal analysis
            if res["temporal"]:
                st.divider()
                st.markdown("### Temporal analysis")
                for rt in res["temporal"]:
                    cols = st.columns(len(rt.windows))
                    for col, win, score in zip(cols, rt.windows, rt.scores_per_window):
                        col.metric(win, f"{score:.1f}")
                    status = ("⚠️ Drift detected" if rt.drift_detected
                              else "✅ Stable over time")
                    st.markdown(
                        f"**{rt.hypothesis_name}:** {status} "
                        f"(trend: {rt.trend:+.2f} points/window)"
                    )
