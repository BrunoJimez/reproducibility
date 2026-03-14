# Reproducibility 🔬

> *"If the patterns of reality are consistent, a true hypothesis must be reproducible."*

An open-source system for quantitative evaluation of scientific hypotheses
formulated in natural language. Write your hypothesis in plain English (or
Portuguese), connect it to real data from open sources, and receive a
reproducibility score validated against published literature.

**No proprietary tools. No fees. No API key required to get started.**

---

## Quick start (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/reproducibility.git
cd reproducibility

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the web interface
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`.

---

## How it works

### Tab 1 — Hypotheses
Write your scientific hypothesis in plain English:
```
Life expectancy increases with GDP per capita
CO₂ emissions grow exponentially with industrial output
Wages vary linearly with years of education and experience
```
The system automatically converts it into an executable mathematical function.

### Tab 2 — Data
Choose your data source:
- **Upload CSV/Excel** — your own dataset
- **World Bank** — global indicators (GDP, life expectancy, CO₂, unemployment…)
- **NASA POWER** — climate data at any geographic coordinate
- **IBGE** — Brazilian national statistics
- **Synthetic dataset** — generated internally (works offline)

### Tab 3 — Run
Map your hypothesis variables to dataset columns, click **Run**, and receive
the full dashboard with four analytical modules.

---

## The four analytical modules

| Module | What it measures |
|--------|-----------------|
| **Simulated reproducibility** | Internal consistency via Monte Carlo (score 0–100) |
| **Empirical fit** | How well the hypothesis explains real data (R², RMSE, AIC) |
| **Hypothesis comparison** | Which competing hypothesis best explains the phenomenon (Akaike weights) |
| **Temporal drift** | Is the hypothesis stable over time, or does it weaken? |

### The composite score

$$S_{\text{comp}} = S_{\text{sim}} \cdot \max(0,\ R^2)$$

A hypothesis must be both *internally consistent* (high $S_{\text{sim}}$) **and**
*explain the data* (positive $R^2$) to achieve a high composite score.

---

## Validation results

The composite score was validated against three published datasets with known
expected behaviour:

| Case | Best hypothesis | Composite score | Expected | |
|---|---|---|---|---|
| CO₂ → Global temperature (IPCC AR6) | Linear | **49.1** | ≥ 40 | ✅ |
| Minimum wage → Employment (Card & Krueger 1994) | Positive effect | **5.0** | ≤ 20 | ✅ |
| Original effect → Replication (OSC 2015) | 50% attenuation | **20.5** | 10–30 | ✅ |

And against four a priori cases:

| Hypothesis | Score | CV |
|---|---|---|
| Newton's Law of Gravitation | **100.0** | 0.000% |
| Constant function y = 5 | **100.0** | 0.000% |
| Pure Meritocracy + structural noise | **0.8** | 48.8% |
| Random noise hypothesis | **0.0** | 2129% |

---

## Module 1 — Natural language translation (Open Science layers)

| Layer | Tool | Requirement | Privacy |
|-------|------|-------------|---------|
| 0 — Always available | Rule-based parser (offline) | None | Full |
| 1 — Free cloud LLM | [Groq](https://console.groq.com) | Free key (no card) | Data sent to Groq |
| 2 — Local LLM | [Ollama](https://ollama.ai) | `ollama pull llama3.2` | Full (local) |

The system tries Ollama → Groq → RuleParser automatically.

---

## Open data sources integrated

| Source | Data | Authentication |
|--------|------|----------------|
| [World Bank API](https://data.worldbank.org) | 12 global indicators | None |
| [NASA POWER](https://power.larc.nasa.gov) | Climate data by coordinate | None |
| [IBGE SIDRA](https://sidra.ibge.gov.br) | Brazilian GDP and more | None |

---

## File structure

```
reproducibility/
├── reproducibility_core.py   # Core logic (no UI dependencies)
├── app.py                    # Streamlit web interface
├── phase3_validation.py      # External validation script
├── tests/
│   └── test_core.py          # 40 automated tests
├── requirements.txt
├── paper.md                  # JOSS article
├── paper.bib                 # References
├── README.md                 # This file
└── LICENSE                   # MIT
```

---

## Running the tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
pytest tests/ --cov=reproducibility_core --cov-report=term-missing
```

All 40 tests should pass. The scientific score validation tests verify that the
instrument correctly distinguishes between physical laws (score = 100), social
constructions with high noise (score < 1), and intermediate empirical cases.

---

## Using with your own hypothesis and data

```python
from reproducibility_core import OpenTranslator, DataFetcher, run_pipeline

# Translate hypothesis
translator = OpenTranslator()
h = translator.translate("Population grows exponentially with the reproduction rate")

# Fetch real data
df = DataFetcher().fetch_worldbank("population", "BR", 2000, 2022)

# Run full pipeline
results = run_pipeline(
    hypotheses    = [h],
    df            = df,
    target_column = "value",
    mappings      = [{"N": "value"}],
    time_column   = "year",
)
print(results["simulation"][0].reproducibility_score)
```

---

## Citation

If you use this software in your research, please cite:

```bibtex
@article{reproducibility2026,
  author  = {[Your Name]},
  title   = {Reproducibility: an open-source tool for computational
             hypothesis testing in natural language},
  journal = {Journal of Open Source Software},
  year    = {2026},
  doi     = {10.21105/joss.XXXXX}
}
```

---

## Licence

MIT — use, modify, and distribute freely.
Science must be open and accessible to all.
