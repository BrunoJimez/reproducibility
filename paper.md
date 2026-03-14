---
title: 'Reproducibility: an open-source tool for computational hypothesis testing in natural language'
tags:
  - Python
  - reproducibility
  - open science
  - hypothesis testing
  - Monte Carlo simulation
  - model comparison
  - natural language processing
authors:
  - name: Bruno Jimez
    orcid: 0009-0002-8745-3418
    affiliation: 1
affiliations:
  - name: Independent Researcher, Dourados, MS, Brazil
    index: 1
date: 2026
bibliography: paper.bib
---

# Summary

We present *Reproducibility*, an open-source software system for quantitative
evaluation of scientific hypotheses formulated in natural language. The system
converts scientific statements written in Portuguese or English into executable
mathematical functions, assesses their internal consistency through Monte Carlo
simulation, and measures their predictive power against real empirical data
from open sources (World Bank, NASA POWER, IBGE) or user-supplied CSV files.
We propose a *composite reproducibility score* — combining simulated consistency
and empirical fit — and demonstrate its validity against three cases with
expected behavior established by the literature: the CO₂–global temperature
relationship (composite score = 49.1; expected ≥ 40), the minimum wage effect
on employment from Card & Krueger (composite score = 5.0; expected ≤ 20), and
the replication rate of psychological studies from the Open Science Collaboration
(composite score = 20.5; expected 10–30). The tool requires no proprietary
software, operating with open-source language models (Llama 3, Mistral) via
Ollama or the free Groq API, or with an entirely offline rule-based parser.

# Statement of Need

The reproducibility crisis in science is a documented, large-scale problem.
@ioannidis2005 showed through probabilistic analysis that the majority of
published findings in biomedical research may be false, considering the combined
effects of insufficient sample sizes, analytical flexibility, and publication
bias. This prediction was empirically confirmed by @osc2015, who conducted
replications of 100 psychology studies and found that only 36% reproduced
statistically significant results, with replication effects approximately half
the original magnitude.

There is an important conceptual distinction between *reproducibility* — obtaining
the same results with the same data — and *replicability* — obtaining equivalent
results with new independent data [@goodman2016]. Our tool operates in the second
domain: given that a researcher formulates a hypothesis, it provides an *a priori*
estimate of whether the mathematical structure of that hypothesis is compatible
with high reproducibility before any data collection occurs.

Existing tools for reproducibility support focus primarily on *post hoc*
verification: the `repro` R package allows previously performed analyses to be
reproduced, and platforms such as OSF [@osf2013] facilitate data and code sharing.
We have not identified in the literature a tool that allows a researcher to write
a hypothesis in natural language and receive, within seconds, a quantitative
estimate of its expected reproducibility against open data. This gap motivates
the present contribution.

# Theoretical Background

## Simulated reproducibility score

The *simulated reproducibility score* $S_{\text{sim}}$ is defined as:

$$S_{\text{sim}} = 100 \cdot \exp\!\left(-\frac{CV}{10}\right)$$

where $CV$ is the Coefficient of Variation (standard deviation / mean × 100%)
of the outputs from $n$ independent Monte Carlo simulation trials. Each trial
samples values of the hypothesis variables from their plausible ranges and
computes the output of the corresponding mathematical function.

The function has the following desirable mathematical properties:

1. $S_{\text{sim}} \in [0, 100]$: normalised score, comparable across hypotheses
2. $S_{\text{sim}}(CV=0) = 100$: zero variation implies perfect reproducibility
3. $S_{\text{sim}}(CV \to \infty) \to 0$: unbounded variation implies non-reproducibility
4. Strict monotonicity: increasing $CV$ implies decreasing $S_{\text{sim}}$

## Composite score

Phase 3 validation revealed a fundamental limitation of $S_{\text{sim}}$ in
isolation: hypotheses with simple deterministic functions obtain
$S_{\text{sim}} = 100$ regardless of their predictive power. To also capture
empirical fit, we define the *composite score*:

$$S_{\text{comp}} = S_{\text{sim}} \cdot \max(0,\ R^2)$$

where $R^2$ is the coefficient of determination obtained by testing the hypothesis
against an empirical dataset. This formula penalises hypotheses that, while
mathematically consistent, do not explain the observed variation in data.

## Hypothesis comparison via Akaike criterion

To rank competing hypotheses explaining the same phenomenon, we use the Akaike
information criterion [@burnham2002]:

$$AIC = 2k - 2\ln(\hat{L})$$

The normalised Akaike weight $w_i$ represents the relative probability that each
hypothesis is the best approximation to reality given the tested model set:

$$w_i = \frac{\exp(-\Delta_i / 2)}{\sum_j \exp(-\Delta_j / 2)}$$

# Software Architecture

The system is organised into two main files. **`reproducibility_core.py`** is a
pure logic module with no UI dependencies, importable in any Python environment.
It implements eight subcomponents: data structures (`Variable`, `Hypothesis`,
result types); `OpenTranslator` for three-layer NLP translation;
`ReproducibilityEngine` for Monte Carlo simulation; `EmpiricalTester` for fit
to real data with R², RMSE, AIC, and BIC; `HypothesisComparator` for Akaike
criterion ranking; `TemporalAnalyser` for drift detection; `DataFetcher` for
unauthenticated open API access; and `generate_dashboard` for matplotlib
visualisation. **`app.py`** is a Streamlit web interface with three tabs:
hypothesis definition, data selection, and execution with result visualisation.

## Module 1: Hypothesis translation (Open Science)

The `OpenTranslator` operates in three layers with automatic fallback.
**Layer 0 — RuleParser** is always available, requires zero dependencies, and
works offline. It uses a regex-based system with a dictionary of 60+ scientific
variables in Portuguese and English, recognising eight mathematical relationship
types. **Layer 1 — Groq API** accesses Llama 3.3 70B [@meta2024llama3] and
Mixtral 8×7B [@mistral2023] via a free API without a credit card. **Layer 2 —
Ollama** runs open-source models locally on the researcher's own machine without
sending data to external servers, recommended for sensitive research data.

## Open data sources

| Source | Type | Authentication | Coverage |
|--------|------|----------------|----------|
| World Bank API | Global indicators | None | 200+ countries, 1960–present |
| NASA POWER | Climate data | None | Any coordinate, 1981–present |
| IBGE SIDRA | Brazilian data | None | Brazil, historical series |
| CSV / Excel | User data | — | Any domain |

# Validation

## Table 1: Cases with known a priori answers

| Hypothesis | Type | $S_{\text{sim}}$ | CV (%) | Expected |
|---|---|---|---|---|
| Newton's Law of Universal Gravitation | Physical law | 100.0 | 0.000 | $\geq 85$ |
| Constant function ($y = 5$) | Constant | 100.0 | 0.000 | $\geq 95$ |
| Pure Meritocracy + high structural noise | Social construction | 0.8 | 48.8 | $\leq 10$ |
| Hypothesis with noise $\sigma = 1000$ | No structure | 0.0 | 2129 | $\leq 10$ |

*$n = 100$ trials $\times$ 500 samples. Seed = 42.*

## Table 2: Validation against published article datasets

| Dataset | Source | Best hypothesis | $S_{\text{sim}}$ | $R^2$ | $S_{\text{comp}}$ | Expected | Status |
|---|---|---|---|---|---|---|---|
| CO₂ → Global temperature | IPCC AR6 / NASA GISS | Linear | 69.0 | 0.711 | **49.1** | $\geq 40$ | ✅ |
| Minimum wage → Employment | Card & Krueger [-@card1994] | Positive effect | 99.6 | 0.003 | **5.0** | $\leq 20$ | ✅ |
| Original effect → Replication | OSC [-@osc2015] | 50% attenuation | 72.2 | 0.285 | **20.5** | 10–30 | ✅ |

*Datasets reconstructed from published statistics. Using `--live` flag fetches
original data via API/OSF.*

The CO₂–temperature case (score = 49.1) confirms the well-documented robustness
of the relationship [@ipcc2021]. The Card & Krueger case (score = 5.0) reflects
that the minimum wage effect on employment is small relative to natural variance,
explaining decades of controversy [@card1994; @neumark2000]. The OSC 2015 case
(score = 20.5) is consistent with the finding that replication effects are
approximately half the original magnitude [@osc2015].

A key methodological finding emerged from Phase 3: the separation between
$S_{\text{sim}}$ and $S_{\text{comp}}$ distinguishes *internal consistency* from
*empirical validity* — two dimensions that the reproducibility literature treats
separately [@goodman2016].

# Comparison with Related Tools

| Feature | **Reproducibility** | repro (R) | ReproducibiliTeachR | OSF |
|---|:---:|:---:|:---:|:---:|
| Natural language hypotheses | ✅ | ❌ | ❌ | ❌ |
| Quantitative reproducibility score | ✅ | ❌ | ❌ | ❌ |
| Hypothesis comparison (AIC) | ✅ | ❌ | ❌ | ❌ |
| Temporal drift analysis | ✅ | ❌ | ❌ | ❌ |
| Integrated open API data | ✅ | ❌ | ❌ | ✅ |
| Web interface | ✅ | ❌ | ✅ | ✅ |
| No proprietary software | ✅ | ✅ | ✅ | ✅ |
| No cost | ✅ | ✅ | ✅ | ✅ |

The fundamental difference from existing tools is one of scope: `repro` and
similar tools verify whether an *already performed* analysis can be reproduced
with the same data. *Reproducibility* allows evaluating whether a *not-yet-tested*
hypothesis has a structure compatible with reproducible results.

# Usage Example

```python
from reproducibility_core import OpenTranslator, DataFetcher, run_pipeline

# 1. Hypothesis in natural language
translator = OpenTranslator()
hypothesis = translator.translate(
    "Life expectancy increases with GDP per capita"
)

# 2. Real data from World Bank (no API key required)
df = DataFetcher().fetch_worldbank_multiple_countries(
    indicators=["life_expectancy", "gdp_per_capita"],
    countries=["BR", "US", "DE", "CN", "IN", "ZA"],
)

# 3. Full pipeline
results = run_pipeline(
    hypotheses    = [hypothesis],
    df            = df,
    target_column = "life_expectancy",
    mappings      = [{"ev": "life_expectancy"}],
)
print(f"R² = {results['empirical'][0].r_squared:.3f}")
print(f"Score = {results['simulation'][0].reproducibility_score:.1f}/100")
```

# Automated Tests

The test suite in `tests/test_core.py` contains 40 tests organised in eight
classes covering all modules. The scientific score validation tests verify not
only computational correctness but the scientific validity of the instrument —
whether it correctly discriminates between the cases with known answers in Table 1.

Run with: `pytest tests/ -v`

# Limitations

The **RuleParser** (Layer 0) operates in a closed domain: variables absent from
the internal dictionary are mapped to generic symbols. For specialised domains,
use of the Groq or Ollama layers is recommended. The **composite score** formula
was validated on seven cases but has not been calibrated against a large corpus
of published hypotheses; qualitative thresholds are provisional. Hypotheses with
variable interactions (moderation, mediation) require the user to supply the
function manually.

# Future Work

- Calibration of the composite score against a corpus of hypotheses with known
  reproducibility (e.g., OSF studies with documented replication rates)
- Integration with open repositories (Zenodo, Figshare, OSF) for automatic
  dataset retrieval by keyword
- Command-line interface (CLI) for use in scientific CI/CD pipelines
- Expanded variable dictionary for specialised domains

# Dependencies

| Package | Min. version | Role |
|---|---|---|
| numpy | 1.24.0 | Vectorised numerical operations |
| scipy | 1.11.0 | Statistics (Shapiro-Wilk, t-CI, regression) |
| pandas | 2.0.0 | Tabular data manipulation |
| matplotlib | 3.7.0 | Dashboard generation (Agg backend) |
| requests | 2.31.0 | Open APIs (World Bank, NASA POWER, IBGE) |
| streamlit | 1.32.0 | Web interface |
| openpyxl | 3.1.0 | Excel file support |

# Acknowledgements

The author thanks the open science initiatives that made this work possible at
no cost: World Bank Open Data, NASA POWER API, IBGE Open Data, Open Science
Framework (Center for Open Science), Groq (free API), Meta AI (Llama 3), and
Mistral AI (Mixtral, Apache 2.0).

# References
