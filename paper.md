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
  - name: Bruno Oliveira Costa Jimez
    orcid: 0009-0002-8745-3418
    affiliation: 1
affiliations:
  - name: Independent Researcher, Dourados, MS, Brazil
    index: 1
date: 2026
archive_doi: https://doi.org/10.5281/zenodo.19042510
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
and empirical fit — and demonstrate its validity against five cases with
expected behaviour established by the literature: the CO₂–global temperature
relationship (composite score = 49.1; expected ≥ 40), the minimum wage effect
on employment from Card & Krueger (composite score = 0.3; expected ≤ 20), the
replication rate of psychological studies from the Open Science Collaboration
(composite score = 20.8; expected 10–30), the Preston Curve relating GDP per
capita to life expectancy (composite score = 56.3; expected ≥ 40), and the
Kuznets Environmental Curve relating GDP per capita to CO₂ emissions per capita
(composite score = 48.3; expected ≥ 40). The tool requires no proprietary
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
results with new independent data [@goodman2016]. Crucially, both concepts
operate *post hoc*: they require data to already exist. Our tool operates in a
third, *a priori* domain: given that a researcher formulates a hypothesis, it
provides a quantitative estimate of whether the mathematical structure of that
hypothesis is compatible with stable, replicable results — *before any data
collection occurs*. This distinction is the primary differentiator from all
existing tools.

Existing tools for reproducibility support focus primarily on *post hoc*
verification: the `repro` R package [@repro2021] allows previously performed
analyses to be reproduced with the same data and code, and platforms such as
OSF [@osf2013] facilitate data and code sharing after publication.
`ReproducibiliTeachR` is an R package designed for teaching reproducibility
concepts rather than evaluating research hypotheses.
We have not identified in the literature a tool that allows a researcher to write
a hypothesis in natural language and receive, within seconds, a quantitative
estimate of its expected reproducibility against open data before the study is
conducted. This gap motivates the present contribution.

# Theoretical Background

## Simulated reproducibility score

The *simulated reproducibility score* $S_{\text{sim}}$ is defined as:

$$S_{\text{sim}} = 100 \cdot \exp\!\left(-\frac{CV}{10}\right)$$

where $CV$ is the Coefficient of Variation (standard deviation / mean × 100%)
of the outputs from $n$ independent Monte Carlo simulation trials. Each trial
samples values of the hypothesis variables from their plausible ranges and
computes the output of the corresponding mathematical function.

To illustrate the extremes: a deterministic physical law such as Newton's
gravitational law has CV ≈ 0% and $S_{\text{sim}} = 100$, while a hypothesis
dominated by structural noise (e.g., a social outcome driven primarily by
factors outside the model) may have CV > 40% and $S_{\text{sim}} < 1$.

The function has the following desirable mathematical properties:

1. $S_{\text{sim}} \in [0, 100]$: normalised score, comparable across hypotheses
2. $S_{\text{sim}}(CV=0) = 100$: zero variation implies perfect reproducibility
3. $S_{\text{sim}}(CV \to \infty) \to 0$: unbounded variation implies non-reproducibility
4. Strict monotonicity: increasing $CV$ implies decreasing $S_{\text{sim}}$

## Composite score

A fundamental limitation of $S_{\text{sim}}$ in isolation is that hypotheses
with simple deterministic functions — including trivially vacuous ones such as
the constant function $y = 5$ — obtain $S_{\text{sim}} = 100$ regardless of
their predictive power over real data. To also capture empirical fit, we define
the *composite score* as the primary reporting metric:

$$S_{\text{comp}} = S_{\text{sim}} \cdot \max(0,\ R^2)$$

where $R^2$ is the coefficient of determination obtained by testing the hypothesis
against an empirical dataset. This formula penalises hypotheses that, while
mathematically consistent, do not explain the observed variation in data. When
$R^2 \leq 0$ — meaning the hypothesis performs worse than simply predicting
the mean — the composite score is zero. The composite score is automatically
computed by `run_pipeline()` and is the recommended metric for all reporting.
$S_{\text{comp}}$ is a scale on $[0, 100]$ and is **not** a probability;
it should not be interpreted as or compared directly with the Akaike weights
$w_i$, which are probabilities summing to one across the competing hypotheses.

## Hypothesis comparison via Akaike criterion

To rank competing hypotheses explaining the same phenomenon, we use the Akaike
information criterion [@burnham2002]:

$$AIC = 2k - 2\ln(\hat{L})$$

AIC was selected over BIC and adjusted-$R^2$ because the hypotheses tested are
typically non-nested — none is a special case of another — a condition for which
AIC was specifically designed [@burnham2002]. The implementation omits additive
constants from the Gaussian log-likelihood ($-\frac{n}{2}\ln(2\pi)$ and
$-\frac{n}{2}$) that cancel in all $\Delta_i$ computations and therefore do
not affect Akaike weights or rankings. Absolute AIC values are consequently
not comparable across datasets of different size $n$; only $\Delta_i$ and
$w_i$ should be interpreted. The normalised Akaike weight $w_i$
represents the relative probability that each hypothesis is the best
approximation to reality given the tested model set:

$$w_i = \frac{\exp(-\Delta_i / 2)}{\sum_j \exp(-\Delta_j / 2)}$$

When $\Delta_i < 2$, competing hypotheses are considered statistically
indistinguishable and both are reported as plausible. This behaviour was observed
empirically in Case 5 of the validation (Section Validation), where linear and
logarithmic hypotheses for CO₂–GDP yielded $\Delta \text{AIC} = 1.9$,
correctly reflecting the genuine functional ambiguity documented in that
literature.

# Software Architecture

The system is organised into two main files. **`reproducibility_core.py`** is a
pure logic module with no UI dependencies, importable in any Python environment
as a standard library — enabling integration into automated pipelines and
research scripts independently of the web interface. It implements eight
subcomponents: data structures (`Variable`, `Hypothesis`, result types);
`OpenTranslator` for three-layer NLP translation; `ReproducibilityEngine` for
Monte Carlo simulation; `EmpiricalTester` for fit to real data with R², RMSE,
AIC, and BIC; `HypothesisComparator` for Akaike criterion ranking;
`TemporalAnalyser` for drift detection; `DataFetcher` for unauthenticated open
API access; and `generate_dashboard` for matplotlib visualisation. **`app.py`**
is a Streamlit web interface with three tabs: hypothesis definition, data
selection, and execution with result visualisation.

## Module 1: Hypothesis translation (Open Science)

The `OpenTranslator` operates in three layers with automatic fallback.
**Layer 0 — RuleParser** is always available, requires zero dependencies, and
works offline. It uses a regex-based system with a dictionary of 60+ scientific
variables in Portuguese (PT-BR) and English, recognising eight mathematical
relationship types (linear, inverse, power, exponential, logarithmic, constant,
quadratic, cubic). The bilingual coverage makes the tool directly accessible to
researchers in Brazil and other Lusophone countries without requiring an LLM.
**Layer 1 — Groq API** accesses Llama 3.3 70B [@meta2024llama3] and
Mixtral 8×7B [@mistral2023] via a free API without a credit card. Both models
are open-source (Meta Llama licence and Apache 2.0, respectively), consistent
with the Open Science principles of the tool. **Layer 2 — Ollama** runs
open-source models locally on the researcher's own machine without sending data
to external servers, recommended for sensitive research data.

## Open data sources

| Source | Type | Authentication | Coverage |
|--------|------|----------------|----------|
| World Bank API | Global indicators | None | 200+ countries, 1960–present |
| NASA POWER | Climate data | None | Any coordinate, 1981–present |
| IBGE SIDRA | Brazilian data | None | Brazil, historical series |
| CSV / Excel | User data | — | Any domain |

# Validation

The validation is structured in two phases. **Phase 1** uses cases with answers
known *a priori* from mathematics or physical theory, verifying that the
instrument is correctly calibrated at the extremes of the scale (score = 100
for deterministic laws; score ≈ 0 for pure noise). **Phase 2** uses published
empirical datasets with expected score ranges established by the literature,
verifying that the instrument discriminates correctly across the full spectrum
from controversial to robust relationships.

## Table 1: Cases with known a priori answers

| Hypothesis | Type | $S_{\text{sim}}$ | CV (%) | Expected |
|---|---|---|---|---|
| Newton's Law of Universal Gravitation | Physical law | 100.0 | 0.000 | $\geq 85$ |
| Constant function ($y = 5$) | Constant | 100.0 | 0.000 | $\geq 95$ |
| Mincer earnings equation with high residual variance [@mincer1974] | High-noise social | 0.8 | 48.8 | $\leq 10$ |
| Hypothesis with noise $\sigma = 1000$ | No structure | 0.0 | 2129 | $\leq 10$ |

*Table 1 parameters: $n = 100$ trials $\times$ 500 samples, seed = 42. These higher-precision parameters are used in the `TestScoreValidation` test class to serve as calibration anchors. Table 2 results were produced with $n = 80$ trials $\times$ 300 samples (matching `phase3_validation.py` defaults). The `run_pipeline()` function uses the same defaults (80 trials, 300 samples); scores obtained interactively are therefore directly comparable to Table 2 but will differ slightly from Table 1.*

The third case implements a simplified Mincer earnings equation
[@mincer1974; @becker1964], modelling income as a linear function of education
and experience with a large stochastic residual ($\sigma = 3000$ monetary units)
representing structural factors outside the model (discrimination, inheritance,
macroeconomic shocks). The dominant noise term produces a high CV and a
near-zero score, confirming that the instrument correctly penalises hypotheses
whose explanatory variables account for only a small fraction of outcome variance.

## Table 2: Validation against published article datasets

| Dataset | Source | Best hypothesis | $S_{\text{sim}}$ | $R^2$ | $S_{\text{comp}}$ | Expected | Status |
|---|---|---|---|---|---|---|---|
| CO₂ → Global temperature | IPCC AR6 / NASA GISS | Linear | 69.0 | 0.711 | **49.1** | $\geq 40$ | ✅ |
| Minimum wage → Employment | Card & Krueger [-@card1994] | Positive effect | 99.6 | 0.003 | **0.3** | $\leq 20$ | ✅ |
| Original effect → Replication | OSC [-@osc2015] | 50% attenuation | 73.1 | 0.285 | **20.8** | 10–30 | ✅ |
| GDP per capita → Life expectancy | World Bank / Preston [-@preston1975] | Logarithmic | 96.4 | 0.584 | **56.3** | $\geq 40$ | ✅ |
| GDP per capita → CO₂ emissions | World Bank / Grossman & Krueger [-@grossman1991] | Linear | 78.2 | 0.617 | **48.3** | $\geq 40$ | ✅ |

*Cases 1–3: datasets reconstructed from published statistics. Cases 4–5: World
Bank Open Data averages (2010–2022 and 2015–2020 respectively). Using the
`--live` flag with Cases 1–3 fetches original data via API/OSF.*

The CO₂–temperature case (score = 49.1) confirms the well-documented robustness
of the relationship [@ipcc2021]. The Card & Krueger case (score = 0.3) reflects
that the minimum wage effect on employment is essentially negligible relative to
the natural variance in restaurant-level employment data — the best hypothesis
(positive effect) explains only 0.3% of the outcome variance ($R^2 = 0.003$),
consistent with decades of controversy about whether the effect is real
[@card1994; @neumark2000]. The near-zero composite score correctly captures the
empirical ambiguity: the treatment (state) is a weak predictor when noise
dominates the outcome. The OSC 2015 case
(score = 20.8) is consistent with the finding that replication effects are
approximately half the original magnitude [@osc2015].

The Preston Curve case (score = 56.3) confirms the well-established relationship
between national income and population health [@preston1975], with the
logarithmic form correctly identified as the best functional form — consistent
with the diminishing marginal returns to health outcomes from income at higher
development levels. The CO₂–GDP case (score = 48.3) is consistent with the
empirical literature on the Kuznets Environmental Curve [@grossman1991]; the
near-equivalent fit of linear and logarithmic hypotheses ($\Delta \text{AIC} = 1.9$)
correctly reflects the genuine functional ambiguity documented in that literature
for the income range covered by the dataset.

A key methodological finding is that the separation between $S_{\text{sim}}$
and $S_{\text{comp}}$ distinguishes *internal consistency* from *empirical
validity* — two dimensions that the reproducibility literature treats separately
[@goodman2016]. A hypothesis can be highly reproducible in simulation yet fail
empirically, as illustrated by the constant-function baseline in each case.

**Note on negative $R^2$:** in Cases 1 and 3, some competing hypotheses yield
$R^2 < 0$, meaning the model performs worse than simply predicting the
unconditional mean. In Case 1, the Arrhenius logarithmic parametrisation,
while physically motivated, is not calibrated to the 1900–2022 range where the
linear approximation holds. In all such cases $S_{\text{comp}} = 0$ by the
$\max(0, R^2)$ term.

# Comparison with Related Tools

| Feature | **Reproducibility** | repro (R) | ReproducibiliTeachR | OSF |
|---|:---:|:---:|:---:|:---:|
| Natural language hypotheses | ✅ | ❌ | ❌ | ❌ |
| Quantitative reproducibility score | ✅ | ❌ | ❌ | ❌ |
| Hypothesis comparison (AIC) | ✅ | ❌ | ❌ | ❌ |
| Temporal drift analysis | ✅ | ❌ | ❌ | ❌ |
| Integrated open API data | ✅ | ❌ | ❌ | ✅ |
| Reproduction of published analyses | ❌ | ✅ | ❌ | ✅ |
| Web interface | ✅ | ❌ | ✅ | ✅ |
| No proprietary software | ✅ | ✅ | ✅ | ✅ |
| No cost | ✅ | ✅ | ✅ | ✅ |

The `repro` package and OSF excel at verifying whether an *already performed*
analysis can be reproduced with the same data — a post hoc use case for which
this tool is not designed. The two approaches are complementary: `repro` and
OSF verify computational reproducibility after a study is complete; this tool
estimates replicability potential before the study begins. Note that only this
tool operates a priori and provides a quantitative score.

# Usage Example

```python
from reproducibility_core import OpenTranslator, DataFetcher, run_pipeline

# 1. Hypothesis in natural language (English or Portuguese)
translator = OpenTranslator()
hypothesis = translator.translate(
    "Life expectancy increases with GDP per capita"
)

# Always inspect variable symbols before constructing the mapping.
# The RuleParser may detect multiple variables; all must be mapped.
for v in hypothesis.variables:
    print(f"  symbol '{v.name}' -> {v.description}")
# Example output:
#   symbol 'ev' -> life expectancy (anos)
#   symbol 'gdp' -> gdp (USD)

# 2. Real data from World Bank (no API key required)
df = DataFetcher().fetch_worldbank_multiple_countries(
    indicators = ["life_expectancy", "gdp_per_capita"],
    countries  = ["BR", "US", "DE", "CN", "IN", "ZA"],
)

# 3. Full pipeline
# mappings: maps every hypothesis symbol to a DataFrame column.
# Both symbols detected above must be mapped; omitting any symbol
# will raise a KeyError when the hypothesis function executes.
results = run_pipeline(
    hypotheses    = [hypothesis],
    df            = df,
    target_column = "life_expectancy",
    mappings      = [{"ev": "life_expectancy", "gdp": "gdp_per_capita"}],
)
print(f"R²              = {results['empirical'][0].r_squared:.3f}")
print(f"Sim. score      = {results['simulation'][0].reproducibility_score:.1f}/100")
print(f"Composite score = {results['empirical'][0].composite_score:.1f}/100")
```

The `mappings` parameter bridges hypothesis variable symbols (assigned
automatically by the translator) to the actual column names in the DataFrame.
Always inspect `hypothesis.variables` before defining the mapping to confirm
which symbols were detected — the RuleParser may identify multiple independent
variables, all of which must appear as keys in the mapping dictionary.
Mapping only a subset of detected variables will cause a `KeyError` at runtime.

# Automated Tests

The test suite in `tests/test_core.py` contains 52 tests organised in eight
classes covering all modules. The `TestScoreValidation` class is the scientific
core of the suite: its six tests verify not only computational correctness but
the scientific validity of the instrument — whether it correctly discriminates
between the cases with known answers in Table 1 (physical laws → score ≥ 85;
high-noise social hypotheses → score ≤ 10; strictly monotonic decrease with
added noise).

Run with: `pytest tests/ -v`

# Limitations

The **RuleParser** (Layer 0) operates in a closed domain: variables absent from
the internal dictionary are mapped to generic symbols. For specialised domains
(biochemistry, astrophysics, clinical pharmacology), use of the Groq or Ollama
layers is recommended. Additionally, when the RuleParser detects multiple
variables in a hypothesis, all are treated as additive independent predictors;
the user must verify the mapping between hypothesis variable symbols and dataset
columns before running the pipeline (inspecting `hypothesis.variables` is
recommended to avoid mapping errors).

The system internally computes a secondary diagnostic metric, `empirical_score`,
combining $R^2$ (weight 0.70) and a residual-normality indicator (weight 0.30).
These weights are heuristic and not derived from a formal criterion; the metric
is exposed in the ranking table for diagnostic purposes only. The **composite
score** ($S_{\text{comp}}$) is the theoretically grounded primary metric and
should be used for all comparisons and reporting.

The **composite score** formula was validated on the ten cases reported in
Tables 1 and 2, but has not been calibrated against a large corpus of published
hypotheses; qualitative thresholds (≥ 40: robust relationship; 10–30: moderate;
≤ 10: weak or controversial) are provisional and may require domain-specific
adjustment. Furthermore, $S_{\text{sim}}$ measures *variability* but not
*systematic bias*: a hypothesis may be highly internally consistent yet
systematically wrong relative to empirical data; $S_{\text{comp}}$ partially
addresses this through $R^2$, but sensitivity to dataset quality and sample
size is not quantified. No confidence interval is currently reported for either
score; differences smaller than approximately 5 points should be interpreted
with caution given Monte Carlo sampling variance. The Shapiro-Wilk
normality test is applied to a maximum of 50 observations (trials or
residuals); with larger samples, the test is known to detect trivially
small deviations that have no practical significance, so this truncation
is intentional [@burnham2002]. Additionally, $S_{\text{sim}}$
assumes a **uniform prior** over each variable's domain: all values in
$[v_{\min}, v_{\max}]$ are sampled with equal probability. This simplification
may over- or under-estimate the true CV when the variable's real-world
distribution is non-uniform (e.g., concentrated near a mode). Users should
set domain bounds to reflect the range of scientific interest rather than
theoretical extremes.

Hypotheses with variable interactions (moderation, mediation) require the user
to supply the functional form manually, as the current NLP layers do not parse
interaction terms. This limits direct applicability to multivariate causal
models common in psychology, medicine, and epidemiology.

When a fixed seed is used, the random number generator is reset to the same
state at the start of each hypothesis evaluation. Consequently, the Monte Carlo
trials of different hypotheses tested in the same session are **correlated**:
they share the same random sequence. This ensures deterministic, reproducible
scores for each hypothesis individually, but means that score differences
between hypotheses do not arise from statistically independent samples.
For applications requiring inter-hypothesis independence (e.g., hypothesis
ranking with formal uncertainty quantification), users should omit the seed
or derive distinct seeds per hypothesis.

# Future Work

- Calibration of the composite score against a corpus of hypotheses with known
  reproducibility (e.g., OSF studies with documented replication rates)
- Bootstrap confidence intervals for $S_{\text{sim}}$ and $S_{\text{comp}}$
- Temporal windows defined by equal time intervals rather than equal row counts,
  to handle datasets with irregular temporal sampling
- `suggest_mapping()` helper function to automatically infer variable-to-column
  mappings from name similarity, reducing a common source of user error
- Integration with open repositories (Zenodo, Figshare, OSF) for automatic
  dataset retrieval by keyword
- Command-line interface (CLI) for use in scientific CI/CD pipelines
- Expanded variable dictionary for specialised domains (biochemistry,
  astrophysics, clinical pharmacology)

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

# License

The software is released under the MIT License.

# Acknowledgements

The author thanks the open science initiatives that made this work possible at
no cost: World Bank Open Data, NASA POWER API, IBGE Open Data, Open Science
Framework (Center for Open Science), Groq (free API), Meta AI (Llama 3), and
Mistral AI (Mixtral, Apache 2.0).

# References
