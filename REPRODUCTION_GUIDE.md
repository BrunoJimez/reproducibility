# REPRODUCTION GUIDE

**Reproducibility — Open-source tool for computational hypothesis testing**
Zenodo DOI: [10.5281/zenodo.19042510](https://doi.org/10.5281/zenodo.19042510)

This guide enables complete, independent reproduction of all results reported
in the paper. It is structured in five phases that correspond to the validation
sections of the paper.

---

## Prerequisites

| Requirement | Version tested | Notes |
|---|---|---|
| Python | 3.10–3.13 | 3.13.x recommended |
| pip | any recent | bundled with Python |
| git | any | to clone the repository |
| Internet | optional | required only for `--live` flag and World Bank API calls |

> **Internet notice:** Steps that use `--live` or call `fetch_worldbank*()`,
> `fetch_nasa_climate()`, or `fetch_ibge_gdp()` require a stable internet
> connection. The embedded datasets in `phase3_validation.py` are the only
> guaranteed offline, deterministic data source for reproducing Table 2
> (Cases 1-3) of the paper.

---

## Phase 1 — Installation

```cmd
:: 1. Clone the repository
git clone https://github.com/BrunoJimez/reproducibility.git
cd reproducibility

:: 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          :: Windows
source .venv/bin/activate       :: Linux / macOS

:: 3. Install all dependencies (including pytest)
pip install -r requirements.txt
pip install pytest
```

**Expected output of `pip install`:**
```
Successfully installed matplotlib numpy pandas scipy streamlit requests openpyxl pytest
```

---

## Phase 2 — Automated test suite (52 tests)

This phase verifies that all eight modules are computationally correct and
that the scoring instrument is scientifically valid.

```cmd
pytest tests\ -v
```

**Expected result:**
```
52 passed, 16 warnings in ~8s
```

The 16 warnings are cosmetic and do not affect results:
- 3 warnings: emoji glyphs missing from DejaVu Sans font (Windows only)
- 4 warnings: tight_layout incompatibility (matplotlib cosmetic)
- 9 warnings: scipy RuntimeWarning when CV = 0 (mathematically expected for
  constant and deterministic hypotheses)

The `TestScoreValidation` class (tests 44-49) verifies the scientific
validity of the instrument. These six tests confirm:
- Physical laws (Newton) produce score >= 85
- Constant function (y = 5) produces score = 100
- Mincer equation with high noise produces score <= 10
- Pure noise (sigma = 1000) produces score ~= 0
- Score strictly decreases as noise increases (monotonicity)

If any of these tests fail, do not proceed — the instrument is not calibrated.

---

## Phase 3 — Validation against published datasets (Table 2, Cases 1-3)

This phase reproduces the three original validation cases using embedded
datasets reconstructed from published statistics.

```cmd
python phase3_validation.py
```

**Expected output (SUMMARY section):**
```
CO2 -> Global temperature       composite= 49.1  expected=>= 40   COMPLIANT
Minimum wage -> Employment (CK  composite=  0.3  expected=<= 20   COMPLIANT
Original effect -> Replication  composite= 20.8  expected=10-30   COMPLIANT
```

**Expected output files:**
- `phase3_dashboard.png` — visual dashboard with 3 cases
- `phase3_results.json` — full numerical results with 15-digit precision

**Key numerical values to verify** (exact, seed = 42):

| Case | Hypothesis | S_sim | R2 | Composite |
|---|---|---|---|---|
| CO2 -> Temperature | Linear (winner) | 69.0 | 0.711 | 49.1 |
| CO2 -> Temperature | Logarithmic | 78.0 | -3.918 | 0.0 |
| CK 1994 | Positive effect (winner) | 99.6 | 0.003 | 0.3 |
| CK 1994 | No effect | 100.0 | ~0.000 | 0.0 |
| OSC 2015 | 50% attenuation (winner) | 73.1 | 0.285 | 20.8 |
| OSC 2015 | Faithful replication | 73.1 | -1.524 | 0.0 |

> **Note on Card & Krueger composite = 0.3:** this is the correct value from
> the formula S_comp = S_sim x max(0, R2) = 99.6 x 0.003 = 0.3. It reflects
> that the state indicator (NJ vs PA) explains less than 0.3% of the variance
> in restaurant-level employment. The case is COMPLIANT (0.3 <= 20).

> **Note on negative R2:** the Logarithmic climate hypothesis and several
> competing hypotheses in Cases 2 and 3 yield R2 < 0, meaning they perform
> worse than simply predicting the unconditional mean. This is mathematically
> correct. S_comp = 0 in all these cases by the max(0, R2) term.

**To fetch live data instead of embedded datasets:**
```cmd
python phase3_validation.py --live
```
Requires internet. Falls back to embedded data automatically if any network
call fails.

---

## Phase 4 — Web interface (Streamlit)

```cmd
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`.

### Recommended test sequence

**Step 1 — Translate a hypothesis:**
In the **Hypotheses** tab, type:
```
Income grows linearly with years of education
```
Click **Translate hypothesis**.

Verify:
- Name: `Hypothesis Linear(income, education)` (English, not Portuguese)
- Origin: `rule_parser` (not `parser_regras`)
- Prediction: `Linear relationship between income, education`

**Step 2 — Inspect variables before mapping:**
Expand the hypothesis card and read the **Variables** section:
```
Y in [0.0, 1000000.0] — income ($)
E in [0.0, 30.0]      — education (anos)
```

> **Critical:** always check which symbol (Y, E, etc.) corresponds to which
> concept before defining the mapping in Step 4. Mapping the wrong variable
> to the wrong column produces nonsensical R2 values without any error
> message. A yellow warning appears if |R2| > 50 — use this as a signal to
> review your mapping.

**Step 3 — Generate synthetic data:**
In the **Data** tab, select **Synthetic dataset**, set Observations = 300,
Seed = 42, click **Generate dataset**.

Verify: `288 observations generated` (the generator produces full years
from 2000 to 2023 at default density, giving 288 rows, not 300).

**Step 4 — Run the full pipeline:**
In the **Run & Results** tab:
- Target column: `renda`
- Time column: `ano`
- Mapping: `Y (income $)` to `renda` ; `E (education anos)` to `estudo`
- Click **Run full analysis**

Verify in the results:
- **Full ranking** table contains a `Composite score` column
- **Details per hypothesis** shows 5 metrics including `Composite score`
- **Temporal analysis** shows 5 time windows (W1-W5, 2000-2023)
- No crash in the Details per hypothesis expander

**Step 5 — Verify the mapping warning (optional):**
Add a second hypothesis:
```
CO2 emissions increase exponentially with industrial output
```
Keep the synthetic dataset. In the mapping, assign `co2` to `ano`
(intentionally wrong).

Click **Run full analysis**.

Verify: a yellow warning appears before the results:
```
Mapping check: one or more hypotheses have extreme R2 values...
```

---

## Phase 5 — Independent validation (Cases 4-5, requires internet)

These cases reproduce the two additional validation cases reported in
Table 2 of the paper, using World Bank Open Data.

```cmd
python teste_hugo.py    :: Case 4 — Preston Curve (life expectancy vs GDP)
python teste_novo.py    :: Case 5 — CO2 emissions vs GDP per capita
```

**Expected results — Case 4 (Preston Curve, 10 countries):**

| Hypothesis | S_sim | R2 | Composite | AIC weight |
|---|---|---|---|---|
| Preston Logarithmic (winner) | 96.4 | 0.584 | 56.3 | 85.0% |
| Preston Linear | 95.7 | 0.403 | 38.6 | 13.9% |
| Preston Null | 100.0 | 0.000 | 0.0 | 1.1% |

The logarithmic form wins, consistent with Preston (1975) and the
diminishing marginal returns to health outcomes from income.

**Expected results — Case 5 (CO2 vs GDP, 15 countries):**

| Hypothesis | S_sim | R2 | Composite | AIC weight |
|---|---|---|---|---|
| CO2 Linear (winner) | 78.2 | 0.617 | 48.3 | 72.1% |
| CO2 Logarithmic | 84.1 | 0.566 | 47.6 | 27.8% |
| CO2 Null | 100.0 | 0.000 | 0.0 | 0.1% |

The near-tie (Delta AIC = 1.9 < 2) correctly reflects the functional
ambiguity in the Kuznets Environmental Curve literature.

> **If the World Bank API is unavailable:** `teste_novo.py` automatically
> falls back to a hardcoded 15-country reference dataset and prints:
> `"using hardcoded reference data"`. Results are slightly different from
> live data but reproduce the same qualitative conclusions.
> `teste_hugo.py` requires internet — retry after a few minutes if it fails.

---

## Troubleshooting

### KeyError during pipeline (e.g. KeyError: 'gdp')

**Cause:** a hypothesis variable symbol is not present in the mapping.
The RuleParser detected multiple variables but the mapping only covered some.

**Fix:** inspect the variables before constructing the mapping:
```python
print([(v.name, v.description) for v in hypothesis.variables])
# e.g. [('Y', 'income ($)'), ('E', 'education (anos)')]
# Map ALL symbols: {"Y": "renda", "E": "estudo"}
```

### R2 is extremely negative (e.g. -3541 or -5.7e15)

**Cause:** a variable with one magnitude range (e.g. CO2 in ppm: 200-800)
was mapped to a column with a very different magnitude (e.g. income: 500-19000).
The hypothesis output is orders of magnitude larger than the target.

**Fix:** check that each variable symbol maps to a column in the same
physical domain. The Streamlit UI displays a yellow warning when |R2| > 50.

### Countries loaded: 0 (teste_novo.py)

**Cause:** the World Bank API was unavailable or returned fewer than 5 countries.

**Fix:** `teste_novo.py` handles this automatically with a fallback dataset.
The output will show `"using hardcoded reference data"` clearly.

### 52 passed becomes N failed after modifying any file

**Cause:** a change broke compatibility between `reproducibility_core.py`
and `test_core.py`. All files must come from the same version.

**Fix:** restore all files from the same commit:
```cmd
git checkout -- .
```

### Streamlit shows use_container_width warnings in the terminal

**Cause:** older version of `app.py` is running. The corrected version
uses `width="stretch"` instead of `use_container_width=True`.

**Fix:** replace `app.py` with the corrected version from the repository.

---

## File version reference

All results in the paper were produced with these file versions:

| File | Key markers to verify |
|---|---|
| `reproducibility_core.py` | `origin="rule_parser"`, field `residuals_p_value`, composite computed in `run_pipeline()` |
| `app.py` | `width="stretch"`, R2 mapping warning, `Composite score` in Full ranking |
| `phase3_validation.py` | composite fallback = 0.0, `mappings` key in return dict |
| `tests/test_core.py` | 52 tests, `TestScoreValidation` class, `"rule_parser"` in assertions |
| `paper.md` | 5 cases in Table 2, Mincer reference, CK composite = 0.3 |

To verify you have the correct version of `reproducibility_core.py`:
```python
from reproducibility_core import OpenTranslator
h = OpenTranslator().translate("income grows with education")
print(h.origin)                  # must print: rule_parser
print("Hypothesis" in h.name)    # must print: True
```

---

## Complete reproduction checklist

```
Phase 1 — Installation
[ ] git clone succeeds
[ ] pip install completes without errors
[ ] python -c "import reproducibility_core" runs without errors

Phase 2 — Tests
[ ] pytest tests\ -v  ->  52 passed, 16 warnings
[ ] Zero FAILED tests

Phase 3 — Published dataset validation
[ ] python phase3_validation.py  ->  3x COMPLIANT
[ ] CO2 composite = 49.1
[ ] CK composite  =  0.3
[ ] OSC composite = 20.8
[ ] phase3_dashboard.png generated
[ ] phase3_results.json generated

Phase 4 — Web interface
[ ] streamlit run app.py  ->  opens at localhost:8501, no use_container_width warnings
[ ] Hypothesis translated to English (Hypothesis Linear..., rule_parser)
[ ] Synthetic dataset generates 288 observations
[ ] Composite score appears in Full ranking table
[ ] Composite score appears in Details per hypothesis (5th metric)
[ ] Temporal analysis shows 5 windows (W1-W5)
[ ] Mapping warning appears for intentionally wrong mapping (|R2| > 50)

Phase 5 — Independent validation (requires internet)
[ ] python teste_hugo.py  ->  Preston Log composite ~= 56.3
[ ] python teste_novo.py  ->  CO2 Linear composite ~= 48.3
```

---

## Citation

```bibtex
@article{jimez2026reproducibility,
  author  = {Jimez, Bruno Oliveira Costa},
  title   = {Reproducibility: an open-source tool for computational
             hypothesis testing in natural language},
  journal = {Journal of Open Source Software},
  year    = {2026},
  doi     = {10.5281/zenodo.19042510}
}
```

*Questions or issues: https://github.com/BrunoJimez/reproducibility/issues*
