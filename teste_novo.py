from reproducibility_core import (
    Variable, Hypothesis, DataFetcher, run_pipeline
)
import numpy as np
import pandas as pd

print("=" * 60)
print("TESTE COM HIPÓTESE E DADOS NOVOS")
print("Hipótese: CO₂ emissions increase with GDP per capita")
print("=" * 60)

fetcher = DataFetcher()

# ── Dataset de referência (valores reais publicados pelo World Bank)
# Fonte: World Bank Open Data, médias 2015-2020
# gdp_per_capita em USD, co2_per_capita em toneladas/pessoa
FALLBACK_DATA = {
    "country":      ["US",  "AU",  "CA",  "DE",  "JP",  "KR",
                     "FR",  "GB",  "MX",  "CN",  "BR",  "ZA",
                     "AR",  "IN",  "NG"],
    "gdp_per_capita":[62530, 54907, 46195, 45229, 40247, 31430,
                      40494, 41330, 10118,  8827,  8897,  6001,
                       9912,  2016,  2230],
    "co2_per_capita": [15.24, 15.37, 15.32,  8.93,  9.13, 11.85,
                        4.81,  5.55,  3.89,  7.38,  2.18,  8.14,
                        3.97,  1.73,  0.59],
}

# ── Tentar World Bank, usar fallback se falhar ───────────────────────
print("\nAttempting World Bank API...")
try:
    df = fetcher.fetch_worldbank_multiple_countries(
        indicators = ["co2_per_capita", "gdp_per_capita"],
        countries  = ["US", "CN", "DE", "BR", "IN", "ZA",
                      "NG", "AR", "MX", "JP", "AU", "CA",
                      "FR", "GB", "KR"],
        year_start = 2010,
        year_end   = 2020,
    )
    if len(df) < 5:
        raise ValueError(f"Too few countries returned: {len(df)}")
    print(f"World Bank API: {len(df)} countries loaded")
except Exception as e:
    print(f"World Bank unavailable ({e}) — using hardcoded reference data")
    df = pd.DataFrame(FALLBACK_DATA)
    df.attrs["source"] = "World Bank (hardcoded reference values, 2015-2020 avg)"

print(f"\nDataset: {len(df)} countries")
print(df[["country", "gdp_per_capita", "co2_per_capita"]].to_string(index=False))

# ── Calibração OLS para parâmetros realistas ────────────────────────
gdp = df["gdp_per_capita"].values.astype(float)
co2 = df["co2_per_capita"].values.astype(float)

a_lin = np.cov(gdp, co2)[0,1] / np.var(gdp)
b_lin = np.mean(co2) - a_lin * np.mean(gdp)

log_gdp = np.log(gdp)
a_log = np.cov(log_gdp, co2)[0,1] / np.var(log_gdp)
b_log = np.mean(co2) - a_log * np.mean(log_gdp)

print(f"\nOLS-calibrated parameters:")
print(f"  Linear:      co2 = {a_lin:.6f} × gdp + {b_lin:.3f}")
print(f"  Logarithmic: co2 = {a_log:.4f} × log(gdp) + {b_log:.3f}")

# ── Hipóteses ────────────────────────────────────────────────────────
h_linear = Hypothesis(
    name        = "CO2 Linear",
    description = "CO₂ emissions increase linearly with GDP per capita",
    function    = lambda v, a=a_lin, b=b_lin: a * v['g'] + b,
    variables   = [Variable("g", 1000.0, 65000.0, "GDP per capita (USD)")],
    origin      = "manual",
)

h_log = Hypothesis(
    name        = "CO2 Logarithmic",
    description = "CO₂ emissions increase logarithmically with GDP per capita",
    function    = lambda v, a=a_log, b=b_log: a * np.log(v['g']) + b,
    variables   = [Variable("g", 1000.0, 65000.0, "GDP per capita (USD)")],
    origin      = "manual",
)

h_null = Hypothesis(
    name        = "CO2 Null (constant)",
    description = "CO₂ emissions are independent of GDP per capita",
    function    = lambda v, m=np.mean(co2): np.ones_like(v['g']) * m,
    variables   = [Variable("g", 1000.0, 65000.0, "GDP per capita (USD)")],
    origin      = "manual",
)

hypotheses = [h_linear, h_log, h_null]
mappings   = [{"g": "gdp_per_capita"}] * 3

# ── Pipeline ─────────────────────────────────────────────────────────
print("\nRunning analysis...")
results = run_pipeline(
    hypotheses    = hypotheses,
    df            = df,
    target_column = "co2_per_capita",
    mappings      = mappings,
    n_trials      = 60,
    sample_size   = 200,
    seed          = 42,
)

# ── Resultados ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS — CO₂ emissions vs GDP per capita")
print("=" * 60)

for sim, emp in zip(results["simulation"], results["empirical"]):
    print(f"\n── {sim.hypothesis_name}")
    print(f"   Sim. score   : {sim.reproducibility_score:.1f}/100")
    print(f"   CV           : {sim.coefficient_of_variation:.2f}%")
    print(f"   R²           : {emp.r_squared:.4f}")
    print(f"   RMSE         : {emp.rmse:.4f} t CO₂/capita")
    print(f"   Composite    : {emp.composite_score:.1f}/100")

print(f"\n── AIC Ranking:")
for r in results["comparison"].ranking:
    flag = "🏆" if r["name"] == results["comparison"].winner["name"] else "  "
    print(f"   {flag} {r['name']:<28} "
          f"AIC={r['aic']:7.1f}  "
          f"Δ={r['delta_aic']:5.1f}  "
          f"weight={r['akaike_weight']:.1%}")

best = max(results["empirical"], key=lambda r: r.composite_score)
print(f"\n── Comparison with paper benchmarks:")
print(f"   Best composite (CO₂ vs GDP) : {best.composite_score:.1f}")
print(f"   Preston Curve (Hugo test)   : 56.3  (robust relationship)")
print(f"   CO₂ → Temperature (paper)   : 49.1  (expected ≥ 40)")
print(f"   OSC 2015 (paper)             : 20.8  (expected 10–30)")
print(f"   Card & Krueger (paper)       :  5.0  (expected ≤ 20)")

# ── Dashboard ────────────────────────────────────────────────────────
results["figure"].savefig(
    "dashboard_co2_gdp.png",
    dpi=150, bbox_inches="tight", facecolor="#0d0d0d"
)
print("\nDashboard saved: dashboard_co2_gdp.png")