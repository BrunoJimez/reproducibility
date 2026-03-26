from reproducibility_core import (
    Variable, Hypothesis, DataFetcher,
    ReproducibilityEngine, EmpiricalTester, HypothesisComparator
)
import numpy as np

print("=" * 60)
print("TESTE DE HUGO — Curva de Preston (versão corrigida)")
print("Life expectancy ~ GDP per capita (10 países, World Bank)")
print("=" * 60)

fetcher = DataFetcher()

# ── Dados reais do World Bank ────────────────────────────────────────
print("\nBuscando dados do World Bank...")
df = fetcher.fetch_worldbank_multiple_countries(
    indicators = ["gdp_per_capita", "life_expectancy"],
    countries  = ["BR", "US", "DE", "CN", "IN", "ZA",
                  "NG", "AR", "MX", "JP"],
    year_start = 2010,
    year_end   = 2022,
)
print(f"Países carregados: {len(df)}")
print(df[["country", "gdp_per_capita", "life_expectancy"]])

# ── Calibração OLS manual para definir parâmetros realistas ─────────
import numpy as np
gdp = df["gdp_per_capita"].values
ev  = df["life_expectancy"].values

# Ajuste linear: ev = a*gdp + b
a_lin = np.cov(gdp, ev)[0,1] / np.var(gdp)
b_lin = np.mean(ev) - a_lin * np.mean(gdp)

# Ajuste log: ev = a*log(gdp) + b
log_gdp = np.log(gdp)
a_log = np.cov(log_gdp, ev)[0,1] / np.var(log_gdp)
b_log = np.mean(ev) - a_log * np.mean(log_gdp)

print(f"\nParâmetros calibrados por OLS:")
print(f"  Linear:      ev = {a_lin:.6f} × gdp + {b_lin:.2f}")
print(f"  Logarítmica: ev = {a_log:.4f} × log(gdp) + {b_log:.2f}")

# ── Hipóteses com parâmetros calibrados ──────────────────────────────
h_linear = Hypothesis(
    name        = "Preston Linear",
    description = "Life expectancy increases linearly with GDP per capita",
    function    = lambda v, a=a_lin, b=b_lin: a * v['g'] + b,
    variables   = [Variable("g", 1000.0, 65000.0, "GDP per capita (USD)")],
    origin      = "manual",
)

h_log = Hypothesis(
    name        = "Preston Logarithmic",
    description = "Life expectancy increases logarithmically with GDP (Preston curve)",
    function    = lambda v, a=a_log, b=b_log: a * np.log(v['g']) + b,
    variables   = [Variable("g", 1000.0, 65000.0, "GDP per capita (USD)")],
    origin      = "manual",
)

h_null = Hypothesis(
    name        = "Preston Null (constant)",
    description = "Life expectancy is independent of GDP per capita",
    function    = lambda v, m=np.mean(ev): np.ones_like(v['g']) * m,
    variables   = [Variable("g", 1000.0, 65000.0, "GDP per capita (USD)")],
    origin      = "manual",
)

hypotheses = [h_linear, h_log, h_null]
mapping    = {"g": "gdp_per_capita"}

# ── Rodar módulos manualmente com cálculo correto do composite ───────
engine     = ReproducibilityEngine(n_trials=60, sample_size=200, seed=42)
tester     = EmpiricalTester()
comparator = HypothesisComparator()

print("\nRodando análise...")
res_sim, res_emp = [], []

for h in hypotheses:
    rs = engine.test(h)
    re = tester.test(h, df, "life_expectancy", mapping)

    # CORREÇÃO DO BUG: calcular composite_score explicitamente
    if re.r_squared > 0:
        re.composite_score = rs.reproducibility_score * re.r_squared
    else:
        re.composite_score = 0.0   # não penalizar com fallback artificial

    res_sim.append(rs)
    res_emp.append(re)

comp = comparator.compare(res_emp, res_sim)

# ── Resultados ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTADOS — Curva de Preston (parâmetros OLS)")
print("=" * 60)

for rs, re in zip(res_sim, res_emp):
    print(f"\n── {rs.hypothesis_name}")
    print(f"   Score simulação : {rs.reproducibility_score:.1f}/100")
    print(f"   CV              : {rs.coefficient_of_variation:.2f}%")
    print(f"   R²              : {re.r_squared:.4f}")
    print(f"   RMSE            : {re.rmse:.2f} anos")
    print(f"   Score composto  : {re.composite_score:.1f}/100  ← corrigido")

print(f"\n── Ranking AIC:")
for r in comp.ranking:
    flag = "🏆" if r["name"] == comp.winner["name"] else "  "
    print(f"   {flag} {r['name']:<30} "
          f"AIC={r['aic']:8.1f}  "
          f"Δ={r['delta_aic']:5.1f}  "
          f"weight={r['akaike_weight']:.1%}")

best = max(res_emp, key=lambda r: r.composite_score)
print(f"\n── Comparação com benchmarks do paper:")
print(f"   Melhor score composto (Preston) : {best.composite_score:.1f}")
print(f"   CO₂ → temperatura (paper)       : 49.1  (esperado ≥ 40)")
print(f"   Card & Krueger (paper)           :  5.0  (esperado ≤ 20)")
print(f"   OSC 2015 (paper)                 : 20.8  (esperado 10–30)")

# ── Dashboard ────────────────────────────────────────────────────────
from reproducibility_core import generate_dashboard, TemporalAnalyser
fig = generate_dashboard(res_sim, res_emp, comp, [])
fig.savefig("dashboard_preston_curve.png",
            dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
print("\nDashboard salvo: dashboard_preston_curve.png")