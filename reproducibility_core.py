"""
reproducibility_core.py
─────────────────────────────────────────────────────────────────────
Pure logic of the Reproducibility system.
No UI dependencies (sem streamlit, sem matplotlib.show()).
Imported by app.py (Streamlit) e por scripts de linha de comando.

Módulos:
  0 — Estruturas de dados (Variable, Hypothesis, results)
  1 — OpenTranslator  (NLP: Parser de Regras / Groq / Ollama)
  2 — ReproducibilityEngine  (simulação)
  3 — EmpiricalTester        (real data)
  4 — HypothesisComparator     (AIC, BIC, Peso de Akaike)
  5 — TemporalAnalyser      (temporal drift)
  6 — DataFetcher           (APIs abertas: World Bank, IBGE, NASA)
  7 — GeradorDashboard        (matplotlib → fig, sem plt.show())
"""

import re
import json
import math
import textwrap
import unicodedata
import warnings
import io
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Tuple, Optional, Any

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")          # backend sem janela — essencial para Streamlit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 0 — ESTRUTURAS DE DADOS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Variable:
    name: str
    minimum: float
    maximum: float
    description: str = ""

    def sample(self, n: int) -> np.ndarray:
        return np.random.uniform(self.minimum, self.maximum, n)


@dataclass
class Hypothesis:
    name: str
    description: str
    function: Callable
    variables: List[Variable]
    prediction: str = ""
    origin: str = "manual"
    generated_code: str = ""
    results: List[float] = field(default_factory=list)


@dataclass
class SimulationResult:
    hypothesis_name: str
    n_trials: int
    sample_size: int
    mean: float
    std_dev: float
    coefficient_of_variation: float
    confidence_interval_95: Tuple
    normality_p_value: float
    reproducibility_score: float
    classification: str
    results_per_trial: List[float]


@dataclass
class EmpiricalResult:
    hypothesis_name: str
    n_samples: int
    r_squared: float
    rmse: float
    mae: float
    residuals: np.ndarray
    p_valor_residuals: float
    aic: float
    bic: float
    empirical_score: float
    composite_score: float = 0.0


@dataclass
class ComparisonResult:
    ranking: List[Dict]
    winner: Dict


@dataclass
class TemporalResult:
    hypothesis_name: str
    windows: List[str]
    scores_per_window: List[float]
    trend: float
    stable: bool
    drift_detected: bool


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 1 — TRADUTOR ABERTO (Open Science)
# ══════════════════════════════════════════════════════════════════════

class RuleParser:
    """
    Converte hipóteses em linguagem natural para funções Python.
    Zero dependências externas. Funciona offline. PT-BR e EN.
    """

    VARIAVEIS = {
        "temperatura": ("T", 0.0, 1500.0, "K"),
        "temperature": ("T", 0.0, 1500.0, "K"),
        "pressao":     ("P", 0.01, 1000.0, "atm"),
        "pressure":    ("P", 0.01, 1000.0, "atm"),
        "volume":      ("V", 0.001, 1000.0, "L"),
        "velocidade":  ("v", 0.0, 1000.0, "m/s"),
        "velocity":    ("v", 0.0, 1000.0, "m/s"),
        "speed":       ("v", 0.0, 1000.0, "m/s"),
        "forca":       ("F", 0.0, 1e5, "N"),
        "force":       ("F", 0.0, 1e5, "N"),
        "massa":       ("m", 0.001, 1e4, "kg"),
        "mass":        ("m", 0.001, 1e4, "kg"),
        "energia":     ("E", 0.0, 1e6, "J"),
        "energy":      ("E", 0.0, 1e6, "J"),
        "distancia":   ("d", 0.0, 1e6, "m"),
        "distance":    ("d", 0.0, 1e6, "m"),
        "altura":      ("h", 0.0, 1e4, "m"),
        "height":      ("h", 0.0, 1e4, "m"),
        "aceleracao":  ("a", 0.0, 1000.0, "m/s²"),
        "acceleration":("a", 0.0, 1000.0, "m/s²"),
        "frequencia":  ("f", 0.001, 1e9, "Hz"),
        "frequency":   ("f", 0.001, 1e9, "Hz"),
        "comprimento": ("L", 0.0, 1000.0, "m"),
        "length":      ("L", 0.0, 1000.0, "m"),
        "tempo":       ("t", 0.0, 1e6, "s"),
        "time":        ("t", 0.0, 1e6, "s"),
        "corrente":    ("I", 0.0, 1000.0, "A"),
        "tensao":      ("U", 0.0, 1e4, "V"),
        "voltage":     ("U", 0.0, 1e4, "V"),
        "resistencia": ("R", 0.001, 1e6, "Ω"),
        "resistance":  ("R", 0.001, 1e6, "Ω"),
        "potencia":    ("W", 0.0, 1e6, "W"),
        "power":       ("W", 0.0, 1e6, "W"),
        "concentracao":("C", 0.0, 100.0, "mol/L"),
        "concentration":("C", 0.0, 100.0, "mol/L"),
        "ph":          ("pH", 0.0, 14.0, ""),
        "densidade":   ("rho", 0.001, 20.0, "g/cm³"),
        "density":     ("rho", 0.001, 20.0, "g/cm³"),
        "populacao":   ("N", 1.0, 1e9, "ind."),
        "population":  ("N", 1.0, 1e9, "ind."),
        "taxa":        ("r", -2.0, 10.0, "%"),
        "rate":        ("r", -2.0, 10.0, "%"),
        "crescimento": ("g", -1.0, 5.0, "%"),
        "growth":      ("g", -1.0, 5.0, "%"),
        "renda":       ("Y", 0.0, 1e6, "R$"),
        "income":      ("Y", 0.0, 1e6, "$"),
        "estudo":      ("E", 0.0, 30.0, "anos"),
        "education":   ("E", 0.0, 30.0, "anos"),
        "experiencia": ("X", 0.0, 40.0, "anos"),
        "experience":  ("X", 0.0, 40.0, "anos"),
        "capital":     ("K", 0.0, 1e8, "R$"),
        "salario":     ("S", 0.0, 1e5, "R$"),
        "salary":      ("S", 0.0, 1e5, "$"),
        "idade":       ("i", 0.0, 120.0, "anos"),
        "age":         ("i", 0.0, 120.0, "anos"),
        "inflacao":    ("pi", -0.5, 5.0, "%"),
        "inflation":   ("pi", -0.5, 5.0, "%"),
        "co2":         ("co2", 200.0, 800.0, "ppm"),
        "temperatura_global": ("Tg", -2.0, 5.0, "°C anomalia"),
        "pib":         ("pib", 0.0, 1e13, "USD"),
        "gdp":         ("gdp", 0.0, 1e13, "USD"),
        "desemprego":  ("u", 0.0, 100.0, "%"),
        "unemployment":("u", 0.0, 100.0, "%"),
        "pobreza":     ("pov", 0.0, 100.0, "%"),
        "poverty":     ("pov", 0.0, 100.0, "%"),
        "expectativa de vida": ("ev", 30.0, 100.0, "anos"),
        "life expectancy":     ("ev", 30.0, 100.0, "anos"),
        "area":        ("A", 0.0, 1e6, "m²"),
        "raio":        ("r", 0.0, 1000.0, "m"),
        "radius":      ("r", 0.0, 1000.0, "m"),
        "angulo":      ("theta", 0.0, 6.28, "rad"),
    }

    RELACOES = [
        (r"quadrado|ao quadrado|segunda pot|quadratic|squared",       "potencia",    2.0),
        (r"cubo|ao cubo|terceira pot|cubic|cubed",                    "potencia",    3.0),
        (r"raiz quadrada|square root",                                "potencia",    0.5),
        (r"exponencial|cresce exponencial|decai exponencial|exponential", "exponencial", None),
        (r"logarítm|logaritmic|log de|\blog\b",                       "logaritmica", None),
        (r"inversamente proporcional|varia inversamente|decresce com|diminui com|inversely proportional", "inversa", -1.0),
        (r"proporcional|cresce com|aumenta com|varia com|depende de|directly proportional|grows with|increases with|varies with|linear", "linear", 1.0),
        (r"constante|independe|n[aã]o (?:depende|varia)|constant|independent", "constante", 0.0),
    ]

    def __init__(self):
        self._re = [(re.compile(p, re.IGNORECASE), t, e) for p, t, e in self.RELACOES]

    def _normalize(self, s: str) -> str:
        nfkd = unicodedata.normalize("NFKD", s.lower())
        return re.sub(r"\s+", " ", "".join(c for c in nfkd if not unicodedata.combining(c))).strip()

    def _extrair_variables(self, texto: str) -> List[Variable]:
        tn = self._normalize(texto)
        encontradas, usados = [], set()
        for palavra, (name, vmin, vmax, un) in self.VARIAVEIS.items():
            if re.search(r"\b" + re.escape(self._normalize(palavra)) + r"\b", tn) and name not in usados:
                encontradas.append(Variable(name, vmin, vmax, f"{palavra} ({un})" if un else palavra))
                usados.add(name)
        if not encontradas:
            for m in re.finditer(r"(?:fun[cç][aã]o d[aeo]s?|depende d[aeo]s?|varia com|proporcional a[os]?)\s+(\w+)", texto, re.IGNORECASE):
                t = m.group(1).lower()
                if len(t) > 2 and t not in usados:
                    encontradas.append(Variable(f"x{len(encontradas)+1}", 0.0, 100.0, t))
                    usados.add(t)
        return encontradas or [Variable("x", 0.0, 100.0, "variável independente")]

    def _identify_relation(self, texto: str) -> Tuple[str, Optional[float]]:
        for regex, tipo, exp in self._re:
            if regex.search(texto):
                return tipo, exp
        return "linear", 1.0

    def _generate_code(self, variables: List[Variable], relacao: str, exp: Optional[float]) -> str:
        names = [v.name for v in variables]
        x = names[0]
        if relacao == "linear":
            expr = " + ".join(f"v['{n}']" for n in names)
        elif relacao == "inversa":
            base = f"1.0 / (v['{x}'] + 1e-10)"
            expr = f"({base}) * (" + " + ".join(f"v['{n}']" for n in names[1:]) + ")" if len(names) > 1 else base
        elif relacao == "potencia":
            e = exp or 2.0
            expr = " + ".join(f"v['{n}'] ** {e}" for n in names)
        elif relacao == "exponencial":
            soma = " + ".join(f"v['{n}']" for n in names)
            expr = f"np.exp(({soma}) / {max(1, len(names)) * 50})"
        elif relacao == "logaritmica":
            expr = " + ".join(f"np.log(np.abs(v['{n}']) + 1)" for n in names)
        elif relacao == "constante":
            expr = f"np.ones_like(v['{x}']) * 1.0"
        else:
            expr = f"v['{x}']"
        return f"def hypothesis(v):\n    return {expr}"

    def parse(self, texto: str) -> Hypothesis:
        variables    = self._extrair_variables(texto)
        relacao, exp = self._identify_relation(texto)
        codigo       = self._generate_code(variables, relacao, exp)
        ns = {"np": np}
        exec(codigo, ns)
        names_var = [v.description.split("(")[0].strip() for v in variables]
        rel_map = {"linear": "linear", "inversa": "inversa", "potencia": f"potência(exp={exp})",
                   "exponencial": "exponencial", "logaritmica": "logarítmica", "constante": "constante"}
        return Hypothesis(
            name=f"Hipótese {rel_map.get(relacao, relacao).capitalize()}({', '.join(names_var)})",
            description=texto,
            function=ns["hypothesis"],
            variables=variables,
            prediction=f"Relação {rel_map.get(relacao, relacao)} entre {', '.join(names_var)}",
            origin="parser_regras",
            generated_code=codigo
        )


_GROQ_SYSTEM = textwrap.dedent("""
    You are a scientific hypothesis formalization assistant.
    Given a hypothesis in natural language (Portuguese or English),
    return ONLY a valid JSON (no markdown, no extra text):
    {
      "name": "Short name",
      "description": "Natural language description",
      "prediction": "What the hypothesis predicts",
      "variables": [{"name":"symbol","minimum":0.0,"maximum":100.0,"description":"meaning (unit)"}],
      "code_function": "def hypothesis(v):\\n    return v['symbol'] * 2.0"
    }
    Function MUST be named 'hypothesis', receive dict 'v' with numpy arrays.
    Use only numpy (imported as np). Return ONLY the JSON.
""")


def _construir_hypothesis_de_json(e: dict, origin: str) -> Hypothesis:
    variables = [Variable(v["name"], v["minimum"], v["maximum"], v.get("description", "")) for v in e["variables"]]
    ns = {"np": np}
    exec(e["code_function"], ns)
    return Hypothesis(name=e["name"], description=e["description"], function=ns["hypothesis"],
                    variables=variables, prediction=e.get("prediction", ""),
                    origin=origin, generated_code=e["code_function"])


def translate_via_groq(texto: str, api_key: str, modelo: str = "llama-3.3-70b-versatile") -> Hypothesis:
    if not _REQUESTS_OK:
        raise RuntimeError("requests não instalado")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": modelo, "temperature": 0.1, "max_tokens": 800,
              "messages": [{"role": "system", "content": _GROQ_SYSTEM},
                           {"role": "user",   "content": texto}]},
        timeout=30
    )
    resp.raise_for_status()
    conteudo = resp.json()["choices"][0]["message"]["content"].strip()
    conteudo = re.sub(r"^```(?:json)?\s*|\s*```$", "", conteudo).strip()
    return _construir_hypothesis_de_json(json.loads(conteudo), f"groq:{modelo}")


def translate_via_ollama(texto: str, modelo: str = "llama3.2") -> Hypothesis:
    if not _REQUESTS_OK:
        raise RuntimeError("requests não instalado")
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": modelo, "stream": False, "options": {"temperature": 0.1},
              "messages": [{"role": "system", "content": _GROQ_SYSTEM},
                           {"role": "user",   "content": texto}]},
        timeout=120
    )
    resp.raise_for_status()
    conteudo = resp.json()["message"]["content"].strip()
    conteudo = re.sub(r"^```(?:json)?\s*|\s*```$", "", conteudo).strip()
    return _construir_hypothesis_de_json(json.loads(conteudo), f"ollama:{modelo}")


def ollama_available() -> bool:
    try:
        return _REQUESTS_OK and requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


class OpenTranslator:
    """
    Orquestrador: Ollama (local) → Groq (gratuito) → RuleParser (offline).
    Sempre produz uma hipótese.
    """
    def __init__(self, groq_key: Optional[str] = None,
                 groq_modelo: str = "llama-3.3-70b-versatile",
                 ollama_model: str = "llama3.2"):
        self.parser      = RuleParser()
        self.groq_key    = groq_key
        self.groq_modelo = groq_modelo
        self.ollama_model = ollama_model

    def translate(self, texto: str, log: Optional[List[str]] = None) -> Hypothesis:
        def _log(msg):
            if log is not None:
                log.append(msg)

        if ollama_available():
            try:
                _log(f"🟢 Ollama ({self.ollama_model}) — LLM local")
                return translate_via_ollama(texto, self.ollama_model)
            except Exception as e:
                _log(f"⚠️ Ollama falhou: {e}")

        if self.groq_key:
            try:
                _log(f"🟡 Groq ({self.groq_modelo}) — LLM gratuito na nuvem")
                return translate_via_groq(texto, self.groq_key, self.groq_modelo)
            except Exception as e:
                _log(f"⚠️ Groq falhou: {e}")

        _log("⚪ Parser de Regras — offline, zero dependências")
        return self.parser.parse(texto)

    def status(self) -> Dict[str, str]:
        return {
            "Camada 0 — Parser de Regras": "✅ sempre disponível",
            "Camada 1 — Groq (gratuito)":  f"✅ {self.groq_modelo}" if self.groq_key else "⚪ sem chave (opcional)",
            "Camada 2 — Ollama (local)":   f"✅ {self.ollama_model}" if ollama_available() else "⚪ não detectado (opcional)",
        }


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 2 — MOTOR DE REPRODUTIBILIDADE (simulação)
# ══════════════════════════════════════════════════════════════════════

class ReproducibilityEngine:
    def __init__(self, n_trials: int = 80, sample_size: int = 300, seed: Optional[int] = None):
        self.n_trials = n_trials
        self.sample_size = sample_size
        if seed is not None:
            np.random.seed(seed)

    def test(self, hypothesis: Hypothesis) -> SimulationResult:
        results = []
        for _ in range(self.n_trials):
            amostras = {v.name: v.sample(self.sample_size) for v in hypothesis.variables}
            r = hypothesis.function(amostras)
            results.append(float(np.mean(r) if hasattr(r, "__len__") else r))
        hypothesis.results = results
        return self._metrics(hypothesis.name, results)

    def _metrics(self, name: str, results: List[float]) -> SimulationResult:
        arr = np.array(results)
        mean, dp = np.mean(arr), np.std(arr)
        cv = (dp / abs(mean) * 100) if mean != 0 else float("inf")
        ic = stats.t.interval(0.95, df=len(arr)-1, loc=mean, scale=stats.sem(arr))
        _, p_norm = stats.shapiro(arr[:50])
        score = min(100.0, 100 * np.exp(-cv / 10))
        cls = ("✅ Highly reproducible" if score >= 85 else
               "🟡 Moderately reproducible" if score >= 60 else
               "🟠 Low reproducibility" if score >= 35 else
               "❌ Not reproducible")
        return SimulationResult(name, self.n_trials, self.sample_size, mean, dp, cv,
                                  ic, p_norm, score, cls, results)


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 3 — TESTE EMPÍRICO (real data)
# ══════════════════════════════════════════════════════════════════════

class EmpiricalTester:
    def test(self, hypothesis: Hypothesis, df: pd.DataFrame,
               target_column: str, mapping: Dict[str, str]) -> EmpiricalResult:
        v = {k: df[col].values.astype(float) for k, col in mapping.items()}
        y_prev = hypothesis.function(v)
        if not hasattr(y_prev, "__len__"):
            y_prev = np.full(len(df), float(y_prev))
        y_prev = np.array(y_prev, dtype=float)
        y_real = df[target_column].values.astype(float)
        mask = ~(np.isnan(y_real) | np.isnan(y_prev))
        y_real, y_prev = y_real[mask], y_prev[mask]
        n = len(y_real)
        res = y_real - y_prev
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        r2   = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        rmse = np.sqrt(np.mean(res ** 2))
        mae  = np.mean(np.abs(res))
        _, p_res = stats.shapiro(res[:50])
        k = len(hypothesis.variables) + 1
        ll = -n / 2 * np.log(ss_res / n + 1e-10)
        aic, bic = 2*k - 2*ll, k*np.log(n) - 2*ll
        score = max(0.0, min(100.0, r2 * 70 + (p_res > 0.05) * 30))
        return EmpiricalResult(hypothesis.name, n, r2, rmse, mae, res, p_res, aic, bic, score, 0.0)


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 4 — COMPARAÇÃO DE HIPÓTESES
# ══════════════════════════════════════════════════════════════════════

class HypothesisComparator:
    def compare(self, results_emp: List[EmpiricalResult],
                 results_sim: Optional[List[SimulationResult]] = None) -> ComparisonResult:
        comparativo = []
        for re_ in results_emp:
            linha = {"name": re_.hypothesis_name, "r2": re_.r_squared, "rmse": re_.rmse,
                     "aic": re_.aic, "bic": re_.bic, "empirical_score": re_.empirical_score,
                     "simulated_score": None}
            if results_sim:
                for rs in results_sim:
                    if rs.hypothesis_name == re_.hypothesis_name:
                        linha["simulated_score"] = rs.reproducibility_score
            comparativo.append(linha)
        comparativo.sort(key=lambda x: x["aic"])
        aic_min = comparativo[0]["aic"]
        for c in comparativo:
            c["delta_aic"]    = c["aic"] - aic_min
            c["akaike_weight"]  = np.exp(-c["delta_aic"] / 2)
        soma = sum(c["akaike_weight"] for c in comparativo)
        for c in comparativo:
            c["akaike_weight"] /= soma
        return ComparisonResult(ranking=comparativo, winner=comparativo[0])


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 5 — ANÁLISE TEMPORAL
# ══════════════════════════════════════════════════════════════════════

class TemporalAnalyser:
    def __init__(self, n_windows: int = 5):
        self.n_windows = n_windows

    def analyse(self, hypothesis: Hypothesis, df: pd.DataFrame,
                 time_column: str, target_column: str,
                 mapping: Dict[str, str]) -> TemporalResult:
        df_ord = df.sort_values(time_column).reset_index(drop=True)
        tam    = max(1, len(df_ord) // self.n_windows)
        windows = [df_ord.iloc[i:i+tam].reset_index(drop=True)
                   for i in range(0, len(df_ord), tam)][:self.n_windows]
        testador = EmpiricalTester()
        scores, labels = [], []
        for i, jan in enumerate(windows):
            if len(jan) < 5:
                continue
            t_min = int(jan[time_column].iat[0])
            t_max = int(jan[time_column].iat[-1])
            labels.append(f"J{i+1} [{t_min}–{t_max}]")
            try:
                r = testador.test(hypothesis, jan, target_column, mapping)
                scores.append(r.empirical_score)
            except Exception:
                scores.append(0.0)
        slope, p_val = (0.0, 1.0) if len(scores) < 2 else stats.linregress(range(len(scores)), scores)[:2][::-1][::-1]
        if len(scores) >= 2:
            res_lr = stats.linregress(np.arange(len(scores)), scores)
            slope, p_val = res_lr.slope, res_lr.pvalue
        stable = abs(slope) < 2.0
        return TemporalResult(hypothesis.name, labels, scores, slope, stable,
                                 not stable and p_val < 0.1)


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 6 — BUSCADOR DE DADOS (APIs abertas)
# ══════════════════════════════════════════════════════════════════════

class DataFetcher:
    """
    Acessa fontes de dados abertas e gratuitas:
      - World Bank API  (indicadores globais, sem chave)
      - IBGE SIDRA      (dados brasileiros, sem chave)
      - NASA POWER      (dados climáticos, sem chave)
      - Our World in Data (CSVs públicos)
    """

    # ── World Bank ──────────────────────────────────────────────────
    WORLDBANK_INDICADORES = {
        "pib_per_capita":     "NY.GDP.PCAP.CD",
        "expectativa_vida":   "SP.DYN.LE00.IN",
        "desemprego":         "SL.UEM.TOTL.ZS",
        "co2_per_capita":     "EN.ATM.CO2E.PC",
        "populacao":          "SP.POP.TOTL",
        "mortalidade_infantil":"SP.DYN.IMRT.IN",
        "acesso_eletricidade":"EG.ELC.ACCS.ZS",
        "pobreza_extrema":    "SI.POV.DDAY",
        "alfabetizacao":      "SE.ADT.LITR.ZS",
        "gasto_saude":        "SH.XPD.CHEX.GD.ZS",
        "gasto_educacao":     "SE.XPD.TOTL.GD.ZS",
        "emissoes_co2":       "EN.ATM.CO2E.KT",
    }

    def list_worldbank_indicators(self) -> Dict[str, str]:
        return self.WORLDBANK_INDICADORES

    def fetch_worldbank(self, indicador: str, pais: str = "BR",
                         ano_inicio: int = 2000, ano_fim: int = 2023) -> pd.DataFrame:
        """
        Busca indicador do World Bank.
        indicador: chave do dict acima OU código direto (ex: 'NY.GDP.PCAP.CD')
        pais: código ISO2 (BR, US, DE, CN, etc.) ou 'all' para todos
        """
        if not _REQUESTS_OK:
            raise RuntimeError("requests não disponível")
        codigo = self.WORLDBANK_INDICADORES.get(indicador, indicador)
        url = (f"https://api.worldbank.org/v2/country/{pais}"
               f"/indicator/{codigo}"
               f"?format=json&per_page=500"
               f"&date={ano_inicio}:{ano_fim}")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        dados = resp.json()
        if len(dados) < 2 or not dados[1]:
            raise ValueError(f"Sem dados para {indicador}/{pais}")
        registros = [{"ano": int(d["date"]), "valor": d["value"], "country": d["country"]["value"]}
                     for d in dados[1] if d["value"] is not None]
        df = pd.DataFrame(registros).sort_values("ano").reset_index(drop=True)
        df.attrs["source"]    = "World Bank"
        df.attrs["indicator"] = codigo
        df.attrs["country"]     = pais
        return df

    def fetch_worldbank_multiple_countries(self, indicadores: List[str],
                                          paises: List[str] = None,
                                          ano_inicio: int = 2000,
                                          ano_fim: int = 2022) -> pd.DataFrame:
        """Busca múltiplos indicadores para múltiplos países e junta em um DataFrame."""
        if paises is None:
            paises = ["BR", "US", "DE", "CN", "IN", "ZA", "NG", "AR", "MX", "JP"]
        dfs = []
        for pais in paises:
            row = {"country": pais}
            for ind in indicadores:
                try:
                    df_ind = self.fetch_worldbank(ind, pais, ano_inicio, ano_fim)
                    row[ind] = df_ind["valor"].mean()
                except Exception:
                    row[ind] = np.nan
            dfs.append(row)
        return pd.DataFrame(dfs).dropna()

    # ── NASA POWER (clima) ──────────────────────────────────────────
    def fetch_nasa_climate(self, lat: float = -15.8, lon: float = -47.9,
                           parametros: List[str] = None,
                           ano_inicio: int = 2000, ano_fim: int = 2022) -> pd.DataFrame:
        """
        Dados climáticos da NASA POWER (resolução anual).
        Default: Brasília, DF.
        Parâmetros comuns: T2M (temperatura), PRECTOTCORR (chuva), ALLSKY_SFC_SW_DWN (radiação)
        """
        if not _REQUESTS_OK:
            raise RuntimeError("requests não disponível")
        if parametros is None:
            parametros = ["T2M", "PRECTOTCORR"]
        params_str = ",".join(parametros)
        url = (f"https://power.larc.nasa.gov/api/temporal/annual/point"
               f"?parameters={params_str}&community=RE"
               f"&longitude={lon}&latitude={lat}"
               f"&start={ano_inicio}&end={ano_fim}&format=JSON")
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        dados = resp.json()["properties"]["parameter"]
        registros = []
        anos = list(dados[parametros[0]].keys())
        for ano in anos:
            row = {"ano": int(ano)}
            for p in parametros:
                row[p] = dados[p].get(ano, np.nan)
            registros.append(row)
        return pd.DataFrame(registros).sort_values("ano").reset_index(drop=True)

    # ── IBGE SIDRA ──────────────────────────────────────────────────
    def fetch_ibge_gdp(self) -> pd.DataFrame:
        """PIB brasileiro anual (IBGE SIDRA tabela 1846)."""
        if not _REQUESTS_OK:
            raise RuntimeError("requests não disponível")
        url = "https://servicodados.ibge.gov.br/api/v3/agregados/1846/periodos/2000|2001|2002|2003|2004|2005|2006|2007|2008|2009|2010|2011|2012|2013|2014|2015|2016|2017|2018|2019|2020|2021|2022/variables/585?localidades=N1[all]"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        dados = resp.json()
        results = dados[0]["results"][0]["series"][0]["serie"]
        registros = [{"ano": int(k), "pib_bilhoes": float(v)/1e9}
                     for k, v in results.items() if v != "..."]
        return pd.DataFrame(registros).sort_values("ano").reset_index(drop=True)

    # ── Fallback: dataset sintético ─────────────────────────────────
    def generate_synthetic(self, n: int = 300, seed: int = 42) -> pd.DataFrame:
        """Gera dataset sintético realista para testes offline."""
        rng = np.random.default_rng(seed)
        registros = []
        for ano in range(2000, 2024):
            for _ in range(n // 24):
                estudo = rng.uniform(4, 20)
                exp    = rng.uniform(0, 30)
                crise  = -2000 if ano in [2008, 2009, 2015, 2020] else 0
                renda  = 800 + 500*estudo + 200*exp + rng.normal(0, 3000) + crise
                registros.append({"ano": ano, "estudo": round(estudo, 1),
                                  "experiencia": round(exp, 1),
                                  "renda": max(500, round(renda, 2))})
        df = pd.DataFrame(registros)
        df.attrs["source"] = "synthetic"
        return df


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 7 — GERADOR DE DASHBOARD (matplotlib, sem plt.show())
# ══════════════════════════════════════════════════════════════════════

DARK  = "#0d0d0d"
PANEL = "#1a1a1a"
GRID  = "#2a2a2a"
CORES = ["#00e5ff", "#76ff03", "#ff6d00", "#e040fb", "#ffea00", "#ff4081"]


def _ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="--")


def generate_dashboard(
    results_sim:  List[SimulationResult],
    results_emp:  List[EmpiricalResult],
    comparacao:      ComparisonResult,
    results_temp: List[TemporalResult],
) -> plt.Figure:
    """
    Retorna uma figura matplotlib completa (sem plt.show()).
    O Streamlit chama st.pyplot(fig).
    """
    n = len(results_sim)
    fig = plt.figure(figsize=(20, 5 + 5*n + 5 + 4 + 4), facecolor=DARK)
    fig.suptitle("REPRODUTIBILIDADE — Dashboard Completo",
                 fontsize=15, color="white", fontweight="bold", y=1.005)

    # ── Seção 1: Simulação ──────────────────────────────────────────
    for i, res in enumerate(results_sim):
        cor = CORES[i % len(CORES)]
        arr = np.array(res.results_per_trial)
        top = 0.97 - i * 0.16
        bot = top - 0.14
        gs  = gridspec.GridSpec(1, 3, figure=fig, top=top, bottom=bot,
                                left=0.05, right=0.95, wspace=0.3)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(arr, color=cor, alpha=0.6, linewidth=0.8)
        ax1.axhline(res.mean, color="white", linestyle="--", linewidth=1.2)
        if not (np.isnan(res.confidence_interval_95[0]) or np.isnan(res.confidence_interval_95[1])):
            ax1.fill_between(range(len(arr)),
                             res.confidence_interval_95[0],
                             res.confidence_interval_95[1],
                             color=cor, alpha=0.1)
        _ax(ax1)
        ax1.set_title(f"{res.hypothesis_name}\nTrials", color="white", fontsize=9)

        ax2 = fig.add_subplot(gs[1])
        ax2.hist(arr, bins=15, color=cor, alpha=0.8, edgecolor=DARK)
        ax2.axvline(res.mean, color="white", linestyle="--", linewidth=1.2)
        _ax(ax2)
        ax2.set_title("Distribuição", color="white", fontsize=9)

        ax3 = fig.add_subplot(gs[2])
        ax3.set_facecolor(PANEL)
        ax3.axis("off")
        score = res.reproducibility_score
        cor_s = ("#76ff03" if score >= 85 else "#ffea00" if score >= 60
                 else "#ff6d00" if score >= 35 else "#ff1744")
        ax3.barh(0.5, score/100, height=0.2, color=cor_s, alpha=0.9, transform=ax3.transAxes)
        ax3.barh(0.5, 1.0,       height=0.2, color=GRID,  alpha=0.7, transform=ax3.transAxes)
        ax3.text(0.5, 0.80, f"{score:.0f}/100",  ha="center", color="white",
                 fontsize=16, fontweight="bold", transform=ax3.transAxes)
        ax3.text(0.5, 0.28, res.classification,  ha="center", color=cor_s,
                 fontsize=8, transform=ax3.transAxes)
        ax3.text(0.5, 0.12, f"CV={res.coefficient_of_variation:.1f}%", ha="center",
                 color="#888", fontsize=8, transform=ax3.transAxes)
        ax3.set_title("Score", color="white", fontsize=9)

    sep = 0.97 - n * 0.16

    # ── Seção 2: Empírico ───────────────────────────────────────────
    et, eb = sep - 0.03, sep - 0.17
    fig.text(0.01, et + 0.005, "Módulo 2 — Dados Empíricos", color="#76ff03", fontsize=10, fontweight="bold")
    gs_e = gridspec.GridSpec(1, 3, figure=fig, top=et-0.02, bottom=eb, left=0.05, right=0.95, wspace=0.3)

    ax_res = fig.add_subplot(gs_e[0:2])
    _ax(ax_res)
    ax_res.set_title("Distribuição dos Resíduos", color="white", fontsize=9)
    for i, re_ in enumerate(results_emp):
        ax_res.hist(re_.residuals, bins=20, alpha=0.6, color=CORES[i % len(CORES)],
                    label=f"{re_.hypothesis_name[:20]} R²={re_.r_squared:.3f}", edgecolor=DARK)
    ax_res.axvline(0, color="white", linestyle="--", linewidth=1.2)
    ax_res.legend(fontsize=7, labelcolor="white", facecolor="#333")

    ax_r2 = fig.add_subplot(gs_e[2])
    _ax(ax_r2)
    names = [re_.hypothesis_name.split()[0] for re_ in results_emp]
    r2s   = [re_.r_squared for re_ in results_emp]
    bars  = ax_r2.barh(names, r2s, color=[CORES[i % len(CORES)] for i in range(len(results_emp))], alpha=0.85)
    ax_r2.set_xlim(-0.1, 1.1)
    ax_r2.set_title("R²", color="white", fontsize=9)
    for bar, val in zip(bars, r2s):
        ax_r2.text(max(val + 0.02, 0.02), bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center", color="white", fontsize=8)

    # ── Seção 3: Comparação ─────────────────────────────────────────
    ct, cb = eb - 0.04, eb - 0.14
    fig.text(0.01, ct + 0.005, "Módulo 3 — Comparação (Peso de Akaike)", color="#ff6d00", fontsize=10, fontweight="bold")
    ax_c = fig.add_axes([0.05, cb, 0.90, ct - cb - 0.01])
    ax_c.set_facecolor(PANEL)
    ax_c.tick_params(colors="#888")
    for spine in ax_c.spines.values():
        spine.set_edgecolor(GRID)
    names_c = [r["name"].split()[0] for r in comparacao.ranking]
    pesos_c = [r["akaike_weight"] * 100 for r in comparacao.ranking]
    bars_c  = ax_c.bar(names_c, pesos_c,
                       color=[CORES[i % len(CORES)] for i in range(len(comparacao.ranking))],
                       alpha=0.85, edgecolor=DARK)
    ax_c.set_ylabel("Probabilidade (%)", color="#888")
    for bar, val in zip(bars_c, pesos_c):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f"{val:.1f}%", ha="center", color="white", fontsize=9)
    ax_c.tick_params(axis="x", colors="#aaa")
    if pesos_c:
        ax_c.text(0, max(pesos_c) * 1.12, "🏆", fontsize=12, ha="center")

    # ── Seção 4: Temporal ───────────────────────────────────────────
    tt, tb = cb - 0.04, cb - 0.13
    fig.text(0.01, tt + 0.005, "Módulo 4 — Dimensão Temporal (deriva?)", color="#e040fb", fontsize=10, fontweight="bold")
    n_temp = max(1, len(results_temp))
    gs_t   = gridspec.GridSpec(1, n_temp, figure=fig, top=tt-0.02, bottom=tb,
                               left=0.05, right=0.95, wspace=0.35)
    for i, rt in enumerate(results_temp):
        cor = CORES[i % len(CORES)]
        ax  = fig.add_subplot(gs_t[i])
        _ax(ax)
        sc  = rt.scores_per_window
        x   = np.arange(len(sc))
        ax.plot(x, sc, "o-", color=cor, linewidth=2, markersize=5)
        ax.fill_between(x, sc, alpha=0.12, color=cor)
        if len(sc) >= 2:
            sl, inter = stats.linregress(x, sc)[:2]
            cor_t = "#ff4081" if rt.drift_detected else "#76ff03"
            ax.plot(x, sl * x + inter, "--", color=cor_t, linewidth=1.5)
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(rt.windows, fontsize=6, color="#aaa", rotation=10)
        status = "⚠️ deriva" if rt.drift_detected else "✅ estável"
        ax.set_title(f"{rt.hypothesis_name[:22]}\n{status}", color="white", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.995])
    return fig


# ══════════════════════════════════════════════════════════════════════
#  PIPELINE COMPLETO (chamado pelo app.py)
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(
    hypothesiss:      List[Hypothesis],
    df:             pd.DataFrame,
    target_column:    str,
    mappings:    List[Dict[str, str]],
    time_column:   Optional[str] = None,
    n_trials:       int = 60,
    sample_size:    int = 200,
    n_windows:      int = 5,
    seed:           int = 42,
) -> Dict[str, Any]:
    """
    Roda todos os módulos e retorna um dict com todos os results.
    O app.py usa esse dict para montar a UI.
    """
    motor      = ReproducibilityEngine(n_trials, sample_size, seed)
    testador   = EmpiricalTester()
    comparador = HypothesisComparator()
    analisador = TemporalAnalyser(n_windows)

    res_sim  = [motor.test(h) for h in hypothesiss]
    res_emp  = [testador.test(h, df, target_column, mapa)
                for h, mapa in zip(hypothesiss, mappings)]
    comp     = comparador.compare(res_emp, res_sim)
    res_temp = []
    if time_column:
        for h, mapa in zip(hypothesiss, mappings):
            res_temp.append(analisador.analyse(h, df, time_column, target_column, mapa))

    fig = generate_dashboard(res_sim, res_emp, comp, res_temp)

    return {
        "simulation":  res_sim,
        "empirical":   res_emp,
        "comparison": comp,
        "temporal":   res_temp,
        "figure":     fig,
    }
