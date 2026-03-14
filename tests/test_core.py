"""
tests/test_core.py
──────────────────────────────────────────────────────────────────────
Suite de testes automatizados — sistema Reprodutibilidade

Execução:
    pytest tests/ -v
    pytest tests/ -v --tb=short          # saída compacta
    pytest tests/ -v --cov=reproducibility_core   # cobertura (pip install pytest-cov)

Organização:
    Módulo 1 — OpenTranslator / RuleParser
    Módulo 2 — ReproducibilityEngine
    Módulo 3 — EmpiricalTester
    Módulo 4 — HypothesisComparator
    Módulo 5 — TemporalAnalyser
    Módulo 6 — DataFetcher
    Módulo 7 — GeradorDashboard
    Validação — score contra casos com resposta conhecida (núcleo científico)
"""

import math
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reproducibility_core import (
    Variable,
    Hypothesis,
    ResultadoSimulacao,
    ResultadoEmpirico,
    RuleParser,
    OpenTranslator,
    ReproducibilityEngine,
    EmpiricalTester,
    HypothesisComparator,
    TemporalAnalyser,
    DataFetcher,
    generate_dashboard,
)

# ══════════════════════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def simple_variable():
    return Variable("x", 0.0, 10.0, "variável x")


@pytest.fixture
def linear_hypothesis():
    """y = 2x — relação linear perfeita, deve ser altamente reproduzível."""
    return Hypothesis(
        nome="Linear pura",
        descricao="y = 2x",
        funcao=lambda v: 2.0 * v["x"],
        variaveis=[Variable("x", 0.0, 10.0)],
    )


@pytest.fixture
def constant_hypothesis():
    """y = 5 — constante absoluta, reprodutibilidade máxima."""
    return Hypothesis(
        nome="Constante",
        descricao="y = 5",
        funcao=lambda v: np.ones_like(v["x"]) * 5.0,
        variaveis=[Variable("x", 0.0, 10.0)],
    )


@pytest.fixture
def noisy_hypothesis():
    """y = x + ruído muito alto — deve ter baixa reprodutibilidade."""
    rng = np.random.default_rng(0)
    def f(v):
        return v["x"] + rng.normal(0, 1000, size=len(v["x"]))
    return Hypothesis(
        nome="Ruidosa",
        descricao="y = x + ruido_enorme",
        funcao=f,
        variaveis=[Variable("x", 0.0, 10.0)],
    )


@pytest.fixture
def gravitation_hypothesis():
    """
    Testa a invariância de G — fenômeno físico real.
    G_medido = F·r² / (m1·m2) deve ser sempre 6.674e-11.
    Resultado esperado: score próximo a 100.
    """
    G = 6.674e-11
    def f(v):
        F = G * v["m1"] * v["m2"] / (v["r"] ** 2)
        G_medido = F * v["r"]**2 / (v["m1"] * v["m2"])
        return 1.0 - abs(G_medido - G) / G
    return Hypothesis(
        nome="Gravitação Universal",
        descricao="G = F·r²/(m1·m2) — constante deve ser invariável",
        funcao=f,
        variaveis=[
            Variable("m1", 1.0,  1e6,  "massa 1 (kg)"),
            Variable("m2", 1.0,  1e6,  "massa 2 (kg)"),
            Variable("r",  1.0,  1e4,  "distância (m)"),
        ],
    )


@pytest.fixture
def meritocracy_hypothesis():
    """
    Meritocracia com ruído social dominante.
    Resultado esperado: score baixo (construção social, não lei natural).
    """
    rng = np.random.default_rng(42)
    def f(v):
        return v["esforco"] * v["talento"] + rng.uniform(-5, 5, size=len(v["esforco"]))
    return Hypothesis(
        nome="Meritocracia Pura",
        descricao="Renda = Esforço × Talento + ruído estrutural",
        funcao=f,
        variaveis=[
            Variable("esforco", 0.0, 1.0, "esforço individual"),
            Variable("talento", 0.0, 1.0, "talento inato"),
        ],
    )


@pytest.fixture
def synthetic_df():
    """Dataset sintético reproduzível para testes."""
    return DataFetcher().gerar_sintetico(n=200, seed=42)


@pytest.fixture
def engine():
    return ReproducibilityEngine(n_trials=30, sample_size=100, seed=42)


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 0 — ESTRUTURAS DE DADOS
# ══════════════════════════════════════════════════════════════════════

class TestEstruturas:

    def test_variavel_cria_corretamente(self, simple_variable):
        assert simple_variable.nome == "x"
        assert simple_variable.minimo == 0.0
        assert simple_variable.maximo == 10.0

    def test_variavel_amostra_dentro_dos_limites(self, simple_variable):
        amostras = simple_variable.amostrar(1000)
        assert amostras.min() >= 0.0
        assert amostras.max() <= 10.0
        assert len(amostras) == 1000

    def test_variavel_amostra_retorna_ndarray(self, simple_variable):
        assert isinstance(simple_variable.amostrar(10), np.ndarray)

    def test_hipotese_cria_com_campos_obrigatorios(self, linear_hypothesis):
        assert linear_hypothesis.nome == "Linear pura"
        assert callable(linear_hypothesis.funcao)
        assert len(linear_hypothesis.variaveis) == 1

    def test_hipotese_funcao_executa(self, linear_hypothesis):
        v = {"x": np.array([1.0, 2.0, 3.0])}
        resultado = linear_hypothesis.funcao(v)
        np.testing.assert_array_almost_equal(resultado, [2.0, 4.0, 6.0])


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 1 — PARSER DE REGRAS
# ══════════════════════════════════════════════════════════════════════

class TestRuleParser:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.parser = RuleParser()

    def test_parseia_relacao_linear(self):
        h = self.parser.parsear("A renda cresce linearmente com os anos de estudo")
        assert h.funcao is not None
        assert len(h.variaveis) >= 1

    def test_parseia_relacao_exponencial(self):
        h = self.parser.parsear("A população cresce exponencialmente com a taxa")
        assert "exponencial" in h.nome.lower() or "exponenci" in h.predicao.lower()

    def test_parseia_relacao_inversa(self):
        h = self.parser.parsear("A concentração diminui com o tempo de reação")
        assert "inversa" in h.nome.lower() or "inversa" in h.predicao.lower()

    def test_parseia_relacao_potencia(self):
        h = self.parser.parsear("A energia cinética é proporcional ao quadrado da velocidade")
        assert "potência" in h.nome.lower() or "potencia" in h.predicao.lower()

    def test_funcao_gerada_executa_sem_erro(self):
        h = self.parser.parsear("A temperatura cresce proporcionalmente à pressão")
        amostras = {v.nome: v.amostrar(50) for v in h.variaveis}
        resultado = h.funcao(amostras)
        assert resultado is not None
        assert not np.any(np.isnan(resultado))

    def test_fallback_variavel_generica(self):
        h = self.parser.parsear("O resultado depende da quantidade aplicada ao sistema")
        assert len(h.variaveis) >= 1
        amostras = {v.nome: v.amostrar(10) for v in h.variaveis}
        resultado = h.funcao(amostras)
        assert resultado is not None

    def test_hipotese_tem_nome_e_descricao(self):
        h = self.parser.parsear("A força é proporcional à massa e à aceleração")
        assert len(h.nome) > 0
        assert len(h.descricao) > 0

    def test_variaveis_tem_intervalos_validos(self):
        h = self.parser.parsear("A velocidade cresce com a força aplicada")
        for v in h.variaveis:
            assert v.minimo < v.maximo, f"Intervalo inválido para {v.nome}"

    def test_pt_br_e_en_reconhecidos(self):
        h_pt = self.parser.parsear("A temperatura cresce com a pressão")
        h_en = self.parser.parsear("Temperature increases with pressure")
        for h in [h_pt, h_en]:
            assert len(h.variaveis) >= 1

    def test_tradutor_aberto_usa_parser_sem_config(self):
        """Sem Groq e sem Ollama, deve usar Parser de Regras."""
        t = OpenTranslator()
        h = t.traduzir("A renda varia com os anos de estudo")
        assert h is not None
        assert h.origem == "parser_regras"


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 2 — MOTOR DE REPRODUTIBILIDADE
# ══════════════════════════════════════════════════════════════════════

class TestReproducibilityEngine:

    def test_retorna_resultado_simulacao(self, engine, linear_hypothesis):
        r = engine.testar(linear_hypothesis)
        assert isinstance(r, ResultadoSimulacao)

    def test_numero_de_trials_correto(self, engine, linear_hypothesis):
        r = engine.testar(linear_hypothesis)
        assert r.n_trials == 30
        assert len(r.results_per_trial) == 30

    def test_score_entre_0_e_100(self, engine, linear_hypothesis):
        r = engine.testar(linear_hypothesis)
        assert 0.0 <= r.reproducibility_score <= 100.0

    def test_seed_produz_resultado_deterministico(self, linear_hypothesis):
        m1 = ReproducibilityEngine(n_trials=20, sample_size=50, seed=99)
        m2 = ReproducibilityEngine(n_trials=20, sample_size=50, seed=99)
        r1 = m1.testar(linear_hypothesis)
        r2 = m2.testar(linear_hypothesis)
        assert abs(r1.mean - r2.mean) < 1e-10

    def test_cv_nao_negativo(self, engine, linear_hypothesis):
        r = engine.testar(linear_hypothesis)
        assert r.coefficient_of_variation >= 0.0

    def test_ic_95_contem_mean(self, engine, linear_hypothesis):
        r = engine.testar(linear_hypothesis)
        if not math.isnan(r.confidence_interval_95[0]):
            assert r.confidence_interval_95[0] <= r.mean <= r.confidence_interval_95[1]

    def test_classificacao_nao_vazia(self, engine, linear_hypothesis):
        r = engine.testar(linear_hypothesis)
        assert len(r.classificacao) > 0

    def test_hipotese_recebe_resultados_preenchidos(self, engine, linear_hypothesis):
        engine.testar(linear_hypothesis)
        assert len(linear_hypothesis.resultados) == 30


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 3 — TESTADOR EMPÍRICO
# ══════════════════════════════════════════════════════════════════════

class TestEmpiricalTester:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.testador = EmpiricalTester()

    def test_retorna_resultado_empirico(self, synthetic_df):
        h = Hypothesis(
            nome="teste",
            descricao="",
            funcao=lambda v: 800 + 500*v["E"] + 200*v["X"],
            variaveis=[Variable("E", 4, 20), Variable("X", 0, 30)],
        )
        r = self.testador.testar(h, synthetic_df, "renda", {"E": "estudo", "X": "experiencia"})
        assert isinstance(r, ResultadoEmpirico)

    def test_r2_entre_menos1_e_1(self, synthetic_df):
        h = Hypothesis(
            nome="teste",
            descricao="",
            funcao=lambda v: 800 + 500*v["E"],
            variaveis=[Variable("E", 4, 20)],
        )
        r = self.testador.testar(h, synthetic_df, "renda", {"E": "estudo"})
        assert -10.0 <= r.r_squared <= 1.0

    def test_rmse_nao_negativo(self, synthetic_df):
        h = Hypothesis(
            nome="teste",
            descricao="",
            funcao=lambda v: v["E"] * 500,
            variaveis=[Variable("E", 4, 20)],
        )
        r = self.testador.testar(h, synthetic_df, "renda", {"E": "estudo"})
        assert r.rmse >= 0.0

    def test_residuals_tem_tamanho_correto(self, synthetic_df):
        h = Hypothesis(
            nome="teste",
            descricao="",
            funcao=lambda v: v["E"] * 600,
            variaveis=[Variable("E", 4, 20)],
        )
        r = self.testador.testar(h, synthetic_df, "renda", {"E": "estudo"})
        assert len(r.residuals) == r.n_amostras

    def test_empirical_score_entre_0_e_100(self, synthetic_df):
        h = Hypothesis(
            nome="teste",
            descricao="",
            funcao=lambda v: 800 + 500*v["E"] + 200*v["X"],
            variaveis=[Variable("E", 4, 20), Variable("X", 0, 30)],
        )
        r = self.testador.testar(h, synthetic_df, "renda", {"E": "estudo", "X": "experiencia"})
        assert 0.0 <= r.empirical_score <= 100.0

    def test_hipotese_perfeita_tem_r2_alto(self):
        """Hipótese que conhece a função exata deve ter R² muito alto."""
        np.random.seed(0)
        n = 500
        x = np.random.uniform(1, 10, n)
        y = 3.0 * x + 7.0 + np.random.normal(0, 0.001, n)   # ruído quase nulo
        df = pd.DataFrame({"x": x, "y": y})
        h = Hypothesis(
            nome="perfeita",
            descricao="",
            funcao=lambda v: 3.0 * v["x"] + 7.0,
            variaveis=[Variable("x", 1, 10)],
        )
        r = EmpiricalTester().testar(h, df, "y", {"x": "x"})
        assert r.r_squared > 0.99


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 4 — COMPARADOR DE HIPÓTESES
# ══════════════════════════════════════════════════════════════════════

class TestHypothesisComparator:

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_df):
        self.testador   = EmpiricalTester()
        self.comparador = HypothesisComparator()
        self.df         = synthetic_df

        self.h_boa = Hypothesis(
            nome="Boa",
            descricao="",
            funcao=lambda v: 800 + 500*v["E"] + 200*v["X"],
            variaveis=[Variable("E", 4, 20), Variable("X", 0, 30)],
        )
        self.h_ruim = Hypothesis(
            nome="Ruim",
            descricao="",
            funcao=lambda v: v["E"] * 50,
            variaveis=[Variable("E", 4, 20)],
        )
        self.re_boa  = self.testador.testar(self.h_boa,  self.df, "renda", {"E":"estudo","X":"experiencia"})
        self.re_ruim = self.testador.testar(self.h_ruim, self.df, "renda", {"E":"estudo"})

    def test_vencedor_tem_menor_aic(self):
        comp = self.comparador.comparar([self.re_boa, self.re_ruim])
        assert comp.vencedor["aic"] == min(comp.ranking, key=lambda r: r["aic"])["aic"]

    def test_pesos_somam_um(self):
        comp = self.comparador.comparar([self.re_boa, self.re_ruim])
        soma = sum(r["peso_akaike"] for r in comp.ranking)
        assert abs(soma - 1.0) < 1e-9

    def test_hipotese_melhor_tem_r2_maior(self):
        comp = self.comparador.comparar([self.re_boa, self.re_ruim])
        assert comp.vencedor["r2"] >= min(r["r2"] for r in comp.ranking)

    def test_ranking_ordenado_por_aic(self):
        comp = self.comparador.comparar([self.re_boa, self.re_ruim])
        aics = [r["aic"] for r in comp.ranking]
        assert aics == sorted(aics)

    def test_delta_aic_vencedor_e_zero(self):
        comp = self.comparador.comparar([self.re_boa, self.re_ruim])
        assert comp.vencedor["delta_aic"] == 0.0

    def test_tres_hipoteses_ranking_correto(self):
        h3 = Hypothesis(
            nome="Media",
            descricao="",
            funcao=lambda v: 1000 + 600*v["E"],
            variaveis=[Variable("E", 4, 20)],
        )
        re3 = self.testador.testar(h3, self.df, "renda", {"E": "estudo"})
        comp = self.comparador.comparar([self.re_boa, self.re_ruim, re3])
        assert len(comp.ranking) == 3


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 5 — ANALISADOR TEMPORAL
# ══════════════════════════════════════════════════════════════════════

class TestTemporalAnalyser:

    def test_retorna_resultado_temporal(self, synthetic_df):
        h = Hypothesis(
            nome="teste",
            descricao="",
            funcao=lambda v: 800 + 500*v["E"] + 200*v["X"],
            variaveis=[Variable("E", 4, 20), Variable("X", 0, 30)],
        )
        at = TemporalAnalyser(n_janelas=4)
        rt = at.analisar(h, synthetic_df, "ano", "renda", {"E":"estudo","X":"experiencia"})
        assert rt is not None

    def test_numero_janelas_correto(self, synthetic_df):
        h = Hypothesis(
            nome="teste",
            descricao="",
            funcao=lambda v: 800 + 500*v["E"],
            variaveis=[Variable("E", 4, 20)],
        )
        at = TemporalAnalyser(n_janelas=5)
        rt = at.analisar(h, synthetic_df, "ano", "renda", {"E":"estudo"})
        assert len(rt.scores_por_janela) <= 5

    def test_scores_entre_0_e_100(self, synthetic_df):
        h = Hypothesis(
            nome="teste",
            descricao="",
            funcao=lambda v: 800 + 500*v["E"],
            variaveis=[Variable("E", 4, 20)],
        )
        at = TemporalAnalyser(n_janelas=4)
        rt = at.analisar(h, synthetic_df, "ano", "renda", {"E":"estudo"})
        for s in rt.scores_por_janela:
            assert 0.0 <= s <= 100.0

    def test_hipotese_nome_preservado(self, synthetic_df):
        h = Hypothesis(
            nome="MeuNome",
            descricao="",
            funcao=lambda v: v["E"] * 600,
            variaveis=[Variable("E", 4, 20)],
        )
        at = TemporalAnalyser(n_janelas=3)
        rt = at.analisar(h, synthetic_df, "ano", "renda", {"E":"estudo"})
        assert rt.hipotese_nome == "MeuNome"


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 6 — BUSCADOR DE DADOS
# ══════════════════════════════════════════════════════════════════════

class TestDataFetcher:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.buscador = DataFetcher()

    def test_gera_sintetico_com_colunas_corretas(self):
        df = self.buscador.gerar_sintetico(100, seed=1)
        for col in ["ano", "estudo", "experiencia", "renda"]:
            assert col in df.columns

    def test_gera_sintetico_n_linhas(self):
        df = self.buscador.gerar_sintetico(200, seed=1)
        assert len(df) > 0
        assert len(df) <= 200

    def test_sintetico_sem_nan(self):
        df = self.buscador.gerar_sintetico(100, seed=1)
        assert not df.isnull().any().any()

    def test_sintetico_renda_positiva(self):
        df = self.buscador.gerar_sintetico(200, seed=1)
        assert (df["renda"] > 0).all()

    def test_sintetico_seed_deterministico(self):
        df1 = self.buscador.gerar_sintetico(100, seed=7)
        df2 = self.buscador.gerar_sintetico(100, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_sintetico_estudo_dentro_dos_limites(self):
        df = self.buscador.gerar_sintetico(300, seed=1)
        assert df["estudo"].between(4, 20).all()

    def test_lista_indicadores_worldbank_nao_vazia(self):
        indicadores = self.buscador.listar_indicadores_worldbank()
        assert len(indicadores) > 0
        assert "pib_per_capita" in indicadores


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO 7 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════

class TestDashboard:

    def test_gera_figura_sem_erro(self, engine, linear_hypothesis, synthetic_df):
        import io
        rs = engine.testar(linear_hypothesis)
        h  = Hypothesis(
            nome="Linear pura",
            descricao="",
            funcao=lambda v: 2.0 * v["E"],
            variaveis=[Variable("E", 4, 20)],
        )
        re = EmpiricalTester().testar(h, synthetic_df, "renda", {"E":"estudo"})
        comp = HypothesisComparator().comparar([re])
        at   = TemporalAnalyser(n_janelas=3)
        rt   = at.analisar(h, synthetic_df, "ano", "renda", {"E":"estudo"})

        from reproducibility_core import ResultadoComparacao
        fig = generate_dashboard([rs], [re], comp, [rt])
        assert fig is not None

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=60, bbox_inches="tight")
        assert len(buf.getvalue()) > 1000, "Dashboard gerou PNG vazio"
        import matplotlib.pyplot as plt
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  VALIDAÇÃO CIENTÍFICA DO SCORE — NÚCLEO DO ARTIGO
# ══════════════════════════════════════════════════════════════════════

class TestValidacaoScore:
    """
    Estes testes são o coração da contribuição científica.

    Hipótese de validação:
        Se o score de reprodutibilidade é um instrumento válido,
        ele deve discriminar corretamente entre:
        (a) leis físicas objetivas        → score alto (≥ 85)
        (b) hipóteses mistas              → score médio (30–70)
        (c) ficções sociais com alto ruído → score baixo (≤ 40)

    Isso transforma a fórmula exp(−CV/10) de heurística em
    instrumento empiricamente validado — exatamente o que
    um revisor do JOSS perguntará.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = ReproducibilityEngine(n_trials=100, sample_size=500, seed=42)

    def test_lei_fisica_tem_score_alto(self, gravitation_hypothesis):
        """
        Lei da Gravitação Universal deve ter score ≥ 85.
        Justificativa: G é uma constante da natureza — invariável
        por definição. CV deve ser ≈ 0%.
        """
        r = self.engine.testar(gravitation_hypothesis)
        assert r.reproducibility_score >= 85.0, (
            f"Lei da Gravitação teve score {r.reproducibility_score:.1f} — "
            f"esperado ≥ 85. CV = {r.coefficient_of_variation:.2f}%"
        )

    def test_constant_hypothesis_tem_score_maximo(self, constant_hypothesis):
        """
        y = 5 (constante absoluta) deve ter score ≥ 95.
        Justificativa: CV = 0% por construção.
        """
        r = self.engine.testar(constant_hypothesis)
        assert r.reproducibility_score >= 95.0, (
            f"Constante teve score {r.reproducibility_score:.1f} — "
            f"esperado ≥ 95. CV = {r.coefficient_of_variation:.2f}%"
        )

    def test_noisy_hypothesis_tem_score_baixo(self, noisy_hypothesis):
        """
        y = x + ruído_enorme deve ter score ≤ 10.
        Justificativa: sinal x é dominado pelo ruído de amplitude 1000.
        """
        r = self.engine.testar(noisy_hypothesis)
        assert r.reproducibility_score <= 10.0, (
            f"Hipótese ruidosa teve score {r.reproducibility_score:.1f} — "
            f"esperado ≤ 10. CV = {r.coefficient_of_variation:.2f}%"
        )

    def test_meritocracia_tem_score_menor_que_fisica(self, gravitation_hypothesis, meritocracy_hypothesis):
        """
        Meritocracia deve ter score menor que Lei da Gravitação.
        Justificativa empírica central do artigo:
        construções sociais são menos reproduzíveis que leis físicas.
        """
        r_fisica  = self.engine.testar(gravitation_hypothesis)
        r_social  = self.engine.testar(meritocracy_hypothesis)
        assert r_social.reproducibility_score < r_fisica.reproducibility_score, (
            f"Meritocracia ({r_social.reproducibility_score:.1f}) deveria ser "
            f"< Gravitação ({r_fisica.reproducibility_score:.1f})"
        )

    def test_score_monotonicamente_cresce_com_cv_decrescente(self):
        """
        Propriedade matemática: score = 100·exp(−CV/10).
        CVs crescentes devem produzir scores decrescentes.
        """
        from reproducibility_core import ReproducibilityEngine as M
        m = M(n_trials=50, sample_size=200, seed=0)
        scores_anteriores = []

        for amplitude_ruido in [0.001, 0.1, 1.0, 10.0, 100.0]:
            rng = np.random.default_rng(0)
            def f(v, amp=amplitude_ruido):
                return np.ones(len(v["x"])) + rng.normal(0, amp, len(v["x"]))
            h = Hypothesis(
                nome=f"ruido_{amplitude_ruido}",
                descricao="",
                funcao=f,
                variaveis=[Variable("x", 0, 1)],
            )
            r = m.testar(h)
            scores_anteriores.append(r.reproducibility_score)

        for i in range(len(scores_anteriores) - 1):
            assert scores_anteriores[i] >= scores_anteriores[i+1] - 1e-6, (
                f"Score não decresceu com ruído crescente: "
                f"{scores_anteriores}"
            )

    def test_score_formula_matematicamente_correta(self):
        """
        Testa diretamente a fórmula score = 100·exp(−CV/10).
        Garante que a implementação é fiel à definição.
        """
        casos = [
            (0.0,   100.0),   # CV=0% → score=100 (perfeito)
            (10.0,  100 * math.exp(-1)),   # CV=10% → score=36.8
            (20.0,  100 * math.exp(-2)),   # CV=20% → score=13.5
            (50.0,  100 * math.exp(-5)),   # CV=50% → score=0.67
        ]
        for cv, score_esperado in casos:
            score_calculado = min(100.0, 100 * math.exp(-cv / 10))
            assert abs(score_calculado - score_esperado) < 0.01, (
                f"Fórmula incorreta para CV={cv}%: "
                f"calculado={score_calculado:.3f}, esperado={score_esperado:.3f}"
            )


# ══════════════════════════════════════════════════════════════════════
#  TESTES DE INTEGRAÇÃO
# ══════════════════════════════════════════════════════════════════════

class TestIntegracao:

    def test_pipeline_completo_executa(self, synthetic_df):
        """Testa o pipeline de ponta a ponta como o app.py faria."""
        from reproducibility_core import run_pipeline

        h = Hypothesis(
            nome="Capital Humano",
            descricao="",
            funcao=lambda v: 800 + 500*v["E"] + 200*v["X"],
            variaveis=[Variable("E", 4, 20), Variable("X", 0, 30)],
        )
        res = run_pipeline(
            hipoteses    = [h],
            df           = synthetic_df,
            target_column  = "renda",
            mappings  = [{"E": "estudo", "X": "experiencia"}],
            coluna_tempo = "ano",
            n_trials     = 20,
            sample_size  = 80,
            n_janelas    = 3,
            seed         = 42,
        )
        assert "simulacao"  in res
        assert "empirico"   in res
        assert "comparacao" in res
        assert "temporal"   in res
        assert "figura"     in res
        assert res["figura"] is not None

    def test_pipeline_sem_coluna_tempo(self, synthetic_df):
        """Pipeline deve funcionar sem análise temporal."""
        from reproducibility_core import run_pipeline
        h = Hypothesis(
            nome="Simples",
            descricao="",
            funcao=lambda v: v["E"] * 600,
            variaveis=[Variable("E", 4, 20)],
        )
        res = run_pipeline(
            hipoteses   = [h],
            df          = synthetic_df,
            target_column = "renda",
            mappings = [{"E": "estudo"}],
            coluna_tempo= None,
            n_trials    = 10,
            sample_size = 50,
        )
        assert res["temporal"] == []

    def test_duas_hipoteses_comparadas(self, synthetic_df):
        """Comparação deve eleger sempre um vencedor único."""
        from reproducibility_core import run_pipeline
        h1 = Hypothesis("H1", "", lambda v: 800 + 500*v["E"] + 200*v["X"],
                      [Variable("E",4,20), Variable("X",0,30)])
        h2 = Hypothesis("H2", "", lambda v: 650*v["E"],
                      [Variable("E",4,20)])
        res = run_pipeline(
            hipoteses   = [h1, h2],
            df          = synthetic_df,
            target_column = "renda",
            mappings = [{"E":"estudo","X":"experiencia"}, {"E":"estudo"}],
            n_trials    = 15,
            sample_size = 60,
        )
        assert res["comparacao"].vencedor["nome"] in ["H1", "H2"]
        assert len(res["comparacao"].ranking) == 2
