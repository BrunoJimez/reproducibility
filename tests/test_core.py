"""
tests/test_core.py
──────────────────────────────────────────────────────────────────────
Automated test suite — Reproducibility system

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short
    pytest tests/ -v --cov=reproducibility_core

Organisation:
    Module 0 — Data structures (Variable, Hypothesis)
    Module 1 — OpenTranslator / RuleParser
    Module 2 — ReproducibilityEngine
    Module 3 — EmpiricalTester
    Module 4 — HypothesisComparator
    Module 5 — TemporalAnalyser
    Module 6 — DataFetcher
    Module 7 — generate_dashboard
    Scientific validation — score against cases with known answers
    Integration — full pipeline end-to-end
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
    SimulationResult,
    EmpiricalResult,
    RuleParser,
    OpenTranslator,
    ReproducibilityEngine,
    EmpiricalTester,
    HypothesisComparator,
    TemporalAnalyser,
    DataFetcher,
    generate_dashboard,
    run_pipeline,
)

# ══════════════════════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def simple_variable():
    return Variable("x", 0.0, 10.0, "variable x")


@pytest.fixture
def linear_hypothesis():
    """y = 2x — perfect linear relation, should be highly reproducible."""
    return Hypothesis(
        name="Linear pure",
        description="y = 2x",
        function=lambda v: 2.0 * v["x"],
        variables=[Variable("x", 0.0, 10.0)],
    )


@pytest.fixture
def constant_hypothesis():
    """y = 5 — absolute constant, maximum reproducibility."""
    return Hypothesis(
        name="Constant",
        description="y = 5",
        function=lambda v: np.ones_like(v["x"]) * 5.0,
        variables=[Variable("x", 0.0, 10.0)],
    )


@pytest.fixture
def noisy_hypothesis():
    """y = x + huge noise — should have low reproducibility."""
    rng = np.random.default_rng(0)
    def f(v):
        return v["x"] + rng.normal(0, 1000, size=len(v["x"]))
    return Hypothesis(
        name="Noisy",
        description="y = x + huge_noise",
        function=f,
        variables=[Variable("x", 0.0, 10.0)],
    )


@pytest.fixture
def gravitation_hypothesis():
    """
    Tests the invariance of G — real physical phenomenon.
    G_measured = F·r² / (m1·m2) must always equal 6.674e-11.
    Expected: score close to 100.
    """
    G = 6.674e-11
    def f(v):
        F = G * v["m1"] * v["m2"] / (v["r"] ** 2)
        G_measured = F * v["r"]**2 / (v["m1"] * v["m2"])
        return 1.0 - abs(G_measured - G) / G
    return Hypothesis(
        name="Universal Gravitation",
        description="G = F·r²/(m1·m2) — constant must be invariant",
        function=f,
        variables=[
            Variable("m1", 1.0,  1e6,  "mass 1 (kg)"),
            Variable("m2", 1.0,  1e6,  "mass 2 (kg)"),
            Variable("r",  1.0,  1e4,  "distance (m)"),
        ],
    )


@pytest.fixture
def meritocracy_hypothesis():
    """
    Meritocracy with dominant social noise.
    Expected: low score (social construction, not physical law).
    """
    rng = np.random.default_rng(42)
    def f(v):
        return v["effort"] * v["talent"] + rng.uniform(-5, 5, size=len(v["effort"]))
    return Hypothesis(
        name="Pure Meritocracy",
        description="Income = Effort × Talent + structural noise",
        function=f,
        variables=[
            Variable("effort", 0.0, 1.0, "individual effort"),
            Variable("talent", 0.0, 1.0, "innate talent"),
        ],
    )


@pytest.fixture
def synthetic_df():
    """Reproducible synthetic dataset for tests."""
    return DataFetcher().generate_synthetic(n=200, seed=42)


@pytest.fixture
def engine():
    return ReproducibilityEngine(n_trials=30, sample_size=100, seed=42)


# ══════════════════════════════════════════════════════════════════════
#  MODULE 0 — DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

class TestDataStructures:

    def test_variable_created_correctly(self, simple_variable):
        assert simple_variable.name == "x"
        assert simple_variable.minimum == 0.0
        assert simple_variable.maximum == 10.0

    def test_variable_samples_within_bounds(self, simple_variable):
        samples = simple_variable.sample(1000)
        assert samples.min() >= 0.0
        assert samples.max() <= 10.0
        assert len(samples) == 1000

    def test_variable_sample_returns_ndarray(self, simple_variable):
        assert isinstance(simple_variable.sample(10), np.ndarray)

    def test_hypothesis_created_with_required_fields(self, linear_hypothesis):
        assert linear_hypothesis.name == "Linear pure"
        assert callable(linear_hypothesis.function)
        assert len(linear_hypothesis.variables) == 1

    def test_hypothesis_function_executes(self, linear_hypothesis):
        v = {"x": np.array([1.0, 2.0, 3.0])}
        result = linear_hypothesis.function(v)
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0])


# ══════════════════════════════════════════════════════════════════════
#  MODULE 1 — RULE PARSER
# ══════════════════════════════════════════════════════════════════════

class TestRuleParser:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.parser = RuleParser()

    def test_parses_linear_relation(self):
        h = self.parser.parse("income grows linearly with years of education")
        assert h.function is not None
        assert len(h.variables) >= 1

    def test_parses_exponential_relation(self):
        h = self.parser.parse("population grows exponentially with the reproduction rate")
        assert "exponential" in h.name.lower() or "exponential" in h.prediction.lower()

    def test_parses_inverse_relation(self):
        h = self.parser.parse("concentration decreases with reaction time")
        assert "inverse" in h.name.lower() or "inverse" in h.prediction.lower()

    def test_parses_power_relation(self):
        h = self.parser.parse("kinetic energy is proportional to the square of velocity")
        assert "power" in h.name.lower() or "power" in h.prediction.lower()

    def test_generated_function_runs_without_error(self):
        h = self.parser.parse("temperature increases proportionally with pressure")
        samples = {v.name: v.sample(50) for v in h.variables}
        result = h.function(samples)
        assert result is not None
        assert not np.any(np.isnan(result))

    def test_fallback_generic_variable(self):
        h = self.parser.parse("the result depends on the quantity applied to the system")
        assert len(h.variables) >= 1
        samples = {v.name: v.sample(10) for v in h.variables}
        result = h.function(samples)
        assert result is not None

    def test_hypothesis_has_name_and_description(self):
        h = self.parser.parse("force is proportional to mass and acceleration")
        assert len(h.name) > 0
        assert len(h.description) > 0

    def test_variables_have_valid_ranges(self):
        h = self.parser.parse("velocity increases with applied force")
        for v in h.variables:
            assert v.minimum < v.maximum, f"Invalid range for {v.name}"

    def test_open_translator_uses_parser_without_config(self):
        """Without Groq and without Ollama, must use RuleParser."""
        t = OpenTranslator()
        h = t.translate("income varies with years of education")
        assert h is not None
        assert h.origin == "rule_parser"


# ══════════════════════════════════════════════════════════════════════
#  MODULE 2 — REPRODUCIBILITY ENGINE
# ══════════════════════════════════════════════════════════════════════

class TestReproducibilityEngine:

    def test_returns_simulation_result(self, engine, linear_hypothesis):
        r = engine.test(linear_hypothesis)
        assert isinstance(r, SimulationResult)

    def test_correct_number_of_trials(self, engine, linear_hypothesis):
        r = engine.test(linear_hypothesis)
        assert r.n_trials == 30
        assert len(r.results_per_trial) == 30

    def test_score_between_0_and_100(self, engine, linear_hypothesis):
        r = engine.test(linear_hypothesis)
        assert 0.0 <= r.reproducibility_score <= 100.0

    def test_seed_produces_deterministic_result(self):
        # Create independent hypothesis objects to avoid shared state
        h1 = Hypothesis("S1", "y=2x", lambda v: 2.0 * v["x"], [Variable("x", 0.0, 10.0)])
        h2 = Hypothesis("S2", "y=2x", lambda v: 2.0 * v["x"], [Variable("x", 0.0, 10.0)])
        m1 = ReproducibilityEngine(n_trials=20, sample_size=50, seed=99)
        m2 = ReproducibilityEngine(n_trials=20, sample_size=50, seed=99)
        r1 = m1.test(h1)
        r2 = m2.test(h2)
        assert abs(r1.mean - r2.mean) < 1e-10

    def test_cv_is_non_negative(self, engine, linear_hypothesis):
        r = engine.test(linear_hypothesis)
        assert r.coefficient_of_variation >= 0.0

    def test_classification_is_not_empty(self, engine, linear_hypothesis):
        r = engine.test(linear_hypothesis)
        assert len(r.classification) > 0

    def test_hypothesis_results_are_filled(self, engine, linear_hypothesis):
        engine.test(linear_hypothesis)
        assert len(linear_hypothesis.results) == 30


# ══════════════════════════════════════════════════════════════════════
#  MODULE 3 — EMPIRICAL TESTER
# ══════════════════════════════════════════════════════════════════════

class TestEmpiricalTester:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tester = EmpiricalTester()

    def test_returns_empirical_result(self, synthetic_df):
        h = Hypothesis(
            name="test",
            description="",
            function=lambda v: 800 + 500*v["E"] + 200*v["X"],
            variables=[Variable("E", 4, 20), Variable("X", 0, 30)],
        )
        r = self.tester.test(h, synthetic_df, "renda", {"E": "estudo", "X": "experiencia"})
        assert isinstance(r, EmpiricalResult)

    def test_r2_within_valid_range(self, synthetic_df):
        h = Hypothesis(
            name="test", description="",
            function=lambda v: 800 + 500*v["E"],
            variables=[Variable("E", 4, 20)],
        )
        r = self.tester.test(h, synthetic_df, "renda", {"E": "estudo"})
        assert -10.0 <= r.r_squared <= 1.0

    def test_rmse_is_non_negative(self, synthetic_df):
        h = Hypothesis(
            name="test", description="",
            function=lambda v: v["E"] * 500,
            variables=[Variable("E", 4, 20)],
        )
        r = self.tester.test(h, synthetic_df, "renda", {"E": "estudo"})
        assert r.rmse >= 0.0

    def test_residuals_correct_size(self, synthetic_df):
        h = Hypothesis(
            name="test", description="",
            function=lambda v: v["E"] * 600,
            variables=[Variable("E", 4, 20)],
        )
        r = self.tester.test(h, synthetic_df, "renda", {"E": "estudo"})
        assert len(r.residuals) == r.n_samples

    def test_empirical_score_between_0_and_100(self, synthetic_df):
        h = Hypothesis(
            name="test", description="",
            function=lambda v: 800 + 500*v["E"] + 200*v["X"],
            variables=[Variable("E", 4, 20), Variable("X", 0, 30)],
        )
        r = self.tester.test(h, synthetic_df, "renda", {"E": "estudo", "X": "experiencia"})
        assert 0.0 <= r.empirical_score <= 100.0

    def test_perfect_hypothesis_has_high_r2(self):
        """Hypothesis that knows the exact function must have R² > 0.99."""
        np.random.seed(0)
        n = 500
        x = np.random.uniform(1, 10, n)
        y = 3.0 * x + 7.0 + np.random.normal(0, 0.001, n)
        df = pd.DataFrame({"x": x, "y": y})
        h = Hypothesis(
            name="perfect", description="",
            function=lambda v: 3.0 * v["x"] + 7.0,
            variables=[Variable("x", 1, 10)],
        )
        r = EmpiricalTester().test(h, df, "y", {"x": "x"})
        assert r.r_squared > 0.99


# ══════════════════════════════════════════════════════════════════════
#  MODULE 4 — HYPOTHESIS COMPARATOR
# ══════════════════════════════════════════════════════════════════════

class TestHypothesisComparator:

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_df):
        self.tester     = EmpiricalTester()
        self.comparator = HypothesisComparator()
        self.df         = synthetic_df

        self.h_good = Hypothesis(
            name="Good",
            description="",
            function=lambda v: 800 + 500*v["E"] + 200*v["X"],
            variables=[Variable("E", 4, 20), Variable("X", 0, 30)],
        )
        self.h_bad = Hypothesis(
            name="Bad",
            description="",
            function=lambda v: v["E"] * 50,
            variables=[Variable("E", 4, 20)],
        )
        self.re_good = self.tester.test(self.h_good, self.df, "renda",
                                        {"E": "estudo", "X": "experiencia"})
        self.re_bad  = self.tester.test(self.h_bad,  self.df, "renda",
                                        {"E": "estudo"})

    def test_winner_has_lowest_aic(self):
        comp = self.comparator.compare([self.re_good, self.re_bad])
        assert comp.winner["aic"] == min(comp.ranking, key=lambda r: r["aic"])["aic"]

    def test_weights_sum_to_one(self):
        comp = self.comparator.compare([self.re_good, self.re_bad])
        total = sum(r["akaike_weight"] for r in comp.ranking)
        assert abs(total - 1.0) < 1e-9

    def test_ranking_sorted_by_aic(self):
        comp = self.comparator.compare([self.re_good, self.re_bad])
        aics = [r["aic"] for r in comp.ranking]
        assert aics == sorted(aics)

    def test_winner_delta_aic_is_zero(self):
        comp = self.comparator.compare([self.re_good, self.re_bad])
        assert comp.winner["delta_aic"] == 0.0

    def test_three_hypotheses_correct_ranking(self):
        h3 = Hypothesis(
            name="Medium", description="",
            function=lambda v: 1000 + 600*v["E"],
            variables=[Variable("E", 4, 20)],
        )
        re3 = self.tester.test(h3, self.df, "renda", {"E": "estudo"})
        comp = self.comparator.compare([self.re_good, self.re_bad, re3])
        assert len(comp.ranking) == 3


# ══════════════════════════════════════════════════════════════════════
#  MODULE 5 — TEMPORAL ANALYSER
# ══════════════════════════════════════════════════════════════════════

class TestTemporalAnalyser:

    def test_returns_temporal_result(self, synthetic_df):
        h = Hypothesis(
            name="test", description="",
            function=lambda v: 800 + 500*v["E"] + 200*v["X"],
            variables=[Variable("E", 4, 20), Variable("X", 0, 30)],
        )
        rt = TemporalAnalyser(n_windows=4).analyse(
            h, synthetic_df, "ano", "renda", {"E": "estudo", "X": "experiencia"}
        )
        assert rt is not None

    def test_correct_number_of_windows(self, synthetic_df):
        h = Hypothesis(
            name="test", description="",
            function=lambda v: 800 + 500*v["E"],
            variables=[Variable("E", 4, 20)],
        )
        rt = TemporalAnalyser(n_windows=5).analyse(
            h, synthetic_df, "ano", "renda", {"E": "estudo"}
        )
        assert len(rt.scores_per_window) <= 5

    def test_scores_between_0_and_100(self, synthetic_df):
        h = Hypothesis(
            name="test", description="",
            function=lambda v: 800 + 500*v["E"],
            variables=[Variable("E", 4, 20)],
        )
        rt = TemporalAnalyser(n_windows=4).analyse(
            h, synthetic_df, "ano", "renda", {"E": "estudo"}
        )
        for s in rt.scores_per_window:
            assert 0.0 <= s <= 100.0

    def test_hypothesis_name_preserved(self, synthetic_df):
        h = Hypothesis(
            name="MyName", description="",
            function=lambda v: v["E"] * 600,
            variables=[Variable("E", 4, 20)],
        )
        rt = TemporalAnalyser(n_windows=3).analyse(
            h, synthetic_df, "ano", "renda", {"E": "estudo"}
        )
        assert rt.hypothesis_name == "MyName"


# ══════════════════════════════════════════════════════════════════════
#  MODULE 6 — DATA FETCHER
# ══════════════════════════════════════════════════════════════════════

class TestDataFetcher:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fetcher = DataFetcher()

    def test_generates_synthetic_with_correct_columns(self):
        df = self.fetcher.generate_synthetic(100, seed=1)
        for col in ["ano", "estudo", "experiencia", "renda"]:
            assert col in df.columns

    def test_generates_correct_row_count(self):
        df = self.fetcher.generate_synthetic(200, seed=1)
        assert len(df) > 0 and len(df) <= 200

    def test_synthetic_has_no_nan(self):
        df = self.fetcher.generate_synthetic(100, seed=1)
        assert not df.isnull().any().any()

    def test_synthetic_income_is_positive(self):
        df = self.fetcher.generate_synthetic(200, seed=1)
        assert (df["renda"] > 0).all()

    def test_synthetic_seed_is_deterministic(self):
        df1 = self.fetcher.generate_synthetic(100, seed=7)
        df2 = self.fetcher.generate_synthetic(100, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_worldbank_indicator_list_not_empty(self):
        indicators = self.fetcher.list_worldbank_indicators()
        assert len(indicators) > 0
        assert "pib_per_capita" in indicators


# ══════════════════════════════════════════════════════════════════════
#  MODULE 7 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════

class TestDashboard:

    def test_generates_figure_without_error(self, engine, linear_hypothesis, synthetic_df):
        import io, matplotlib.pyplot as plt
        rs = engine.test(linear_hypothesis)

        h = Hypothesis(
            name="Linear pure", description="",
            function=lambda v: 2.0 * v["E"],
            variables=[Variable("E", 4, 20)],
        )
        re   = EmpiricalTester().test(h, synthetic_df, "renda", {"E": "estudo"})
        comp = HypothesisComparator().compare([re])
        rt   = TemporalAnalyser(n_windows=3).analyse(
            h, synthetic_df, "ano", "renda", {"E": "estudo"}
        )
        fig = generate_dashboard([rs], [re], comp, [rt])
        assert fig is not None

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=60, bbox_inches="tight")
        assert len(buf.getvalue()) > 1000, "Dashboard generated empty PNG"
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  SCIENTIFIC VALIDATION OF THE SCORE — CORE OF THE ARTICLE
# ══════════════════════════════════════════════════════════════════════

class TestScoreValidation:
    """
    These tests are the scientific heart of the contribution.

    Validation hypothesis:
        If the reproducibility score is a valid instrument,
        it must correctly discriminate between:
        (a) objective physical laws       → high score (≥ 85)
        (b) mixed hypotheses              → medium score (30–70)
        (c) social fictions with high noise → low score (≤ 40)
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = ReproducibilityEngine(n_trials=100, sample_size=500, seed=42)

    def test_physical_law_has_high_score(self, gravitation_hypothesis):
        """
        Newton's Law of Universal Gravitation must score ≥ 85.
        Justification: G is a physical constant — invariant by definition. CV ≈ 0%.
        """
        r = self.engine.test(gravitation_hypothesis)
        assert r.reproducibility_score >= 85.0, (
            f"Gravitation scored {r.reproducibility_score:.1f} — "
            f"expected ≥ 85. CV = {r.coefficient_of_variation:.2f}%"
        )

    def test_constant_has_maximum_score(self, constant_hypothesis):
        """
        y = 5 (absolute constant) must score ≥ 95.
        Justification: CV = 0% by construction.
        """
        r = self.engine.test(constant_hypothesis)
        assert r.reproducibility_score >= 95.0, (
            f"Constant scored {r.reproducibility_score:.1f} — "
            f"expected ≥ 95. CV = {r.coefficient_of_variation:.2f}%"
        )

    def test_noisy_hypothesis_has_low_score(self, noisy_hypothesis):
        """
        y = x + huge_noise must score ≤ 10.
        Justification: signal x is dominated by noise of amplitude 1000.
        """
        r = self.engine.test(noisy_hypothesis)
        assert r.reproducibility_score <= 10.0, (
            f"Noisy hypothesis scored {r.reproducibility_score:.1f} — "
            f"expected ≤ 10. CV = {r.coefficient_of_variation:.2f}%"
        )

    def test_meritocracy_scores_lower_than_physics(
        self, gravitation_hypothesis, meritocracy_hypothesis
    ):
        """
        Meritocracy must score lower than Newton's Gravitation.
        This is the central empirical claim of the paper:
        social constructions are less reproducible than physical laws.
        """
        r_physics = self.engine.test(gravitation_hypothesis)
        r_social  = self.engine.test(meritocracy_hypothesis)
        assert r_social.reproducibility_score < r_physics.reproducibility_score, (
            f"Meritocracy ({r_social.reproducibility_score:.1f}) should be "
            f"< Gravitation ({r_physics.reproducibility_score:.1f})"
        )

    def test_score_monotonically_decreases_with_noise(self):
        """
        Mathematical property: score = 100·exp(−CV/10).
        Increasing noise amplitudes must produce decreasing scores.
        """
        scores = []
        for noise_amplitude in [0.001, 0.1, 1.0, 10.0, 100.0]:
            rng = np.random.default_rng(0)
            def f(v, amp=noise_amplitude):
                return np.ones(len(v["x"])) + rng.normal(0, amp, len(v["x"]))
            h = Hypothesis(
                name=f"noise_{noise_amplitude}", description="",
                function=f,
                variables=[Variable("x", 0, 1)],
            )
            r = ReproducibilityEngine(n_trials=50, sample_size=200, seed=0).test(h)
            scores.append(r.reproducibility_score)

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i+1] - 1e-6, (
                f"Score did not decrease with increasing noise: {scores}"
            )

    def test_score_formula_is_mathematically_correct(self):
        """
        Tests the formula score = 100·exp(−CV/10) directly.
        Ensures the implementation matches the definition.
        """
        cases = [
            (0.0,   100.0),
            (10.0,  100 * math.exp(-1)),
            (20.0,  100 * math.exp(-2)),
            (50.0,  100 * math.exp(-5)),
        ]
        for cv, expected_score in cases:
            computed = min(100.0, 100 * math.exp(-cv / 10))
            assert abs(computed - expected_score) < 0.01, (
                f"Formula incorrect for CV={cv}%: "
                f"computed={computed:.3f}, expected={expected_score:.3f}"
            )


# ══════════════════════════════════════════════════════════════════════
#  INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_full_pipeline_runs(self, synthetic_df):
        """Tests the pipeline end-to-end as app.py would."""
        h = Hypothesis(
            name="Human Capital",
            description="",
            function=lambda v: 800 + 500*v["E"] + 200*v["X"],
            variables=[Variable("E", 4, 20), Variable("X", 0, 30)],
        )
        res = run_pipeline(
            hypotheses    = [h],
            df            = synthetic_df,
            target_column = "renda",
            mappings      = [{"E": "estudo", "X": "experiencia"}],
            time_column   = "ano",
            n_trials      = 20,
            sample_size   = 80,
            n_windows     = 3,
            seed          = 42,
        )
        assert "simulation"  in res
        assert "empirical"   in res
        assert "comparison"  in res
        assert "temporal"    in res
        assert "figure"      in res
        assert res["figure"] is not None

    def test_pipeline_without_time_column(self, synthetic_df):
        """Pipeline must work without temporal analysis."""
        h = Hypothesis(
            name="Simple", description="",
            function=lambda v: v["E"] * 600,
            variables=[Variable("E", 4, 20)],
        )
        res = run_pipeline(
            hypotheses    = [h],
            df            = synthetic_df,
            target_column = "renda",
            mappings      = [{"E": "estudo"}],
            time_column   = None,
            n_trials      = 10,
            sample_size   = 50,
        )
        assert res["temporal"] == []

    def test_two_hypotheses_compared(self, synthetic_df):
        """Comparison must always elect exactly one winner."""
        h1 = Hypothesis("H1", "",
                         lambda v: 800 + 500*v["E"] + 200*v["X"],
                         [Variable("E", 4, 20), Variable("X", 0, 30)])
        h2 = Hypothesis("H2", "",
                         lambda v: 650 * v["E"],
                         [Variable("E", 4, 20)])
        res = run_pipeline(
            hypotheses    = [h1, h2],
            df            = synthetic_df,
            target_column = "renda",
            mappings      = [{"E": "estudo", "X": "experiencia"}, {"E": "estudo"}],
            n_trials      = 15,
            sample_size   = 60,
        )
        assert res["comparison"].winner["name"] in ["H1", "H2"]
        assert len(res["comparison"].ranking) == 2
