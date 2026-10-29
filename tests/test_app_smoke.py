"""Smoke tests for the AutomataLearn Streamlit application using AppTest.

Verifies end-to-end UI rendering and execution flows:
- Initial startup and default session state
- Angluin's L* active learning (Exact and Approximate modes)
- RPNI state merging from positive and negative samples
- Feature extraction and ML classifiers training (DT, RF, LR)
- Full comparative benchmark suite execution and result rendering
"""

from __future__ import annotations
import os
import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "automatalearn", "app.py"))
if not os.path.exists(APP_PATH):
    # Fallback if tests/ is inside automatalearn/
    APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "app.py"))
    if not os.path.exists(APP_PATH):
        APP_PATH = "automatalearn/app.py"


def test_app_initial_render():
    """Verify AutomataLearn launches cleanly without any uncaught exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App launched with unhandled exception: {at.exception}"
    assert "current_language" in at.session_state
    assert "splits" in at.session_state
    assert at.session_state["splits"] is not None


def test_app_lstar_exact_and_approx():
    """Verify Angluin's L* execution in both Exact and Approximate modes."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception

    # 1. Exact mode (default)
    btn_lstar = [b for b in at.button if "L*" in b.label][0]
    btn_lstar.click().run()
    assert not at.exception, f"L* Exact failed: {at.exception}"
    lstar_res = at.session_state["lstar_result"]
    assert lstar_res is not None
    assert hasattr(lstar_res, "final_S")
    assert hasattr(lstar_res, "final_E")
    assert len(lstar_res.final_S) > 0

    # 2. Approximate mode
    eq_sb = [s for s in at.selectbox if "Equivalence Query" in s.label][0]
    eq_sb.select_index(1).run()
    assert not at.exception

    btn_lstar = [b for b in at.button if "L*" in b.label][0]
    btn_lstar.click().run()
    assert not at.exception, f"L* Approx failed: {at.exception}"
    assert at.session_state["lstar_result"] is not None


def test_app_rpni_learning():
    """Verify RPNI state merging executes cleanly and updates session state."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception

    rad = [r for r in at.radio if "Learning Paradigm" in r.label][0]
    rad.set_value("RPNI (Passive State Merging from Labeled Strings)").run()
    assert not at.exception

    btn_rpni = [b for b in at.button if "RPNI" in b.label][0]
    btn_rpni.click().run()
    assert not at.exception, f"RPNI failed: {at.exception}"

    rpni_res = at.session_state["rpni_result"]
    assert rpni_res is not None
    assert rpni_res.pta_state_count > 0
    assert rpni_res.final_state_count > 0


def test_app_ml_classifiers_training():
    """Verify ML classifiers train, profile, and render without attribute errors."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception

    btn_ml = [b for b in at.button if "ML Models" in b.label][0]
    btn_ml.click().run()
    assert not at.exception, f"ML training failed: {at.exception}"

    ml_res = at.session_state["ml_results"]
    assert ml_res is not None
    assert len(ml_res) >= 3
    for m_name, m_eval in ml_res.items():
        assert hasattr(m_eval, "complexity_metric")
        assert isinstance(m_eval.complexity_metric, str)
        assert hasattr(m_eval, "cv_mean_accuracy")
        assert hasattr(m_eval, "cv_std_accuracy")


def test_app_comparative_benchmark():
    """Verify full comparative benchmark runs and populates suite with all metrics."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    btn_bench = [b for b in at.button if "Comparative Benchmark" in b.label][0]
    btn_bench.click().run()
    assert not at.exception, f"Comparative Benchmark failed: {at.exception}"

    suite = at.session_state["benchmark_suite"]
    assert suite is not None
    assert hasattr(suite, "alphabet")
    assert len(suite.alphabet) > 0
    assert hasattr(suite, "num_samples")
    assert suite.num_samples > 0
    assert hasattr(suite, "gen_samples")
    assert suite.gen_samples > 0
    assert len(suite.entries) >= 6


def test_app_classify_tab_evaluator_demo():
    """Verify Classify tab evaluator demo: auto-training, validation, quick-picks, random button, and results table."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, f"App launched with unhandled exception: {at.exception}"

    # 1. On first use, auto-train should have populated RPNI, L*, and all 6 ML models
    assert at.session_state["rpni_result"] is not None
    assert at.session_state["lstar_result"] is not None
    ml_res = at.session_state["ml_results"]
    assert ml_res is not None
    for required_model in [
        "Decision Tree (Basic)", "Random Forest (Basic)", "Logistic Regression (Basic)",
        "Decision Tree (Generic)", "Random Forest (Generic)", "Logistic Regression (Generic)",
    ]:
        assert required_model in ml_res
        assert hasattr(ml_res[required_model], "feature_extractor")
        assert ml_res[required_model].feature_extractor is not None

    # Check results table exists on initial load
    assert len(at.dataframe) > 0

    # 2. Test input validation with invalid alphabet characters (e.g., 'xyz')
    inp = at.text_input(key="classify_input_widget")
    inp.input("xyz")
    btn_run = [b for b in at.button if "Run on all systems" in b.label][0]
    btn_run.click().run()
    assert not at.exception
    # Assert error banner appears for invalid input
    assert any("Invalid Input" in e.value for e in at.error)

    # 3. Test empty string input (allowed as ε)
    inp = at.text_input(key="classify_input_widget")
    inp.input("")
    btn_run = [b for b in at.button if "Run on all systems" in b.label][0]
    btn_run.click().run()
    assert not at.exception
    assert not any("Invalid Input" in e.value for e in at.error)
    assert at.session_state["classify_eval_str"] == ""

    # 4. Test "Random string" button
    btn_rand = [b for b in at.button if "Random string" in b.label][0]
    btn_rand.click().run()
    assert not at.exception
    rand_str = at.session_state["classify_eval_str"]
    assert 14 <= len(rand_str) <= 22
    assert all(c in at.session_state["current_language"].alphabet for c in rand_str)

    # 5. Test Quick-pick buttons
    qp_btns = [b for b in at.button if b.key and b.key.startswith("btn_qp_")]
    assert len(qp_btns) >= 5
    qp_btns[0].click().run()
    assert not at.exception


def test_app_tab1_user_input_tester():
    """Verify Tab 1 User Input string tester and custom regex expander."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception

    # 1. Verify Tab 1 has User Input string tester widget
    inp = at.text_input(key="tab1_user_input_widget")
    assert inp is not None
    assert inp.value in ["ab", "0101", "01"]

    # 2. Test inputting a valid string and clicking Test String
    inp.input("aab")
    btn_test = [b for b in at.button if b.key == "btn_tab1_run"][0]
    btn_test.click().run()
    assert not at.exception
    assert at.session_state["tab1_eval_str"] == "aab"

    # 3. Test quick picks on Tab 1
    tab1_qp_btns = [b for b in at.button if b.key and b.key.startswith("btn_tab1_qp_")]
    assert len(tab1_qp_btns) >= 5
    tab1_qp_btns[0].click().run()
    assert not at.exception

    # 4. Test invalid input validation on Tab 1
    inp = at.text_input(key="tab1_user_input_widget")
    inp.input("xyz123")
    btn_test = [b for b in at.button if b.key == "btn_tab1_run"][0]
    btn_test.click().run()
    assert not at.exception
    assert any("Invalid Input" in e.value for e in at.error)


def test_app_custom_regex_pipeline():
    """Verify complete end-to-end UI custom regex compilation, acceptance, error recovery, and alphabet changes."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    # 1. Compile valid custom regex: (a+b)*abb(a+b)*
    re_input = at.text_input(key="custom_re_input")
    re_input.input("(a+b)*abb(a+b)*")
    btn_compile = [b for b in at.button if b.key == "btn_compile_custom_re"][0]
    btn_compile.click().run()
    assert not at.exception

    cur_lang = at.session_state["current_language"]
    assert cur_lang.key == "custom_regex"
    assert "(a+b)*abb(a+b)*" in cur_lang.name

    # 2. Evaluate accepted strings: abb, aabb, babba
    for s_acc in ["abb", "aabb", "babba"]:
        inp = at.text_input(key="tab1_user_input_widget")
        inp.input(s_acc)
        btn_test = [b for b in at.button if b.key == "btn_tab1_run"][0]
        btn_test.click().run()
        assert not at.exception
        assert at.session_state["tab1_eval_str"] == s_acc
        assert any("ACCEPTED" in m.value for m in at.markdown)

    # 3. Evaluate rejected strings: a, b, ab, ba
    for s_rej in ["a", "b", "ab", "ba"]:
        inp = at.text_input(key="tab1_user_input_widget")
        inp.input(s_rej)
        btn_test = [b for b in at.button if b.key == "btn_tab1_run"][0]
        btn_test.click().run()
        assert not at.exception
        assert any("REJECTED" in m.value for m in at.markdown)

    # 4. Enter invalid regex and verify readable validation error appears
    re_input = at.text_input(key="custom_re_input")
    re_input.input("(a+b")
    btn_compile = [b for b in at.button if b.key == "btn_compile_custom_re"][0]
    btn_compile.click().run()
    assert not at.exception
    assert any("Failed to compile regex" in e.value for e in at.error)
    # 7. Failed compilation must NOT replace language with new failed regex
    assert at.session_state["current_language"].regex == "(a+b)*abb(a+b)*"

    # 5. Enter input containing symbol outside configured alphabet
    inp = at.text_input(key="tab1_user_input_widget")
    inp.input("xyz")
    btn_test = [b for b in at.button if b.key == "btn_tab1_run"][0]
    btn_test.click().run()
    assert not at.exception
    assert any("Invalid Input" in e.value for e in at.error)

    # 6 & 8. Change configured alphabet to '0, 1' and compile (0+1)*01
    alpha_input = at.text_input(key="custom_re_alpha")
    alpha_input.input("0, 1")
    re_input = at.text_input(key="custom_re_input")
    re_input.input("(0+1)*01")
    btn_compile = [b for b in at.button if b.key == "btn_compile_custom_re"][0]
    btn_compile.click().run()
    assert not at.exception

    cur_lang2 = at.session_state["current_language"]
    assert cur_lang2.alphabet == ["0", "1"]
    assert cur_lang2.dfa.accepts("01")
    assert cur_lang2.dfa.accepts("001")
    assert not cur_lang2.dfa.accepts("0")
    assert not cur_lang2.dfa.accepts("abb")




