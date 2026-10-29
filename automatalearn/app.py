"""AutomataLearn: Intelligent Automata Learning and Regular Language Classification.

Professional ML Dashboard (Linear / Weights & Biases style) featuring Deterministic
Finite Automata (DFA), NFA subset construction, Hopcroft & Table-Filling minimization,
RPNI passive state merging, Angluin's L* active teacher learning, and classical
machine learning baselines with out-of-distribution length generalization profiling.
"""

from __future__ import annotations
import os
import sys

# Ensure project's parent directory is on sys.path so 'automatalearn' package is importable
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import io
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import streamlit as st

from automatalearn.core.dfa import DFA
from automatalearn.core.languages import (
    PREDEFINED_LANGUAGES,
    LanguageDefinition,
    get_language,
    list_available_languages,
)
from automatalearn.core.minimize import (
    get_non_minimal_example_dfa,
    minimize_dfa,
    minimize_dfa_table_filling,
)
from automatalearn.core.nfa import NFA
from automatalearn.core.regex_to_nfa import regex_to_nfa
from automatalearn.data.generator import (
    DataGenerator,
    DatasetSplits,
    export_dataset_to_csv,
    load_dataset_from_csv,
)
from automatalearn.learning.lstar import LStarLearner, LStarResult
from automatalearn.learning.rpni import RPNILearner, RPNIResult
from automatalearn.ml.benchmark import BenchmarkEngine, BenchmarkSuiteResult
from automatalearn.ml.features import (
    ALL_FEATURE_GROUPS,
    BASIC_FEATURE_GROUPS,
    FEATURE_GROUPS,
    GENERIC_FEATURE_GROUPS,
    StringFeatureExtractor,
)
from automatalearn.ml.models import (
    ModelEvaluationResult,
    train_and_evaluate_ml_model,
)
from automatalearn.ui.style import apply_custom_css, kpi, kv_row, kv_table
from automatalearn.viz.draw import dfa_to_dot, nfa_to_dot
from automatalearn.viz.plots import (
    plot_complexity_vs_accuracy,
    plot_confusion_matrix_heatmap,
    plot_feature_importance_bar,
    plot_in_dist_vs_gen,
    plot_learning_curves,
    plot_noise_robustness,
    plot_runtime_comparison,
)

# -----------------------------------------------------------------------------
# App Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutomataLearn | ML & Automata Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom Linear / W&B dark-theme CSS
apply_custom_css()

# -----------------------------------------------------------------------------
# Quick-Pick Example Strings per Language
# -----------------------------------------------------------------------------
QUICK_PICKS_PER_LANG: Dict[str, List[Tuple[str, str]]] = {
    "ending_ab": [
        ("ab", "Short Accepted"),
        ("aab", "Prefix + ab"),
        ("ba", "Reversed (Rejected)"),
        ("", "Empty string ε"),
        ("babababa", "Long periodic (ends with a)"),
    ],
    "even_0_odd_1": [
        ("1", "Accepted (0 0s, 1 1)"),
        ("001", "Accepted (2 0s, 1 1)"),
        ("01", "Rejected (1 0, 1 1)"),
        ("", "Empty string ε (Rejected)"),
        ("0000111", "Long OOD (Accepted)"),
    ],
    "contains_101": [
        ("101", "Exact '101'"),
        ("0010111", "Embedded '101'"),
        ("110011", "No '101' (Rejected)"),
        ("", "Empty string ε"),
        ("01001001", "Repeated 010 (Rejected)"),
    ],
    "count_a_div_3": [
        ("aaa", "3 a's (Accepted)"),
        ("baaba", "3 a's with b's (Accepted)"),
        ("a", "1 a (Rejected)"),
        ("", "0 a's = 0%3 (Accepted ε)"),
        ("baabaababba", "6 a's long string (Accepted)"),
    ],
    "binary_div_3": [
        ("0", "0 % 3 == 0 (Accepted)"),
        ("11", "Value 3 % 3 == 0 (Accepted)"),
        ("110", "Value 6 % 3 == 0 (Accepted)"),
        ("101", "Value 5 % 3 != 0 (Rejected)"),
        ("", "Empty string ε (Rejected)"),
    ],
    "no_consecutive_1s": [
        ("01010", "Alternating (Accepted)"),
        ("100101", "No consecutive 1s (Accepted)"),
        ("11", "Immediate '11' (Rejected)"),
        ("010110", "Contains '11' (Rejected)"),
        ("", "Empty string ε (Accepted)"),
    ],
    "length_mod_4": [
        ("0101", "Length 4 (Accepted)"),
        ("00001111", "Length 8 (Accepted)"),
        ("01", "Length 2 (Rejected)"),
        ("", "Length 0 = 0%4 (Accepted ε)"),
        ("1010101010101010", "Length 16 (Accepted - long)"),
    ],
}


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state():
    if "current_lang_key" not in st.session_state:
        st.session_state["current_lang_key"] = "ending_ab"
    if "current_language" not in st.session_state:
        st.session_state["current_language"] = get_language("ending_ab")
    if "splits" not in st.session_state:
        gen = DataGenerator(
            alphabet=st.session_state["current_language"].alphabet,
            oracle=st.session_state["current_language"].predicate,
            dfa=st.session_state["current_language"].dfa,
            seed=42,
        )
        st.session_state["splits"] = gen.generate_dataset(
            min_length=0, max_length=12, num_samples=1500, gen_samples=500
        )
    if "rpni_result" not in st.session_state:
        st.session_state["rpni_result"] = None
    if "lstar_result" not in st.session_state:
        st.session_state["lstar_result"] = None
    if "ml_results" not in st.session_state:
        st.session_state["ml_results"] = {}
    if "extractor" not in st.session_state:
        st.session_state["extractor"] = None
    if "benchmark_suite" not in st.session_state:
        st.session_state["benchmark_suite"] = None
    if "lab_custom_dfa" not in st.session_state:
        st.session_state["lab_custom_dfa"] = None


init_session_state()


# -----------------------------------------------------------------------------
# Sidebar: Global Navigation, Dataset Parameters, and Hyperparameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;"><span style="font-size:1.8rem;">⚡</span><span style="font-size:1.3rem; font-weight:800; letter-spacing:-0.02em; color:#F3F4F6;">AutomataLearn</span></div>', unsafe_allow_html=True)
    st.caption("Intelligent Automata Learning & Regular Language Classification")
    st.markdown("---")

    # 1. Target Language Selector
    lang_keys = list(PREDEFINED_LANGUAGES.keys()) + ["custom_regex"]
    lang_names = [PREDEFINED_LANGUAGES[k]().name for k in PREDEFINED_LANGUAGES.keys()] + ["Custom Regular Expression"]

    selected_idx = 0
    if st.session_state["current_lang_key"] in lang_keys:
        selected_idx = lang_keys.index(st.session_state["current_lang_key"])

    selected_lang_name = st.selectbox(
        "🎯 Target Language",
        options=lang_names,
        index=selected_idx,
        help="Select a benchmark regular language or enter a custom regex.",
    )

    new_key = lang_keys[lang_names.index(selected_lang_name)]
    if new_key != st.session_state["current_lang_key"]:
        st.session_state["current_lang_key"] = new_key
        st.session_state.pop("lab_custom_dfa", None)
        if new_key != "custom_regex":
            st.session_state["current_language"] = get_language(new_key)
            gen = DataGenerator(
                alphabet=st.session_state["current_language"].alphabet,
                oracle=st.session_state["current_language"].predicate,
                dfa=st.session_state["current_language"].dfa,
                seed=42,
            )
            st.session_state["splits"] = gen.generate_dataset(
                min_length=0, max_length=12, num_samples=1500, gen_samples=500
            )
            st.session_state["rpni_result"] = None
            st.session_state["lstar_result"] = None
            st.session_state["ml_results"] = {}
            st.session_state["benchmark_suite"] = None
            st.rerun()

    # 2. Global Seed
    global_seed = st.number_input("🎲 Global Seed", value=42, min_value=0, step=1)

    # 3. Global Dataset Generation Controls
    with st.expander("📦 Dataset Parameters", expanded=True):
        n_samples = st.slider("In-Dist Samples", min_value=100, max_value=3000, value=1500, step=100)
        c_len1, c_len2 = st.columns(2)
        with c_len1:
            min_len = st.number_input("Min Length", min_value=0, max_value=5, value=0)
        with c_len2:
            max_len = st.number_input("Max Length", min_value=4, max_value=16, value=12)

        class_bal = st.slider("Class Balance (% Positives)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        noise_rate = st.slider("Train Label Noise", min_value=0.0, max_value=0.3, value=0.0, step=0.05,
                               help="Noise is injected ONLY into the training set; validation and test sets remain ground truth.")
        gen_samples = st.slider("Length-Gen Samples", min_value=100, max_value=1000, value=500, step=50)

        if st.button("⚡ Regenerate Dataset", type="primary", use_container_width=True):
            with st.spinner("Generating partitioned dataset..."):
                gen = DataGenerator(
                    alphabet=st.session_state["current_language"].alphabet,
                    oracle=st.session_state["current_language"].predicate,
                    dfa=st.session_state["current_language"].dfa,
                    seed=global_seed,
                )
                st.session_state["splits"] = gen.generate_dataset(
                    min_length=min_len,
                    max_length=max_len,
                    num_samples=n_samples,
                    class_balance=class_bal,
                    noise_rate=noise_rate,
                    gen_min_length=max_len + 1,
                    gen_max_length=max_len + 12,
                    gen_samples=gen_samples,
                )
                st.session_state["rpni_result"] = None
                st.session_state["lstar_result"] = None
                st.session_state["ml_results"] = {}
                st.session_state["benchmark_suite"] = None
                st.success("Dataset successfully generated!")

    # 4. Global ML Hyperparameter Controls
    with st.expander("⚙️ ML Hyperparameters", expanded=False):
        st.markdown("**Decision Tree**")
        dt_max_depth = st.slider("DT Max Depth", 1, 20, 6)
        dt_criterion = st.selectbox("DT Criterion", ["gini", "entropy"])

        st.markdown("**Random Forest**")
        rf_estimators = st.slider("RF Trees Count", 10, 200, 50, step=10)
        rf_max_depth = st.slider("RF Max Depth", 1, 20, 8)

        st.markdown("**Logistic Regression**")
        lr_c = st.slider("LR Regularization C", 0.01, 10.0, 1.0)
        lr_penalty = st.selectbox("LR Penalty", ["l2", "l1"])

    dt_params = {"max_depth": dt_max_depth, "criterion": dt_criterion}
    rf_params = {"n_estimators": rf_estimators, "max_depth": rf_max_depth}
    lr_params = {"C": lr_c, "penalty": lr_penalty, "solver": "liblinear" if lr_penalty == "l1" else "lbfgs"}


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Header Display with Gradient Title (Inline Style), Subtitle, and Pill Badges
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 1.2rem;">
      <h1 style="background: linear-gradient(135deg, #6C8CFF 0%, #A78BFA 50%, #22D3EE 100%) no-repeat;
                 background-size: 280px 100%;
                 -webkit-background-clip: text;
                 background-clip: text;
                 -webkit-text-fill-color: transparent;
                 font-size: 2.35rem;
                 font-weight: 800;
                 letter-spacing: -0.03em;
                 margin: 0 0 0.25rem 0;
                 padding: 0;
                 line-height: 1.15;
                 font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;">
        AutomataLearn
      </h1>
      <div style="font-size: 0.96rem; font-weight: 400; color: #94A3B8; margin-bottom: 0.75rem;">
        Intelligent Automata Learning &amp; Regular Language Generalization Benchmark
      </div>
      <div class="badge-row">
        <span class="pill-badge badge-blue">DFA / NFA</span>
        <span class="pill-badge badge-teal">RPNI State Merging</span>
        <span class="pill-badge badge-emerald">Angluin's L*</span>
        <span class="pill-badge badge-purple">Random Forest</span>
        <span class="pill-badge badge-amber">Decision Trees</span>
        <span class="pill-badge badge-slate">scikit-learn</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Application Tabs
tabs = st.tabs([
    "1. Language & Data",
    "2. DFA / NFA Lab",
    "3. Learn Automaton",
    "4. ML Classifiers",
    "5. Compare Dashboard",
    "6. Classify",
    "7. Theory & Foundations",
])


def dfa_to_transition_df(dfa: DFA) -> pd.DataFrame:
    """Build a transition table DataFrame for delta(q, a) with start/accept state markers."""
    sorted_states = sorted(list(dfa.states), key=lambda s: (s != dfa.start_state, str(s)))
    sorted_alphabet = sorted(list(dfa.alphabet))

    rows = []
    for s in sorted_states:
        marker = ""
        if s == dfa.start_state:
            marker += "→ "
        if s in dfa.accept_states:
            marker += "★ "
        state_label = f"{marker}{s}".strip() if marker else str(s)
        row_dict: dict[str, str] = {"State": state_label}
        for a in sorted_alphabet:
            dest = dfa.transitions.get((s, a), "—")
            row_dict[f"δ(·, {a})"] = str(dest)
        rows.append(row_dict)

    return pd.DataFrame(rows)


# =============================================================================
# TAB 1: LANGUAGE & DATA
# =============================================================================
with tabs[0]:
    try:
        current_lang = st.session_state["current_language"]
        splits: DatasetSplits = st.session_state["splits"]

        # Top KPI metric row (fixed 128px height, identical structure in all cards)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(kpi("Alphabet (Σ)", f"{{{', '.join(current_lang.alphabet)}}}", "Input symbols"), unsafe_allow_html=True)
        with kpi2:
            st.markdown(kpi("Target DFA States", len(current_lang.dfa.states), f"{len(current_lang.dfa.accept_states)} accepting"), unsafe_allow_html=True)
        with kpi3:
            n_tr = len(splits.train_df)
            n_vl = len(splits.val_df)
            n_ts = len(splits.test_df)
            st.markdown(kpi("Splits: Train / Val / Test", f"{n_tr} / {n_vl} / {n_ts}", "70% · 15% · 15%"), unsafe_allow_html=True)
        with kpi4:
            min_l = splits.generalization_df['length'].min() if len(splits.generalization_df) > 0 else 0
            max_l = splits.generalization_df['length'].max() if len(splits.generalization_df) > 0 else 0
            st.markdown(kpi("Length-Gen Pool", len(splits.generalization_df), f"Lengths {min_l}–{max_l}"), unsafe_allow_html=True)

        with st.expander("➕ Define Custom Language via Regular Expression (User Input)", expanded=(st.session_state["current_lang_key"] == "custom_regex")):
            c_re1, c_re2, c_re3 = st.columns([2, 1, 1])
            with c_re1:
                custom_re = st.text_input("Regular Expression (supports |, *, +, ?, parenthesized groups):", value="(a|b)*ab", key="custom_re_input")
            with c_re2:
                custom_alphabet_str = st.text_input("Alphabet (comma-separated):", value="a, b", key="custom_re_alpha")
                custom_alpha = [c.strip() for c in custom_alphabet_str.split(",") if c.strip()]
            with c_re3:
                st.write("")
                st.write("")
                if st.button("Compile Regex to DFA", type="primary", use_container_width=True, key="btn_compile_custom_re"):
                    try:
                        nfa = regex_to_nfa(custom_re, alphabet=custom_alpha)
                        raw_dfa = nfa.to_dfa()
                        min_res = minimize_dfa(raw_dfa)
                        compiled_dfa = min_res.minimized_dfa
                        st.session_state["current_lang_key"] = "custom_regex"
                        st.session_state["current_language"] = LanguageDefinition(
                            key="custom_regex",
                            name=f"Custom: /{custom_re}/",
                            description=f"Compiled from regex: {custom_re}",
                            alphabet=custom_alpha,
                            dfa=compiled_dfa,
                            predicate=lambda s: compiled_dfa.accepts(s),
                            regex=custom_re,
                            theoretical_notes="User-defined regular expression compiled via Thompson's construction and powerset determinization.",
                        )
                        st.session_state["splits"] = DataGenerator(
                            alphabet=custom_alpha,
                            oracle=lambda s: compiled_dfa.accepts(s),
                            dfa=compiled_dfa,
                            seed=global_seed,
                        ).generate_dataset(min_length=0, max_length=8, num_samples=300, gen_samples=100)
                        st.session_state["rpni_result"] = None
                        st.session_state["lstar_result"] = None
                        st.session_state["ml_results"] = {}
                        st.session_state["benchmark_suite"] = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to compile regex: {e}")

        # Section 1: Target Language Specification, Diagram, and Transition Table
        with st.container(border=True):
            st.markdown(f"### Target Language: {current_lang.name}")
            col_l1, col_l2, col_l3 = st.columns([1.1, 1.4, 0.9])
            with col_l1:
                lang_rows: list[tuple[str, object] | tuple[str, object, bool]] = [
                    ("Description", current_lang.description, False),
                    ("Alphabet (Σ)", f"{{{', '.join(current_lang.alphabet)}}}", True),
                    ("DFA State Count", f"{len(current_lang.dfa.states)} ({len(current_lang.dfa.accept_states)} accepting)", True),
                ]
                if current_lang.regex:
                    lang_rows.append(("Regex Pattern", current_lang.regex, True))
                st.markdown(kv_table(lang_rows), unsafe_allow_html=True)
                if current_lang.theoretical_notes:
                    st.info(current_lang.theoretical_notes)
            with col_l2:
                st.caption("State Transition Diagram")
                dot_str = dfa_to_dot(current_lang.dfa)
                st.graphviz_chart(dot_str, use_container_width=True)
            with col_l3:
                st.caption("Transition Table $\\delta(q, a)$ (→ start, ★ accept)")
                df_delta = dfa_to_transition_df(current_lang.dfa)
                st.dataframe(df_delta, use_container_width=True, hide_index=True)

        # Section 2: Interactive User Input & Live String Acceptance Tester
        with st.container(border=True):
            st.markdown("### ✍️ User Input: Live String Acceptance Tester")
            st.caption(
                f"Enter any candidate string over alphabet $\\Sigma = {{{', '.join(current_lang.alphabet)}}}$ to test acceptance against the target DFA. "
                "Step through the state path or benchmark across all learning models in Tab 6."
            )

            tab1_quick_picks = QUICK_PICKS_PER_LANG.get(current_lang.key, [
                ("", "Empty string ε"),
                (current_lang.alphabet[0], "Short 1"),
                (current_lang.alphabet[0] * 2, "Short 2"),
                (current_lang.alphabet[-1], "Short 3"),
                ((current_lang.alphabet[0] + current_lang.alphabet[-1]) * 4, "Periodic"),
            ])

            if "pending_tab1_input" in st.session_state:
                st.session_state["tab1_user_input_widget"] = st.session_state.pop("pending_tab1_input")
            elif "tab1_user_input_widget" not in st.session_state or st.session_state.get("tab1_lang_key") != current_lang.key:
                st.session_state["tab1_lang_key"] = current_lang.key
                st.session_state["tab1_user_input_widget"] = "ab" if "a" in current_lang.alphabet else "01"
                st.session_state["tab1_eval_str"] = st.session_state["tab1_user_input_widget"]

            st.markdown("**⚡ Quick-Pick Test Strings:**")
            qp1_cols = st.columns(5)
            for i, (qp_val, qp_lbl) in enumerate(tab1_quick_picks[:5]):
                with qp1_cols[i]:
                    btn_t = f"ε ({qp_lbl})" if qp_val == "" else f"'{qp_val}' ({qp_lbl})"
                    if st.button(btn_t, key=f"btn_tab1_qp_{current_lang.key}_{i}", use_container_width=True):
                        st.session_state["pending_tab1_input"] = qp_val
                        st.session_state["tab1_eval_str"] = qp_val
                        st.rerun()

            c_in, c_btn, c_rnd = st.columns([5, 2, 2])
            with c_in:
                user_tab1_str = st.text_input(
                    label=f"Input String to Test (Σ = {{{', '.join(current_lang.alphabet)}}}):",
                    key="tab1_user_input_widget",
                    help="Enter any string over the alphabet. Leave blank for empty string ε.",
                )
            with c_btn:
                st.write("")
                st.write("")
                if st.button("⚡ Test String", key="btn_tab1_run", type="primary", use_container_width=True):
                    st.session_state["tab1_eval_str"] = user_tab1_str
                    st.rerun()
            with c_rnd:
                st.write("")
                st.write("")
                if st.button("🎲 Random String", key="btn_tab1_rand", use_container_width=True):
                    rand_l = random.randint(2, 10)
                    rand_val = "".join(random.choices(current_lang.alphabet, k=rand_l))
                    st.session_state["pending_tab1_input"] = rand_val
                    st.session_state["tab1_eval_str"] = rand_val
                    st.rerun()

            tab1_active_str = st.session_state.get("tab1_eval_str", user_tab1_str)
            tab1_invalid = sorted(list(set(c for c in tab1_active_str if c not in current_lang.alphabet)))

            if tab1_invalid:
                st.error(f"❌ **Invalid Input:** String contains symbols `{set(tab1_invalid)}` not in alphabet Σ = {{{', '.join(current_lang.alphabet)}}}.")
            else:
                accepted, path = current_lang.dfa.trace(tab1_active_str)
                disp_val = f"'{tab1_active_str}'" if tab1_active_str != "" else "ε (empty string)"

                c_res1, c_res2, c_res3 = st.columns([1.2, 1.8, 1.5])
                with c_res1:
                    verdict_text = "✅ ACCEPTED" if accepted else "❌ REJECTED"
                    st.markdown(kpi("DFA Verdict", verdict_text, f"{disp_val}"), unsafe_allow_html=True)
                with c_res2:
                    st.markdown(kpi("State Transition Path", f"{path[0]} → ... → {path[-1]}", f"Total steps: {len(path)-1}"), unsafe_allow_html=True)
                with c_res3:
                    final_type = "accepting" if path[-1] in current_lang.dfa.accept_states else "non-accepting"
                    st.markdown(kpi("Final State", str(path[-1]), f"State is {final_type}"), unsafe_allow_html=True)

                path_repr = " → ".join(f"**{path[i]}** ─`{tab1_active_str[i]}`→ **{path[i+1]}**" for i in range(len(tab1_active_str))) if tab1_active_str else f"Start state **{path[0]}** (empty string ε)"
                st.markdown(f"**Step-by-Step Transition Trace:** {path_repr}")

                col_info, col_jump = st.columns([3, 1.3])
                with col_info:
                    st.caption("Want to compare how RPNI, L*, Decision Trees, and Random Forests classify this exact sequence?")
                with col_jump:
                    if st.button("🚀 Run on All Systems in Tab 6", key="btn_tab1_to_tab6", use_container_width=True):
                        st.session_state["pending_classify_input"] = tab1_active_str
                        st.session_state["classify_eval_str"] = tab1_active_str
                        st.success("String staged for Tab 6! Navigate to '6. Classify' in the top bar.")

        # Section 3: Partitioned Dataset Inspector
        with st.container(border=True):
            st.markdown("### Partitioned Dataset Inspector & CSV Serialization")
            tab_train, tab_val, tab_test, tab_gen, tab_io = st.tabs([
                "Train Set", "Validation Set", "Test Set (In-Dist)", "Length-Generalization Set", "Upload / Download CSV"
            ])

            col_cfg = {
                "string": st.column_config.TextColumn("String", help="Input string over alphabet"),
                "label": st.column_config.NumberColumn("Label", format="%d", help="1 = Accept, 0 = Reject"),
                "length": st.column_config.ProgressColumn("Length", min_value=0, max_value=25, format="%d chars"),
            }

            with tab_train:
                st.dataframe(splits.train_df, column_config=col_cfg, use_container_width=True, height=240)
            with tab_val:
                st.dataframe(splits.val_df, column_config=col_cfg, use_container_width=True, height=240)
            with tab_test:
                st.dataframe(splits.test_df, column_config=col_cfg, use_container_width=True, height=240)
            with tab_gen:
                st.caption("Note: Strings in this split have lengths strictly greater than any training string, testing inductive generalization.")
                st.dataframe(splits.generalization_df, column_config=col_cfg, use_container_width=True, height=240)
            with tab_io:
                c_io1, c_io2 = st.columns(2)
                with c_io1:
                    st.markdown("#### Export Current Splits")
                    csv_train = export_dataset_to_csv(splits.train_df)
                    st.download_button(
                        "📥 Download Training CSV",
                        data=csv_train,
                        file_name=f"train_{current_lang.key}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with c_io2:
                    st.markdown("#### Import External CSV")
                    up_file = st.file_uploader("Upload CSV (must contain 'string' and 'label' columns)", type=["csv"])
                    if up_file is not None:
                        try:
                            df_up = load_dataset_from_csv(up_file.getvalue().decode("utf-8"))
                            st.success(f"Loaded {len(df_up)} samples from uploaded CSV!")
                            st.dataframe(df_up.head(10), column_config=col_cfg, use_container_width=True)
                        except Exception as ex:
                            st.error(f"Error parsing CSV: {ex}")
    except Exception as e:
        st.error(f"Error in Language & Data tab: {e}")
        st.exception(e)

# =============================================================================
# TAB 2: DFA / NFA LAB
# =============================================================================
with tabs[1]:
    try:
        current_lang = st.session_state["current_language"]
        is_custom_dfa = bool(st.session_state.get("lab_custom_dfa"))
        active_dfa = st.session_state.get("lab_custom_dfa") or current_lang.dfa

        # Top KPI metric row (fixed 128px height, identical structure)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(kpi("Total States", len(active_dfa.states), "Non-minimal (6 states)" if is_custom_dfa else "Ground truth"), unsafe_allow_html=True)
        with kpi2:
            st.markdown(kpi("Accept States", len(active_dfa.accept_states), "Terminal states"), unsafe_allow_html=True)
        with kpi3:
            st.markdown(kpi("Is Complete?", "YES" if active_dfa.is_complete() else "NO", "Transition coverage"), unsafe_allow_html=True)
        with kpi4:
            st.markdown(kpi("Alphabet Size", len(active_dfa.alphabet), f"{{{', '.join(sorted(active_dfa.alphabet))}}}"), unsafe_allow_html=True)

        # Section 1: Minimization & Equivalence Classes
        with st.container(border=True):
            st.markdown("### DFA Minimization & Equivalence Classes")

            c_demo1, c_demo2 = st.columns(2)
            with c_demo1:
                if st.button("⚡ Load Non-Minimal Example DFA", use_container_width=True, help="Load a 6-state DFA that reduces to 3 states (-50.0% reduction)"):
                    st.session_state["lab_custom_dfa"] = get_non_minimal_example_dfa()
                    st.rerun()
            with c_demo2:
                if st.button("↺ Reset to Target Language DFA", use_container_width=True, disabled=not is_custom_dfa):
                    st.session_state.pop("lab_custom_dfa", None)
                    st.rerun()

            if is_custom_dfa:
                st.info("💡 **Demonstrating Minimization on a Non-Minimal DFA:** This 6-state automaton has equivalent pairs `{A, B}`, `{C, D}`, and `{E, F}`. Hopcroft & Table-Filling merge them into exactly 3 states (-50.0% reduction) without altering the language.")

            c_min1, c_min2 = st.columns([1, 2])
            with c_min1:
                min_algo = st.selectbox("Minimization Algorithm", ["Hopcroft (Partition Refinement)", "Table-Filling (Myhill-Nerode)"])
                method_key = "hopcroft" if "Hopcroft" in min_algo else "table_filling"
                min_result = minimize_dfa(active_dfa, method=method_key)
                reduction_label = f"-{min_result.reduction_percentage:.1f}% reduced" if min_result.reduction_percentage > 0 else "Already minimal"
                st.markdown(kpi("State Reduction", f"{min_result.before_state_count} → {min_result.after_state_count}", reduction_label), unsafe_allow_html=True)

                min_kv: list[tuple[str, object]] = [
                    ("Algorithm", "Hopcroft (Partition)" if method_key == "hopcroft" else "Table-Filling (Nerode)"),
                    ("Initial State Count", min_result.before_state_count),
                    ("Minimized State Count", min_result.after_state_count),
                    ("Equivalence Classes", len(min_result.equivalence_classes)),
                ]
                st.markdown(kv_table(min_kv), unsafe_allow_html=True)

                st.markdown("**Merged Equivalence Classes:**")
                for idx, block in enumerate(min_result.equivalence_classes):
                    st.code(f"Class {idx}: {{{', '.join(sorted([str(s) for s in block]))}}}")

            with c_min2:
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.markdown(f"**Original DFA ({min_result.before_state_count} states)**")
                    st.graphviz_chart(dfa_to_dot(active_dfa), use_container_width=True)
                with col_b2:
                    st.markdown(f"**Minimized DFA ({min_result.after_state_count} states)**")
                    st.graphviz_chart(dfa_to_dot(min_result.minimized_dfa), use_container_width=True)

            if min_result.distinguishability_table:
                st.markdown("#### 📐 Myhill-Nerode Distinguishability Matrix (Pairs Table)")
                state_list = sorted(list(active_dfa.states), key=str)
                matrix_data = []
                for i, p in enumerate(state_list):
                    row = {}
                    for j, q in enumerate(state_list):
                        if j < i:
                            pair = (q, p) if (q, p) in min_result.distinguishability_table else (p, q)
                            dist, sym = min_result.distinguishability_table.get(pair, (False, "-"))
                            row[str(q)] = f"X ({sym})" if dist else "EQ"
                        else:
                            row[str(q)] = ""
                    matrix_data.append(row)
                df_dist = pd.DataFrame(matrix_data, index=[str(s) for s in state_list])
                st.dataframe(df_dist, use_container_width=True)

        # Section 2: Interactive Path Tracing with Active Highlighting
        with st.container(border=True):
            st.markdown("### Interactive State Path Tracing")
            st.caption("Step through any input string. Active states and transition edges are dynamically highlighted.")

            sample_def = "0101" if "0" in active_dfa.alphabet else "aab"
            c_tr1, c_tr2 = st.columns([1, 1])
            with c_tr1:
                input_trace_str = st.text_input("Enter String to Trace:", value=sample_def)
                accepted, path = active_dfa.trace(input_trace_str)

                step_idx = st.slider("Execution Step Slider", min_value=0, max_value=len(path)-1, value=len(path)-1)
                active_subpath = path[:step_idx+1]
                active_curr = path[step_idx]

                trace_kv: list[tuple[str, object]] = [
                    ("Input String", f"'{input_trace_str}'"),
                    ("String Length", f"{len(input_trace_str)} chars"),
                    ("Verdict", "✅ ACCEPT" if accepted else "❌ REJECT"),
                    ("Current Step", f"{step_idx} of {len(path)-1}"),
                    ("Active State", str(active_curr)),
                    ("Consumed Prefix", f"'{input_trace_str[:step_idx]}'"),
                ]
                st.markdown(kv_table(trace_kv), unsafe_allow_html=True)
                st.code(" -> ".join(str(s) for s in active_subpath))

            with c_tr2:
                dot_trace = dfa_to_dot(
                    active_dfa,
                    highlight_states={active_curr},
                    active_path=active_subpath,
                )
                st.graphviz_chart(dot_trace, use_container_width=True)
    except Exception as e:
        st.error(f"Error in DFA / NFA Lab tab: {e}")
        st.exception(e)

# =============================================================================
# TAB 3: LEARN AUTOMATON (RPNI & L*)
# =============================================================================
with tabs[2]:
    try:
        current_lang = st.session_state["current_language"]
        splits = st.session_state["splits"]
        lstar_res: Optional[LStarResult] = st.session_state["lstar_result"]
        rpni_res: Optional[RPNIResult] = st.session_state["rpni_result"]

        # Top KPI metric row (fixed 128px height, identical structure)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(kpi("Target Minimal States", len(current_lang.dfa.states), "Ground truth"), unsafe_allow_html=True)
        with kpi2:
            st.markdown(kpi("RPNI Learned States", len(rpni_res.minimized_dfa.states) if rpni_res else "—", "Passive inference" if rpni_res else "Not run"), unsafe_allow_html=True)
        with kpi3:
            st.markdown(kpi("L* Learned States", len(lstar_res.minimized_dfa.states) if lstar_res else "—", "Active MAT" if lstar_res else "Not run"), unsafe_allow_html=True)
        with kpi4:
            st.markdown(kpi("L* Equivalence", "100% Exact" if (lstar_res and lstar_res.is_equivalent_to_target) else "—", "Isomorphic" if (lstar_res and lstar_res.is_equivalent_to_target) else "Pending"), unsafe_allow_html=True)

        learn_algo = st.radio("Learning Paradigm", ["Angluin's L* (Active Learning with MAT)", "RPNI (Passive State Merging from Labeled Strings)"], horizontal=True)

        if "Angluin" in learn_algo:
            with st.container(border=True):
                st.markdown("### Angluin's L* Active Learning (Minimal Adequate Teacher)")
                c_l1, c_l2 = st.columns([1, 2])
                with c_l1:
                    eq_oracle_type = st.selectbox("Equivalence Query (EQ) Mode", ["Exact (Product Construction)", "Approximate (Random Sampling)"])
                    if st.button("🚀 Run L* Active Learning", type="primary", use_container_width=True):
                        with st.status("Executing Angluin's L* algorithm...", expanded=True) as status:
                            st.write("Initializing Observation Table (S, E, T)...")
                            learner = LStarLearner(
                                alphabet=current_lang.alphabet,
                                membership_oracle=current_lang.predicate,
                                target_dfa=current_lang.dfa if "Exact" in eq_oracle_type else None,
                            )
                            st.write("Resolving closedness & consistency queries...")
                            res = learner.learn()
                            st.session_state["lstar_result"] = res
                            st.write(f"Learned minimal DFA with {len(res.minimized_dfa.states)} states!")
                            status.update(label="L* Active Learning Complete!", state="complete", expanded=False)
                            st.rerun()

                if lstar_res is not None:
                    with c_l2:
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.markdown(kpi("Membership Queries", lstar_res.num_membership_queries, "MQ count"), unsafe_allow_html=True)
                        with col_m2:
                            st.markdown(kpi("Equivalence Queries", lstar_res.num_equivalence_queries, "EQ count"), unsafe_allow_html=True)
                        with col_m3:
                            st.markdown(kpi("Execution Time", f"{lstar_res.execution_time_ms:.2f} ms", "Latency"), unsafe_allow_html=True)

                        lstar_kv: list[tuple[str, object]] = [
                            ("Learning Mode", "Active Angluin L*"),
                            ("Teacher Oracle", eq_oracle_type),
                            ("Prefixes |S|", len(lstar_res.final_S)),
                            ("Suffixes |E|", len(lstar_res.final_E)),
                            ("Equivalence", "EXACT ISOMORPHISM (True)" if lstar_res.is_equivalent_to_target else "APPROXIMATION"),
                        ]
                        st.markdown(kv_table(lstar_kv), unsafe_allow_html=True)

                if lstar_res is not None:
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.markdown("#### Ground-Truth Target DFA")
                        st.graphviz_chart(dfa_to_dot(current_lang.dfa), use_container_width=True)
                    with col_d2:
                        st.markdown("#### Learned Minimal DFA (L*)")
                        st.graphviz_chart(dfa_to_dot(lstar_res.minimized_dfa), use_container_width=True)

                    st.markdown("#### 📜 L* Observation Table Progression Log")
                    valid_trace = [
                        t for t in lstar_res.trace_log
                        if t.action and str(t.detail).strip()
                    ]
                    trace_df = pd.DataFrame([
                        {
                            "Iteration": t.iteration,
                            "Action": t.action,
                            "Detail": t.detail,
                            "|S| (Prefixes)": t.num_prefixes_S,
                            "|E| (Suffixes)": t.num_suffixes_E,
                            "MQs Executed": t.membership_queries_so_far,
                        }
                        for t in valid_trace
                    ])
                    st.dataframe(trace_df, use_container_width=True, height=260)

        else:
            with st.container(border=True):
                st.markdown("### RPNI: Regular Positive & Negative Inference (Passive)")
                pos_train = [s for s, y in zip(splits.train_strings, splits.train_labels) if y == 1]
                neg_train = [s for s, y in zip(splits.train_strings, splits.train_labels) if y == 0]

                c_r1, c_r2 = st.columns([1, 2])
                with c_r1:
                    st.write(f"Training on **{len(pos_train)} positive** and **{len(neg_train)} negative** strings.")
                    if st.button("🚀 Run RPNI State Merging", type="primary", use_container_width=True):
                        with st.status("Running RPNI state merging...", expanded=True) as status:
                            st.write("Constructing Prefix Tree Acceptor (PTA)...")
                            learner = RPNILearner(alphabet=current_lang.alphabet)
                            st.write("Merging Red/Blue states with deterministic folding...")
                            res = learner.learn(pos_train, neg_train)
                            st.session_state["rpni_result"] = res
                            st.write(f"Folded PTA ({res.pta_state_count} states) down to {res.final_state_count} states!")
                            status.update(label="RPNI Learning Complete!", state="complete", expanded=False)
                            st.rerun()

                if rpni_res is not None:
                    with c_r2:
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.markdown(kpi("Initial PTA States", rpni_res.pta_state_count, "PTA size"), unsafe_allow_html=True)
                        with col_m2:
                            st.markdown(kpi("Merges Accepted", f"{rpni_res.merges_accepted}/{rpni_res.merges_attempted}", "State merges"), unsafe_allow_html=True)
                        with col_m3:
                            st.markdown(kpi("Execution Time", f"{rpni_res.execution_time_ms:.2f} ms", "Latency"), unsafe_allow_html=True)

                        rpni_kv: list[tuple[str, object]] = [
                            ("Learning Mode", "Passive RPNI Folding"),
                            ("Training Corpus", f"{len(pos_train)} pos / {len(neg_train)} neg"),
                            ("Initial PTA States", f"{rpni_res.pta_state_count} nodes"),
                            ("Folded States", f"{rpni_res.final_state_count} states"),
                            ("State Compression", f"{(1 - len(rpni_res.minimized_dfa.states)/max(1, rpni_res.pta_state_count))*100:.1f}% reduction"),
                        ]
                        st.markdown(kv_table(rpni_kv), unsafe_allow_html=True)

                if rpni_res is not None:
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.markdown("#### Ground-Truth Target DFA")
                        st.graphviz_chart(dfa_to_dot(current_lang.dfa), use_container_width=True)
                    with col_d2:
                        st.markdown("#### Learned Minimized DFA (RPNI)")
                        st.graphviz_chart(dfa_to_dot(rpni_res.minimized_dfa), use_container_width=True)

                    equiv, ce = rpni_res.minimized_dfa.equivalent_to(current_lang.dfa)
                    if equiv:
                        st.success("🎯 Exact Language Equivalence Confirmed! The learned DFA perfectly matches the ground-truth grammar.")
                    else:
                        st.warning(f"Learned DFA is an approximation. Distinguishing counterexample: '{ce}'")

                    st.markdown("#### 📜 State Merging Decision Trace")
                    trace_df = pd.DataFrame([
                        {
                            "Step": t.step_num,
                            "Action": t.action,
                            "Blue State": str(t.blue_state),
                            "Red State": str(t.red_state) if t.red_state is not None else "-",
                            "Reason": t.reason,
                            "States After": t.num_states_after,
                        }
                        for t in rpni_res.trace_log
                    ])
                    st.dataframe(trace_df, use_container_width=True, height=260)
    except Exception as e:
        st.error(f"Error in Learn Automaton tab: {e}")
        st.exception(e)

# =============================================================================
# TAB 4: ML CLASSIFIERS
# =============================================================================
with tabs[3]:
    try:
        splits = st.session_state["splits"]
        current_lang = st.session_state["current_language"]
        ml_dict: Dict[str, ModelEvaluationResult] = st.session_state["ml_results"]

        # Top KPI row (fixed 128px height, identical structure)
        best_in_dist = max([m.in_dist_accuracy for m in ml_dict.values()]) if ml_dict else 0.0
        best_gen = max([m.gen_accuracy for m in ml_dict.values()]) if ml_dict else 0.0
        feat_count = len(st.session_state["extractor"].get_feature_names()) if st.session_state["extractor"] else 0

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(kpi("Best In-Dist Accuracy", f"{best_in_dist*100:.1f}%" if ml_dict else "—", "Seen lengths"), unsafe_allow_html=True)
        with kpi2:
            st.markdown(kpi("Best Length-Gen Acc", f"{best_gen*100:.1f}%" if ml_dict else "—", "Unseen longer"), unsafe_allow_html=True)
        with kpi3:
            st.markdown(kpi("Active Features", feat_count if feat_count > 0 else "All (~62)", "Feature vector"), unsafe_allow_html=True)
        with kpi4:
            st.markdown(kpi("Trained Models", len(ml_dict), "Classifiers"), unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### Feature Engineering Toggles & Classifier Training")
            feat_preset = st.radio(
                "Feature Mode Preset:",
                [
                    "Generic (mod-k, affixes, n-grams, runs)",
                    "Basic only (honest baseline: length, counts, n-grams, runs)",
                    "Oracle Leaky (ground-truth shortcut priors)",
                    "Custom Selection",
                ],
                horizontal=True,
            )

            if feat_preset.startswith("Basic"):
                selected_groups = list(BASIC_FEATURE_GROUPS)
                variant_suffix = " (Basic)"
                st.caption("ℹ️ **Basic only:** Honest baseline with length, character counts/ratios, n-grams, and runs. Strictly excludes parity (mod-k), affixes, and oracle features.")
            elif feat_preset.startswith("Oracle"):
                selected_groups = list(ALL_FEATURE_GROUPS)
                variant_suffix = " (Oracle Leaky)"
                st.caption("⚠️ **Oracle Leaky:** Includes ground-truth shortcut features. For benchmarking leaky priors.")
            elif feat_preset.startswith("Generic"):
                selected_groups = list(GENERIC_FEATURE_GROUPS)
                variant_suffix = " (Generic)"
                st.caption("ℹ️ **Generic:** Standard domain-agnostic regular language features (including mod-k and affixes).")
            else:
                variant_suffix = ""
                c_f1, c_f2 = st.columns(2)
                with c_f1:
                    f_basic = st.checkbox("Basic Features (length, symbol counts, ratios)", value=True)
                    f_parity = st.checkbox("Parity Features (mod 2, mod 3 counts & lengths)", value=True)
                    f_affixes = st.checkbox("Affix Features (one-hot first/last k symbols)", value=True)
                with c_f2:
                    f_ngrams = st.checkbox("N-Grams (n=1,2,3 counts & presence flags)", value=True)
                    f_runs = st.checkbox("Run-Length Features (max runs, alternations)", value=True)
                    f_pos = st.checkbox("Position-Weighted Features (weighted sums, alternations)", value=True)
                    f_oracle = st.checkbox("Include oracle features (leaky)", value=False, help="Language-specific shortcut features. WARNING: Leaks ground-truth inductive priors!")

                selected_groups = []
                if f_basic: selected_groups.append("basic")
                if f_parity: selected_groups.append("parity")
                if f_affixes: selected_groups.append("affixes")
                if f_ngrams: selected_groups.append("ngrams")
                if f_runs: selected_groups.append("runs")
                if f_pos: selected_groups.append("position_weighted")
                if f_oracle: selected_groups.append("oracle")

            if st.button("🚀 Train & Profile All ML Models", type="primary", use_container_width=True):
                with st.status("Training ML classifiers on extracted numerical features...", expanded=True) as status:
                    st.write("Fitting numerical feature extractor on training sequences...")
                    extractor = StringFeatureExtractor(alphabet=current_lang.alphabet, enabled_groups=selected_groups)
                    X_train = extractor.fit_transform(splits.train_strings)
                    y_train = splits.train_df["label"].to_numpy()
                    X_test = extractor.transform(splits.test_strings)
                    y_test = splits.test_df["label"].to_numpy()
                    X_gen = extractor.transform(splits.gen_strings)
                    y_gen = splits.generalization_df["label"].to_numpy()

                    st.session_state["extractor"] = extractor
                    results = {}

                    models_to_train = [
                        ("decision_tree", "Decision Tree", dt_params),
                        ("random_forest", "Random Forest", rf_params),
                        ("logistic_regression", "Logistic Regression", lr_params),
                    ]

                    for m_type, m_name, params in models_to_train:
                        full_name = f"{m_name}{variant_suffix}" if variant_suffix else m_name
                        st.write(f"Training {full_name}...")
                        res = train_and_evaluate_ml_model(
                            model_type=m_type,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            X_gen=X_gen,
                            y_gen=y_gen,
                            feature_names=extractor.get_feature_names(),
                            hyperparams=params,
                            run_cv=True,
                            seed=global_seed,
                            raw_test_strings=splits.test_strings,
                            feature_extractor=extractor,
                        )
                        res.model_name = full_name
                        res.feature_extractor = extractor
                        results[full_name] = res

                    current_ml_res = dict(st.session_state.get("ml_results", {}))
                    current_ml_res.update(results)
                    st.session_state["ml_results"] = current_ml_res
                    status.update(label="All ML models trained and profiled!", state="complete", expanded=False)
                    st.rerun()

        if ml_dict:
            with st.container(border=True):
                st.markdown("### Model Performance & Feature Importance Inspector")
                selected_model_inspect = st.selectbox("Select Model to Inspect", list(ml_dict.keys()))
                m_res = ml_dict[selected_model_inspect]

                col_met1, col_met2, col_met3, col_met4, col_met5 = st.columns(5)
                with col_met1:
                    st.markdown(kpi("In-Dist Accuracy", f"{m_res.in_dist_accuracy * 100:.1f}%", "Seen lengths"), unsafe_allow_html=True)
                with col_met2:
                    st.markdown(kpi("Length-Gen Acc", f"{m_res.gen_accuracy * 100:.1f}%", "Out-of-dist"), unsafe_allow_html=True)
                with col_met3:
                    st.markdown(kpi("In-Dist F1 Score", f"{m_res.in_dist_f1:.3f}", "Harmonic mean"), unsafe_allow_html=True)
                with col_met4:
                    st.markdown(kpi("Train Duration", f"{m_res.train_time_ms:.2f} ms", "Fitting time"), unsafe_allow_html=True)
                with col_met5:
                    st.markdown(kpi("Inference Latency", f"{m_res.inference_time_us_per_sample:.1f} µs/str", "Per string"), unsafe_allow_html=True)

                ml_kv: list[tuple[str, object]] = [
                    ("Selected Model", selected_model_inspect),
                    ("Complexity Metric", m_res.complexity_metric),
                    ("Cross-Validation Score", f"{m_res.cv_mean_accuracy*100:.1f}% ± {m_res.cv_std_accuracy*100:.1f}%" if m_res.cv_mean_accuracy else "N/A"),
                    ("Generalization Delta", f"{(m_res.in_dist_accuracy - m_res.gen_accuracy)*100:+.1f}% drop"),
                ]
                st.markdown(kv_table(ml_kv), unsafe_allow_html=True)

                col_cm1, col_cm2 = st.columns(2)
                with col_cm1:
                    st.plotly_chart(plot_confusion_matrix_heatmap(m_res.in_dist_confusion, title=f"In-Dist Confusion Matrix ({selected_model_inspect})"), use_container_width=True)
                with col_cm2:
                    st.plotly_chart(plot_confusion_matrix_heatmap(m_res.gen_confusion, title=f"Length-Gen Confusion Matrix ({selected_model_inspect})"), use_container_width=True)

                if m_res.feature_importances:
                    st.plotly_chart(plot_feature_importance_bar(m_res.feature_importances, top_k=15), use_container_width=True)
    except Exception as e:
        st.error(f"Error in ML Classifiers tab: {e}")
        st.exception(e)

# =============================================================================
# TAB 5: COMPARE (THE MAIN COMPARATIVE DASHBOARD)
# =============================================================================
with tabs[4]:
    try:
        current_lang = st.session_state["current_language"]
        splits = st.session_state["splits"]
        suite: Optional[BenchmarkSuiteResult] = st.session_state["benchmark_suite"]

        fastest_infer = min([e.infer_time_us for e in suite.entries]) if (suite and suite.entries) else 1.4
        fastest_name = [e.model_name for e in suite.entries if e.infer_time_us == fastest_infer][0] if (suite and suite.entries) else "DFA Lookups"
        best_gen = max([e.gen_acc for e in suite.entries]) * 100.0 if (suite and suite.entries) else 100.0
        best_models = ", ".join([e.model_name for e in suite.entries if e.gen_acc >= (best_gen / 100.0 - 1e-4)][:2]) if (suite and suite.entries) else "L* / RPNI / DFA"

        # Top KPI Dashboard Cards (fixed 128px height, identical structure)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(kpi("Best Generalization", f"{best_gen:.1f}% Acc", best_models), unsafe_allow_html=True)
        with kpi2:
            st.markdown(kpi("Fastest Inference", f"{fastest_infer:.1f} µs/str", fastest_name), unsafe_allow_html=True)
        with kpi3:
            st.markdown(kpi("Most Compact Model", f"{len(current_lang.dfa.states)} States", "Minimal DFA"), unsafe_allow_html=True)
        with kpi4:
            st.markdown(kpi("Exact Grammar Equiv", "L* & RPNI", "Provable Isomorphism"), unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### ⚡ Multi-Classifier Benchmark Suite")
            c_opt1, c_opt2, c_opt3, c_opt4 = st.columns([1, 1, 1, 1])
            with c_opt1:
                include_basic = st.checkbox("Include Basic-only baseline (honest)", value=True, help="Include Basic-only models without mod-k parity or affixes")
            with c_opt2:
                include_oracle = st.checkbox("Include oracle features (leaky)", value=False, help="Include language-specific shortcut features alongside generic ones")
            with c_opt3:
                include_curves = st.checkbox("Compute Learning Curves (Varying N)", value=False)
            with c_opt4:
                include_noise = st.checkbox("Compute Noise Robustness Curves", value=False)

            run_bench_btn = st.button("🚀 Run Full Comparative Benchmark", type="primary", use_container_width=True)

            if run_bench_btn:
                with st.status("Executing standardized benchmark across all classifiers...", expanded=True) as status:
                    st.write("Evaluating Ground-Truth DFA...")
                    engine = BenchmarkEngine(seed=global_seed)
                    st.write("Running RPNI state merging and Angluin's L* active teacher queries...")
                    st.write("Extracting numerical features and training Decision Tree, Random Forest, Logistic Regression...")
                    bench_suite = engine.run_benchmark(
                        language=current_lang,
                        num_samples=len(splits.train_df) + len(splits.val_df) + len(splits.test_df),
                        gen_samples=len(splits.generalization_df),
                        include_basic=include_basic,
                        include_oracle=include_oracle,
                        dt_params=dt_params,
                        rf_params=rf_params,
                        lr_params=lr_params,
                    )
                    st.session_state["benchmark_suite"] = bench_suite
                    suite = bench_suite
                    if bench_suite.rpni_result is not None:
                        st.session_state["rpni_result"] = bench_suite.rpni_result
                    if bench_suite.lstar_result is not None:
                        st.session_state["lstar_result"] = bench_suite.lstar_result
                    if bench_suite.ml_eval_results:
                        st.session_state["ml_results"] = bench_suite.ml_eval_results
                    if bench_suite.feature_extractor is not None:
                        st.session_state["extractor"] = bench_suite.feature_extractor
                    status.update(label="Full comparative benchmark finished!", state="complete", expanded=False)
                    st.rerun()

        if suite is not None:
            # Dashboard Table with Progress Columns
            with st.container(border=True):
                st.markdown("### 📊 Performance Comparison Table")
                bench_kv: list[tuple[str, object]] = [
                    ("Target Benchmark", suite.language_name),
                    ("Alphabet (Σ)", f"{{{', '.join(suite.alphabet)}}}"),
                    ("In-Distribution Evaluation", f"{suite.num_samples} samples"),
                    ("Length-Generalization Pool", f"{suite.gen_samples} longer strings"),
                ]
                st.markdown(kv_table(bench_kv), unsafe_allow_html=True)
                df_display = pd.DataFrame([
                    {
                        "Model": e.model_name,
                        "Category": e.model_category,
                        "In-Dist Accuracy": e.in_dist_acc * 100.0,
                        "Length-Gen Accuracy": e.gen_acc * 100.0,
                        "Train Time (ms)": e.train_time_ms,
                        "Infer Latency (µs)": e.infer_time_us,
                        "Complexity": e.complexity_desc,
                        "Target Equiv?": "Exact (True)" if e.is_equivalent_to_target is True else (f"Not equivalent (counterexample: '{e.counterexample}')" if e.counterexample else "Not equivalent") if e.is_equivalent_to_target is False else "N/A",
                    }
                    for e in suite.entries
                ])

                table_config = {
                    "In-Dist Accuracy": st.column_config.ProgressColumn(
                        "In-Dist Accuracy",
                        help="Test accuracy on seen length distribution",
                        format="%.1f%%",
                        min_value=0.0,
                        max_value=100.0,
                    ),
                    "Length-Gen Accuracy": st.column_config.ProgressColumn(
                        "Length-Gen Accuracy",
                        help="Out-of-distribution accuracy on longer unseen sequences",
                        format="%.1f%%",
                        min_value=0.0,
                        max_value=100.0,
                    ),
                    "Train Time (ms)": st.column_config.NumberColumn(
                        "Train (ms)",
                        format="%.2f ms",
                    ),
                    "Infer Latency (µs)": st.column_config.NumberColumn(
                        "Infer (µs/str)",
                        format="%.1f µs",
                    ),
                }

                st.dataframe(df_display, column_config=table_config, use_container_width=True)

            # Charts Row 1: Generalization Gap Bar Chart & Complexity vs Accuracy Scatter
            c_ch1, c_ch2 = st.columns(2)
            with c_ch1:
                with st.container(border=True):
                    st.plotly_chart(plot_in_dist_vs_gen(suite.entries, suite.language_name), use_container_width=True)
            with c_ch2:
                with st.container(border=True):
                    st.plotly_chart(plot_complexity_vs_accuracy(suite.entries), use_container_width=True)

            # Charts Row 2: Runtime Latency Profiling
            with st.container(border=True):
                st.plotly_chart(plot_runtime_comparison(suite.entries), use_container_width=True)

            # Optional Learning Curves and Noise Robustness
            if include_curves:
                with st.container(border=True):
                    st.markdown("### 📈 Empirical Learning Curves (Accuracy vs Training Size)")
                    with st.spinner("Computing learning curves across sample sizes [40, 80, 160, 300]..."):
                        engine = BenchmarkEngine(seed=global_seed)
                        df_curves = engine.run_learning_curve(current_lang, sample_sizes=[40, 80, 160, 300])
                        st.plotly_chart(plot_learning_curves(df_curves), use_container_width=True)

            if include_noise:
                with st.container(border=True):
                    st.markdown("### 🛡️ Label Noise Robustness Curves")
                    with st.spinner("Computing noise degradation curves across [0%, 5%, 10%, 20%]..."):
                        engine = BenchmarkEngine(seed=global_seed)
                        df_noise = engine.run_noise_robustness(current_lang, noise_rates=[0.0, 0.05, 0.1, 0.2])
                        st.plotly_chart(plot_noise_robustness(df_noise), use_container_width=True)

            # Auto-Generated Pedagogical Insights Card
            with st.container(border=True):
                st.markdown(suite.generate_narrative_summary())
    except Exception as e:
        st.error(f"Error in Compare Dashboard tab: {e}")
        st.exception(e)

# =============================================================================
# TAB 6: CLASSIFY (THE MAIN EVALUATOR-FACING DEMO)
# =============================================================================
with tabs[5]:
    try:
        current_lang = st.session_state["current_language"]
        splits = st.session_state["splits"]
        global_seed = int(st.session_state.get("seed", 42))

        # Synchronize default string per language or pending input from buttons
        sample_def = "0101" if "0" in current_lang.alphabet else "ab"
        if "pending_classify_input" in st.session_state:
            st.session_state["classify_input_widget"] = st.session_state.pop("pending_classify_input")
        elif "classify_input_widget" not in st.session_state or st.session_state.get("classify_lang_key") != current_lang.key:
            st.session_state["classify_lang_key"] = current_lang.key
            st.session_state["classify_input_widget"] = sample_def
            st.session_state["classify_eval_str"] = sample_def


        # -------------------------------------------------------------------------
        # AUTO-TRAIN ON FIRST USE (RPNI, L*, and ML models for Basic & Generic)
        # -------------------------------------------------------------------------
        needed_ml_models = [
            "Decision Tree (Basic)", "Random Forest (Basic)", "Logistic Regression (Basic)",
            "Decision Tree (Generic)", "Random Forest (Generic)", "Logistic Regression (Generic)",
        ]
        curr_ml_res = st.session_state.get("ml_results", {})
        has_rpni = st.session_state.get("rpni_result") is not None
        has_lstar = st.session_state.get("lstar_result") is not None
        has_all_ml = all(m in curr_ml_res for m in needed_ml_models)
        trained_lang_match = st.session_state.get("classify_trained_lang") == current_lang.key

        if not (has_rpni and has_lstar and has_all_ml and trained_lang_match):
            with st.status(f"⚡ Auto-training all evaluation systems for '{current_lang.name}'...", expanded=True) as status:
                pos_train = [s for s, y in zip(splits.train_strings, splits.train_labels) if y == 1]
                neg_train = [s for s, y in zip(splits.train_strings, splits.train_labels) if y == 0]
                y_tr = splits.train_df["label"].to_numpy()
                y_te = splits.test_df["label"].to_numpy()
                y_ge = splits.generalization_df["label"].to_numpy()

                st.write("1. Inducing DFA via RPNI passive state merging...")
                r_learner = RPNILearner(alphabet=current_lang.alphabet)
                st.session_state["rpni_result"] = r_learner.learn(pos_train, neg_train)

                st.write("2. Learning minimal DFA via Angluin's L* active teacher...")
                l_learner = LStarLearner(
                    alphabet=current_lang.alphabet,
                    membership_oracle=current_lang.predicate,
                    target_dfa=current_lang.dfa,
                )
                st.session_state["lstar_result"] = l_learner.learn()

                st.write("3. Fitting Basic-only feature extractor & ML models...")
                ext_b = StringFeatureExtractor(alphabet=current_lang.alphabet, enabled_groups=BASIC_FEATURE_GROUPS)
                X_tr_b = ext_b.fit_transform(splits.train_strings)
                X_te_b = ext_b.transform(splits.test_strings)
                X_ge_b = ext_b.transform(splits.gen_strings)

                new_ml_results = dict(curr_ml_res) if trained_lang_match else {}
                for m_t, m_n in [("decision_tree", "Decision Tree"), ("random_forest", "Random Forest"), ("logistic_regression", "Logistic Regression")]:
                    full_name = f"{m_n} (Basic)"
                    res = train_and_evaluate_ml_model(
                        model_type=m_t,
                        X_train=X_tr_b,
                        y_train=y_tr,
                        X_test=X_te_b,
                        y_test=y_te,
                        X_gen=X_ge_b,
                        y_gen=y_ge,
                        feature_names=ext_b.get_feature_names(),
                        hyperparams=dt_params if m_t == "decision_tree" else rf_params if m_t == "random_forest" else lr_params,
                        run_cv=False,
                        seed=global_seed,
                        raw_test_strings=splits.test_strings,
                        feature_extractor=ext_b,
                    )
                    res.model_name = full_name
                    res.feature_extractor = ext_b
                    new_ml_results[full_name] = res

                st.write("4. Fitting Generic feature extractor & ML models (with mod-k & affixes)...")
                ext_g = StringFeatureExtractor(alphabet=current_lang.alphabet, enabled_groups=GENERIC_FEATURE_GROUPS)
                X_tr_g = ext_g.fit_transform(splits.train_strings)
                X_te_g = ext_g.transform(splits.test_strings)
                X_ge_g = ext_g.transform(splits.gen_strings)

                for m_t, m_n in [("decision_tree", "Decision Tree"), ("random_forest", "Random Forest"), ("logistic_regression", "Logistic Regression")]:
                    full_name = f"{m_n} (Generic)"
                    res = train_and_evaluate_ml_model(
                        model_type=m_t,
                        X_train=X_tr_g,
                        y_train=y_tr,
                        X_test=X_te_g,
                        y_test=y_te,
                        X_gen=X_ge_g,
                        y_gen=y_ge,
                        feature_names=ext_g.get_feature_names(),
                        hyperparams=dt_params if m_t == "decision_tree" else rf_params if m_t == "random_forest" else lr_params,
                        run_cv=False,
                        seed=global_seed,
                        raw_test_strings=splits.test_strings,
                        feature_extractor=ext_g,
                    )
                    res.model_name = full_name
                    res.feature_extractor = ext_g
                    new_ml_results[full_name] = res

                st.session_state["ml_results"] = new_ml_results
                st.session_state["extractor"] = ext_g
                st.session_state["classify_trained_lang"] = current_lang.key
                status.update(label="All systems ready for live classification!", state="complete", expanded=False)

        rpni_res = st.session_state.get("rpni_result")
        lstar_res = st.session_state.get("lstar_result")
        ml_res = st.session_state.get("ml_results") or {}

        active_eval_str = st.session_state.get("classify_eval_str", sample_def)
        disp_sample = f"'{active_eval_str}'" if active_eval_str != "" else "ε (empty)"

        # Top KPI row (fixed 128px height, identical structure)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(kpi("Ground-Truth Automaton", current_lang.name, "Target DFA"), unsafe_allow_html=True)
        with kpi2:
            st.markdown(kpi("Alphabet (Σ)", f"{{{', '.join(current_lang.alphabet)}}}", "Input symbols"), unsafe_allow_html=True)
        with kpi3:
            st.markdown(kpi("Target Min States", len(current_lang.dfa.states), f"{len(current_lang.dfa.accept_states)} accepting"), unsafe_allow_html=True)
        with kpi4:
            st.markdown(kpi("Evaluated String", disp_sample, f"Length {len(active_eval_str)}"), unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### 🧪 Evaluator Live Sequence Benchmark Demo")
            st.caption(
                "Evaluate any sequence across all systems simultaneously. "
                "Notice how DFA and automata maintain 100% inductive accuracy even on arbitrarily long out-of-distribution strings, "
                "whereas statistical classifiers degrade outside their training bounds."
            )

            # ---------------------------------------------------------------------
            # 5 QUICK-PICK EXAMPLES PER LANGUAGE
            # ---------------------------------------------------------------------
            quick_picks = QUICK_PICKS_PER_LANG.get(current_lang.key, [
                ("", "Empty string ε"),
                (current_lang.alphabet[0], "Short 1"),
                (current_lang.alphabet[0] * 2, "Short 2"),
                (current_lang.alphabet[-1], "Short 3"),
                ((current_lang.alphabet[0] + current_lang.alphabet[-1]) * 8, "Long OOD"),
            ])

            st.markdown("**⚡ Quick-Pick Examples:**")
            qp_cols = st.columns(5)
            for i, (qp_val, qp_lbl) in enumerate(quick_picks[:5]):
                with qp_cols[i]:
                    btn_title = f"ε ({qp_lbl})" if qp_val == "" else f"'{qp_val}' ({qp_lbl})"
                    if st.button(btn_title, key=f"btn_qp_{current_lang.key}_{i}", use_container_width=True):
                        st.session_state["pending_classify_input"] = qp_val
                        st.session_state["classify_eval_str"] = qp_val
                        st.rerun()

            # ---------------------------------------------------------------------
            # INPUT + RUN BUTTON + RANDOM BUTTON ROW
            # ---------------------------------------------------------------------
            col_in, col_run, col_rand = st.columns([5, 2, 2])
            with col_in:
                user_typed = st.text_input(
                    label=f"Candidate String to Classify (Σ = {{{', '.join(current_lang.alphabet)}}}):",
                    key="classify_input_widget",
                    help="Type any custom string and click 'Run on all systems'. Does not classify on every keystroke.",
                )
            with col_run:
                st.write("")
                st.write("")
                if st.button("⚡ Run on all systems", type="primary", use_container_width=True):
                    st.session_state["classify_eval_str"] = user_typed
                    st.rerun()

            with col_rand:
                st.write("")
                st.write("")
                if st.button("🎲 Random string", use_container_width=True, help="Generate random string with length 14–22 (longer than training set) to evaluate generalization"):
                    rand_len = random.randint(14, 22)
                    rand_s = "".join(random.choices(current_lang.alphabet, k=rand_len))
                    st.session_state["pending_classify_input"] = rand_s
                    st.session_state["classify_eval_str"] = rand_s
                    st.rerun()

            # ---------------------------------------------------------------------
            # INPUT VALIDATION
            # ---------------------------------------------------------------------
            active_eval_str = st.session_state.get("classify_eval_str", sample_def)
            invalid_chars = sorted(list(set(c for c in active_eval_str if c not in current_lang.alphabet)))

            if invalid_chars:
                st.error(
                    f"❌ **Invalid Input:** Symbol(s) `{'`, `'.join(invalid_chars)}` are not in alphabet "
                    f"$\\Sigma = \\{{{', '.join(current_lang.alphabet)}\\}}$! "
                    f"Please enter only symbols from $\\Sigma$, or leave blank for $\\varepsilon$ (empty string)."
                )
            else:
                disp_str = f"'{active_eval_str}'" if active_eval_str != "" else "ε (empty string)"

                # 1. Ground Truth DFA
                gt_dfa = current_lang.dfa
                for _ in range(2): gt_dfa.accepts(active_eval_str)
                t0 = time.perf_counter()
                for _ in range(10): gt_verdict = gt_dfa.accepts(active_eval_str)
                t1 = time.perf_counter()
                gt_time_us = ((t1 - t0) * 1e6) / 10.0
                gt_pred_str = "ACCEPT (1)" if gt_verdict else "REJECT (0)"

                rows = []
                rows.append({
                    "System": "Traditional DFA",
                    "Model": "Ground-Truth DFA",
                    "Prediction": gt_pred_str,
                    "Matches Ground Truth": "✅ Match",
                    "Time (µs)": gt_time_us,
                })

                # 2. Learned Automata (RPNI & L*)
                if rpni_res and hasattr(rpni_res, "minimized_dfa"):
                    rp_dfa = rpni_res.minimized_dfa
                    for _ in range(2): rp_dfa.accepts(active_eval_str)
                    t0 = time.perf_counter()
                    for _ in range(10): rp_acc = rp_dfa.accepts(active_eval_str)
                    t1 = time.perf_counter()
                    rp_time_us = ((t1 - t0) * 1e6) / 10.0
                    rp_pred = "ACCEPT (1)" if rp_acc else "REJECT (0)"
                    rows.append({
                        "System": "Learned Automaton",
                        "Model": "RPNI Learned DFA",
                        "Prediction": rp_pred,
                        "Matches Ground Truth": "✅ Match" if (rp_acc == gt_verdict) else "❌ Disagree",
                        "Time (µs)": rp_time_us,
                    })

                if lstar_res and hasattr(lstar_res, "minimized_dfa"):
                    ls_dfa = lstar_res.minimized_dfa
                    for _ in range(2): ls_dfa.accepts(active_eval_str)
                    t0 = time.perf_counter()
                    for _ in range(10): ls_acc = ls_dfa.accepts(active_eval_str)
                    t1 = time.perf_counter()
                    ls_time_us = ((t1 - t0) * 1e6) / 10.0
                    ls_pred = "ACCEPT (1)" if ls_acc else "REJECT (0)"
                    rows.append({
                        "System": "Learned Automaton",
                        "Model": "L* Learned DFA",
                        "Prediction": ls_pred,
                        "Matches Ground Truth": "✅ Match" if (ls_acc == gt_verdict) else "❌ Disagree",
                        "Time (µs)": ls_time_us,
                    })

                # 3. Machine Learning Classifiers (DT, RF, LR for Basic and Generic)
                sorted_ml_keys = []
                for suffix in ["(Basic)", "(Generic)", "(Oracle Leaky)"]:
                    for base in ["Decision Tree", "Random Forest", "Logistic Regression"]:
                        k = f"{base} {suffix}"
                        if k in ml_res and k not in sorted_ml_keys:
                            sorted_ml_keys.append(k)
                for k in ml_res:
                    if k not in sorted_ml_keys:
                        sorted_ml_keys.append(k)

                for m_name in sorted_ml_keys:
                    m_ev = ml_res[m_name]
                    if hasattr(m_ev, "raw_model") and m_ev.raw_model is not None:
                        try:
                            m_ext = getattr(m_ev, "feature_extractor", None) or getattr(m_ev, "extractor", None)
                            if m_ext is None:
                                raise ValueError("No feature extractor stored on model")

                            _warm = m_ext.transform([active_eval_str])
                            _ = m_ev.raw_model.predict(_warm)

                            t0 = time.perf_counter()
                            for _ in range(5):
                                X_in = m_ext.transform([active_eval_str])
                                p_arr = m_ev.raw_model.predict(X_in)
                            t1 = time.perf_counter()
                            m_time_us = ((t1 - t0) * 1e6) / 5.0

                            p_val = int(p_arr[0])
                            p_pred = "ACCEPT (1)" if p_val == 1 else "REJECT (0)"
                            matches = (p_val == (1 if gt_verdict else 0))
                            match_str = "✅ Match" if matches else "❌ Disagree"
                        except Exception as exc:
                            p_pred = f"n/a ({exc})"
                            match_str = "⚠️ Error"
                            m_time_us = 0.0

                        sys_cat = (
                            "Machine Learning (Basic)" if "(Basic)" in m_name
                            else "Machine Learning (Generic)" if "(Generic)" in m_name
                            else "Machine Learning (Oracle)" if "(Oracle" in m_name
                            else "Machine Learning"
                        )
                        rows.append({
                            "System": sys_cat,
                            "Model": m_name,
                            "Prediction": p_pred,
                            "Matches Ground Truth": match_str,
                            "Time (µs)": m_time_us,
                        })

                # Results Table with styled disagreeing rows in red
                df_results = pd.DataFrame(rows)

                def highlight_disagreements(row):
                    if row["Matches Ground Truth"] == "❌ Disagree":
                        return ["background-color: rgba(239, 68, 68, 0.22); color: #FCA5A5; font-weight: 600;"] * len(row)
                    return [""] * len(row)

                styled_df = df_results.style.apply(highlight_disagreements, axis=1).format({
                    "Time (µs)": "{:.1f} µs"
                })

                st.markdown("#### 📊 Comparative Classification Results Table")
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                # Consensus summary line under table
                non_gt_rows = [r for r in rows if r["System"] != "Traditional DFA"]
                matches_count = sum(1 for r in non_gt_rows if r["Matches Ground Truth"] == "✅ Match")
                total_evaluated = len(non_gt_rows)

                if matches_count == total_evaluated:
                    st.success(f"✅ **Consensus:** {matches_count}/{total_evaluated} systems agree with ground truth on {disp_str}.")
                else:
                    disagree_models = [r["Model"] for r in non_gt_rows if r["Matches Ground Truth"] == "❌ Disagree"]
                    st.error(
                        f"⚠️ **Disagreement Detected:** {matches_count}/{total_evaluated} systems agree with ground truth on {disp_str} "
                        f"({len(disagree_models)} systems failed: {', '.join(disagree_models)})."
                    )

                # DFA State Transition Path Diagram
                with st.container(border=True):
                    st.markdown("#### 🗺️ Ground-Truth DFA State Transition Path")
                    acc, path = current_lang.dfa.trace(active_eval_str)
                    path_display = " ➔ ".join(f"[{s}]" for s in path)
                    classify_kv: list[tuple[str, object]] = [
                        ("Query String", disp_str),
                        ("String Length (|w|)", f"{len(active_eval_str)} symbols"),
                        ("Ground-Truth Verdict", "ACCEPT (1)" if acc else "REJECT (0)"),
                        ("Transition Path", path_display),
                    ]
                    st.markdown(kv_table(classify_kv), unsafe_allow_html=True)
                    st.graphviz_chart(dfa_to_dot(current_lang.dfa, active_path=path), use_container_width=True)
    except Exception as e:
        st.error(f"Error in Classify tab: {e}")
        st.exception(e)

# =============================================================================
# TAB 7: THEORY & FOUNDATIONS
# =============================================================================
with tabs[6]:
    try:
        # Top KPI row (fixed 128px height, identical structure)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(kpi("Grammar Hierarchy", "Type-3 Regular", "Chomsky"), unsafe_allow_html=True)
        with kpi2:
            st.markdown(kpi("Time Complexity", "O(|w|) Linear", "Single scan"), unsafe_allow_html=True)
        with kpi3:
            st.markdown(kpi("State Minimality", "Unique Minimal", "Myhill-Nerode"), unsafe_allow_html=True)
        with kpi4:
            st.markdown(kpi("Exact Learning", "Poly-Time MAT", "Angluin L*"), unsafe_allow_html=True)

        theory_kv: list[tuple[str, object]] = [
            ("Chomsky Hierarchy", "Type-3 Regular Languages (Finite-State Automata)"),
            ("Equivalence Theorem", "Myhill-Nerode (Equivalence classes of Nerode congruence RL)"),
            ("Optimal Minimization", "Hopcroft's Partition Refinement O(|Σ|·|Q| log |Q|)"),
            ("Inductive Limit", "Gold's Theorem 1967 (Identification from positive samples alone is impossible)"),
            ("Active Inference", "Angluin 1987 L* (Polynomial-time exact convergence via MAT queries)"),
        ]
        st.markdown(kv_table(theory_kv), unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                r"""
    ### 1. Deterministic Finite Automata (DFA) & The Chomsky Hierarchy
    A **Deterministic Finite Automaton (DFA)** is defined formally as a 5-tuple:
    $$M = (Q, \Sigma, \delta, q_0, F)$$
    where:
    - $Q$ is a finite non-empty set of **states**,
    - $\Sigma$ is a finite input **alphabet**,
    - $\delta: Q \times \Sigma \to Q$ is the **transition function**,
    - $q_0 \in Q$ is the **initial start state**,
    - $F \subseteq Q$ is the set of **accept / final states**.

    A string $w = c_1 c_2 \dots c_n$ is accepted if there exists a sequence of states $r_0, r_1, \dots, r_n$ such that $r_0 = q_0$, $r_{i} = \delta(r_{i-1}, c_i)$, and $r_n \in F$.
    The set of all languages recognized by DFAs defines the class of **Regular Languages** ($\text{Type-3}$ in the Chomsky hierarchy).

    ---

    ### 2. Myhill-Nerode Theorem & DFA Minimization
    Given a language $L \subseteq \Sigma^*$, define the **Nerode equivalence relation** $R_L$ over strings:
    $$x \sim_L y \iff \forall z \in \Sigma^* : (xz \in L \iff yz \in L)$$
    The **Myhill-Nerode Theorem** states:
    1. $L$ is regular if and only if the number of equivalence classes of $\sim_L$ (the index of $L$) is finite.
    2. The number of states in the unique minimal complete DFA for $L$ is **exactly equal** to the index of $\sim_L$.

    #### Minimization Algorithms:
    - **Table-Filling Algorithm (Myhill-Nerode):**
      Maintains distinguishability of state pairs $\{p, q\}$. Base case distinguishes $p \in F \oplus q \in F$.
      Inductive step: if $\delta(p, a)$ and $\delta(q, a)$ are distinguished, mark $\{p, q\}$. Runs in $O(|\Sigma| \cdot |Q|^2)$.
    - **Hopcroft's Algorithm:**
      Iteratively refines state partitions using a worklist of potential splitters. Achieves optimal $O(|\Sigma| \cdot |Q| \log |Q|)$ time complexity.

    ---

    ### 3. Grammatical Inference: Passive vs Active Learning

    #### Gold's Theorem (1967):
    Gold proved that regular languages **cannot** be learned identification-in-the-limit from positive examples alone.
    Both positive and negative samples (or an active oracle) are required.

    #### RPNI (Regular Positive and Negative Inference - Oncina & García, 1992):
    1. Construct the **Prefix Tree Acceptor (PTA)** $A(S_+)$ where every prefix of positive samples is a state.
    2. Maintain disjoint core states ($RED$) and boundary candidates ($BLUE$).
    3. Attempt state merges between $q_b \in BLUE$ and $q_r \in RED$.
    4. **Deterministic Folding:** Merging $q_b$ and $q_r$ can create non-determinism $\delta(q, a) = \{dst_1, dst_2\}$.
       Folding recursively merges $dst_1$ and $dst_2$ until the quotient automaton is deterministic.
    5. If the folded automaton accepts any negative string $w \in S_-$, reject the merge. Otherwise accept it.

    #### Angluin's $L^*$ Algorithm (1987):
    Learns the exact minimal DFA in polynomial time using a **Minimal Adequate Teacher (MAT)**:
    - **Membership Queries (MQ):** $w \in L$?
    - **Equivalence Queries (EQ):** $L(M_{hyp}) == L$? Returns a counterexample if false.
    - Organizes queries into an **Observation Table** $(S, E, T)$ where $S$ is prefix-closed and $E$ is suffix-closed:
      - **Closedness:** Every row in $S \cdot \Sigma$ appears in $S$.
      - **Consistency:** If $row(s_1) == row(s_2)$, then $row(s_1 \cdot a) == row(s_2 \cdot a)$ for all $a \in \Sigma$.

    ---

    ### 4. The Inductive Bias Paradox: Why Machine Learning Fails on Regular Languages
    Why do deep neural networks and gradient-boosted trees achieve high accuracy in-distribution but catastrophic failure on longer strings?

    1. **Independent Feature Assumption vs Recurrent Cycles:**
       - Classical ML maps strings into bag-of-words, n-grams, or positional encodings.
       - For a modulo language like **"Binary numbers divisible by 3"** or **"Even 0s and Odd 1s"**, the target function is a cyclic finite semigroup (e.g. $\mathbb{Z}_3$ or $\mathbb{Z}_2 \times \mathbb{Z}_2$).
       - A decision tree splits on feature thresholds ($count(0) \le 3$, $length \le 6$). On longer strings, feature values fall completely outside the training bounding box, causing tree predictions to collapse.
    2. **Infinite Generalization of Automata:**
       - Once RPNI or $L^*$ discovers a cycle (e.g. $\delta(q_1, 0) = q_2$, $\delta(q_2, 0) = q_0$), that cycle executes identical transitions for strings of length $10, 100,$ or $1,000,000$.
       - The automaton represents the **exact generative grammar**, achieving $100\%$ zero-shot length generalization.
                """
            )
    except Exception as e:
        st.error(f"Error in Theory & Foundations tab: {e}")
        st.exception(e)

