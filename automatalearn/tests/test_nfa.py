import pytest
from automatalearn.core.nfa import NFA
from automatalearn.core.dfa import DFA
from automatalearn.core.regex_to_nfa import regex_to_nfa, RegexParser
from automatalearn.core.minimize import minimize_dfa


def test_nfa_epsilon_closure():
    """Verify epsilon closure computation."""
    transitions = {
        (0, ""): {1, 2},
        (1, ""): {3},
        (2, "a"): {2},
        (3, ""): {4},
    }
    nfa = NFA(
        states={0, 1, 2, 3, 4},
        alphabet={"a"},
        transitions=transitions,
        start_states={0},
        accept_states={4},
    )
    closure = nfa.epsilon_closure({0})
    assert closure == {0, 1, 2, 3, 4}

    closure_2 = nfa.epsilon_closure({2})
    assert closure_2 == {2}


def test_nfa_subset_construction():
    """Verify conversion of NFA to DFA preserves language equivalence."""
    # Language: strings ending in 'ab'
    transitions = {
        (0, "a"): {0, 1},
        (0, "b"): {0},
        (1, "b"): {2},
    }
    nfa = NFA(
        states={0, 1, 2},
        alphabet={"a", "b"},
        transitions=transitions,
        start_states={0},
        accept_states={2},
    )
    dfa = nfa.to_dfa()

    test_strings = ["ab", "aab", "bab", "bbaab", "", "a", "b", "aba", "abb"]
    for s in test_strings:
        assert nfa.accepts(s) == dfa.accepts(s)


def test_regex_to_nfa_thompson():
    """Test compiling regular expressions to NFA and determinizing."""
    # Language (a|b)*ab
    nfa = regex_to_nfa("(a|b)*ab")
    dfa = nfa.to_dfa()

    assert dfa.accepts("ab")
    assert dfa.accepts("aab")
    assert dfa.accepts("babab")
    assert not dfa.accepts("")
    assert not dfa.accepts("a")
    assert not dfa.accepts("ba")


def test_regex_complex():
    """Test regex with union (+), optional (?), and nested expressions."""
    # Language a + b?c: accepts 'a' OR ('b'? . 'c' => 'bc' or 'c')
    nfa = regex_to_nfa("a+b?c")
    dfa = nfa.to_dfa()

    assert dfa.accepts("a")
    assert dfa.accepts("bc")
    assert dfa.accepts("c")
    assert not dfa.accepts("abc")
    assert not dfa.accepts("aaabc")
    assert not dfa.accepts("ab")
    assert not dfa.accepts("")


def test_regex_union_equivalence():
    """Verify that '+' and '|' both behave as standard regular-expression union operators."""
    dfa_plus = regex_to_nfa("(a+b)*ab").to_dfa()
    dfa_pipe = regex_to_nfa("(a|b)*ab").to_dfa()

    is_equiv, ce = dfa_plus.equivalent_to(dfa_pipe)
    assert is_equiv, f"DFAs for (a+b)*ab and (a|b)*ab should be equivalent, counterexample: {ce}"


def test_regex_operator_precedence():
    """Verify TOC operator precedence: * / ? > concat > + / |."""
    # ab+c means (ab)+c
    dfa1 = regex_to_nfa("ab+c").to_dfa()
    assert dfa1.accepts("ab")
    assert dfa1.accepts("c")
    assert not dfa1.accepts("abc")
    assert not dfa1.accepts("a")
    assert not dfa1.accepts("b")

    # a+bc means a+(bc)
    dfa2 = regex_to_nfa("a+bc").to_dfa()
    assert dfa2.accepts("a")
    assert dfa2.accepts("bc")
    assert not dfa2.accepts("abc")
    assert not dfa2.accepts("b")
    assert not dfa2.accepts("c")

    # (a+b)c means (a+b)c
    dfa3 = regex_to_nfa("(a+b)c").to_dfa()
    assert dfa3.accepts("ac")
    assert dfa3.accepts("bc")
    assert not dfa3.accepts("abc")
    assert not dfa3.accepts("a")
    assert not dfa3.accepts("c")

    # ab(c+d) means ab(c+d)
    dfa4 = regex_to_nfa("ab(c+d)").to_dfa()
    assert dfa4.accepts("abc")
    assert dfa4.accepts("abd")
    assert not dfa4.accepts("ab")
    assert not dfa4.accepts("c")
    assert not dfa4.accepts("d")


def test_regex_problematic_target():
    """Verify target regex (abb)ba(a+b) accepts exactly L = {abbbaa, abbbab}."""
    dfa = regex_to_nfa("(abb)ba(a+b)").to_dfa()
    min_dfa = minimize_dfa(dfa).minimized_dfa

    # Exactly these two strings must be accepted
    assert dfa.accepts("abbbaa")
    assert dfa.accepts("abbbab")
    assert min_dfa.accepts("abbbaa")
    assert min_dfa.accepts("abbbab")

    # Explicit rejection cases specified by user prompt
    reject_cases = [
        "", "a", "b", "ab", "abb", "abba", "abbba", "abbbb",
        "abbaba", "abbabb", "abbbaaa", "abbbaab", "abbbaaba",
        "abbbaba", "abbbaaab", "abbbabb", "aabbbaa", "babbbaa",
    ]
    for s in reject_cases:
        assert not dfa.accepts(s), f"DFA unexpectedly accepted: {s!r}"
        assert not min_dfa.accepts(s), f"Min DFA unexpectedly accepted: {s!r}"


def test_regex_syntax_validation():
    """Verify that syntax errors are caught and raise clear descriptive ValueErrors."""
    parser = RegexParser()

    with pytest.raises(ValueError, match="leading binary operator"):
        parser.compile("+a")
    with pytest.raises(ValueError, match="leading binary operator"):
        parser.compile("|a")
    with pytest.raises(ValueError, match="leading repetition operator"):
        parser.compile("*a")
    with pytest.raises(ValueError, match="trailing binary operator"):
        parser.compile("a+")
    with pytest.raises(ValueError, match="consecutive operators"):
        parser.compile("a++b")
    with pytest.raises(ValueError, match="empty parentheses"):
        parser.compile("a()b")
    with pytest.raises(ValueError, match="unclosed parenthesis"):
        parser.compile("(a+b")
    with pytest.raises(ValueError, match="unexpected closing parenthesis"):
        parser.compile("a+b)")
    with pytest.raises(ValueError, match="repetition operator"):
        parser.compile("a+*b")
