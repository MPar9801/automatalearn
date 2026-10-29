"""Comprehensive Verification Suite for Regex-to-DFA Repair.

Covers:
- Phase 4: Independent reference tester (Brzozowski derivative engine) over all
  511 binary strings of lengths 0 to 8 for (abb)ba(a+b).
- Phase 5: Formal DFA 5-tuple M = (Q, Σ, δ, q0, F) mathematical verification.
- Phase 6: Adversarial and stress tests (long inputs up to 10,000 characters,
  single-symbol mutations, prefixes, suffixes, invalid symbols).
- Phase 7: End-to-end integration and consistency.
"""

from __future__ import annotations
import itertools
import pytest
from typing import Dict, List, Optional, Set, Tuple

from automatalearn.core.dfa import DFA, State
from automatalearn.core.nfa import NFA
from automatalearn.core.regex_to_nfa import RegexParser, regex_to_nfa
from automatalearn.core.minimize import minimize_dfa


# =====================================================================
# PHASE 4: INDEPENDENT REFERENCE IMPLEMENTATION (Brzozowski Derivatives)
# =====================================================================

class _RefNode:
    """Base node for independent reference AST."""
    pass


class _RefEmpty(_RefNode):
    def __repr__(self): return "∅"
    def __eq__(self, other): return isinstance(other, _RefEmpty)
    def __hash__(self): return hash("empty")


class _RefEps(_RefNode):
    def __repr__(self): return "ε"
    def __eq__(self, other): return isinstance(other, _RefEps)
    def __hash__(self): return hash("eps")


class _RefChar(_RefNode):
    def __init__(self, c: str): self.c = c
    def __repr__(self): return self.c
    def __eq__(self, other): return isinstance(other, _RefChar) and self.c == other.c
    def __hash__(self): return hash(("char", self.c))


class _RefUnion(_RefNode):
    def __init__(self, left: _RefNode, right: _RefNode): self.left = left; self.right = right
    def __repr__(self): return f"({self.left}+{self.right})"
    def __eq__(self, other): return isinstance(other, _RefUnion) and (self.left, self.right) == (other.left, other.right)
    def __hash__(self): return hash(("union", self.left, self.right))


class _RefConcat(_RefNode):
    def __init__(self, left: _RefNode, right: _RefNode): self.left = left; self.right = right
    def __repr__(self): return f"({self.left}.{self.right})"
    def __eq__(self, other): return isinstance(other, _RefConcat) and (self.left, self.right) == (other.left, other.right)
    def __hash__(self): return hash(("concat", self.left, self.right))


class _RefStar(_RefNode):
    def __init__(self, inner: _RefNode): self.inner = inner
    def __repr__(self): return f"({self.inner}*)"
    def __eq__(self, other): return isinstance(other, _RefStar) and self.inner == other.inner
    def __hash__(self): return hash(("star", self.inner))


class _RefOpt(_RefNode):
    def __init__(self, inner: _RefNode): self.inner = inner
    def __repr__(self): return f"({self.inner}?)"
    def __eq__(self, other): return isinstance(other, _RefOpt) and self.inner == other.inner
    def __hash__(self): return hash(("opt", self.inner))


def _ref_nullable(r: _RefNode) -> bool:
    if isinstance(r, _RefEmpty): return False
    if isinstance(r, _RefEps): return True
    if isinstance(r, _RefChar): return False
    if isinstance(r, _RefUnion): return _ref_nullable(r.left) or _ref_nullable(r.right)
    if isinstance(r, _RefConcat): return _ref_nullable(r.left) and _ref_nullable(r.right)
    if isinstance(r, _RefStar): return True
    if isinstance(r, _RefOpt): return True
    return False


def _ref_simplify(r: _RefNode) -> _RefNode:
    if isinstance(r, _RefUnion):
        l, rt = _ref_simplify(r.left), _ref_simplify(r.right)
        if isinstance(l, _RefEmpty): return rt
        if isinstance(rt, _RefEmpty): return l
        if l == rt: return l
        return _RefUnion(l, rt)
    if isinstance(r, _RefConcat):
        l, rt = _ref_simplify(r.left), _ref_simplify(r.right)
        if isinstance(l, _RefEmpty) or isinstance(rt, _RefEmpty): return _RefEmpty()
        if isinstance(l, _RefEps): return rt
        if isinstance(rt, _RefEps): return l
        return _RefConcat(l, rt)
    if isinstance(r, _RefStar):
        inner = _ref_simplify(r.inner)
        if isinstance(inner, (_RefEmpty, _RefEps)): return _RefEps()
        if isinstance(inner, _RefStar): return inner
        return _RefStar(inner)
    if isinstance(r, _RefOpt):
        inner = _ref_simplify(r.inner)
        if isinstance(inner, _RefEmpty): return _RefEps()
        return _RefOpt(inner)
    return r


def _ref_derivative(r: _RefNode, c: str) -> _RefNode:
    if isinstance(r, _RefEmpty): return _RefEmpty()
    if isinstance(r, _RefEps): return _RefEmpty()
    if isinstance(r, _RefChar): return _RefEps() if r.c == c else _RefEmpty()
    if isinstance(r, _RefUnion):
        return _ref_simplify(_RefUnion(_ref_derivative(r.left, c), _ref_derivative(r.right, c)))
    if isinstance(r, _RefConcat):
        d_left = _ref_simplify(_RefConcat(_ref_derivative(r.left, c), r.right))
        if _ref_nullable(r.left):
            return _ref_simplify(_RefUnion(d_left, _ref_derivative(r.right, c)))
        return d_left
    if isinstance(r, _RefStar):
        return _ref_simplify(_RefConcat(_ref_derivative(r.inner, c), r))
    if isinstance(r, _RefOpt):
        return _ref_derivative(r.inner, c)
    return _RefEmpty()


def ref_matches(r: _RefNode, s: str) -> bool:
    curr = r
    for c in s:
        curr = _ref_derivative(curr, c)
        if isinstance(curr, _RefEmpty):
            return False
    return _ref_nullable(curr)


def parse_independent_reference_regex(pattern: str) -> _RefNode:
    """Independent Recursive-Descent Parser for Reference Regex."""
    cleaned = "".join(pattern.split())
    tokens = []
    i = 0
    while i < len(cleaned):
        if cleaned[i:i + 2] == r"\e":
            tokens.append(r"\e")
            i += 2
        else:
            tokens.append(cleaned[i])
            i += 1

    pos = 0

    def peek() -> Optional[str]:
        return tokens[pos] if pos < len(tokens) else None

    def get() -> str:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def parse_E() -> _RefNode:
        node = parse_T()
        while peek() in ("+", "|"):
            get()
            rt = parse_T()
            node = _RefUnion(node, rt)
        return node

    def parse_T() -> _RefNode:
        node = parse_F()
        while True:
            nxt = peek()
            if nxt in (None, ")", "+", "|"):
                break
            if nxt == ".":
                get()
            rt = parse_F()
            node = _RefConcat(node, rt)
        return node

    def parse_F() -> _RefNode:
        node = parse_Atom()
        while peek() in ("*", "?"):
            op = get()
            if op == "*":
                node = _RefStar(node)
            else:
                node = _RefOpt(node)
        return node

    def parse_Atom() -> _RefNode:
        tok = peek()
        if tok is None:
            raise ValueError("Unexpected end of pattern in reference parser")
        if tok == "(":
            get()
            node = parse_E()
            if peek() != ")":
                raise ValueError("Expected ')' in reference parser")
            get()
            return node
        if tok in ("ε", r"\e"):
            get()
            return _RefEps()
        if tok in ("+", "|", "*", "?", ")", "."):
            raise ValueError(f"Unexpected operator '{tok}' in reference parser")
        return _RefChar(get())

    if not tokens:
        return _RefEps()
    tree = parse_E()
    if pos < len(tokens):
        raise ValueError(f"Unconsumed tokens starting at '{tokens[pos]}'")
    return tree


# =====================================================================
# TEST CASES
# =====================================================================

def test_phase4_exhaustive_511_binary_strings():
    """Phase 4: Test all 511 binary strings (lengths 0-8) against independent reference."""
    pattern = "(abb)ba(a+b)"
    ref_tree = parse_independent_reference_regex(pattern)

    nfa = regex_to_nfa(pattern)
    raw_dfa = nfa.to_dfa()
    min_dfa = minimize_dfa(raw_dfa).minimized_dfa

    alphabet = ["a", "b"]
    count = 0
    accepted_count = 0
    mismatches = []

    for length in range(9):  # 0 through 8 inclusive
        for combo in itertools.product(alphabet, repeat=length):
            s = "".join(combo)
            count += 1

            expected = ref_matches(ref_tree, s)
            actual_raw = raw_dfa.accepts(s)
            actual_min = min_dfa.accepts(s)

            if expected:
                accepted_count += 1

            if actual_raw != expected or actual_min != expected:
                _, path = raw_dfa.trace(s)
                mismatches.append((s, expected, actual_raw, actual_min, path))

    # Assert exactly 511 strings were evaluated
    assert count == 511, f"Expected 511 tests, got {count}"
    # Exactly 2 strings must be accepted: 'abbbaa' and 'abbbab'
    assert accepted_count == 2, f"Expected 2 accepted strings, got {accepted_count}"
    # Assert zero mismatches across all 511 cases
    assert len(mismatches) == 0, f"Detected mismatches: {mismatches}"


def test_phase5_dfa_formal_properties():
    """Phase 5: Verify DFA formal 5-tuple M = (Q, Σ, δ, q0, F) properties."""
    pattern = "(abb)ba(a+b)"
    dfa = regex_to_nfa(pattern).to_dfa()
    min_dfa = minimize_dfa(dfa).minimized_dfa

    for d, label in [(dfa, "Determinized DFA"), (min_dfa, "Minimized DFA")]:
        # 1. Q is non-empty
        assert len(d.states) > 0, f"{label}: Q must be non-empty"
        # 2. q0 belongs to Q
        assert d.start_state in d.states, f"{label}: q0 ({d.start_state}) must belong to Q"
        # 3. F is a subset of Q
        assert d.accept_states.issubset(d.states), f"{label}: F must be subset of Q"
        assert len(d.accept_states) > 0, f"{label}: Language is non-empty, so F must have >= 1 state"
        # 4. Transitions map valid Q x Σ -> Q
        for (st, sym), dst in d.transitions.items():
            assert st in d.states, f"{label}: Transition source {st} not in Q"
            assert sym in d.alphabet, f"{label}: Transition symbol {sym} not in Σ"
            assert dst in d.states, f"{label}: Transition target {dst} not in Q"
            assert sym != "", f"{label}: Final DFA must contain NO epsilon transitions"
        # 5. Completeness: every reachable state has exactly one transition for each symbol in Σ
        for st in d.states:
            for sym in d.alphabet:
                dst = d.transition(st, sym)
                assert dst is not None, f"{label}: State {st} lacks transition on symbol '{sym}'"
                assert dst in d.states, f"{label}: Transition target {dst} not in Q"
        # 6. Empty string acceptance matches initial state acceptance status
        assert d.accepts("") == (d.start_state in d.accept_states)
        # 7. Unreachable states are eliminated or consistent
        cleaned = d.remove_unreachable()
        assert len(cleaned.states) == len(d.states), f"{label} had unreachable states"


def test_phase6_adversarial_and_stress():
    """Phase 6: Adversarial and stress tests."""
    pattern = "(abb)ba(a+b)"
    dfa = regex_to_nfa(pattern).to_dfa()
    min_dfa = minimize_dfa(dfa).minimized_dfa

    # 1. Long strings of repeated 'a' and 'b'
    assert not dfa.accepts("a" * 100)
    assert not dfa.accepts("a" * 1000)
    assert not dfa.accepts("b" * 100)
    assert not dfa.accepts("b" * 1000)

    # 2. Alternating sequences
    assert not dfa.accepts("ab" * 50)
    assert not dfa.accepts("ba" * 50)

    # 3. Repeated substrings
    assert not dfa.accepts("abbbaabbbaabbba")
    assert not dfa.accepts("abbbab" * 3)

    # 4. Valid strings with extra characters appended
    assert not dfa.accepts("abbbaaa")
    assert not dfa.accepts("abbbaab")
    assert not dfa.accepts("abbbaba")
    assert not dfa.accepts("abbbabb")

    # 5. Valid strings with one symbol deleted (all length 5 prefixes/substrings)
    assert not dfa.accepts("abbba")
    assert not dfa.accepts("bbbaa")
    assert not dfa.accepts("abbab")

    # 6. Valid strings with one symbol inserted
    assert not dfa.accepts("aabbbaa")
    assert not dfa.accepts("abbbbaa")
    assert not dfa.accepts("abbbaaa")

    # 7. Valid strings with one symbol substituted
    assert not dfa.accepts("bbbbaa")
    assert not dfa.accepts("aabbaa")
    assert not dfa.accepts("abbbaa".replace("a", "b", 1))

    # 8. Prefixes and suffixes
    for pfx in ["", "a", "ab", "abb", "abbb", "abbba"]:
        assert not dfa.accepts(pfx)
    for sfx in ["", "a", "aa", "baa", "bbaa", "bbbaa"]:
        if sfx != "abbbaa":
            assert not dfa.accepts(sfx)

    # 9. Invalid alphabet symbols (must return False or reject)
    for invalid in ["c", "0", "1", "abbbaac", "cabbbaa", "abb_baa"]:
        assert not dfa.accepts(invalid)

    # 10. Long inputs (terminates quickly without infinite loop)
    long_input_100 = "a" * 50 + "b" * 50
    long_input_1k = ("ab" * 500)
    long_input_10k = ("a" * 5000 + "b" * 5000)

    assert not dfa.accepts(long_input_100)
    assert not dfa.accepts(long_input_1k)
    assert not dfa.accepts(long_input_10k)
    assert not min_dfa.accepts(long_input_10k)


def test_phase7_custom_re_determinism():
    """Verify that repeatedly compiling and executing custom regular expressions remains deterministic."""
    for _ in range(5):
        dfa = regex_to_nfa("(abb)ba(a+b)").to_dfa()
        assert dfa.accepts("abbbaa")
        assert dfa.accepts("abbbab")
        assert not dfa.accepts("abbbaab")
        assert not dfa.accepts("")


def test_kleene_star_targeted_cases():
    """Verify Kleene Star (*) on target expressions with positive and negative test cases."""
    alphabet = {"a", "b"}
    cases = [
        {
            "regex": "a*",
            "accept": ["", "a", "aa", "aaaaa"],
            "reject": ["b", "ab", "ba", "aab"],
        },
        {
            "regex": "(ab)*",
            "accept": ["", "ab", "abab", "ababab"],
            "reject": ["a", "b", "aba", "abb", "ba"],
        },
        {
            "regex": "(a+b)*",
            "accept": ["", "a", "b", "ab", "ba", "aabbab", "bbbbaaaa"],
            "reject": [],
        },
        {
            "regex": "a*b",
            "accept": ["b", "ab", "aab", "aaaab"],
            "reject": ["", "a", "ba", "abb"],
        },
        {
            "regex": "(ab)*a",
            "accept": ["a", "aba", "ababa", "abababa"],
            "reject": ["", "ab", "aa", "ba"],
        },
        {
            "regex": "a*b*",
            "accept": ["", "a", "b", "aaa", "bbb", "aaabbb", "ab"],
            "reject": ["ba", "abab", "baba"],
        },
        {
            "regex": "(abb)*",
            "accept": ["", "abb", "abbabb", "abbabbabb"],
            "reject": ["a", "b", "ab", "bb", "abbab"],
        },
        {
            "regex": "ab*",
            "accept": ["a", "ab", "abb", "abbbbb"],
            "reject": ["", "b", "ba", "aba", "aab"],
        },
        {
            "regex": "a+b*",
            "accept": ["a", "", "b", "bb", "bbbb"],
            "reject": ["ba", "ab", "aa", "aab"],
        },
        {
            "regex": "(a+b)*abb(a+b)*",
            "accept": ["abb", "aabb", "abba", "babba", "bbabbbb", "ababbab"],
            "reject": ["", "a", "b", "ab", "ba", "aab", "bab", "bbba", "abab"],
        },
    ]

    for c in cases:
        reg = c["regex"]
        raw_dfa = regex_to_nfa(reg, alphabet=alphabet).to_dfa()
        min_dfa = minimize_dfa(raw_dfa).minimized_dfa

        for s in c["accept"]:
            assert raw_dfa.accepts(s), f"{reg} expected to accept {s!r}"
            assert min_dfa.accepts(s), f"{reg} (min) expected to accept {s!r}"

        for s in c["reject"]:
            assert not raw_dfa.accepts(s), f"{reg} expected to reject {s!r}"
            assert not min_dfa.accepts(s), f"{reg} (min) expected to reject {s!r}"


def test_kleene_star_exhaustive_reference_comparison():
    """Verify Kleene Star (*) across 5,110 exhaustive tests against independent reference."""
    alphabet = {"a", "b"}
    regexes = [
        "a*",
        "(ab)*",
        "(a+b)*",
        "a*b",
        "(ab)*a",
        "a*b*",
        "(abb)*",
        "ab*",
        "a+b*",
        "(a+b)*abb(a+b)*",
    ]

    for reg in regexes:
        raw_dfa = regex_to_nfa(reg, alphabet=alphabet).to_dfa()
        min_dfa = minimize_dfa(raw_dfa).minimized_dfa
        ref_tree = parse_independent_reference_regex(reg)

        for length in range(9):  # 0 through 8 inclusive
            for combo in itertools.product(["a", "b"], repeat=length):
                s = "".join(combo)
                expected = ref_matches(ref_tree, s)
                actual_raw = raw_dfa.accepts(s)
                actual_min = min_dfa.accepts(s)

                assert actual_raw == expected, f"{reg} raw mismatch on {s!r}: got {actual_raw}, expected {expected}"
                assert actual_min == expected, f"{reg} min mismatch on {s!r}: got {actual_min}, expected {expected}"


def test_alphabet_validation_and_enforcement():
    """Verify alphabet validation across regex compilation, NFA, raw DFA, and minimized DFA."""
    # 1. Configured alphabet {a, b}
    alpha_ab = {"a", "b"}
    dfa_raw = regex_to_nfa("(a+b)*abb(a+b)*", alphabet=alpha_ab).to_dfa()
    dfa_min = minimize_dfa(dfa_raw).minimized_dfa

    # Both DFAs must have alphabet {a, b}
    assert dfa_raw.alphabet == alpha_ab
    assert dfa_min.alphabet == alpha_ab

    # Valid strings over {a, b}
    assert dfa_raw.accepts("abb")
    assert dfa_min.accepts("abb")
    assert not dfa_raw.accepts("ab")
    assert not dfa_min.accepts("ab")

    # Empty string handling
    assert not dfa_raw.accepts("")
    assert not dfa_min.accepts("")

    # Inputs containing unsupported symbols: 'c', 'x', '0', '1', '_', spaces
    unsupported_samples = [
        "c", "x", "0", "1", "_", " ", "abb c", "abb1", "abb_a", "xabb", "abb\t", "a b b"
    ]
    for s in unsupported_samples:
        assert not dfa_raw.accepts(s), f"Raw DFA must not accept unsupported symbol string {s!r}"
        assert not dfa_min.accepts(s), f"Min DFA must not accept unsupported symbol string {s!r}"
        acc, path = dfa_raw.trace(s)
        assert not acc
        assert "DEAD_CHAR" in path, f"Raw DFA trace must record DEAD_CHAR for string {s!r}"
        acc_min, path_min = dfa_min.trace(s)
        assert not acc_min
        assert "DEAD_CHAR" in path_min, f"Min DFA trace must record DEAD_CHAR for string {s!r}"

    # 2. Configured alphabet {0, 1}
    alpha_01 = {"0", "1"}
    dfa_01_raw = regex_to_nfa("(0+1)*01", alphabet=alpha_01).to_dfa()
    dfa_01_min = minimize_dfa(dfa_01_raw).minimized_dfa

    assert dfa_01_raw.alphabet == alpha_01
    assert dfa_01_min.alphabet == alpha_01

    assert dfa_01_raw.accepts("01")
    assert dfa_01_raw.accepts("001")
    assert dfa_01_raw.accepts("1101")
    assert dfa_01_min.accepts("01")
    assert dfa_01_min.accepts("001")
    assert dfa_01_min.accepts("1101")

    assert not dfa_01_raw.accepts("")
    assert not dfa_01_raw.accepts("0")
    assert not dfa_01_raw.accepts("1")
    assert not dfa_01_raw.accepts("10")
    assert not dfa_01_min.accepts("0")

    # In {0, 1} alphabet, 'a' and 'b' are unsupported symbols
    for s in ["a", "b", "01a", "b01", "01_"]:
        assert not dfa_01_raw.accepts(s)
        assert not dfa_01_min.accepts(s)

    # 3. Reject regex literals outside the configured alphabet during compilation
    parser = RegexParser()
    with pytest.raises(ValueError, match="Symbol 'c' not in specified alphabet"):
        parser.compile("(a+b)*c", alphabet=alpha_ab)
    with pytest.raises(ValueError, match="Symbol 'a' not in specified alphabet"):
        parser.compile("0+a", alphabet=alpha_01)

    # 4. Epsilon expressions: 'ε' and '\\e'
    for eps_pattern in ["ε", r"\e"]:
        nfa_eps = parser.compile(eps_pattern, alphabet=alpha_ab)
        dfa_eps_raw = nfa_eps.to_dfa()
        dfa_eps_min = minimize_dfa(dfa_eps_raw).minimized_dfa

        # Language of ε is {""}
        assert dfa_eps_raw.accepts("")
        assert dfa_eps_min.accepts("")
        assert not dfa_eps_raw.accepts("a")
        assert not dfa_eps_raw.accepts("b")
        assert not dfa_eps_min.accepts("a")
        assert not dfa_eps_min.accepts("b")

        # Epsilon is not included as an ordinary alphabet literal
        assert "ε" not in dfa_eps_raw.alphabet
        assert r"\e" not in dfa_eps_raw.alphabet


def test_malformed_regex_handling():
    """Verify malformed expressions raise descriptive ValueErrors and valid edge cases succeed."""
    parser = RegexParser()

    # Empty expression is valid in automata theory: represents language {ε}
    nfa_empty = parser.compile("")
    dfa_empty = nfa_empty.to_dfa()
    assert dfa_empty.accepts("")
    assert not dfa_empty.accepts("a")

    nfa_spaces = parser.compile("   ")
    dfa_spaces = nfa_spaces.to_dfa()
    assert dfa_spaces.accepts("")
    assert not dfa_spaces.accepts("a")

    # Malformed expressions with expected descriptive error patterns
    malformed_cases = [
        ("()", "empty parentheses"),
        ("(a+b", "unclosed parenthesis"),
        ("a+b)", "unexpected closing parenthesis"),
        ("+a", "unexpected leading binary operator"),
        ("a+", "unexpected trailing binary operator"),
        ("a++b", "consecutive operators"),
        ("a+|b", "consecutive operators"),
        ("*a", "unexpected leading repetition operator"),
        ("a**", "consecutive repetition operators"),  # Intentionally forbidden duplicate repetition
        ("a*.", "unexpected trailing binary operator"),
        ("a..b", "consecutive operators"),
        ("a(b", "unclosed parenthesis"),
        ("a)", "unexpected closing parenthesis"),
        ("a+*b", "repetition operator"),
    ]

    for pattern, expected_err in malformed_cases:
        with pytest.raises(ValueError, match=expected_err):
            parser.compile(pattern)

    # Valid edge cases must compile and determinize cleanly
    valid_edge_cases = ["a*", "(a+b)*", "(ab)*", "a*b", "a+b*", "ε", r"\e"]
    for pattern in valid_edge_cases:
        nfa = parser.compile(pattern)
        raw_dfa = nfa.to_dfa()
        min_dfa = minimize_dfa(raw_dfa).minimized_dfa
        assert len(raw_dfa.states) > 0
        assert len(min_dfa.states) > 0


