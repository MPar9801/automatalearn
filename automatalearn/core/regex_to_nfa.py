"""Thompson's Construction: Regular Expression to NFA compiler.

Converts regular expressions with union (+ and |), concatenation,
Kleene star (*), optional (?), and grouping parentheses into
an equivalent NFA with epsilon transitions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from automatalearn.core.dfa import State
from automatalearn.core.nfa import NFA


@dataclass
class _Fragment:
    """Internal helper representing a sub-NFA with single start and accept state."""
    start: int
    accept: int
    transitions: Dict[Tuple[int, str], Set[int]]
    alphabet: Set[str]


class RegexParser:
    """Parser and Thompson compiler for regular expressions."""

    def __init__(self) -> None:
        self._state_counter = 0

    def _new_state(self) -> int:
        st = self._state_counter
        self._state_counter += 1
        return st

    def _tokenize(self, regex: str) -> List[str]:
        """Tokenize the regex string, stripping whitespace and identifying escape tokens."""
        cleaned = "".join(regex.split())
        tokens: List[str] = []
        i = 0
        while i < len(cleaned):
            if cleaned[i:i + 2] == r"\e":
                tokens.append(r"\e")
                i += 2
            else:
                tokens.append(cleaned[i])
                i += 1
        return tokens

    def _validate_tokens(self, tokens: List[str], alphabet: Optional[Set[str]] = None) -> None:
        """Validate token syntax, operator positions, and parentheses matching."""
        if not tokens:
            return

        binary_ops = {"+", "|", "."}
        postfix_ops = {"*", "?"}
        depth = 0

        for tok in tokens:
            if alphabet is not None and tok not in binary_ops and tok not in postfix_ops and tok not in {"(", ")", "ε", r"\e"}:
                if tok not in alphabet:
                    raise ValueError(f"Symbol '{tok}' not in specified alphabet {sorted(alphabet)}")
            if tok == "(":
                depth += 1
            elif tok == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError("Malformed regex: unexpected closing parenthesis ')'")
        if depth != 0:
            raise ValueError("Malformed regex: unclosed parenthesis '('")

        if tokens[0] in binary_ops:
            raise ValueError(f"Malformed regex: unexpected leading binary operator '{tokens[0]}'")
        if tokens[0] in postfix_ops:
            raise ValueError(f"Malformed regex: unexpected leading repetition operator '{tokens[0]}'")
        if tokens[0] == ")":
            raise ValueError("Malformed regex: unexpected leading closing parenthesis ')'")

        if tokens[-1] in binary_ops:
            raise ValueError(f"Malformed regex: unexpected trailing binary operator '{tokens[-1]}'")
        if tokens[-1] == "(":
            raise ValueError("Malformed regex: unclosed parenthesis '(' at end of expression")

        for i in range(len(tokens) - 1):
            curr = tokens[i]
            nxt = tokens[i + 1]

            if curr == "(":
                if nxt == ")":
                    raise ValueError("Malformed regex: empty parentheses '()'")
                if nxt in binary_ops:
                    raise ValueError(f"Malformed regex: binary operator '{nxt}' immediately follows '('")
                if nxt in postfix_ops:
                    raise ValueError(f"Malformed regex: repetition operator '{nxt}' immediately follows '('")
            elif curr in binary_ops:
                if nxt in binary_ops:
                    raise ValueError(f"Malformed regex: consecutive operators '{curr}{nxt}'")
                if nxt in postfix_ops:
                    raise ValueError(f"Malformed regex: repetition operator '{nxt}' follows binary operator '{curr}'")
                if nxt == ")":
                    raise ValueError(f"Malformed regex: binary operator '{curr}' before closing parenthesis ')'")
            elif curr in postfix_ops:
                if nxt in postfix_ops:
                    raise ValueError(f"Malformed regex: consecutive repetition operators '{curr}{nxt}'")

    def _insert_explicit_concat(self, tokens: List[str]) -> List[str]:
        """Insert explicit concatenation '.' operators between adjacent subexpressions."""
        if not tokens:
            return []
        output: List[str] = []
        op_ends = {")", "*", "?"}
        op_starts = {"("}
        binary_ops = {"+", "|", "."}
        postfix_ops = {"*", "?"}

        for i, tok in enumerate(tokens):
            output.append(tok)
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                # Can tok end an operand?
                is_curr_end = (tok in op_ends) or (tok not in binary_ops and tok not in {"("})
                # Can nxt start an operand?
                is_nxt_start = (nxt in op_starts) or (nxt not in binary_ops and nxt not in postfix_ops and nxt not in {")"})

                if is_curr_end and is_nxt_start:
                    output.append(".")

        return output

    def _to_postfix(self, tokens: List[str]) -> List[str]:
        """Convert token stream with explicit concatenation to postfix via Shunting-Yard."""
        precedence = {"*": 3, "?": 3, ".": 2, "+": 1, "|": 1}
        output: List[str] = []
        op_stack: List[str] = []

        for tok in tokens:
            if tok in {"*", "?"}:
                # Unary postfix operators apply immediately to the operand on top of stack
                output.append(tok)
            elif tok in {".", "+", "|"}:
                while op_stack and op_stack[-1] != "(" and precedence.get(op_stack[-1], 0) >= precedence[tok]:
                    output.append(op_stack.pop())
                op_stack.append(tok)
            elif tok == "(":
                op_stack.append(tok)
            elif tok == ")":
                while op_stack and op_stack[-1] != "(":
                    output.append(op_stack.pop())
                if not op_stack or op_stack[-1] != "(":
                    raise ValueError("Malformed regex: unbalanced parentheses")
                op_stack.pop()
            else:
                # Literal or epsilon token
                output.append(tok)

        while op_stack:
            top = op_stack.pop()
            if top in {"(", ")"}:
                raise ValueError("Malformed regex: unbalanced parentheses")
            output.append(top)

        return output

    def compile(self, regex_str: str, alphabet: Optional[Iterable[str]] = None) -> NFA:
        """Compile a regex string into an equivalent NFA via Thompson's Construction.

        Args:
            regex_str: Regular expression string (e.g., '(abb)ba(a+b)').
            alphabet: Optional explicit alphabet to enforce and include.

        Returns:
            Equivalent NFA.
        """
        self._state_counter = 0
        exp_alpha: Optional[Set[str]] = set(alphabet) if alphabet is not None else None

        tokens = self._tokenize(regex_str)
        if not tokens:
            # Empty regular expression: accepts only the empty string ε
            s = self._new_state()
            return NFA(
                states={s},
                alphabet=exp_alpha or set(),
                transitions={},
                start_states={s},
                accept_states={s},
                name="NFA_regex_empty",
            )

        self._validate_tokens(tokens, exp_alpha)
        with_concat = self._insert_explicit_concat(tokens)
        postfix = self._to_postfix(with_concat)

        stack: List[_Fragment] = []

        for token in postfix:
            if token == ".":
                # Concatenation: A . B
                if len(stack) < 2:
                    raise ValueError("Malformed regex: concatenation missing operands")
                frag_b = stack.pop()
                frag_a = stack.pop()
                trans = dict(frag_a.transitions)
                for k, v in frag_b.transitions.items():
                    trans.setdefault(k, set()).update(v)
                # Connect A's accept to B's start with epsilon transition
                trans.setdefault((frag_a.accept, ""), set()).add(frag_b.start)
                stack.append(_Fragment(
                    start=frag_a.start,
                    accept=frag_b.accept,
                    transitions=trans,
                    alphabet=frag_a.alphabet | frag_b.alphabet,
                ))

            elif token in {"|", "+"}:
                # Union (OR): A + B / A | B
                if len(stack) < 2:
                    raise ValueError(f"Malformed regex: union operator '{token}' missing operands")
                frag_b = stack.pop()
                frag_a = stack.pop()
                s_start = self._new_state()
                s_accept = self._new_state()
                trans = dict(frag_a.transitions)
                for k, v in frag_b.transitions.items():
                    trans.setdefault(k, set()).update(v)

                trans.setdefault((s_start, ""), set()).update({frag_a.start, frag_b.start})
                trans.setdefault((frag_a.accept, ""), set()).add(s_accept)
                trans.setdefault((frag_b.accept, ""), set()).add(s_accept)

                stack.append(_Fragment(
                    start=s_start,
                    accept=s_accept,
                    transitions=trans,
                    alphabet=frag_a.alphabet | frag_b.alphabet,
                ))

            elif token == "*":
                # Kleene Star: A*
                if not stack:
                    raise ValueError("Malformed regex: Kleene star '*' missing operand")
                frag = stack.pop()
                s_start = self._new_state()
                s_accept = self._new_state()
                trans = dict(frag.transitions)

                trans.setdefault((s_start, ""), set()).update({frag.start, s_accept})
                trans.setdefault((frag.accept, ""), set()).update({frag.start, s_accept})

                stack.append(_Fragment(
                    start=s_start,
                    accept=s_accept,
                    transitions=trans,
                    alphabet=frag.alphabet,
                ))

            elif token == "?":
                # Zero or one: A?
                if not stack:
                    raise ValueError("Malformed regex: optional operator '?' missing operand")
                frag = stack.pop()
                s_start = self._new_state()
                s_accept = self._new_state()
                trans = dict(frag.transitions)

                trans.setdefault((s_start, ""), set()).update({frag.start, s_accept})
                trans.setdefault((frag.accept, ""), set()).add(s_accept)

                stack.append(_Fragment(
                    start=s_start,
                    accept=s_accept,
                    transitions=trans,
                    alphabet=frag.alphabet,
                ))

            else:
                # Literal character or epsilon
                s_start = self._new_state()
                s_accept = self._new_state()
                char = "" if token in {"ε", r"\e"} else token
                alpha = set() if char == "" else {char}
                trans = {(s_start, char): {s_accept}}
                stack.append(_Fragment(
                    start=s_start,
                    accept=s_accept,
                    transitions=trans,
                    alphabet=alpha,
                ))

        if len(stack) != 1:
            raise ValueError("Malformed regex: unbalanced expression")

        root = stack.pop()
        final_alpha = root.alphabet if exp_alpha is None else (root.alphabet | exp_alpha)

        # Collect all reachable states in the fragment
        all_states: Set[State] = {root.start, root.accept}
        for (st, _), dsts in root.transitions.items():
            all_states.add(st)
            all_states.update(dsts)

        return NFA(
            states=all_states,
            alphabet=final_alpha,
            transitions=root.transitions,
            start_states={root.start},
            accept_states={root.accept},
            name=f"NFA_regex_{regex_str}",
        )


def regex_to_nfa(regex_str: str, alphabet: Optional[Iterable[str]] = None) -> NFA:
    """Convenience function compiling regex to NFA.

    Args:
        regex_str: Regular expression string (e.g., '(abb)ba(a+b)').
        alphabet: Optional explicit alphabet.

    Returns:
        Equivalent NFA.
    """
    parser = RegexParser()
    return parser.compile(regex_str, alphabet=alphabet)
