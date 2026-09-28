"""Precedence-correct arithmetic checking for <<a op b=c>> annotations.

THE BUG THIS REPLACES (audit finding, confirmed):
stage32_uprm_toy1.eval_equation folds operands strictly left-to-right, so
`<<2+3*4=14>>` evaluates to 20 and is reported FALSE although 14 is correct.
Measured false-positive rate on the real cache: 66.2% -- of 8,371 flagged
chains, 6,434 mix */ with +-, and 5,538 of those are actually true.

That function is the GROUND-TRUTH LABEL for stage37 (T1/T4/T5) and
stage39/40 (the "lucky guess" definition), so the bug propagated into:
  * stage37 T1 gap  +0.067 -> -0.014  (inverts; kills the H_ERRORTYPE read)
  * stage40 headline 2.09% -> 1.26%  becomes  0.50% -> 0.39%
  * stage39/40 P1 eligible pools 2,619 -> 526

This module is NEW rather than an edit, so every existing result stays
reproducible and the two implementations can be compared directly.

Evaluation: Decimal throughout (never float), standard precedence via a
two-pass shunting approach, unary minus supported, division by zero and
non-evaluable input return None. Tolerance matches the project convention:
exact or <= 0.005 relative.
"""

from __future__ import annotations

import re
from decimal import Decimal, DivisionByZero, InvalidOperation

from ..q02_can_the_model_judge_itself.verdict_judge_gate import norm

_EQUATION_RE = re.compile(r"<<([^>]{1,120}?)>>")
# token stream: numbers (with optional decimal part) or single operators
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[+\-*/]")
TOLERANCE = Decimal("0.005")


def _tokenize(lhs):
    """['48','/','2'] from '48/2'. None if the string holds anything else."""
    cleaned = (lhs.replace("×", "*").replace("÷", "/")
                  .replace("−", "-").replace(",", "").replace(" ", ""))
    if not cleaned:
        return None
    tokens = _TOKEN_RE.findall(cleaned)
    # reject if tokenizing lost characters (letters, parentheses, %, ...)
    if "".join(tokens) != cleaned:
        return None
    return tokens


def _to_rpn(tokens):
    """Shunting-yard with * / binding tighter than + -. None if malformed."""
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    output, operators = [], []
    expect_operand = True
    for token in tokens:
        if token in precedence:
            if expect_operand:
                # leading/unary minus: treat as 0 - x; anything else is bad
                if token == "-":
                    output.append(Decimal(0))
                    expect_operand = False
                else:
                    return None
            while (operators
                   and precedence[operators[-1]] >= precedence[token]):
                output.append(operators.pop())
            operators.append(token)
            expect_operand = True
        else:
            if not expect_operand:
                return None            # two numbers in a row
            try:
                output.append(Decimal(token))
            except InvalidOperation:
                return None
            expect_operand = False
    if expect_operand:
        return None                    # trailing operator
    while operators:
        output.append(operators.pop())
    return output


def _eval_rpn(rpn):
    stack = []
    for item in rpn:
        if isinstance(item, Decimal):
            stack.append(item)
            continue
        if len(stack) < 2:
            return None
        b, a = stack.pop(), stack.pop()
        try:
            if item == "+":
                stack.append(a + b)
            elif item == "-":
                stack.append(a - b)
            elif item == "*":
                stack.append(a * b)
            elif item == "/":
                if b == 0:
                    return None
                stack.append(a / b)
            else:
                return None
        except (InvalidOperation, DivisionByZero):
            return None
    return stack[0] if len(stack) == 1 else None


def eval_equation(lhs, rhs):
    """Is <<lhs=rhs>> arithmetically true, honouring operator precedence?
    True / False / None (not evaluable)."""
    tokens = _tokenize(lhs)
    if tokens is None:
        return None
    rpn = _to_rpn(tokens)
    if rpn is None:
        return None
    total = _eval_rpn(rpn)
    if total is None:
        return None
    right = norm(rhs)
    if right is None:
        return None
    try:
        want = Decimal(right)
    except InvalidOperation:
        return None
    if total == want:
        return True
    scale = max(abs(total), abs(want))
    if scale == 0:
        return False
    return abs(total - want) / scale <= TOLERANCE


def first_bad_step(steps):
    """1-based index of the first step containing a FALSE equation, else None.
    Uses stage8's norm() so absurd magnitudes return None rather than raising
    (the InvalidOperation crash that killed the first stage37 run)."""
    for index, step in enumerate(steps, start=1):
        for body in _EQUATION_RE.findall(step):
            if "=" not in body:
                continue
            lhs, _, rhs = body.rpartition("=")
            if eval_equation(lhs, rhs) is False:
                return index
    return None


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- the precedence bug that motivated this module ---")
    check("2+3*4=14 is TRUE", eval_equation("2+3*4", "14"), True)
    check("2+3*4=20 is FALSE (left-to-right answer)",
          eval_equation("2+3*4", "20"), False)
    check("100-100/2=50 is TRUE", eval_equation("100-100/2", "50"), True)
    check("5*10+7*5=85 is TRUE", eval_equation("5*10+7*5", "85"), True)
    check("10000+0.4*10000=14000 is TRUE",
          eval_equation("10000+0.4*10000", "14000"), True)

    print("--- simple cases still behave ---")
    check("48/2=24 true", eval_equation("48/2", "24"), True)
    check("48/2=25 false", eval_equation("48/2", "25"), False)
    check("6*2*3=36 true", eval_equation("6*2*3", "36"), True)
    check("40-2=38 true", eval_equation("40-2", "38"), True)
    check("left-assoc subtraction 10-3-2=5",
          eval_equation("10-3-2", "5"), True)
    check("left-assoc division 100/5/2=10",
          eval_equation("100/5/2", "10"), True)

    print("--- tolerance and formatting ---")
    check("0.5% tolerance holds", eval_equation("100/3", "33.333"), True)
    check("outside tolerance fails", eval_equation("100/3", "34"), False)
    check("comma thousands in rhs", eval_equation("1000+234", "1,234"), True)
    check("unicode times", eval_equation("6×2", "12"), True)
    check("unicode divide", eval_equation("48÷2", "24"), True)
    check("spaces ignored", eval_equation("6 * 2", "12"), True)

    print("--- non-evaluable returns None, never a false accusation ---")
    check("divide by zero", eval_equation("5/0", "1"), None)
    check("letters", eval_equation("5+x", "7"), None)
    check("parentheses unsupported -> None",
          eval_equation("(2+3)*4", "20"), None)
    check("percent unsupported -> None", eval_equation("50%2", "1"), None)
    check("trailing operator", eval_equation("5+", "5"), None)
    check("empty", eval_equation("", "5"), None)
    check("absurd magnitude does not raise",
          eval_equation("1", "9" * 400), None)
    check("unary minus", eval_equation("-5+8", "3"), True)

    print("--- first_bad_step ---")
    check("clean chain",
          first_bad_step(["a 6*2 = <<6*2=12>>12", "b 12*3 = <<12*3=36>>36"]),
          None)
    check("finds the broken step",
          first_bad_step(["a 6*2 = <<6*2=12>>12", "b 12*3 = <<12*3=30>>30"]),
          2)
    check("precedence chain is NOT flagged",
          first_bad_step(["total = <<2+3*4=14>>14"]), None)
    check("no annotations", first_bad_step(["He had 5 apples"]), None)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    self_test()
