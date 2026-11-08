"""쿼리봇 계산기 (2026-11-08). 모델이 <<calc: 식>>을 내면 이 계산기로 정확히 계산한다.

- 아무 코드나 실행하지 않는다: 식을 파이썬 문법 나무(ast)로 읽어서 숫자·사칙연산·거듭제곱·나머지와
  아래 함수 목록만 계산한다(이름·속성·인덱싱·람다 등은 전부 거부).
- 서버를 멈추게 하는 식(10**10**10, 100000!)은 크기 제한으로 막는다.
"""
import ast
import math
import operator

MAX_POW_EXP = 1000        # 정수 거듭제곱 지수 상한
MAX_INT_DIGITS = 400      # 정수 결과 자릿수 상한
MAX_FACTORIAL = 300
MAX_EXPR_CHARS = 200

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _deg(f):
    return lambda x: f(math.radians(x))


def _factorial(n):
    if n != int(n) or n < 0 or n > MAX_FACTORIAL:
        raise ValueError(f"팩토리얼은 0~{MAX_FACTORIAL}의 정수만")
    return math.factorial(int(n))


def _root(x, n=2):
    if n == 0:
        raise ValueError("0제곱근은 없음")
    if x < 0 and int(n) == n and int(n) % 2 == 1:
        return -((-x) ** (1 / n))
    r = x ** (1 / n)
    return round(r) if abs(r - round(r)) < 1e-9 else r


FUNCS = {
    "sqrt": math.sqrt, "cbrt": lambda x: _root(x, 3), "root": _root, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sind": _deg(math.sin), "cosd": _deg(math.cos), "tand": _deg(math.tan),  # 각도(도) 단위
    "log": math.log, "ln": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
    "factorial": _factorial, "gcd": math.gcd, "lcm": math.lcm, "comb": math.comb, "perm": math.perm,
    "radians": math.radians, "degrees": math.degrees, "hypot": math.hypot, "min": min, "max": max,
}
CONSTS = {"pi": math.pi, "e": math.e}


def _ev(node):
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in CONSTS:
        return CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_ev(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(b) > MAX_POW_EXP:
                raise ValueError("지수가 너무 큼")
            if isinstance(a, int) and isinstance(b, int) and b > 0 and abs(a) > 1 and b * math.log10(abs(a)) > MAX_INT_DIGITS:
                raise ValueError("결과가 너무 큼")
        if isinstance(node.op, ast.Mult) and isinstance(a, int) and isinstance(b, int) and a and b \
                and math.log10(abs(a)) + math.log10(abs(b)) > MAX_INT_DIGITS:
            raise ValueError("결과가 너무 큼")
        return _BIN[type(node.op)](a, b)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords:
        return FUNCS[node.func.id](*[_ev(a) for a in node.args])
    raise ValueError("허용되지 않는 식")


def normalize(expr: str) -> str:
    """모델이 흔히 쓰는 기호를 계산기 문법으로: × ÷ ^ √ , (천 단위 쉼표) 등."""
    s = expr.strip().strip("`").replace("×", "*").replace("÷", "/").replace("^", "**").replace("−", "-")
    s = s.replace("√", "sqrt").replace("π", "pi")
    import re
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 1,234,567 → 1234567
    s = re.sub(r"(\d+(?:\.\d+)?)\s*!", r"factorial(\1)", s)  # 5! → factorial(5)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)  # 15% → (15/100)
    return s


def calc(expr: str) -> str:
    """계산 결과를 글자로. 실패하면 ValueError(이유)."""
    s = normalize(expr)
    if not s or len(s) > MAX_EXPR_CHARS:
        raise ValueError("식이 비었거나 너무 김")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise ValueError("식을 읽지 못함") from e
    v = _ev(tree)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("값이 정해지지 않음")
        if abs(v - round(v)) < 1e-12 and abs(v) < 1e15:
            return str(int(round(v)))
        return f"{v:.12g}"
    if isinstance(v, int) and len(str(abs(v))) > MAX_INT_DIGITS:
        raise ValueError("결과가 너무 큼")
    return str(v)


if __name__ == "__main__":  # 간단 시험
    for e in ["987654321*123456789", "2**64", "sqrt(7.3)", "17!", "1.05**30", "sind(30)", "log2(1000000)", "cbrt(1000000)",
              "15% * 84500", "1,234,567 × 3", "10**10**10", "__import__('os')", "(2**100)**100", "factorial(100000)", "1/0"]:
        try:
            print(e, "=", calc(e))
        except Exception as x:  # noqa: BLE001
            print(e, "→ 거부:", x)
