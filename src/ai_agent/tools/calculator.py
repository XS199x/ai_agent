"""Calculator：安全的数学表达式求值工具。

实现：
- AST 白名单：只允许 数字 / 二元运算 / 一元负号 / 括号
- 不执行任何 Python 代码、不调用 eval
- 支持：+ - * / ** //、以及 Python 原生运算的优先级

为什么手写 AST：
- eval 太危险，可能执行任意代码
- 简单计算不需要 numexpr 等依赖
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict

from src.ai_agent.tools.base import BaseTool


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = (
        "执行数学计算。当用户问数学问题、需要计算表达式、需要精确数值时使用。"
        "支持：加减乘除（+ - * /）、幂运算（**）、括号、负数。"
        "不要用它做文字处理或非数学问题。"
    )
    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "一个可计算的数学表达式。例如：'123 * 456'、'2^10 - 1'、'(100 + 50) / 3'。注意使用 ** 表示幂运算。",
            }
        },
        "required": ["expression"],
    }

    # ---- 允许的 AST 节点 ----
    _ALLOWED_BIN_OPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }
    _ALLOWED_UNARY_OPS = {
        ast.UAdd: lambda v: +v,
        ast.USub: lambda v: -v,
    }

    def _preprocess(self, expr: str) -> str:
        # 把常见的中文符号转成英文；把 ^ 换成 **
        expr = expr.strip()
        expr = expr.replace("^", "**")
        expr = expr.replace("，", ",")
        expr = expr.replace("×", "*")
        expr = expr.replace("÷", "/")
        expr = expr.replace("（", "(")
        expr = expr.replace("）", ")")
        return expr

    def _safe_eval(self, node: ast.AST):
        if isinstance(node, ast.Expression):
            return self._safe_eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的常量：{node.value!r}")
        # Python < 3.8 用 Num，3.8+ 用 Constant；这里兼容
        if isinstance(node, getattr(ast, "Num", type(None))):
            return node.n  # type: ignore[attr-defined]
        if isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            op_type = type(node.op)
            if op_type not in self._ALLOWED_BIN_OPS:
                raise ValueError(f"不支持的运算：{op_type.__name__}")
            if op_type == ast.Pow and abs(right) > 200:
                raise ValueError("指数太大，拒绝计算以保护资源")
            if op_type in (ast.Div, ast.FloorDiv) and right == 0:
                raise ZeroDivisionError("除数为 0")
            return self._ALLOWED_BIN_OPS[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand)
            op_type = type(node.op)
            if op_type not in self._ALLOWED_UNARY_OPS:
                raise ValueError(f"不支持的一元运算：{op_type.__name__}")
            return self._ALLOWED_UNARY_OPS[op_type](operand)
        raise ValueError(f"不支持的表达式节点：{type(node).__name__}")

    def run(self, args: Dict[str, Any]) -> str:
        expr = args.get("expression")
        if not expr or not isinstance(expr, str):
            raise ValueError(f"缺少参数 expression 或类型不对：{expr!r}")
        expr = self._preprocess(expr)
        try:
            tree = ast.parse(expr, mode="eval")
            result = self._safe_eval(tree)
        except ZeroDivisionError as e:
            return f"计算错误：{e}"
        except (SyntaxError, ValueError) as e:
            return f"无法解析表达式 {expr!r}：{e}"
        except OverflowError:
            return "结果太大，无法表示"

        if isinstance(result, float) and math.isnan(result):
            return "结果为 NaN"
        if isinstance(result, float) and math.isinf(result):
            return "结果为无穷大"

        # 浮点数如果接近整数，就转成整数展示
        if isinstance(result, float) and abs(result - round(result)) < 1e-12:
            return str(int(round(result)))
        return str(result)
