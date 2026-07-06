"""TextStats：统计文本的字数、字符数、行数等。

设计原则：
- 纯字符串处理，不依赖网络/文件
- 区分"中文字符"（按表意单位）和"单词"（按空白分隔）
- 输出简洁，方便 LLM 继续整理

典型用法：
- 用户问"这段话有多少字？" → 中文字符数
- 用户问"这篇文章有多少词？" → 单词数（英文分词）
- 用户问"这段有多少行？" → 行数
"""

from __future__ import annotations

import re
from typing import Any, Dict

from ai_agent.tools.base import BaseTool


class TextStatsTool(BaseTool):
    name: str = "text_stats"
    description: str = (
        "统计一段文本的字数、字符数、行数等。"
        "当用户问'这段有多少字'、'统计字数'、'多少行'、'多少字符'时使用。"
        "不要用它做数学计算或时间查询。"
    )
    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "要统计的文本内容。用户给出的原文直接传进来。",
            },
            "action": {
                "type": "string",
                "enum": ["count", "detail"],
                "description": "统计模式：'count'=只给核心数字（字数/行数），'detail'=详细统计（默认）。",
            },
        },
        "required": ["text"],
    }

    _CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")  # 中日韩表意字符

    @staticmethod
    def _count_cjk(text: str) -> int:
        return len(TextStatsTool._CJK_RE.findall(text))

    def run(self, args: Dict[str, Any]) -> str:
        text = args.get("text") or ""
        action = (args.get("action") or "detail").strip().lower()

        total_chars = len(text)
        total_chars_no_space = len(text.replace(" ", "").replace("\n", ""))
        cjk_chars = self._count_cjk(text)
        lines = len(text.splitlines()) if text else 0
        non_empty_lines = sum(1 for line in text.splitlines() if line.strip())

        # 英文单词数：按空白切分，去掉空字符串
        words = [w for w in re.split(r"\s+", text) if w]
        word_count = len(words)

        if action == "count":
            return (
                f"统计结果：\n"
                f"• 总字符数：{total_chars}\n"
                f"• 中文字符数：{cjk_chars}\n"
                f"• 行数：{lines}"
            )

        # 默认 detail
        return (
            f"文本统计结果：\n"
            f"• 总字符数（含空格换行）：{total_chars}\n"
            f"• 总字符数（不含空格换行）：{total_chars_no_space}\n"
            f"• 中文表意字符数：{cjk_chars}\n"
            f"• 英文/数字单词数：{word_count}\n"
            f"• 总行数：{lines}\n"
            f"• 非空行数：{non_empty_lines}"
        )
