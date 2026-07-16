"""Skill（技能）能力包。

Skill = 高内聚的能力单元 = 工具组 + system prompt 片段 + 专属上下文提供者 + 生命周期

典型 Skill 例子：
- 代码生成技能：tools(python_runner/git_fs) + prompt(代码规范片段) + context_provider(读取项目结构)
- 数据分析技能：tools(sql_query/chart) + prompt(数据分析流程) + context_provider(数据源schema)
- 文档问答技能：tools(rag_search/file_read) + prompt(引用说明)

使用方式（由 SkillManager 统一管理）：
    from ai_agent.skills import SkillManager, BaseSkill

    class MySkill(BaseSkill):
        def name(self): return "my_skill"
        def prompt_snippet(self): return "..."
        def tools(self): return [tool1, tool2]

    sm = SkillManager([MySkill(), AnotherSkill()])
    await sm.setup()
    sm.enable("my_skill")
    # 把 sm 作为 ToolProvider 注入 CompositeToolProvider
    # 把 sm.get_system_prompt_snippets() 返回的内容拼接给 Planner
"""

from ai_agent.skills.base import BaseSkill, SkillManager

__all__ = ["BaseSkill", "SkillManager"]
