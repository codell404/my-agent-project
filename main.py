from typing import Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
import os

# =========================
# 1. 智谱 LLM（GLM-4）
# =========================
llm = ChatZhipuAI(
    api_key="f7ef6136aa894f70a26d1f2b8c0f1b92.t8xfpmLtuMINQHfB",
    model="glm-4-flash"
)

# =========================
# 2. 搜索工具（Tavily）
# =========================
os.environ["TAVILY_API_KEY"] = "你的TAVILY_KEY"
search_tool = TavilySearchResults(max_results=3)

# =========================
# 3. State 定义
# =========================
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    topic: str
    outline: str
    draft: str
    feedback: str
    next: str
    current_time: str
    current_location: str


# =========================
# 4. Researcher（调研）
# =========================
def researcher_node(state: AgentState):
    print("🔍 Researcher running...")

    try:
        query = f"{state['current_location']} {state['current_time']} {state['topic']} 最新趋势"
        search_result = search_tool.invoke(query)
        context = str(search_result)
    except Exception as e:
        context = f"搜索失败，使用模型知识: {e}"

    prompt = f"""
你是一名专业内容策划。

时间：{state['current_time']}
地点：{state['current_location']}

参考资料：
{context}

请为主题「{state['topic']}」生成文章大纲（结构化）。
"""

    res = llm.invoke([HumanMessage(content=prompt)])

    return {
        "outline": res.content,
        "next": "writer"
    }


# =========================
# 5. Writer（写作）
# =========================
def writer_node(state: AgentState):
    print("✍️ Writer running...")

    prompt = f"""
你是一名专业写手。

时间：{state['current_time']}
地点：{state['current_location']}

根据大纲写一篇 500 字 Markdown 文章：

{state['outline']}
"""

    res = llm.invoke([HumanMessage(content=prompt)])

    return {
        "draft": res.content,
        "next": "editor"
    }


# =========================
# 6. Editor（审核）
# =========================
def editor_node(state: AgentState):
    print("🧐 Editor running...")

    prompt = f"""
你是一名严格主编。

请检查文章：

{state['draft']}

标准：
1. 是否逻辑清晰
2. 是否有问题
3. 是否符合 Markdown

如果通过：只输出 APPROVED
否则：指出问题
"""

    res = llm.invoke([HumanMessage(content=prompt)])
    feedback = res.content

    if "APPROVED" in feedback:
        return {
            "feedback": "通过",
            "next": END
        }
    else:
        return {
            "feedback": feedback,
            "next": "writer"
        }


# =========================
# 7. 路由函数
# =========================
def router(state: AgentState):
    return state["next"]


# =========================
# 8. 构建 LangGraph
# =========================
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

workflow.set_entry_point("researcher")

workflow.add_conditional_edges("researcher", router, {
    "writer": "writer"
})

workflow.add_conditional_edges("writer", router, {
    "editor": "editor"
})

workflow.add_conditional_edges("editor", router, {
    "writer": "writer",
    END: END
})

app = workflow.compile()


# =========================
# 9. 运行测试
# =========================
if __name__ == "__main__":

    inputs = {
        "topic": "AI Agent 在自动化办公中的应用",
        "messages": [],
        "current_time": "2026-05-06 星期三",
        "current_location": "中国 山西省 晋中市",
        "outline": "",
        "draft": "",
        "feedback": "",
        "next": ""
    }

    print("🚀 开始运行...\n")

    final_state = None

    for step in app.stream(inputs):
        for k, v in step.items():
            final_state = v

    print("\n\n✅ 最终文章：\n")
    print(final_state.get("draft", "生成失败"))