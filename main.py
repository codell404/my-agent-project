from typing import Sequence, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.chat_models import ChatZhipuAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
import os
import operator  # 新增：用于处理列表追加

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
# 注意：请确保这里填入了你真实的 TAVILY_API_KEY
os.environ["TAVILY_API_KEY"] = "tvly-你的真实KEY"
search_tool = TavilySearch(max_results=3)

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
    # 新增一个字段，专门用来存最终结果，防止被覆盖
    final_result: str

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
        context = f"搜索失败: {e}"

    prompt = f"""
你是一名专业内容策划。
时间：{state['current_time']}
地点：{state['current_location']}
参考资料：{context}
请为主题「{state['topic']}」生成文章大纲（结构化）。
"""
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"outline": res.content, "next": "writer"}

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
    # 写入草稿
    return {"draft": res.content, "next": "editor"}

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
        # 【关键修改】审核通过时，必须把 draft 也传下去，或者存入 final_result
        # 否则最后一步拿不到文章内容
        return {
            "feedback": "通过",
            "next": END,
            "final_result": state['draft'] # 把草稿存入最终结果字段
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

workflow.add_conditional_edges("researcher", router, {"writer": "writer"})
workflow.add_conditional_edges("writer", router, {"editor": "editor"})
workflow.add_conditional_edges("editor", router, {"writer": "writer", END: END})

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
        "next": "",
        "final_result": ""
    }

    print("🚀 开始运行...\n")

    final_output = None

    # 使用 stream 模式运行
    for step in app.stream(inputs, config={"recursion_limit": 50}):
        # step 是一个字典，key 是节点名，value 是节点返回的数据
        for node_name, node_data in step.items():
            print(f"-> 当前节点: {node_name}")
            # 实时保存最新的草稿，防止最后一步丢失
            if "draft" in node_data:
                final_output = node_data["draft"]
            # 如果我们定义了 final_result，也保存它
            if "final_result" in node_data:
                final_output = node_data["final_result"]

    print("\n" + "="*30)
    if final_output:
        print("✅ 最终文章：\n")
        print(final_output)
    else:
        print("❌ 生成失败：未能获取到任何草稿内容。")
    print("="*30)
