import os
from IPython.display import Image, display
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

# 导入本地定义的消息格式
from agent.tools_and_schemas import SearchQueryList, Reflection
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
# 导入配置和提示词
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
# 导入工具
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

# 加载环境变量，设置GEMINI_API_KEY
load_dotenv()
if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# 定义查询生成节点，基于原始查询生成若干搜索查询
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    # 获取配置
    configurable = Configuration.from_runnable_config(config)

    # 检查是否设置了初始搜索查询数量，如果没有则设置为配置中的默认值
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # 初始化Gemini 2.0 Flash模型
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    # 使用预定义的结构化输出，将模型输出转换为SearchQueryList类型
    structured_llm = llm.with_structured_output(SearchQueryList)

    # 格式化提示词
    current_date = get_current_date() # 获取当前日期
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # 调用模型生成搜索查询
    result = structured_llm.invoke(formatted_prompt)
    # 返回搜索查询，search_query是Query类型，query是list[str]
    return {"search_query": result.query}

# 将生成的多个搜索查询并行给web_research节点，并行处理
def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    # 发送搜索查询到web_research节点，id是搜索查询的索引
    # Send(node_name, data) 发送消息到指定节点，data是传递给节点的数据
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]

# 网络搜索节点，使用Google Search API工具执行网络搜索
def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    # 填充网络搜索提示词
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
    # 使用Google GenAI客户端作为LangChain客户端，因为LangChain客户端不返回grounding元数据
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # 将长URL转换为短URL，以节省token和时间
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    # 获取引用并添加到生成的文本中
    citations = get_citations(response, resolved_urls)
    # 将引用添加到生成的文本中
    modified_text = insert_citation_markers(response.text, citations)
    # 将引用添加到sources_gathered中
    sources_gathered = [item for citation in citations for item in citation["segments"]]
    # 返回搜索结果，sources_gathered是引用列表，search_query是搜索查询，web_research_result是搜索结果
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }

# 反思节点，识别知识缺口并生成潜在的后续查询
def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    # 增加研究循环计数并获取推理模型
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # 填充提示词
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # 初始化推理模型
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    # 使用预定义的结构化输出，将模型输出转换为Reflection类型，并继续推理
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
    # 返回反思结果，is_sufficient是是否足够，knowledge_gap是知识缺口，follow_up_queries是后续查询，research_loop_count是研究循环计数，number_of_ran_queries是已运行的查询数量
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }

# 评估研究节点，决定是否继续收集信息或总结
def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    # 获取最大研究循环次数
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    # 如果已经足够或达到最大研究循环次数，则返回总结节点
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    # 否则，返回网络搜索节点
    else:
        # 发送后续查询到网络搜索节点，id是后续查询的索引
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]
    

# 总结节点，准备最终输出，去重并格式化引用，然后与运行中的总结结合，创建一个结构化的研究报告，并添加正确的引用
def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    # 获取推理模型
    reasoning_model = state.get("reasoning_model") or configurable.answer_model
    # 填充提示词
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )
    # 初始化推理模型
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    # 进行推理
    result = llm.invoke(formatted_prompt)
    # 将短URL替换为原始URL，并将所有使用的URL添加到sources_gathered中
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)
    # 返回最终结果，messages是最终答案，sources_gathered是引用列表
    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }

# 创建我们的Agent Graph 
builder = StateGraph(OverallState, config_schema=Configuration)
# 定义节点
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# 设置入口点为generate_query，这意味着这个节点是第一个被调用的
builder.add_edge(START, "generate_query")
# 添加条件边，继续执行搜索查询的并行分支
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# 反思网络搜索结果
builder.add_edge("web_research", "reflection")
# 评估研究
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# 总结答案
builder.add_edge("finalize_answer", END)
# 编译图
graph = builder.compile(name="pro-search-agent")

# 绘制图
display(Image(graph.get_graph().draw_mermaid_png()))