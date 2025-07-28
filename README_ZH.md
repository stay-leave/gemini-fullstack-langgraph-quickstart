# 逻辑编排 graph.py
## 节点
- generate_query：根据用户原始查询生成若干搜索查询
- web_research：使用谷歌大模型进行谷歌搜索
- reflection：评估知识缺口，并生成后续查询
- finalize_answer：根据搜索到的信息的总结内容，创建研究报告

## 图编排
- 入口：generate_query
- 条件边：continue_to_web_research函数作为条件，将生成的若干搜索，发送到web_research，并行执行
- 普通边：web_research的结果，发送到reflection，进行评估，如有必要生成新的查询
- 条件边：evaluate_research函数作为条件，评估是否继续进入web_research，或是直接进入finalize_answer
- 终端节点：finalize_answer

# 配置 configuration.py
使用pydantic定义结构化类
使用cls实现工厂方法，优先采用环境变量的配置

# 方法 utils.py
- get_research_topic：从聊天记录中获取研究主题
- resolve_urls：将原始的长url映射到短url
- insert_citation_markers：将大模型生成的文本，添加md格式的链接，返回文本
- get_citations：从大模型中获取引用，返回一个list

# 工具模式定义 tools_and_schemas.py
- SearchQueryList：生成的搜索查询结果格式
- Reflection：反思结果格式

# 提示词  prompts.py
- get_current_date函数：获取当前时间
- query_writer_instructions：查询改写指令
- web_searcher_instructions：网络搜索指令
- reflection_instructions：反思指令
- answer_instructions：最终回复指令

# 状态定义  state.py
- OverallState：整个agent的上下文状态格式
- ReflectionState: 反思节点的输出格式




