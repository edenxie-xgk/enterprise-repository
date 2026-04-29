from __future__ import annotations

import re

from core.settings import settings
from src.agent.action_registry import INITIAL_ACTION_NAMES, REASONING_ACTION_NAMES, TERMINAL_ACTION_NAMES, TOOL_ACTION_NAMES
from src.graph.planner import looks_like_financial_graph_query
from src.types.agent_state import State
from src.types.policy_type import InitialActionDecision, InputGuardDecision, RetrievalPolicyPlan
from src.types.rag_type import RagContext


CURRENT_REASONING_ACTIONS = list(REASONING_ACTION_NAMES)

CURRENT_TOOL_ACTIONS = list(TOOL_ACTION_NAMES)

FUTURE_TOOL_ACTIONS = [
    "graph_rag",
    "web_search",
    "db_search",
    "export_rag_report",
]

TERMINAL_ACTIONS = list(TERMINAL_ACTION_NAMES)
INITIAL_ACTIONS = set(INITIAL_ACTION_NAMES)
RAG_RETRY_FAIL_REASONS = {"no_data", "low_recall", "bad_ranking", "verification_failed"}

COMPLEX_QUERY_MARKERS = [
    "对比",
    "比较",
    "分析",
    "总结",
    "方案",
    "风险",
    "以及",
    "并且",
    "同时",
    "how to",
]

REWRITE_QUERY_MARKERS = [
    "这个",
    "那个",
    "它",
    "怎么理解",
    "怎么看",
    "如何看",
]

DECOMPOSE_HINT_MARKERS = [
    "对比",
    "比较",
    "共同点",
    "差异",
    "区别",
    "分别",
    "各自",
    "以及",
    "同时",
    "and",
    "compare",
    "difference",
    "differences",
    "respectively",
]


def is_web_search_allowed(state: State) -> bool:
    profile = state.user_profile or {}
    return bool(profile.get("allow_web_search", False))


def _should_direct_answer(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False

    direct_markers = [
        "是什么",
        "什么意思",
        "解释一下",
        "介绍一下",
        "帮我润色",
        "帮我改写",
        "帮我翻译",
        "what is",
        "explain",
        "introduce",
        "translate",
        "rewrite",
        "polish",
        "summarize this",
    ]
    enterprise_markers = [
        "文档",
        "文件",
        "资料",
        "制度",
        "部门",
        "角色",
        "权限",
        "知识库",
        "上传",
        "公司",
        "内部",
        "report",
        "document",
        "knowledge base",
        "department",
        "permission",
        "uploaded",
        "internal",
    ]

    if (
        _looks_like_external_query(normalized)
        or _looks_like_structured_db_query(normalized)
        or _looks_like_graph_query(normalized)
    ):
        return False
    if any(marker in normalized for marker in enterprise_markers):
        return False
    if any(marker in normalized for marker in direct_markers):
        return True
    return len(normalized) <= 18 and not is_complex_query(normalized)


def _looks_like_external_query(text: str) -> bool:
    normalized = (text or "").strip().lower()
    markers = [
        "最新",
        "最近",
        "今日",
        "今天",
        "本周",
        "本月",
        "今年",
        "新闻",
        "公告",
        "政策",
        "市场动态",
        "latest",
        "recent",
        "today",
        "current",
        "news",
        "policy",
        "market update",
    ]
    return any(marker in normalized for marker in markers)


def _looks_like_structured_db_query(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False

    file_terms = ["文件", "文档", "资料", "上传", "file", "document", "uploaded"]
    dept_terms = ["部门", "department", "科室", "组织"]
    role_terms = ["角色", "role", "岗位"]
    permission_terms = ["权限", "可访问", "访问范围", "scope", "permission", "accessible"]
    list_terms = ["多少", "数量", "几个", "几条", "列表", "清单", "最近", "最新", "有哪些", "count", "how many", "list", "recent", "latest"]
    first_person_terms = ["我", "我的", "我能", "我可以", "my", "mine", "当前用户"]

    mentions_file = any(term in normalized for term in file_terms)
    mentions_department = any(term in normalized for term in dept_terms)
    mentions_role = any(term in normalized for term in role_terms)
    mentions_permission = any(term in normalized for term in permission_terms)
    mentions_list_intent = any(term in normalized for term in list_terms)
    mentions_first_person = any(term in normalized for term in first_person_terms)

    if mentions_role and (mentions_department or mentions_permission):
        return True
    if mentions_department and (mentions_permission or mentions_first_person or mentions_list_intent):
        return True
    if mentions_file and (mentions_list_intent or mentions_first_person):
        return True
    if mentions_permission and mentions_first_person:
        return True
    return False


def _looks_like_graph_query(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    if _looks_like_structured_db_query(normalized) or _looks_like_external_query(normalized):
        return False
    return looks_like_financial_graph_query(normalized)


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_illegal_cyber_query(text: str) -> bool:
    malicious_verbs = [
        "盗取",
        "窃取",
        "破解",
        "爆破",
        "撞库",
        "绕过",
        "入侵",
        "攻击",
        "挂马",
        "投毒",
        "勒索",
        "注入攻击",
        "steal",
        "hack",
        "exploit",
        "bypass",
        "exfiltrate",
        "phish",
        "malware",
        "ransomware",
    ]
    cyber_targets = [
        "密码",
        "账号",
        "cookie",
        "token",
        "凭证",
        "数据库",
        "后台",
        "服务器",
        "内网",
        "权限",
        "验证码",
        "系统",
        "password",
        "credential",
        "account",
        "admin",
        "database",
        "server",
        "internal network",
    ]
    return _contains_any(text, malicious_verbs) and _contains_any(text, cyber_targets)


def _is_privacy_exfiltration_query(text: str) -> bool:
    exfiltration_verbs = [
        "批量导出",
        "批量获取",
        "爬取",
        "倒库",
        "导出全部",
        "收集",
        "泄露",
        "卖掉",
        "dump",
        "scrape",
        "export all",
        "bulk extract",
    ]
    sensitive_targets = [
        "身份证",
        "手机号",
        "住址",
        "银行卡",
        "邮箱",
        "家庭地址",
        "人脸",
        "隐私",
        "客户名单",
        "employee list",
        "phone number",
        "id card",
        "bank card",
        "personal data",
        "pii",
    ]
    return _contains_any(text, exfiltration_verbs) and _contains_any(text, sensitive_targets)


def _is_illegal_deception_query(text: str) -> bool:
    deception_markers = [
        "钓鱼邮件",
        "钓鱼网站",
        "伪造合同",
        "伪造公章",
        "绕过审核",
        "伪造发票",
        "social engineering",
        "phishing email",
        "fake invoice",
        "fake contract",
        "bypass review",
    ]
    return _contains_any(text, deception_markers)


def guard_input(query: str) -> InputGuardDecision:
    normalized = (query or "").strip()
    lowered = normalized.lower()

    if not normalized:
        return InputGuardDecision(
            is_valid=False,
            reason="empty_query",
            response="请输入明确的问题后再继续。",
        )

    if len(normalized) > 2000:
        return InputGuardDecision(
            is_valid=False,
            reason="query_too_long",
            response="输入内容过长，请先缩小问题范围或分段提问。",
        )

    if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", normalized):
        return InputGuardDecision(
            is_valid=False,
            reason="non_semantic_input",
            response="输入内容缺少可识别的业务语义，请重新描述你的问题。",
        )

    if _is_illegal_cyber_query(lowered):
        return InputGuardDecision(
            is_valid=False,
            reason="illegal_cyber_activity",
            response="我不能帮助进行越权访问、攻击系统、窃取凭据或投放恶意程序等违法行为。请改为合规、安全防护或授权测试场景的问题。",
        )

    if _is_privacy_exfiltration_query(lowered):
        return InputGuardDecision(
            is_valid=False,
            reason="privacy_exfiltration",
            response="我不能协助批量获取、导出或泄露个人隐私和敏感数据。请改为数据脱敏、权限治理或合规处理相关问题。",
        )

    if _is_illegal_deception_query(lowered):
        return InputGuardDecision(
            is_valid=False,
            reason="illegal_deception_request",
            response="我不能帮助生成用于欺骗、伪造、钓鱼或绕过审核的内容。请改为合规沟通、风险识别或审计防护相关问题。",
        )

    suspicious_patterns = [
        "忽略以上指令",
        "ignore previous instructions",
        "system prompt",
        "你现在是",
        "act as",
    ]
    if any(pattern in normalized or pattern in lowered for pattern in suspicious_patterns):
        return InputGuardDecision(
            is_valid=False,
            reason="prompt_injection_like_input",
            response="当前输入更像系统指令操控而不是业务问题，请直接描述你的业务需求。",
        )

    if len(set(normalized)) <= 2 and len(normalized) >= 10:
        return InputGuardDecision(
            is_valid=False,
            reason="repetitive_noise_input",
            response="输入内容疑似无效噪声，请重新输入明确问题。",
        )

    return InputGuardDecision(is_valid=True)


def decide_initial_action(state: State) -> InitialActionDecision:
    query = (state.working_query or state.resolved_query or state.query or "").strip()
    if not query:
        return InitialActionDecision(
            next_action="clarify_question",
            reason="query_is_empty",
            clarification_question="请告诉我你想查询的具体问题、对象或范围。",
        )

    if _should_direct_answer(query):
        return InitialActionDecision(
            next_action="direct_answer",
            reason="direct_answer_preferred",
        )

    decision = _fallback_initial_action(query, allow_web_search=is_web_search_allowed(state))
    if decision.next_action not in INITIAL_ACTIONS:
        return InitialActionDecision(
            next_action="rag",
            reason="fallback_direct_rag",
        )
    return decision


def _fallback_initial_action(query: str, *, allow_web_search: bool = True) -> InitialActionDecision:
    if _should_direct_answer(query):
        return InitialActionDecision(
            next_action="direct_answer",
            reason="fallback_direct_answer",
        )
    if _looks_like_structured_db_query(query):
        return InitialActionDecision(
            next_action="db_search",
            reason="fallback_structured_db_query",
        )
    if _looks_like_graph_query(query):
        return InitialActionDecision(
            next_action="graph_rag",
            reason="fallback_financial_fact_graph_query",
        )
    if allow_web_search and _looks_like_external_query(query):
        return InitialActionDecision(
            next_action="web_search",
            reason="fallback_external_query",
        )
    if should_decompose_query(query):
        return InitialActionDecision(
            next_action="decompose_query",
            reason="fallback_complex_query",
        )
    if should_rewrite_query(query):
        return InitialActionDecision(
            next_action="rewrite_query",
            reason="fallback_needs_rewrite",
        )
    return InitialActionDecision(
        next_action="rag",
        reason="fallback_direct_rag",
    )


def _has_history_event(state: State, event_name: str) -> bool:
    return any(event.name == event_name for event in state.action_history)


def _append_action(actions: list[str], action_name: str, *, condition: bool = True) -> None:
    if condition and action_name not in actions:
        actions.append(action_name)


def _build_initial_allowed_actions(state: State, query: str) -> list[str]:
    primary = decide_initial_action(state).next_action
    actions = [primary]

    if primary in {"clarify_question", "db_search", "web_search"}:
        return actions

    if primary == "direct_answer":
        _append_action(actions, "rag")
        _append_action(actions, "rewrite_query", condition=should_rewrite_query(query))
        return actions

    if primary == "graph_rag":
        _append_action(actions, "rag")
        _append_action(actions, "rewrite_query", condition=should_rewrite_query(query))
        return actions

    if primary == "decompose_query":
        _append_action(actions, "rewrite_query", condition=should_rewrite_query(query))
        _append_action(actions, "rag")
        return actions

    if primary == "rewrite_query":
        _append_action(actions, "rag")
        _append_action(actions, "decompose_query", condition=should_decompose_query(query))
        return actions

    _append_action(actions, "rewrite_query", condition=should_rewrite_query(query))
    _append_action(actions, "decompose_query", condition=should_decompose_query(query))
    return actions


def _build_post_reasoning_allowed_actions(query: str, *, allow_web_search: bool) -> list[str]:
    if _looks_like_structured_db_query(query):
        return ["db_search"]
    if _looks_like_graph_query(query):
        return ["graph_rag", "rag"]
    if allow_web_search and _looks_like_external_query(query):
        return ["web_search"]
    return ["rag"]


def _build_graph_retry_actions(state: State, query: str) -> list[str]:
    actions: list[str] = []
    _append_action(
        actions,
        "rewrite_query",
        condition=should_rewrite_query(query) and not _has_history_event(state, "rewrite_query"),
    )
    _append_action(actions, "rag", condition=not _has_history_event(state, "rag"))
    _append_action(
        actions,
        "decompose_query",
        condition=should_decompose_query(query) and not _has_history_event(state, "decompose_query"),
    )
    _append_action(actions, "finish")
    return actions


def _build_web_retry_actions(state: State, query: str) -> list[str]:
    actions: list[str] = []
    _append_action(
        actions,
        "db_search",
        condition=_looks_like_structured_db_query(query) and not _has_history_event(state, "db_search"),
    )
    _append_action(actions, "finish")
    return actions


def _build_rag_retry_actions(
    state: State,
    query: str,
    *,
    allow_web_search: bool,
    include_rag_retry: bool = False,
) -> list[str]:
    actions: list[str] = []
    _append_action(actions, "rag", condition=include_rag_retry)
    _append_action(
        actions,
        "graph_rag",
        condition=_looks_like_graph_query(query) and not _has_history_event(state, "graph_rag"),
    )
    _append_action(
        actions,
        "db_search",
        condition=_looks_like_structured_db_query(query) and not _has_history_event(state, "db_search"),
    )
    _append_action(
        actions,
        "web_search",
        condition=allow_web_search and _looks_like_external_query(query) and not _has_history_event(state, "web_search"),
    )
    _append_action(
        actions,
        "rewrite_query",
        condition=should_rewrite_query(query) and not _has_history_event(state, "rewrite_query"),
    )
    _append_action(
        actions,
        "decompose_query",
        condition=should_decompose_query(query) and not _has_history_event(state, "decompose_query"),
    )
    _append_action(actions, "finish")
    return actions


def get_allowed_actions(state: State) -> list[str]:
    current_query = state.working_query or state.resolved_query or state.query or ""
    allow_web_search = is_web_search_allowed(state)
    reasoning_history = [
        event for event in state.action_history
        if event.kind == "reasoning" and event.name != "resolved_query"
    ]
    last_event = state.action_history[-1] if state.action_history else None
    last_tool = next((event for event in reversed(state.action_history) if event.kind == "tool"), None)

    if not reasoning_history and not last_tool:
        return _build_initial_allowed_actions(state, current_query)

    if last_event and last_event.kind == "reasoning":
        return _build_post_reasoning_allowed_actions(current_query, allow_web_search=allow_web_search)

    if not last_tool:
        return ["rag"]

    if last_tool.name == "db_search":
        if getattr(last_tool.output, "is_sufficient", False):
            return ["finalize", "finish"]
        return ["finish"]

    if last_tool.name == "graph_rag":
        if getattr(last_tool.output, "documents", None):
            return ["finalize", "finish"]
        return _build_graph_retry_actions(state, current_query)

    if last_tool.name == "web_search":
        if getattr(last_tool.output, "documents", None):
            return ["finalize", "finish"]
        return _build_web_retry_actions(state, current_query)

    if last_tool.name == "rag":
        if getattr(last_tool.output, "is_sufficient", False):
            return ["finalize", "finish"]
        if getattr(last_tool.output, "fail_reason", None) in RAG_RETRY_FAIL_REASONS:
            return _build_rag_retry_actions(
                state,
                current_query,
                allow_web_search=allow_web_search,
                include_rag_retry=True,
            )
        if getattr(last_tool.output, "documents", None):
            return ["finalize", "finish"]
        return _build_rag_retry_actions(state, current_query, allow_web_search=allow_web_search)

    return ["finish"]


def should_force_finish(state: State) -> tuple[bool, str | None]:
    recent_events = [event for event in state.action_history if event.name != "resolved_query"]
    recent_names = [event.name for event in recent_events]
    last_tools = [event for event in recent_events if event.kind == "tool"]
    last_rag_result = state.last_rag_result

    if len(last_tools) >= 2:
        last_two_tools = last_tools[-2:]
        if last_two_tools[0].name == last_two_tools[1].name:
            fail_reasons = [getattr(event.output, "fail_reason", None) for event in last_two_tools]
            if fail_reasons[0] and fail_reasons[0] == fail_reasons[1]:
                if fail_reasons[0] in {"no_data", "low_recall", "bad_ranking", "tool_error"}:
                    return True, f"repeated_{last_two_tools[0].name}_failure:{fail_reasons[0]}"

    if last_rag_result and last_rag_result.fail_reason == "no_data":
        if "rewrite_query" in recent_names and "expand_query" in recent_names and recent_names.count("rag") >= 2:
            return True, "rag_exhausted_after_rewrite_expand"

    if state.decompose_query and state.sub_query_results and last_rag_result:
        if last_rag_result.fail_reason in {"no_data", "tool_error"}:
            return True, f"decompose_path_exhausted:{last_rag_result.fail_reason}"

    return False, None


def is_complex_query(query: str) -> bool:
    lowered = query.lower()
    punctuation_count = sum(query.count(symbol) for symbol in ["，", ",", "；", ";", "。"])
    if len(query) >= 30 and punctuation_count >= 1:
        return True
    if punctuation_count >= 2:
        return True
    return any(marker in lowered or marker in query for marker in COMPLEX_QUERY_MARKERS)


def needs_rewrite_first(query: str) -> bool:
    lowered = query.lower()
    if len(query) <= 8:
        return True
    return any(marker in lowered or marker in query for marker in REWRITE_QUERY_MARKERS)


def should_rewrite_query(query: str) -> bool:
    normalized = (query or "").strip()
    if not normalized:
        return False
    if (
        _looks_like_structured_db_query(normalized)
        or _looks_like_external_query(normalized)
        or _looks_like_graph_query(normalized)
    ):
        return False
    if should_decompose_query(normalized):
        return False
    return needs_rewrite_first(normalized)


def should_decompose_query(query: str) -> bool:
    normalized = (query or "").strip()
    if not normalized:
        return False
    if (
        _looks_like_structured_db_query(normalized)
        or _looks_like_external_query(normalized)
        or _looks_like_graph_query(normalized)
    ):
        return False

    lowered = normalized.lower()
    punctuation_count = sum(normalized.count(symbol) for symbol in ["，", ",", "；", ";", "。"])
    if len(normalized) >= 18 and punctuation_count >= 2 and is_complex_query(normalized):
        return True
    return any(marker in lowered or marker in normalized for marker in DECOMPOSE_HINT_MARKERS)


def build_retrieval_plan(state: State, previous_context: RagContext | None = None) -> RetrievalPolicyPlan:
    retrieval_top_k = previous_context.retrieval_top_k if previous_context else settings.retriever_top_k
    rerank_top_k = previous_context.rerank_top_k if previous_context else settings.reranker_top_k
    use_retrieval = True if previous_context is None else previous_context.use_retrieval
    use_rerank = True if previous_context is None else previous_context.use_rerank

    needs_more_recall = False
    needs_more_precision = False
    strategy_reason = "default_rag_strategy"

    last_result = state.last_rag_result
    if not last_result:
        return RetrievalPolicyPlan(
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
            use_retrieval=use_retrieval,
            use_rerank=use_rerank,
            needs_more_recall=needs_more_recall,
            needs_more_precision=needs_more_precision,
            strategy_reason=strategy_reason,
        )

    fail_reason = last_result.fail_reason
    if fail_reason in {"no_data", "low_recall"}:
        retrieval_top_k = max(retrieval_top_k, settings.retriever_top_k) + 3
        use_retrieval = True
        use_rerank = True
        needs_more_recall = True
        strategy_reason = "increase_recall_after_sparse_results"
    elif fail_reason in {"bad_ranking", "verification_failed"}:
        rerank_top_k = max(rerank_top_k, settings.reranker_top_k) + 2
        use_retrieval = False
        use_rerank = True
        needs_more_precision = True
        strategy_reason = "increase_precision_after_ranking_issue"
    elif fail_reason == "ambiguous_query":
        use_retrieval = True
        use_rerank = True
        strategy_reason = "query_is_ambiguous_retry_after_query_transform"

    return RetrievalPolicyPlan(
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        use_retrieval=use_retrieval,
        use_rerank=use_rerank,
        needs_more_recall=needs_more_recall,
        needs_more_precision=needs_more_precision,
        strategy_reason=strategy_reason,
    )
