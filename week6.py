"""Week6 Day1～Day5：一个文件内完成只读业务查询练习。

Day1～Day3 的契约、Client、Service 和 MCP 注册沿用已有实现。
Day5 的三个重点函数：
1. redact_sensitive_data：复制并脱敏 JSON 中的敏感字段。
2. prepare_tool_result：将 MCP 业务结果精简到字符预算内。
3. request_json_with_retry：对只读 HTTP 查询执行有上限的重试。

三个函数已接入 MCP 出站与 HTTP 查询路径，使用本地 Mock 做离线验收。
学习内容、函数契约与各项验收见 docs/week6_day5.md。

分层仍保留为类和函数，不拆分文件：MCP → Service → Client → Mock HTTP。
所有数据和 Token 都是本地合成值，不访问公司系统，不调用 LLM。
测试及各检查点说明见 tests/test_week6.py。
"""

import copy
import json
import re
from collections.abc import Callable
from curses.ascii import isdigit
from typing import Literal, Annotated, Any
from urllib.parse import quote

import httpx
from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, Field


MOCK_TIME = "2026-10-28T00:00:00Z"
MAX_LIMIT = 100
SOURCE = "mock_task_api"
MAX_TOOL_RESULT_CHARS = 6000
SENSITIVE_FIELDS = {
    "authorization", "token", "access_token", "refresh_token",
    "api_key", "password", "secret", "client_secret",
}
TASK_PUBLIC_FIELDS = {"task_id", "state", "engineCode", "params", "tags"}

API_CONTRACTS = {
    "query_task": {
        "method": "GET", "path": "/tasks/{task_id}",
        "required_params": ["task_id"], "permission": "task.read",
        "possible_errors": [
            "MISSING_PARAMETER", "INVALID_ARGUMENT", "FORBIDDEN", "NOT_FOUND",
            "TIMEOUT", "CONNECTION_ERROR", "UPSTREAM_HTTP_ERROR", "INVALID_RESPONSE",
        ],
    },
    "query_tasks_by_engine": {
        "method": "GET", "path": "/tasks",
        "required_params": ["engine_code"], "permission": "task.read",
        "default_limit": MAX_LIMIT, "max_limit": MAX_LIMIT,
        "possible_errors": [
            "MISSING_PARAMETER", "INVALID_ARGUMENT", "FORBIDDEN", "NOT_FOUND",
            "TIMEOUT", "CONNECTION_ERROR", "UPSTREAM_HTTP_ERROR", "INVALID_RESPONSE",
        ],
    },
}

# 复用 Day1 数据：alice 有 12 个 import_std 任务，bob 有 1 个。
TASKS = {
    "12345": {
        "task_id": "12345", "state": "RUNNING", "engineCode": "import_std",
        "params": {"source": "mock/a.pcd"},
        "tags": {"upload_date": "2026-10-20"}, "owner": "alice",
    },
    "23456": {
        "task_id": "23456", "state": "SUCCESS", "engineCode": "import_standard",
        "params": {"source": "mock/b.pcd"}, "tags": {}, "owner": "alice",
    },
    "34567": {
        "task_id": "34567", "state": "FAILED", "engineCode": "import_std",
        "params": {}, "tags": {}, "owner": "bob",
    },
}
for number in range(11):
    task_id = str(20000 + number)
    TASKS[task_id] = {
        "task_id": task_id, "state": "SUCCESS", "engineCode": "import_std",
        "params": {}, "tags": {}, "owner": "alice",
    }

# 仅用于 Mock 鉴权；固定在程序侧，不能成为模型填写的工具参数。
MOCK_TOKENS = {"alice": "mock-alice-token", "bob": "mock-bob-token"}


class BusinessResult(BaseModel):
    """统一响应：status 表示查询结果，任务运行状态放在 data.state 中。

    success：有完整数据；empty：正常响应但无数据；partial：列表被 limit 截断。
    needs_input：缺必填参数；error：参数非法、无权限、网络或接口错误。
    列表 data 为 {items: [...], total: 可见匹配总数}，详情 data 为任务字典。
    本练习不实现翻页，所以 next_cursor 始终为 None。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["success", "empty", "partial", "error", "needs_input"]
    source: str = SOURCE
    data: dict | None = None
    fetched_at: str = MOCK_TIME
    truncated: bool = False
    error_code: str | None = None
    next_cursor: str | None = None


def redact_sensitive_data(value: Any) -> Any:
    """复制并递归脱敏一个 JSON 值，不修改输入。

    输入：JSON 可表示的 dict/list/str/int/float/bool/None。
    输出：同样的数据结构；字典键忽略大小写后若属于 SENSITIVE_FIELDS，
    对应值整体替换成 '[REDACTED]'；其他字典值、列表元素继续递归处理。
    普通标量原样返回。不扫描任意正文推测秘密，不截断或改写业务标识符。
    """
    if isinstance(value, dict):
        # 重新构造字典，避免修改输入；JSON 对象的键都是字符串。
        result = {}
        for key, item in value.items():
            if key.lower() in SENSITIVE_FIELDS:
                # 敏感字段的整个值都隐藏，即使该值本身是字典或列表。
                result[key] = "[REDACTED]"
            else:
                result[key] = redact_sensitive_data(item)
        return result

    if isinstance(value, list):
        # 列表中也可能嵌套字典或列表，逐项递归并保持原顺序。
        result = []
        for item in value:
            result.append(redact_sensitive_data(item))
        return result

    # JSON 标量不可变，可直接返回；字符串即使像 JSON，也不额外解析。
    return value


def prepare_tool_result(
    result: BusinessResult, max_chars: int = MAX_TOOL_RESULT_CHARS,
) -> BusinessResult:
    """在 MCP 返回前，复制、脱敏、精简并限长业务结果。

    核心逻辑：先筛选公开字段，再删除大字段，最后从列表末尾移除记录。
    每个阶段都重新测量处理后的完整结果，满足预算就立即返回。

    输入：合法 BusinessResult；max_chars 为至少 512 的真正 int，不接受 bool。
    输出：新的 BusinessResult，原输入不变。预算包含整个外层响应，计算方式为
    len(json.dumps(result.model_dump(), ensure_ascii=True))，与 SDK 的结构化
    工具输出序列化方式一致；这是字符预算，不是 Token 数。

    处理规则：
    - 调用 redact_sensitive_data。任务顶层只保留 TASK_PUBLIC_FIELDS；列表
      data 只保留 items/total。去掉其他业务字段也算截断。
    - 先保留完整公开字段；超预算时删除任务的 params/tags，再从列表末尾
      移除记录直到满足预算，保持剩余记录顺序。total 保留原匹配总数。
    - 有字段或记录被删时设置 truncated=True；success/partial 改为 partial。
      已经 partial 的结果不能恢复为 success，也不能把截断列表标成 empty。
    - source/fetched_at/error_code/next_cursor 保留；本练习 next_cursor=None。
      无数据的 empty/error/needs_input 保持原有状态。
    - 不截断 task_id/state/engineCode 等标识值；详情的最小字段仍放不下时，
      返回 error/RESULT_TOO_LARGE、data=None、truncated=True，并保留来源与时刻。
      若连基础元数据都超过预算，抛 ValueError，不返回超预算响应。
    """
    if type(max_chars) is not int or max_chars < 512:
        raise ValueError("max_chars 必须是至少 512 的整数")
    prepared = result.model_copy(deep=True)
    prepared.data = redact_sensitive_data(prepared.data)

    def within_budget() -> bool:
        # 始终测量 prepared，并包含来源、状态等外层字段；不能只测量 data。
        return len(json.dumps(prepared.model_dump(), ensure_ascii=True)) <= max_chars

    def mark_truncated():
        prepared.truncated = True
        # 错误和缺参不会因为裁剪就变成成功查询的部分结果。
        if prepared.status in ("success", "partial"):
            prepared.status = "partial"

    # 错误或空详情可能没有 data；保留其原有状态和错误代码。
    if prepared.data is None:
        if not within_budget():
            raise ValueError("基础元数据超过字符预算")
        return prepared

    # 列表外层保留 items/total，任务字段白名单应用到每一条记录上。
    is_list = "items" in prepared.data
    if is_list:
        for key in list(prepared.data):
            if key not in {"items", "total"}:
                del prepared.data[key]
                mark_truncated()
        tasks = prepared.data["items"]
    else:
        tasks = [prepared.data]

    # 第一阶段：删除内部字段。遍历键的副本，避免边遍历字典边删除键。
    for task in tasks:
        for key in list(task):
            if key not in TASK_PUBLIC_FIELDS:
                del task[key]
                mark_truncated()
    if within_budget():
        return prepared

    # 第二阶段：优先去掉参数和备注，保留 ID、状态、流程编码及记录数量。
    for task in tasks:
        for key in ("params", "tags"):
            if key in task:
                del task[key]
                mark_truncated()
    if within_budget():
        return prepared

    # 第三阶段：只从列表末尾删除，每次重新测量，保留原顺序的前缀。
    # 即使所有记录都被裁掉，total 也不变，仍表示 partial 而不是 empty。
    if is_list:
        while prepared.data["items"] and not within_budget():
            prepared.data["items"].pop()
            mark_truncated()
        if within_budget():
            return prepared

    # 不截断业务 ID 或来源来凑预算；无法表示详情时明确返回限长错误。
    prepared.status = "error"
    prepared.error_code = "RESULT_TOO_LARGE"
    prepared.data = None
    prepared.truncated = True
    if not within_budget():
        raise ValueError("基础元数据超过字符预算")

    return prepared


def request_json_with_retry(
    client: httpx.Client, path: str, *, params: dict | None = None,
    max_attempts: int = 1,
) -> dict:
    """通过有限循环重试只读 GET，成功后返回解析后的 JSON。

    核心逻辑：每轮重新发送请求；成功立即返回；遇到不可重试错误，或最后
    一次尝试仍失败时，立即抛出原异常并结束循环。

    path 是固定接口路径，params 是查询参数；身份仍由 Client 请求头管理。
    max_attempts 是含首次请求在内的总尝试次数，允许 1～3 的真正 int。
    仅重试 TimeoutException、ConnectError、HTTP 429 / 5xx；403/404/其他
    4xx、其他请求异常和 JSON 解析失败立即抛出。达到上限后继续抛原异常，
    由 Service 映射成业务 error。每次重试都必须重新执行 client.get。
    本地学习不做 sleep，不实现后台重试；成功或空数据正常返回。
    """
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts 必须是 1～3 的整数")
    # attempt 包括首次请求，例如 max_attempts=2 表示最多发送两次 GET。
    for attempt in range(1, max_attempts + 1):
        try:
            result = client.get(path, params=params)
            # 先检查 HTTP 状态，避免将 403/404 的响应正文当成查询成功。
            result.raise_for_status()
        except (httpx.TimeoutException, httpx.ConnectError):
            # 临时网络故障可以进入下一轮；最后一次失败则保留原异常类型和对象。
            if attempt == max_attempts:
                raise
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            retryable = status == 429 or 500 <= status < 600
            if not retryable or attempt == max_attempts:
                raise
        else:
            # HTTP 成功后只解析一次；空数据正常返回，JSON 解析失败也不重试。
            return result.json()


def response(status: str, data=None, error_code=None, *, truncated=False) -> dict:
    """构造 Mock API 的 JSON 响应；通过模型保证外层字段一致。"""
    return BusinessResult(
        status=status, data=data, error_code=error_code, truncated=truncated,
    ).model_dump()


def visible_task(task: dict) -> dict:
    """返回独立的公开数据副本，内部 owner 字段不进入查询结果。"""
    return copy.deepcopy({key: value for key, value in task.items() if key != "owner"})


def mock_handler(request: httpx.Request) -> httpx.Response:
    """模拟两个只读 HTTP 接口；不会进行真实网络请求或 sleep。

    路径：GET /tasks/{task_id}，GET /tasks?engine_code=...&limit=...。
    故障 ID：11111=空数据，22222/88888=ReadTimeout，33333=403，
    44444=ConnectError，55555=500，66666=非法 JSON；这些不是业务记录。
    Mock API 自身也执行权限校验，避免绕过 Service 后泄漏其他用户的数据。
    """
    if request.method != "GET":
        return httpx.Response(405, json=response("error", error_code="METHOD_NOT_ALLOWED"))

    token = request.headers.get("Authorization", "")
    user_id = next(
        (user for user, value in MOCK_TOKENS.items() if token == f"Bearer {value}"),
        None,
    )
    if user_id is None:
        return httpx.Response(403, json=response("error", error_code="FORBIDDEN"))

    path = request.url.path
    if path.startswith("/tasks/"):
        task_id = path.removeprefix("/tasks/")
        if not re.fullmatch(r"[0-9]+", task_id):
            return httpx.Response(400, json=response("error", error_code="INVALID_ARGUMENT"))
        if task_id in {"22222", "88888"}:
            raise httpx.ReadTimeout("Mock read timeout", request=request)
        if task_id == "44444":
            raise httpx.ConnectError("Mock connection failed", request=request)
        if task_id == "33333":
            return httpx.Response(403, json=response("error", error_code="FORBIDDEN"))
        if task_id == "55555":
            return httpx.Response(500, json=response("error", error_code="UPSTREAM_HTTP_ERROR"))
        if task_id == "66666":
            return httpx.Response(200, text="not valid json")
        if task_id == "11111":
            return httpx.Response(200, json=response("empty"))

        task = TASKS.get(task_id)
        if task is None:
            return httpx.Response(404, json=response("error", error_code="NOT_FOUND"))
        if task["owner"] != user_id:
            return httpx.Response(403, json=response("error", error_code="FORBIDDEN"))
        return httpx.Response(200, json=response("success", data=visible_task(task)))

    if path == "/tasks":
        engine_code = request.url.params.get("engine_code")
        raw_limit = request.url.params.get("limit", str(MAX_LIMIT))
        if not engine_code or not engine_code.strip():
            return httpx.Response(400, json=response("error", error_code="INVALID_ARGUMENT"))
        if not re.fullmatch(r"[0-9]+", raw_limit) or not 1 <= int(raw_limit) <= MAX_LIMIT:
            return httpx.Response(400, json=response("error", error_code="INVALID_ARGUMENT"))

        matches = [
            visible_task(task) for task in TASKS.values()
            if task["owner"] == user_id and task["engineCode"] == engine_code
        ]
        items = matches[:int(raw_limit)]
        truncated = len(items) < len(matches)
        status = "partial" if truncated else ("success" if matches else "empty")
        return httpx.Response(200, json=response(
            status, data={"items": items, "total": len(matches)}, truncated=truncated,
        ))

    return httpx.Response(404, json=response("error", error_code="NOT_FOUND"))


def create_mock_http_client(user_id: str = "alice") -> httpx.Client:
    """创建离线 HTTP 客户端；调用方用 with 或 close() 释放资源。

    user_id 仅由本地程序选择，决定发送哪个合成 Bearer Token。
    未知用户使用无效 Token，以便测试 403。trust_env=False 避免本机代理干扰。
    """
    return httpx.Client(
        base_url="https://mock.example",
        headers={"Authorization": f"Bearer {MOCK_TOKENS.get(user_id, 'invalid-mock-token')}"},
        transport=httpx.MockTransport(mock_handler),
        timeout=5.0,
        trust_env=False,
    )


class BusinessAPIClient:
    """Day2：只负责 HTTP 请求与 JSON 解析，空结果正常返回。

    注入的 httpx.Client 由调用方管理生命周期。HTTP 错误继续抛出，交给
    Day3 Service 统一转换；默认单次请求，Day5 可配置最多 3 次尝试。
    """

    def __init__(self, client: httpx.Client, *, max_attempts: int = 1):
        self.client = client
        self.max_attempts = max_attempts

    def query_task(self, task_id: str) -> dict:
        """GET /tasks/{task_id}，返回包含 status/data 等字段的原始字典。"""
        return request_json_with_retry(
            self.client, f"/tasks/{quote(str(task_id), safe='')}",
            max_attempts=self.max_attempts,
        )

    def query_tasks_by_engine(self, engine_code: str, limit: int = MAX_LIMIT) -> dict:
        """按 engineCode 精确查询列表；limit 默认 100，合法范围为 1～100。"""
        return request_json_with_retry(
            self.client, "/tasks", params={"engine_code": engine_code, "limit": limit},
            max_attempts=self.max_attempts,
        )


def map_api_error(error: httpx.HTTPError | ValueError) -> BusinessResult:
    """将一次 Client 异常转换成统一错误结果，不重试。

    输入：_call_api 捕获的异常对象；不能把 str(error)、URL、请求头或响应正文
    原样返回给模型。输出：BusinessResult(status='error', error_code=...)。

    映射规则（TimeoutException 必须先于 RequestError 判断）：
    - HTTPStatusError：403→FORBIDDEN，404→NOT_FOUND，其他→UPSTREAM_HTTP_ERROR。
    - TimeoutException：TIMEOUT。
    - RequestError：CONNECTION_ERROR。
    - ValueError（含 JSON 解析失败 / Pydantic 响应校验失败）：INVALID_RESPONSE。
    空结果通过正常响应表示，不通过 ValueError 表示。
    """
    if isinstance(error, httpx.TimeoutException):
        return BusinessResult(status="error", error_code="TIMEOUT")
    elif isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code == 403:
            return BusinessResult(status="error", error_code="FORBIDDEN")
        elif error.response.status_code == 404:
            return BusinessResult(status="error", error_code="NOT_FOUND")
        else:
            return BusinessResult(status="error", error_code="UPSTREAM_HTTP_ERROR")
    elif isinstance(error, httpx.RequestError):
        return BusinessResult(status="error", error_code="CONNECTION_ERROR")
    elif isinstance(error, ValueError):
        return BusinessResult(status="error", error_code="INVALID_RESPONSE")
    else:
        return BusinessResult(status="error", error_code="UNKNOWN_ERROR")


class BusinessService:
    """Day3：集中校验和错误映射，MCP 包装层不再重复这些规则。"""

    def __init__(self, api_client: BusinessAPIClient, *, can_read_tasks: bool = True):
        """can_read_tasks 是程序侧固定权限，不是 MCP 工具参数。

        本地无读取权限时，必须在请求 API 之前拒绝。任务 owner 权限由 Mock
        API 根据客户端 Token 校验；Service 将上游 403 转为 FORBIDDEN。
        """
        self.api_client = api_client
        self.can_read_tasks = can_read_tasks

    def _call_api(self, execute: Callable[[], dict]) -> BusinessResult:
        """已提供：执行无参数回调、校验响应，并调用待实现的错误映射。"""
        try:
            return BusinessResult.model_validate(execute())
        except (httpx.HTTPError, ValueError) as error:
            return map_api_error(error)

    def query_task(self, task_id: str | None) -> BusinessResult:
        """查询详情，输出统一 BusinessResult。

        输入：task_id 允许 None，以便直接测试缺参；合法值为非空 ASCII 数字串，
        不自动转换整数、去空格或接收 bool。
        校验顺序与输出：
        1. None / '' → needs_input + MISSING_PARAMETER。
        2. 其他类型 / 非 ASCII 数字串 → error + INVALID_ARGUMENT。
        3. can_read_tasks=False → error + FORBIDDEN。
        上述分支均不得请求 API。
        4. 合法时用 self._call_api(lambda: self.api_client.query_task(task_id))。
        保留上游 success / empty；接口故障由 map_api_error 转换。
        """
        if task_id is None or task_id == '':
            return BusinessResult(status="needs_input", error_code="MISSING_PARAMETER")
        # 先检查类型；or 短路可避免整数、布尔值进入正则匹配。
        elif not isinstance(task_id, str) or not re.fullmatch(r"[0-9]+", task_id):
            return BusinessResult(status="error", error_code="INVALID_ARGUMENT")

        if not self.can_read_tasks:
            return BusinessResult(status="error", error_code="FORBIDDEN")

        return self._call_api(lambda: self.api_client.query_task(task_id))



    def query_tasks_by_engine(
        self, engine_code: str | None, limit: int = MAX_LIMIT,
    ) -> BusinessResult:
        """按 engineCode 精确查询，输出统一 BusinessResult。

        输入：engine_code 为非空且非纯空白的字符串，不自动 strip 或修改内容；
        limit 默认 100，仅允许真正的 int，范围 1～100，bool 也必须拒绝。
        校验顺序：缺 engine_code(None/'')→needs_input/MISSING_PARAMETER；
        非法 engine_code 或 limit→error/INVALID_ARGUMENT；
        can_read_tasks=False→error/FORBIDDEN。以上均不得请求 API。
        合法时通过 _call_api 调用 Client.query_tasks_by_engine(engine_code, limit)。
        保留 data.items、data.total、status、truncated，不在此重复筛选或发第二次请求。
        """
        if engine_code is None or engine_code == '':
            return BusinessResult(status="needs_input", error_code="MISSING_PARAMETER")
        if not isinstance(engine_code, str) or not engine_code.strip():
            return BusinessResult(status="error", error_code="INVALID_ARGUMENT")
        # bool 是 int 的子类，使用 type 严格限制为整数。
        if type(limit) is not int or not 1 <= limit <= MAX_LIMIT:
            return BusinessResult(status="error", error_code="INVALID_ARGUMENT")

        if not self.can_read_tasks:
            return BusinessResult(status="error", error_code="FORBIDDEN")

        return self._call_api(lambda: self.api_client.query_tasks_by_engine(engine_code, limit))



def create_mcp_server(service: BusinessService) -> MCPServer:
    """创建并返回 MCPServer，注册两个只读工具，不在这里 run。

    当前环境 MCP SDK 2.1.1：使用 from mcp.server import MCPServer。
    先创建 mcp = MCPServer('week6_business')，再使用 @mcp.tool() 注册：
    - query_task(task_id: str) -> BusinessResult
    - query_tasks_by_engine(engine_code: str, limit: int = 100) -> BusinessResult

    两个函数仅委托给传入的 service 对应方法，并写中文说明以生成工具描述。
    不把 Token、user_id、can_read_tasks、httpx.Client 暴露为工具参数。
    必填参数缺失可由 MCP 参数校验拒绝；直接调用 Service 则返回 needs_input。
    生成的 Schema 必须拒绝错误类型；limit 应使用 Annotated[int, Field(...)]
    表达 strict=True、ge=1、le=100，服务层仍要独立校验，不依赖模型遵守 Schema。
    返回已注册的 mcp；stdio 生命周期由 main 管理，调试日志不能写 stdout。
    """
    from mcp.server import MCPServer
    mcp = MCPServer('week6_business')

    @mcp.tool()
    def query_task(task_id: str) -> BusinessResult:
        """
        通过任务ID查询相应的任务信息
        """
        return prepare_tool_result(service.query_task(task_id))


    @mcp.tool()
    def query_tasks_by_engine(engine_code: str, limit: Annotated[
        int, Field(strict=True, ge=1, le=100)
    ] = MAX_LIMIT) -> BusinessResult:
        """按实际 engine_code 精确查询任务列表，最多返回 limit 条。

        engine_code 是必填参数，只能使用用户或当前已确认上下文提供的实际值。
        仅知道 limit 时不能调用本工具；缺少编码时应先询问用户，
        不得使用占位符、示例值或编造值，也不得调用本工具探测或确认缺参。
        limit 可省略，默认 100，范围 1～100；仅限制返回条数，不代替 engine_code。
        """
        return prepare_tool_result(service.query_tasks_by_engine(engine_code, limit))

    return mcp


if __name__ == "__main__":
    # 完成 Day3 后：.venv/bin/python -B week6.py；该命令启动离线 stdio 服务。
    with create_mock_http_client() as http_client:
        service = BusinessService(BusinessAPIClient(http_client, max_attempts=2))
        create_mcp_server(service).run(transport="stdio")
