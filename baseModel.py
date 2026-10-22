from typing import Any, Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class AgentInput(BaseModel):
    """供程序组装本轮 Prompt 的任务数据，不替代 Session 和 TaskState。

    session_id 由调用程序管理；续接任务时，可将 TaskState.todos 填入 todos。
    workspace 只提供任务背景，实际工具授权仍由运行上下文决定。
    """

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    goal: str = Field(description="本轮任务目标", min_length=1)
    constraints: list[str] = Field(default_factory=list, description="本轮必须遵守的约束")
    workspace: str | None = Field(default=None, description="项目工作目录；普通聊天可不提供", min_length=1)
    verification: str = Field(default="", description="验收方法；未指定时为空，不代表验证已通过")
    todos: list[str] = Field(default_factory=list, description="本轮执行前已有的待办事项")

    @field_validator("constraints", "todos")
    @classmethod
    def check_nonempty_items(cls, values: list[str]) -> list[str]:
        """允许空列表，但不接受空白的约束或待办事项。"""
        if any(not value.strip() for value in values):
            raise ValueError("约束和待办事项不能包含空白内容")
        return values


def build_prompt(task: AgentInput) -> str:
    """将已校验的任务数据组装为分段中文 Prompt。

    普通对话只需提供 goal，未填写的可选信息不生成对应段落。
    本函数不修改输入，也不读取 Session、任务状态或调用 LLM。
    """
    sections = [f"任务目标：\n{task.goal}"]

    if task.workspace:
        sections.append(f"工作目录：\n{task.workspace}")
    if task.constraints:
        constraints = "\n".join(f"- {item}" for item in task.constraints)
        sections.append(f"任务约束：\n{constraints}")
    if task.todos:
        todos = "\n".join(f"- {item}" for item in task.todos)
        sections.append(f"已有待办事项：\n{todos}")
    if task.verification:
        sections.append(f"验收方法：\n{task.verification}")

    return "\n\n".join(sections)


class AgentOutput(BaseModel):
    """LLM 的本轮最终回答，所有字段必填；无修改或待办时显式返回空列表。

    unfinished_work 依据输入待办和本轮实际结果更新，不是简单复制输入。
    test_status 只表示测试结果，不等同于整个任务完成；usage 和耗时由程序记录。
    """

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    summary: str = Field(description="本轮处理结论，必须依据实际执行结果", min_length=1)
    changed_files: list[str] = Field(description="实际修改的文件，路径相对项目工作目录")
    test_status: Literal["not_run", "passed", "failed"] = Field(description="本轮测试的实际结果")
    unfinished_work: list[str] = Field(description="本轮结束后仍未完成的事项")

    @field_validator("changed_files")
    @classmethod
    def check_relative_paths(cls, values: list[str]) -> list[str]:
        """统一用 / 分隔的相对路径；这项格式检查不能替代工具层权限校验。"""
        for value in values:
            if (not value.strip() or "\\" in value or ":" in value
                    or any(part in ("", ".", "..") for part in value.split("/"))):
                raise ValueError("修改文件必须使用相对项目目录的规范路径，例如 src/report.py")
        if len(values) != len(set(values)):
            raise ValueError("修改文件列表不能包含重复路径")
        return values

    @field_validator("unfinished_work")
    @classmethod
    def check_unfinished_work(cls, values: list[str]) -> list[str]:
        """允许没有未完成事项，但每条事项必须有实际内容。"""
        if any(not value.strip() for value in values):
            raise ValueError("未完成事项不能包含空白内容")
        return values


class ToolOutputBase(BaseModel):
    """工具输出的公共字段，具体结果由子类扩展。

    success 表示工具是否正常执行，不代表测试通过等业务目标已经达成。
    致命错误仍由工具抛出异常；retryable 只提供信息，不触发自动重试。
    """

    # 不统一去除字符串空白，避免改变子类中的文件原文、代码和日志。
    model_config = ConfigDict(extra="forbid", strict=True)

    success: bool = Field(description="工具是否正常完成执行，不等同于业务结果通过")
    message: str = Field(description="简短执行说明或错误原因", default="")
    error_code: str | None = Field(description="错误码", default=None)
    retryable: bool = Field(description="失败后是否允许修正参数或条件后再次尝试，不表示自动重试", default=False)
    source: str | None = Field(description="结果来源或完整日志位置", default=None)
    truncated: bool = Field(description="返回内容是否发生截断", default=False)

    @field_validator("error_code", "source")
    @classmethod
    def check_optional_text(cls, value: str | None) -> str | None:
        """没有错误码或来源时用 None，有值时不能只有空白。"""
        if value is not None and not value.strip():
            raise ValueError("错误码和来源不能是空白字符串，未提供时请使用 None")
        return value

    @model_validator(mode="after")
    def check_execution_result(self) -> "ToolOutputBase":
        """检查成功状态、错误信息和重试标记是否一致。"""
        if self.success:
            if self.error_code is not None:
                raise ValueError("工具正常执行时 error_code 必须为 None")
            if self.retryable:
                raise ValueError("工具正常执行时 retryable 必须为 False")
        else:
            if self.error_code is None:
                raise ValueError("工具执行失败时必须提供 error_code")
            if not self.message.strip():
                raise ValueError("工具执行失败时必须提供非空的错误说明")
        return self


class FileReadOutput(ToolOutputBase):
    """readFile 和 readFileRange 共用的读取结果，保留原有正文与续读字段。"""

    source: str = Field(description="实际读取的文件路径", min_length=1)
    excerpt: str = Field(description="本次返回的 UTF-8 文本，不去除首尾空白")
    complete: bool = Field(description="本次请求的内容是否返回完整，不代表读完了整份文件")
    start_line: int = Field(description="本次读取的起始行，从 1 开始", ge=1)
    start_column: int = Field(description="首行的起始列，按字符计数，从 1 开始", ge=1)
    next_line: int | None = Field(default=None, description="截断后的续读行，未截断时为 None", ge=1)
    next_column: int | None = Field(default=None, description="截断后的续读列，未截断时为 None", ge=1)
