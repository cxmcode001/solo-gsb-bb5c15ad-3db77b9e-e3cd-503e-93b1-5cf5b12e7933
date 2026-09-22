"""请求 / 响应模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ExperimentCreate(BaseModel):
    """创建交叉实验的请求体。"""

    conditions: list[str] = Field(
        ..., min_length=2, description="互异的实验条件，至少 2 个"
    )
    subject_ids: list[str] = Field(
        ..., min_length=1, description="互异受试者 ID"
    )
    seed: int = Field(..., description="随机种子，相同输入与种子产生相同分配")
    washout_seconds: int = Field(
        default=0,
        ge=0,
        description=(
            "相邻周期间的洗脱期秒数（非负）。上一周期完成后必须再经过该时长"
            "才能推进下一周期；默认 0 表示只需登记完成即可推进"
        ),
    )


class AssignmentOut(BaseModel):
    """受试者名册条目；不含条件序列，避免提前揭盲。"""

    subject_id: str


class ExperimentOut(BaseModel):
    experiment_id: str
    conditions: list[str]
    seed: int
    group_size: int
    num_subjects: int
    num_periods: int
    washout_seconds: int
    status: Literal["active", "paused"]
    state_version: int
    created_at: str
    assignments: list[AssignmentOut]


class ExperimentSummary(BaseModel):
    experiment_id: str
    conditions: list[str]
    seed: int
    group_size: int
    num_subjects: int
    num_periods: int
    washout_seconds: int
    status: Literal["active", "paused"]
    state_version: int
    created_at: str


class AdvanceRequest(BaseModel):
    """按周期受控揭盲的推进请求。"""

    subject_id: str = Field(..., min_length=1, description="受试者 ID")
    expected_period: int = Field(
        ...,
        ge=1,
        description="调用方认为该受试者所处的下一周期（从 1 开始）",
    )
    request_id: str = Field(
        ..., min_length=1, description="幂等键，相同 request_id 重试不重复推进"
    )


class AdvanceOut(BaseModel):
    experiment_id: str
    subject_id: str
    period: int
    condition: str
    request_id: str
    revealed_at: str


class AdvanceBatchMember(BaseModel):
    """批量揭盲中的一个成员：受试者与其预期下一周期。"""

    subject_id: str = Field(..., min_length=1, description="受试者 ID")
    expected_period: int = Field(
        ...,
        ge=1,
        description="调用方认为该受试者所处的下一周期（从 1 开始）",
    )


class AdvanceBatchRequest(BaseModel):
    """批量受控揭盲请求：1～100 个互异的 subject_id / expected_period 对。"""

    batch_id: str = Field(
        ...,
        min_length=1,
        description="批量幂等键，相同 batch_id 与相同成员列表重试返回首次结果",
    )
    members: list[AdvanceBatchMember] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="批量成员（1～100 个），按该列表顺序揭盲与推进",
    )

    @model_validator(mode="after")
    def _members_must_be_distinct(self) -> "AdvanceBatchRequest":
        seen: set[tuple[str, int]] = set()
        for member in self.members:
            key = (member.subject_id, member.expected_period)
            if key in seen:
                raise ValueError(
                    "members 必须互异：存在重复的 subject_id / expected_period"
                )
            seen.add(key)
        return self


class AdvanceBatchItemOut(AdvanceOut):
    """批量揭盲中单个成员的揭盲结果。"""


class AdvanceBatchOut(BaseModel):
    experiment_id: str
    batch_id: str
    # 结果严格按请求 members 的输入顺序返回
    results: list[AdvanceBatchItemOut]


class CompleteRequest(BaseModel):
    """周期完成登记请求。"""

    subject_id: str = Field(..., min_length=1, description="受试者 ID")
    period: int = Field(
        ...,
        ge=1,
        description="要登记完成的周期（必须是已揭盲且尚未完成的当前周期）",
    )
    request_id: str = Field(
        ..., min_length=1, description="幂等键，相同 request_id 重试返回原结果"
    )


class CompleteOut(BaseModel):
    experiment_id: str
    subject_id: str
    period: int
    request_id: str
    completed_at: str


class RevealOut(BaseModel):
    """一条已揭盲记录；completed_at 为该周期完成登记时间（未登记为 null）。"""

    period: int
    condition: str
    revealed_at: str
    completed_at: str | None = None


class ProgressOut(BaseModel):
    """受试者进度：只含已揭盲记录、下一周期与下一周期可推进时间。"""

    experiment_id: str
    subject_id: str
    next_period: int
    num_periods: int
    washout_seconds: int
    completed: bool
    next_available_at: str | None = None
    reveals: list[RevealOut]


class PauseRequest(BaseModel):
    """暂停实验请求：携带唯一 operation_id 与暂停原因。"""

    operation_id: str = Field(
        ...,
        min_length=1,
        description="本次暂停的唯一操作键（如 UUID），相同操作重试必须复用",
    )
    reason: str = Field(
        ..., min_length=1, description="暂停原因（必填），随审计记录持久化"
    )


class ResumeRequest(BaseModel):
    """恢复实验请求：携带期望状态版本做原子 compare-and-set。"""

    expected_version: int = Field(
        ...,
        ge=0,
        description=(
            "调用方看到的暂停状态版本；服务端仅在当前版本与之相等且实验已暂停时"
            "才原子恢复，否则 409 并回传当前版本"
        ),
    )
    operation_id: str | None = Field(
        default=None,
        min_length=1,
        description="可选的唯一操作键；携带时相同操作重试返回首次结果",
    )
    reason: str | None = Field(
        default=None,
        min_length=1,
        description="可选的恢复原因，随审计记录持久化",
    )


class StateEventOut(BaseModel):
    """一条实验状态变更审计记录。"""

    id: int
    experiment_id: str
    operation_id: str | None = None
    action: Literal["pause", "resume"]
    from_status: Literal["active", "paused"]
    to_status: Literal["active", "paused"]
    state_version: int
    reason: str | None = None
    created_at: str


class StateEventsOut(BaseModel):
    """状态变更审计列表，按时间倒序（最新在前）。"""

    experiment_id: str
    events: list[StateEventOut]


class StateTransitionOperation(BaseModel):
    """批量状态切换中的单项操作：目标实验、动作、期望版本与操作幂等键。"""

    experiment_id: str = Field(..., min_length=1, description="目标实验 ID")
    action: Literal["pause", "resume"] = Field(
        ..., description="目标动作：pause（暂停）或 resume（恢复）"
    )
    expected_version: int = Field(
        ...,
        ge=0,
        description=(
            "调用方看到的该实验状态版本；服务端仅在当前版本与之相等且状态"
            "允许目标动作时才执行，否则整批 409"
        ),
    )
    operation_id: str = Field(
        ...,
        min_length=1,
        description="本项操作的唯一幂等键（如 UUID），与单条暂停 / 恢复共享幂等空间",
    )
    reason: str | None = Field(
        default=None,
        min_length=1,
        description="可选的暂停 / 恢复原因，随审计记录持久化",
    )


class StateTransitionBatchRequest(BaseModel):
    """批量状态切换请求：1～100 项操作，同一实验每批最多出现一次。"""

    batch_operation_id: str = Field(
        ...,
        min_length=1,
        description=(
            "整批幂等键（如 UUID），相同 batch_operation_id 与相同操作列表"
            "重试返回首次结果"
        ),
    )
    operations: list[StateTransitionOperation] = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "状态切换操作列表（1～100 项），全部校验通过才按输入顺序执行，"
            "任一项冲突整批回滚"
        ),
    )

    @model_validator(mode="after")
    def _experiments_must_be_distinct(self) -> "StateTransitionBatchRequest":
        seen: set[str] = set()
        for operation in self.operations:
            if operation.experiment_id in seen:
                raise ValueError(
                    "operations 必须互异：同一实验在同一批中最多出现一次"
                )
            seen.add(operation.experiment_id)
        return self


class StateTransitionBatchOut(BaseModel):
    batch_operation_id: str
    # 结果严格按请求 operations 的输入顺序返回
    results: list[StateEventOut]
