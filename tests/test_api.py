"""API 与 Williams 序列的自动化测试。"""

from __future__ import annotations

import sqlite3
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app
from app.williams import group_size, williams_sequences


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as test_client:
        yield test_client


def assert_balanced(sequences: list[list[str]], n: int) -> None:
    """校验位置平衡与有向相邻对平衡。"""
    size = group_size(n)
    assert len(sequences) % size == 0
    k = len(sequences) // size  # 完整组数
    per_position = k * (1 if n % 2 == 0 else 2)

    # 每个条件在每个位置出现次数相等
    for pos in range(n):
        counts = Counter(seq[pos] for seq in sequences)
        assert len(counts) == n
        assert all(v == per_position for v in counts.values())

    # 每个有向相邻条件对出现次数相等
    pairs: Counter = Counter()
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            pairs[(a, b)] += 1
    assert len(pairs) == n * (n - 1)
    assert len(set(pairs.values())) == 1


def create_experiment(
    client: TestClient,
    conditions: list[str] | None = None,
    subject_ids: list[str] | None = None,
    seed: int = 42,
    washout_seconds: int = 0,
):
    conditions = conditions or ["A", "B", "C"]
    subject_ids = subject_ids or [f"s{i}" for i in range(6)]
    resp = client.post(
        "/experiments",
        json={
            "conditions": conditions,
            "subject_ids": subject_ids,
            "seed": seed,
            "washout_seconds": washout_seconds,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json(), subject_ids


def complete_period(
    client: TestClient,
    experiment_id: str,
    subject_id: str,
    period: int,
    request_id: str | None = None,
):
    resp = client.post(
        f"/experiments/{experiment_id}/complete",
        json={
            "subject_id": subject_id,
            "period": period,
            "request_id": request_id or f"{subject_id}-c{period}",
        },
    )
    return resp


def reveal_period(
    client: TestClient,
    experiment_id: str,
    subject_id: str,
    period: int,
    req_prefix: str = "",
):
    return client.post(
        f"/experiments/{experiment_id}/advance",
        json={
            "subject_id": subject_id,
            "expected_period": period,
            "request_id": f"{req_prefix}{subject_id}-p{period}",
        },
    )


def reveal_sequence(
    client: TestClient,
    experiment_id: str,
    subject_id: str,
    num_periods: int,
    req_prefix: str = "",
) -> list[str]:
    """通过推进接口逐周期揭盲，重建该受试者的条件序列。

    默认实验 washout=0，因此每个周期揭盲后登记完成即可立刻推进下一周期。
    """
    sequence: list[str] = []
    for period in range(1, num_periods + 1):
        resp = reveal_period(
            client, experiment_id, subject_id, period, req_prefix
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["period"] == period
        sequence.append(body["condition"])
        if period < num_periods:
            done = complete_period(
                client,
                experiment_id,
                subject_id,
                period,
                f"{req_prefix}{subject_id}-c{period}",
            )
            assert done.status_code == 200, done.text
    return sequence


def assert_no_sequence_leak(payload: object) -> None:
    """响应中任何层级都不得出现条件序列或序列下标。"""
    forbidden = {"sequence", "sequence_index"}
    if isinstance(payload, dict):
        assert forbidden.isdisjoint(payload.keys())
        for value in payload.values():
            assert_no_sequence_leak(value)
    elif isinstance(payload, list):
        for item in payload:
            assert_no_sequence_leak(item)


# ---------- Williams 序列单元测试 ----------

@pytest.mark.parametrize("n", range(2, 9))
def test_williams_sequences_balanced(n: int) -> None:
    sequences = [[str(i) for i in seq] for seq in williams_sequences(n)]
    assert len(sequences) == group_size(n)
    assert_balanced(sequences, n)


# ---------- 创建 / 查询接口（不再下发完整序列） ----------

def test_create_even_conditions(client: TestClient) -> None:
    body, subject_ids = create_experiment(
        client,
        conditions=["A", "B", "C", "D"],
        subject_ids=[f"s{i}" for i in range(8)],
    )
    assert body["group_size"] == 4
    assert body["num_subjects"] == 8
    assert body["num_periods"] == 4
    assert len(body["assignments"]) == 8
    # 名册只含受试者 ID，且顺序与提交一致
    assert [a["subject_id"] for a in body["assignments"]] == subject_ids
    assert set(body["assignments"][0].keys()) == {"subject_id"}
    assert_no_sequence_leak(body)

    # 序列只能通过逐周期揭盲重建，且仍满足 Williams 平衡
    sequences = [
        reveal_sequence(client, body["experiment_id"], s, 4)
        for s in subject_ids
    ]
    assert_balanced(sequences, 4)
    # 2 个完整组：每条 Williams 序列恰好被使用 2 次
    assert sorted(Counter(tuple(seq) for seq in sequences).values()) == [
        2, 2, 2, 2
    ]


def test_create_odd_conditions(client: TestClient) -> None:
    body, subject_ids = create_experiment(
        client,
        conditions=["A", "B", "C"],
        subject_ids=[f"s{i}" for i in range(6)],
        seed=7,
    )
    assert body["group_size"] == 6
    assert body["num_periods"] == 3
    assert_no_sequence_leak(body)
    sequences = [
        reveal_sequence(client, body["experiment_id"], s, 3)
        for s in subject_ids
    ]
    assert_balanced(sequences, 3)


def test_same_input_and_seed_gives_same_assignment(client: TestClient) -> None:
    payload = {
        "conditions": ["A", "B", "C"],
        "subject_ids": [f"s{i}" for i in range(12)],
        "seed": 123,
    }
    first = client.post("/experiments", json=payload).json()
    second = client.post("/experiments", json=payload).json()
    assert first["assignments"] == second["assignments"]
    # 逐周期揭盲重建的序列也完全一致（确定性分配）
    first_seqs = [
        reveal_sequence(client, first["experiment_id"], s, 3, "first-")
        for s in payload["subject_ids"]
    ]
    second_seqs = [
        reveal_sequence(client, second["experiment_id"], s, 3, "second-")
        for s in payload["subject_ids"]
    ]
    assert first_seqs == second_seqs


def test_subject_count_not_multiple_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/experiments",
        json={
            "conditions": ["A", "B", "C"],
            "subject_ids": [f"s{i}" for i in range(7)],
            "seed": 1,
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["group_size"] == 6
    assert detail["subject_count"] == 7
    assert detail["nearest_acceptable"] == [6, 12]


def test_subject_count_below_one_group_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/experiments",
        json={
            "conditions": ["A", "B", "C", "D"],
            "subject_ids": ["s1", "s2"],
            "seed": 1,
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["nearest_acceptable"] == [4]


def test_duplicate_conditions_rejected(client: TestClient) -> None:
    resp = client.post(
        "/experiments",
        json={
            "conditions": ["A", "A", "B"],
            "subject_ids": [f"s{i}" for i in range(6)],
            "seed": 1,
        },
    )
    assert resp.status_code == 422


def test_duplicate_subjects_rejected(client: TestClient) -> None:
    resp = client.post(
        "/experiments",
        json={
            "conditions": ["A", "B", "C"],
            "subject_ids": ["s1", "s1", "s2", "s3", "s4", "s5"],
            "seed": 1,
        },
    )
    assert resp.status_code == 422


def test_too_few_conditions_rejected(client: TestClient) -> None:
    resp = client.post(
        "/experiments",
        json={"conditions": ["A"], "subject_ids": ["s1"], "seed": 1},
    )
    assert resp.status_code == 422


def test_get_experiment_does_not_leak_sequence(client: TestClient) -> None:
    created, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=5
    )
    experiment_id = created["experiment_id"]

    resp = client.get(f"/experiments/{experiment_id}")
    assert resp.status_code == 200
    assert resp.json() == created
    assert_no_sequence_leak(resp.json())

    listing = client.get("/experiments").json()
    entry = next(e for e in listing if e["experiment_id"] == experiment_id)
    assert entry["num_periods"] == 2
    assert_no_sequence_leak(entry)
    assert "assignments" not in entry


def test_get_unknown_experiment_returns_404(client: TestClient) -> None:
    assert client.get("/experiments/nope").status_code == 404


# ---------- 按周期受控揭盲 ----------

def test_advance_reveals_one_period_and_progresses(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B", "C"], subject_ids=[f"s{i}" for i in range(6)]
    )
    eid = body["experiment_id"]

    progress = client.get(f"/experiments/{eid}/subjects/s0/progress").json()
    assert progress["next_period"] == 1
    assert progress["num_periods"] == 3
    assert progress["washout_seconds"] == 0
    assert progress["completed"] is False
    assert progress["next_available_at"] is None
    assert progress["reveals"] == []

    first = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s0", "expected_period": 1, "request_id": "r1"},
    )
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["period"] == 1
    assert first_body["condition"] in {"A", "B", "C"}
    assert first_body["request_id"] == "r1"
    assert "revealed_at" in first_body

    progress = client.get(f"/experiments/{eid}/subjects/s0/progress").json()
    assert progress["next_period"] == 2
    assert progress["next_available_at"] is None
    assert progress["reveals"] == [
        {
            "period": 1,
            "condition": first_body["condition"],
            "revealed_at": first_body["revealed_at"],
            "completed_at": None,
        }
    ]
    assert_no_sequence_leak(progress)


def test_stale_or_skipped_period_returns_409_and_reveals_nothing(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=3
    )
    eid = body["experiment_id"]

    # 跳过周期 1 直接请求周期 2
    resp = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 2, "request_id": "skip"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "period_mismatch"
    assert detail["next_period"] == 1
    assert "condition" not in detail

    # 失败不推进、不留幂等记录：同一 request_id 在正确周期上仍可首次成功
    ok = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "skip"},
    )
    assert ok.status_code == 200

    # 周期 1 已揭盲后，过期请求周期 1 再次冲突
    stale = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "late"},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["next_period"] == 2

    progress = client.get(f"/experiments/{eid}/subjects/s1/progress").json()
    assert progress["next_period"] == 2
    assert len(progress["reveals"]) == 1


def test_same_request_id_retry_is_idempotent(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=9
    )
    eid = body["experiment_id"]
    payload = {
        "subject_id": "s1",
        "expected_period": 1,
        "request_id": "idem-1",
    }

    first = client.post(f"/experiments/{eid}/advance", json=payload)
    assert first.status_code == 200
    first_body = first.json()

    for _ in range(3):
        retry = client.post(f"/experiments/{eid}/advance", json=payload)
        assert retry.status_code == 200
        assert retry.json() == first_body

    # 重试不重复推进
    progress = client.get(f"/experiments/{eid}/subjects/s1/progress").json()
    assert progress["next_period"] == 2
    assert len(progress["reveals"]) == 1

    # 登记周期 1 完成（washout=0，完成后即可推进）
    done = complete_period(client, eid, "s1", 1, "c1")
    assert done.status_code == 200

    # 推进到周期 2 后，用旧 request_id 重放仍返回首次（周期 1）结果
    second = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 2, "request_id": "idem-2"},
    )
    assert second.status_code == 200
    assert second.json()["period"] == 2

    replay = client.post(f"/experiments/{eid}/advance", json=payload)
    assert replay.status_code == 200
    assert replay.json() == first_body


def test_request_id_reused_with_different_payload_returns_409(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=4
    )
    eid = body["experiment_id"]

    first = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "shared"},
    )
    assert first.status_code == 200

    # 同一 request_id 绑定不同受试者
    other_subject = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s2", "expected_period": 1, "request_id": "shared"},
    )
    assert other_subject.status_code == 409
    assert other_subject.json()["detail"]["reason"] == "request_id_conflict"

    # s2 未因此被推进或揭盲
    progress = client.get(f"/experiments/{eid}/subjects/s2/progress").json()
    assert progress["next_period"] == 1
    assert progress["reveals"] == []


def test_concurrent_advance_has_single_winner(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "conc.db"))
    with TestClient(app) as setup:
        created = setup.post(
            "/experiments",
            json={
                "conditions": ["A", "B"],
                "subject_ids": ["s1", "s2"],
                "seed": 11,
            },
        ).json()
    eid = created["experiment_id"]

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    statuses: list[int] = []
    statuses_lock = threading.Lock()

    def worker(i: int) -> None:
        # 每个线程使用独立 client（独立门户线程），共享同一个 SQLite 文件
        with TestClient(app) as local_client:
            barrier.wait()
            resp = local_client.post(
                f"/experiments/{eid}/advance",
                json={
                    "subject_id": "s1",
                    "expected_period": 1,
                    "request_id": f"race-{i}",
                },
            )
            with statuses_lock:
                statuses.append(resp.status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert statuses.count(200) == 1
    assert statuses.count(409) == n_threads - 1

    with TestClient(app) as verify:
        progress = verify.get(
            f"/experiments/{eid}/subjects/s1/progress"
        ).json()
    assert progress["next_period"] == 2
    assert len(progress["reveals"]) == 1


def test_advance_after_final_period_returns_409(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=8
    )
    eid = body["experiment_id"]

    sequence = reveal_sequence(client, eid, "s1", 2)
    assert len(sequence) == 2

    done = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 3, "request_id": "over"},
    )
    assert done.status_code == 409
    assert done.json()["detail"]["reason"] == "experiment_complete"
    assert done.json()["detail"]["next_period"] == 3

    progress = client.get(f"/experiments/{eid}/subjects/s1/progress").json()
    assert progress["completed"] is True
    assert progress["next_period"] == 3
    assert [r["period"] for r in progress["reveals"]] == [1, 2]


def test_advance_unknown_experiment_or_subject_returns_404(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=6
    )
    eid = body["experiment_id"]
    payload = {
        "subject_id": "ghost",
        "expected_period": 1,
        "request_id": "x",
    }

    assert client.post("/experiments/missing/advance", json=payload).status_code == 404
    assert client.post(f"/experiments/{eid}/advance", json=payload).status_code == 404
    assert client.get(f"/experiments/{eid}/subjects/ghost/progress").status_code == 404


def test_invalid_advance_payload_rejected(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=2
    )
    eid = body["experiment_id"]

    resp = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 0, "request_id": "r"},
    )
    assert resp.status_code == 422

    resp = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": ""},
    )
    assert resp.status_code == 422


# ---------- 周期完成与洗脱期门禁 ----------

def test_washout_configured_on_create_and_defaults_zero(client: TestClient) -> None:
    body, _ = create_experiment(
        client,
        conditions=["A", "B"],
        subject_ids=["s1", "s2"],
        washout_seconds=900,
    )
    assert body["washout_seconds"] == 900
    listing = client.get("/experiments").json()
    assert next(
        e for e in listing if e["experiment_id"] == body["experiment_id"]
    )["washout_seconds"] == 900

    default_body, _ = create_experiment(
        client,
        conditions=["A", "B"],
        subject_ids=["s3", "s4"],
        seed=3,
    )
    assert default_body["washout_seconds"] == 0

    # 负数洗脱期被拒绝
    resp = client.post(
        "/experiments",
        json={
            "conditions": ["A", "B"],
            "subject_ids": ["s5", "s6"],
            "seed": 1,
            "washout_seconds": -1,
        },
    )
    assert resp.status_code == 422


def test_complete_only_allowed_for_revealed_current_period(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=3
    )
    eid = body["experiment_id"]

    # 尚未揭盲任何周期
    early = complete_period(client, eid, "s1", 1, "early")
    assert early.status_code == 409
    detail = early.json()["detail"]
    assert detail["reason"] == "period_out_of_order"
    assert detail["current_period"] == 0

    first = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    )
    assert first.status_code == 200

    # 只能登记当前已揭盲周期，不能提前 / 补登
    skip = complete_period(client, eid, "s1", 2, "skip")
    assert skip.status_code == 409
    assert skip.json()["detail"]["reason"] == "period_out_of_order"
    assert skip.json()["detail"]["current_period"] == 1

    ok = complete_period(client, eid, "s1", 1, "c1")
    assert ok.status_code == 200
    completed_body = ok.json()
    assert completed_body["period"] == 1
    assert completed_body["request_id"] == "c1"
    assert "completed_at" in completed_body

    # 同一周期重复登记（换 request_id）→ 409，completed_at 不被改写
    again = complete_period(client, eid, "s1", 1, "c1-again")
    assert again.status_code == 409
    assert again.json()["detail"]["reason"] == "period_already_completed"
    assert again.json()["detail"]["current_period"] == 1


def test_complete_request_id_is_idempotent_and_replays(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=9
    )
    eid = body["experiment_id"]
    client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    )

    payload = {"subject_id": "s1", "period": 1, "request_id": "done-1"}
    first = client.post(f"/experiments/{eid}/complete", json=payload)
    assert first.status_code == 200
    for _ in range(3):
        replay = client.post(f"/experiments/{eid}/complete", json=payload)
        assert replay.status_code == 200
        assert replay.json() == first.json()

    # 同一完成幂等键绑定不同周期 / 受试者 → 409
    other_period = client.post(
        f"/experiments/{eid}/complete",
        json={"subject_id": "s1", "period": 2, "request_id": "done-1"},
    )
    assert other_period.status_code == 409
    assert other_period.json()["detail"]["reason"] == "request_id_conflict"
    other_subject = client.post(
        f"/experiments/{eid}/complete",
        json={"subject_id": "s2", "period": 1, "request_id": "done-1"},
    )
    assert other_subject.status_code == 409

    # 推进与完成的 request_id 命名空间互不影响
    second_advance = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 2, "request_id": "a2"},
    )
    assert second_advance.status_code == 200


def test_advance_blocked_until_prior_period_completed(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=4
    )
    eid = body["experiment_id"]
    client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    )

    # washout=0 但上一周期尚未完成 → 409，且不泄露下一周期条件
    blocked = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 2, "request_id": "a2"},
    )
    assert blocked.status_code == 409
    detail = blocked.json()["detail"]
    assert detail["reason"] == "prior_period_not_completed"
    assert "condition" not in detail
    assert "next_period" not in detail

    # 门禁失败不消耗幂等键、不推进：登记完成后原 request_id 首次成功
    done = complete_period(client, eid, "s1", 1, "c1")
    assert done.status_code == 200
    ok = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 2, "request_id": "a2"},
    )
    assert ok.status_code == 200
    assert ok.json()["period"] == 2


def test_washout_gate_uses_server_time_and_returns_retry_after(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "washout.db"))
    t0 = datetime(2026, 9, 21, 8, 0, 0, tzinfo=timezone.utc)

    class Clock:
        def __init__(self) -> None:
            self.now = t0

        def __call__(self) -> datetime:
            return self.now

    clock = Clock()
    monkeypatch.setattr(db, "utc_now", clock)

    with TestClient(app) as local:
        created = local.post(
            "/experiments",
            json={
                "conditions": ["A", "B"],
                "subject_ids": ["s1", "s2"],
                "seed": 1,
                "washout_seconds": 3600,
            },
        ).json()
        eid = created["experiment_id"]

        first = local.post(
            f"/experiments/{eid}/advance",
            json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
        )
        assert first.status_code == 200

        clock.now += timedelta(seconds=120)
        done = local.post(
            f"/experiments/{eid}/complete",
            json={"subject_id": "s1", "period": 1, "request_id": "c1"},
        )
        assert done.status_code == 200
        completed_at = datetime.fromisoformat(done.json()["completed_at"])

        # 进度只给出下一周期最早可推进时间，不含下一周期条件
        progress = local.get(f"/experiments/{eid}/subjects/s1/progress").json()
        assert progress["washout_seconds"] == 3600
        assert progress["next_period"] == 2
        assert progress["next_available_at"] == (
            completed_at + timedelta(seconds=3600)
        ).isoformat()
        assert progress["reveals"][0]["completed_at"] == done.json()["completed_at"]
        assert_no_sequence_leak(progress)

        advance_payload = {
            "subject_id": "s1",
            "expected_period": 2,
            "request_id": "a2",
        }

        # 刚完成：洗脱期还差 3600 秒
        blocked = local.post(f"/experiments/{eid}/advance", json=advance_payload)
        assert blocked.status_code == 409
        detail = blocked.json()["detail"]
        assert detail["reason"] == "washout_not_elapsed"
        assert detail["retry_after_seconds"] == 3600
        assert "condition" not in detail
        assert "next_period" not in detail

        # 差 1 秒到期
        clock.now = completed_at + timedelta(seconds=3599)
        blocked = local.post(f"/experiments/{eid}/advance", json=advance_payload)
        assert blocked.status_code == 409
        assert blocked.json()["detail"]["retry_after_seconds"] == 1

        # 到期瞬间放行；门禁期间被拒的幂等键仍可首次成功
        clock.now = completed_at + timedelta(seconds=3600)
        ok = local.post(f"/experiments/{eid}/advance", json=advance_payload)
        assert ok.status_code == 200
        assert ok.json()["period"] == 2

        progress = local.get(f"/experiments/{eid}/subjects/s1/progress").json()
        assert progress["next_available_at"] is None


def test_final_period_can_be_completed(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=8
    )
    eid = body["experiment_id"]
    reveal_sequence(client, eid, "s1", 2)

    final = complete_period(client, eid, "s1", 2, "c2")
    assert final.status_code == 200

    progress = client.get(f"/experiments/{eid}/subjects/s1/progress").json()
    assert progress["completed"] is True
    assert progress["next_period"] == 3
    assert progress["next_available_at"] is None
    assert [r["period"] for r in progress["reveals"]] == [1, 2]
    assert all(r["completed_at"] is not None for r in progress["reveals"])


def test_complete_unknown_experiment_or_subject_returns_404(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=6
    )
    eid = body["experiment_id"]
    payload = {"subject_id": "ghost", "period": 1, "request_id": "x"}

    assert client.post(
        "/experiments/missing/complete", json=payload
    ).status_code == 404
    assert client.post(
        f"/experiments/{eid}/complete", json=payload
    ).status_code == 404


def test_invalid_complete_payload_rejected(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=2
    )
    eid = body["experiment_id"]

    resp = client.post(
        f"/experiments/{eid}/complete",
        json={"subject_id": "s1", "period": 0, "request_id": "r"},
    )
    assert resp.status_code == 422
    resp = client.post(
        f"/experiments/{eid}/complete",
        json={"subject_id": "s1", "period": 1, "request_id": ""},
    )
    assert resp.status_code == 422


def test_concurrent_complete_has_single_winner(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "conc-complete.db"))
    with TestClient(app) as setup:
        created = setup.post(
            "/experiments",
            json={
                "conditions": ["A", "B"],
                "subject_ids": ["s1", "s2"],
                "seed": 12,
            },
        ).json()
        setup.post(
            f"/experiments/{created['experiment_id']}/advance",
            json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
        )
    eid = created["experiment_id"]

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    statuses: list[int] = []
    statuses_lock = threading.Lock()

    def worker(i: int) -> None:
        with TestClient(app) as local_client:
            barrier.wait()
            resp = local_client.post(
                f"/experiments/{eid}/complete",
                json={"subject_id": "s1", "period": 1, "request_id": f"c-{i}"},
            )
            with statuses_lock:
                statuses.append(resp.status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert statuses.count(200) == 1
    assert statuses.count(409) == n_threads - 1

    with TestClient(app) as verify:
        progress = verify.get(
            f"/experiments/{eid}/subjects/s1/progress"
        ).json()
    assert progress["next_period"] == 2
    assert progress["reveals"][0]["completed_at"] is not None
    # 完成只登记一次：完成后下一周期立刻可推进（washout=0）
    with TestClient(app) as verify:
        advanced = verify.post(
            f"/experiments/{eid}/advance",
            json={"subject_id": "s1", "expected_period": 2, "request_id": "a2"},
        )
    assert advanced.status_code == 200


def test_concurrent_advance_after_completion_has_single_winner(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "conc-advance.db"))
    with TestClient(app) as setup:
        created = setup.post(
            "/experiments",
            json={
                "conditions": ["A", "B"],
                "subject_ids": ["s1", "s2"],
                "seed": 13,
            },
        ).json()
        eid = created["experiment_id"]
        setup.post(
            f"/experiments/{eid}/advance",
            json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
        )
        setup.post(
            f"/experiments/{eid}/complete",
            json={"subject_id": "s1", "period": 1, "request_id": "c1"},
        )

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    statuses: list[int] = []
    statuses_lock = threading.Lock()

    def worker(i: int) -> None:
        with TestClient(app) as local_client:
            barrier.wait()
            resp = local_client.post(
                f"/experiments/{eid}/advance",
                json={
                    "subject_id": "s1",
                    "expected_period": 2,
                    "request_id": f"race2-{i}",
                },
            )
            with statuses_lock:
                statuses.append(resp.status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert statuses.count(200) == 1
    assert statuses.count(409) == n_threads - 1
    with TestClient(app) as verify:
        progress = verify.get(
            f"/experiments/{eid}/subjects/s1/progress"
        ).json()
    assert progress["next_period"] == 3
    assert len(progress["reveals"]) == 2


# ---------- 批量受控揭盲 ----------

def advance_batch(
    client: TestClient,
    experiment_id: str,
    members: list[dict],
    batch_id: str,
):
    return client.post(
        f"/experiments/{experiment_id}/advance-batch",
        json={"batch_id": batch_id, "members": members},
    )


def test_batch_advances_all_members_in_input_order(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=5
    )
    eid = body["experiment_id"]

    members = [
        {"subject_id": "s2", "expected_period": 1},
        {"subject_id": "s1", "expected_period": 1},
    ]
    resp = advance_batch(client, eid, members, "batch-1")
    assert resp.status_code == 200, resp.text
    batch = resp.json()
    assert batch["experiment_id"] == eid
    assert batch["batch_id"] == "batch-1"
    assert [r["subject_id"] for r in batch["results"]] == ["s2", "s1"]
    for result, member in zip(batch["results"], members):
        assert result["period"] == 1
        assert result["condition"] in {"A", "B"}
        assert result["request_id"].startswith("batch:batch-1:")
        assert "revealed_at" in result

    # 进度按整批推进
    for subject_id in ("s1", "s2"):
        progress = client.get(f"/experiments/{eid}/subjects/{subject_id}/progress").json()
        assert progress["next_period"] == 2
        assert len(progress["reveals"]) == 1
        assert progress["reveals"][0]["completed_at"] is None


def test_batch_allows_adjacent_periods_of_same_subject_in_order(
    client: TestClient,
) -> None:
    # 同一受试者的相邻周期可放在同一批中：周期 1 揭盲后需先完成才能揭盲周期 2，
    # 因此只有在之前已登记周期 1 完成时整批才通过
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=5
    )
    eid = body["experiment_id"]
    first = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    ).json()
    assert complete_period(client, eid, "s1", 1, "c1").status_code == 200

    members = [
        {"subject_id": "s1", "expected_period": 2},
        {"subject_id": "s2", "expected_period": 1},
    ]
    resp = advance_batch(client, eid, members, "batch-multi")
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert [r["subject_id"] for r in results] == ["s1", "s2"]
    assert [r["period"] for r in results] == [2, 1]
    # 顺序揭盲：s1 的周期 2 条件即其完整序列第二项，与历史一致
    assert results[0]["condition"] != first["condition"]


def test_batch_failure_is_atomic_and_lists_failures_without_conditions(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=5
    )
    eid = body["experiment_id"]
    # s1 先揭盲周期 1，但不登记完成
    assert client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    ).status_code == 200

    members = [
        # s2 周期 1：本应通过
        {"subject_id": "s2", "expected_period": 1},
        # s1 周期 2：上一周期未完成 → 拒绝
        {"subject_id": "s1", "expected_period": 2},
        # ghost：受试者不存在 → 拒绝
        {"subject_id": "ghost", "expected_period": 1},
    ]
    resp = advance_batch(client, eid, members, "batch-fail")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "batch_member_rejected"
    assert detail["batch_id"] == "batch-fail"
    failures = detail["failures"]
    assert {f["subject_id"]: f["reason"] for f in failures} == {
        "s1": "prior_period_not_completed",
        "ghost": "subject_not_found",
    }
    # 失败项保持输入顺序（s1 在 ghost 之前）
    assert [f["subject_id"] for f in failures] == ["s1", "ghost"]
    # 任何失败项都不得携带条件信息
    assert_no_sequence_leak(detail)

    # 整批零写入：s2 也未被揭盲
    progress = client.get(f"/experiments/{eid}/subjects/s2/progress").json()
    assert progress["next_period"] == 1
    assert progress["reveals"] == []
    # 失败不消耗 batch_id：修正成员后同 batch_id 首次成功
    ok = advance_batch(
        client,
        eid,
        [{"subject_id": "s2", "expected_period": 1}],
        "batch-fail",
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["results"][0]["subject_id"] == "s2"


def test_batch_reports_period_mismatch_and_experiment_complete(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=8
    )
    eid = body["experiment_id"]
    reveal_sequence(client, eid, "s1", 2)  # s1 已完成全部周期

    resp = advance_batch(
        client,
        eid,
        [
            {"subject_id": "s2", "expected_period": 2},   # 超前 → mismatch
            {"subject_id": "s1", "expected_period": 3},   # 已结束
        ],
        "batch-stale",
    )
    assert resp.status_code == 409
    reasons = {f["subject_id"]: f for f in resp.json()["detail"]["failures"]}
    assert reasons["s2"]["reason"] == "period_mismatch"
    assert reasons["s2"]["next_period"] == 1
    assert reasons["s1"]["reason"] == "experiment_complete"
    assert reasons["s1"]["next_period"] == 3
    assert_no_sequence_leak(resp.json())


def test_batch_washout_failure_returns_retry_after_and_zero_writes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "batch-washout.db"))
    t0 = datetime(2026, 9, 21, 8, 0, 0, tzinfo=timezone.utc)

    class Clock:
        def __init__(self) -> None:
            self.now = t0

        def __call__(self) -> datetime:
            return self.now

    clock = Clock()
    monkeypatch.setattr(db, "utc_now", clock)

    with TestClient(app) as local:
        created = local.post(
            "/experiments",
            json={
                "conditions": ["A", "B"],
                "subject_ids": ["s1", "s2"],
                "seed": 1,
                "washout_seconds": 3600,
            },
        ).json()
        eid = created["experiment_id"]
        # s1 已揭盲周期 1，并在 2 分钟后登记完成（距洗脱期届满还差 3480 秒）
        local.post(
            f"/experiments/{eid}/advance",
            json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
        )
        clock.now += timedelta(seconds=120)
        assert local.post(
            f"/experiments/{eid}/complete",
            json={"subject_id": "s1", "period": 1, "request_id": "c1"},
        ).status_code == 200

        # 再走 240 秒：完成于 t0+120，门禁时刻 t0+360，距届满还差 3360 秒
        clock.now += timedelta(seconds=240)
        resp = advance_batch(
            local,
            eid,
            [
                {"subject_id": "s1", "expected_period": 2},
                {"subject_id": "s2", "expected_period": 1},
            ],
            "batch-w",
        )
        assert resp.status_code == 409
        failures = resp.json()["detail"]["failures"]
        assert len(failures) == 1
        assert failures[0]["subject_id"] == "s1"
        assert failures[0]["reason"] == "washout_not_elapsed"
        assert failures[0]["retry_after_seconds"] == 3360
        assert_no_sequence_leak(resp.json())

        # 零写入：s2 未揭盲、s1 仍停在周期 2
        progress_s2 = local.get(f"/experiments/{eid}/subjects/s2/progress").json()
        assert progress_s2["reveals"] == []

        # 到期后用同一 batch_id 首次成功（门禁失败不消耗幂等键）
        clock.now += timedelta(seconds=3360)
        ok = advance_batch(
            local,
            eid,
            [
                {"subject_id": "s1", "expected_period": 2},
                {"subject_id": "s2", "expected_period": 1},
            ],
            "batch-w",
        )
        assert ok.status_code == 200, ok.text
        assert [r["subject_id"] for r in ok.json()["results"]] == ["s1", "s2"]


def test_batch_same_batch_id_replays_first_result(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=9
    )
    eid = body["experiment_id"]
    members = [
        {"subject_id": "s1", "expected_period": 1},
        {"subject_id": "s2", "expected_period": 1},
    ]
    first = advance_batch(client, eid, members, "idem-batch")
    assert first.status_code == 200
    first_body = first.json()

    for _ in range(3):
        retry = advance_batch(client, eid, members, "idem-batch")
        assert retry.status_code == 200
        assert retry.json() == first_body

    # 重放不重复推进
    for subject_id in ("s1", "s2"):
        progress = client.get(f"/experiments/{eid}/subjects/{subject_id}/progress").json()
        assert progress["next_period"] == 2
        assert len(progress["reveals"]) == 1

    # 即使成员已推进到更后周期，重放仍返回首次结果，不重复揭盲
    assert complete_period(client, eid, "s1", 1, "c1").status_code == 200
    assert client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 2, "request_id": "a2"},
    ).status_code == 200
    replay = advance_batch(client, eid, members, "idem-batch")
    assert replay.status_code == 200
    assert replay.json() == first_body


def test_batch_same_batch_id_different_members_returns_409(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=4
    )
    eid = body["experiment_id"]

    first = advance_batch(
        client,
        eid,
        [{"subject_id": "s1", "expected_period": 1}],
        "shared-batch",
    )
    assert first.status_code == 200

    # 不同成员集合
    other_members = advance_batch(
        client,
        eid,
        [{"subject_id": "s2", "expected_period": 1}],
        "shared-batch",
    )
    assert other_members.status_code == 409
    detail = other_members.json()["detail"]
    assert detail["reason"] == "batch_id_conflict"
    assert detail["batch_id"] == "shared-batch"

    # 相同成员但周期不同
    other_period = advance_batch(
        client,
        eid,
        [{"subject_id": "s1", "expected_period": 2}],
        "shared-batch",
    )
    assert other_period.status_code == 409
    assert other_period.json()["detail"]["reason"] == "batch_id_conflict"

    # 相同成员但顺序不同也视为不同请求（使用全新实验避免与上面的状态叠加）
    body2, _ = create_experiment(
        client, conditions=["A", "B", "C", "D"],
        subject_ids=[f"t{i}" for i in range(8)], seed=7,
    )
    eid2 = body2["experiment_id"]
    two = [
        {"subject_id": "t1", "expected_period": 1},
        {"subject_id": "t2", "expected_period": 1},
    ]
    assert advance_batch(client, eid2, two, "ordered-batch").status_code == 200
    reordered = advance_batch(
        client, eid2, list(reversed(two)), "ordered-batch"
    )
    assert reordered.status_code == 409
    assert reordered.json()["detail"]["reason"] == "batch_id_conflict"

    # 冲突的失败路径零写入：t1 / t2 只保留首批各一条揭盲，未重复推进
    for subject_id in ("t1", "t2"):
        progress = client.get(
            f"/experiments/{eid2}/subjects/{subject_id}/progress"
        ).json()
        assert progress["next_period"] == 2
        assert len(progress["reveals"]) == 1


def test_batch_members_validation(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=2
    )
    eid = body["experiment_id"]

    # 空成员列表
    resp = advance_batch(client, eid, [], "empty")
    assert resp.status_code == 422

    # 重复的 subject_id / expected_period 对
    resp = advance_batch(
        client,
        eid,
        [
            {"subject_id": "s1", "expected_period": 1},
            {"subject_id": "s1", "expected_period": 1},
        ],
        "dup",
    )
    assert resp.status_code == 422

    # expected_period < 1 / batch_id 为空
    assert advance_batch(
        client, eid, [{"subject_id": "s1", "expected_period": 0}], "bad"
    ).status_code == 422
    assert advance_batch(
        client, eid, [{"subject_id": "s1", "expected_period": 1}], ""
    ).status_code == 422

    # 超过 100 个成员（受试者名册只建了 2 人，校验在入库前发生）
    too_many = [
        {"subject_id": f"u{i}", "expected_period": 1} for i in range(101)
    ]
    assert advance_batch(client, eid, too_many, "too-many").status_code == 422


def test_batch_boundary_size_of_100_accepted(client: TestClient) -> None:
    # 100 个互异成员（2 条件一组 2 人，100 为整组倍数）
    subject_ids = [f"u{i}" for i in range(100)]
    body, _ = create_experiment(
        client,
        conditions=["A", "B"],
        subject_ids=subject_ids,
        seed=1,
    )
    eid = body["experiment_id"]
    members = [{"subject_id": s, "expected_period": 1} for s in subject_ids]
    resp = advance_batch(client, eid, members, "batch-100")
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["results"]) == 100
    assert [r["subject_id"] for r in resp.json()["results"]] == subject_ids


def test_batch_unknown_experiment_returns_404(client: TestClient) -> None:
    resp = advance_batch(
        client,
        "missing",
        [{"subject_id": "s1", "expected_period": 1}],
        "b",
    )
    assert resp.status_code == 404


def test_concurrent_batches_are_serialized(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "conc-batch.db"))
    with TestClient(app) as setup:
        created = setup.post(
            "/experiments",
            json={
                "conditions": ["A", "B"],
                "subject_ids": ["s1", "s2"],
                "seed": 21,
            },
        ).json()
    eid = created["experiment_id"]

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    statuses: list[int] = []
    statuses_lock = threading.Lock()

    def worker(i: int) -> None:
        with TestClient(app) as local_client:
            barrier.wait()
            resp = advance_batch(
                local_client,
                eid,
                [{"subject_id": "s1", "expected_period": 1}],
                f"batch-race-{i}",
            )
            with statuses_lock:
                statuses.append(resp.status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # BEGIN IMMEDIATE 串行化：只有一批成功，其余看到 period_mismatch
    assert statuses.count(200) == 1
    assert statuses.count(409) == n_threads - 1
    with TestClient(app) as verify:
        progress = verify.get(f"/experiments/{eid}/subjects/s1/progress").json()
    assert progress["next_period"] == 2
    assert len(progress["reveals"]) == 1


# ---------- 旧库启动时自动补齐表结构 ----------

def test_startup_migrates_legacy_database(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "legacy.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    # 模拟旧版本数据库：只有 experiments / assignments 两张表
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE experiments (
                id TEXT PRIMARY KEY,
                conditions TEXT NOT NULL,
                seed INTEGER NOT NULL,
                group_size INTEGER NOT NULL,
                num_subjects INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE assignments (
                experiment_id TEXT NOT NULL REFERENCES experiments(id),
                subject_id TEXT NOT NULL,
                sequence_index INTEGER NOT NULL,
                sequence TEXT NOT NULL,
                PRIMARY KEY (experiment_id, subject_id)
            );
            """
        )
        conn.execute(
            "INSERT INTO experiments "
            "(id, conditions, seed, group_size, num_subjects, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy-1", '["A","B","C"]', 1, 6, 6, "2026-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO assignments "
            "(experiment_id, subject_id, sequence_index, sequence) "
            "VALUES (?, ?, ?, ?)",
            ("legacy-1", "s0", 0, '["A","C","B"]'),
        )

    # 应用启动（lifespan）自动补齐表结构，旧数据仍可查询与揭盲
    with TestClient(app) as migrated:
        detail = migrated.get("/experiments/legacy-1").json()
        assert detail["num_subjects"] == 6
        # 旧实验缺少 washout_seconds 列：迁移后按 0 秒处理
        assert detail["washout_seconds"] == 0
        assert_no_sequence_leak(detail)

        progress = migrated.get(
            "/experiments/legacy-1/subjects/s0/progress"
        ).json()
        assert progress["next_period"] == 1
        assert progress["num_periods"] == 3
        assert progress["washout_seconds"] == 0
        assert progress["next_available_at"] is None
        assert progress["reveals"] == []

        resp = migrated.post(
            "/experiments/legacy-1/advance",
            json={
                "subject_id": "s0",
                "expected_period": 1,
                "request_id": "boot-1",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["condition"] == "A"

        # 未完成上一周期不能推进；旧实验 washout=0，完成后立即放行
        blocked = migrated.post(
            "/experiments/legacy-1/advance",
            json={
                "subject_id": "s0",
                "expected_period": 2,
                "request_id": "boot-2",
            },
        )
        assert blocked.status_code == 409
        assert blocked.json()["detail"]["reason"] == "prior_period_not_completed"

        done = migrated.post(
            "/experiments/legacy-1/complete",
            json={"subject_id": "s0", "period": 1, "request_id": "done-1"},
        )
        assert done.status_code == 200

        second = migrated.post(
            "/experiments/legacy-1/advance",
            json={
                "subject_id": "s0",
                "expected_period": 2,
                "request_id": "boot-2",
            },
        )
        assert second.status_code == 200
        assert second.json()["condition"] == "C"


# ---------- 实验级暂停 / 恢复 ----------

def pause_experiment(
    client: TestClient,
    experiment_id: str,
    operation_id: str,
    reason: str = "suspension review",
):
    return client.post(
        f"/experiments/{experiment_id}/pause",
        json={"operation_id": operation_id, "reason": reason},
    )


def resume_experiment(
    client: TestClient,
    experiment_id: str,
    expected_version: int,
    operation_id: str | None = None,
    reason: str | None = None,
):
    body: dict = {"expected_version": expected_version}
    if operation_id is not None:
        body["operation_id"] = operation_id
    if reason is not None:
        body["reason"] = reason
    return client.post(
        f"/experiments/{experiment_id}/resume", json=body
    )


def test_new_experiment_is_active_at_version_zero(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    assert body["status"] == "active"
    assert body["state_version"] == 0

    detail = client.get(f"/experiments/{body['experiment_id']}").json()
    assert detail["status"] == "active"
    assert detail["state_version"] == 0
    entry = next(
        e for e in client.get("/experiments").json()
        if e["experiment_id"] == body["experiment_id"]
    )
    assert entry["status"] == "active"
    assert entry["state_version"] == 0
    assert_no_sequence_leak(detail)


def test_pause_and_resume_lifecycle(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]

    paused = pause_experiment(client, eid, "op-pause-1", "safety hold")
    assert paused.status_code == 200, paused.text
    event = paused.json()
    assert event["action"] == "pause"
    assert event["from_status"] == "active"
    assert event["to_status"] == "paused"
    assert event["state_version"] == 1
    assert event["operation_id"] == "op-pause-1"
    assert event["reason"] == "safety hold"
    assert "created_at" in event
    assert "id" in event

    detail = client.get(f"/experiments/{eid}").json()
    assert detail["status"] == "paused"
    assert detail["state_version"] == 1

    # 期望版本过期：compare-and-set 失败并回传服务端当前版本与状态
    stale = resume_experiment(client, eid, expected_version=0)
    assert stale.status_code == 409
    stale_detail = stale.json()["detail"]
    assert stale_detail["reason"] == "state_version_mismatch"
    assert stale_detail["state_version"] == 1
    assert stale_detail["current_status"] == "paused"
    assert stale_detail["expected_version"] == 0
    # 版本冲突不改变状态、不产生审计记录
    assert client.get(f"/experiments/{eid}").json()["status"] == "paused"

    resumed = resume_experiment(
        client, eid, expected_version=1, operation_id="op-resume-1",
        reason="review finished",
    )
    assert resumed.status_code == 200, resumed.text
    resume_event = resumed.json()
    assert resume_event["action"] == "resume"
    assert resume_event["from_status"] == "paused"
    assert resume_event["to_status"] == "active"
    assert resume_event["state_version"] == 2

    detail = client.get(f"/experiments/{eid}").json()
    assert detail["status"] == "active"
    assert detail["state_version"] == 2

    # 恢复后揭盲恢复正常
    ok = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    )
    assert ok.status_code == 200


def test_pause_is_idempotent_per_operation_id(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]
    payload = {"operation_id": "op-idem", "reason": "same reason"}

    first = client.post(f"/experiments/{eid}/pause", json=payload)
    assert first.status_code == 200
    for _ in range(3):
        replay = client.post(f"/experiments/{eid}/pause", json=payload)
        assert replay.status_code == 200
        assert replay.json() == first.json()

    # 重放不重复递增版本、不产生额外审计记录
    assert client.get(f"/experiments/{eid}").json()["state_version"] == 1
    events = client.get(f"/experiments/{eid}/state-events").json()["events"]
    assert len(events) == 1


def test_repeated_pause_with_new_operation_returns_409(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]
    assert pause_experiment(client, eid, "op-1").status_code == 200

    again = pause_experiment(client, eid, "op-2", "different hold")
    assert again.status_code == 409
    detail = again.json()["detail"]
    assert detail["reason"] == "already_paused"
    assert detail["state_version"] == 1
    assert detail["current_status"] == "paused"
    # 失败路径不写入审计记录
    events = client.get(f"/experiments/{eid}/state-events").json()["events"]
    assert [e["operation_id"] for e in events] == ["op-1"]


def test_pause_operation_id_conflicts_on_different_reason(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]
    first = pause_experiment(client, eid, "shared-op", "first reason")
    assert first.status_code == 200

    conflict = client.post(
        f"/experiments/{eid}/pause",
        json={"operation_id": "shared-op", "reason": "different reason"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["reason"] == "operation_id_conflict"
    assert conflict.json()["detail"]["operation_id"] == "shared-op"


def test_operation_id_namespace_is_shared_between_pause_and_resume(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]
    assert pause_experiment(client, eid, "shared-op").status_code == 200

    # 暂停的 operation_id 不能用于恢复
    conflict = resume_experiment(client, eid, 1, operation_id="shared-op")
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["reason"] == "operation_id_conflict"

    # 冲突不改变状态：正确版本恢复仍成功
    ok = resume_experiment(client, eid, 1, operation_id="resume-op")
    assert ok.status_code == 200

    # 恢复的 operation_id 也不能复用于再次暂停
    reused = pause_experiment(client, eid, "resume-op")
    assert reused.status_code == 409
    assert reused.json()["detail"]["reason"] == "operation_id_conflict"


def test_advance_and_batch_return_409_while_paused_without_leak(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]
    assert pause_experiment(client, eid, "hold").status_code == 200

    advance = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    )
    assert advance.status_code == 409
    detail = advance.json()["detail"]
    assert detail["reason"] == "experiment_paused"
    # 不泄露任何条件 / 后续安排
    assert "condition" not in detail
    assert "next_period" not in detail
    assert_no_sequence_leak(detail)

    batch = advance_batch(
        client,
        eid,
        [
            {"subject_id": "s1", "expected_period": 1},
            {"subject_id": "s2", "expected_period": 1},
        ],
        "b1",
    )
    assert batch.status_code == 409
    batch_detail = batch.json()["detail"]
    assert batch_detail["reason"] == "experiment_paused"
    assert "failures" not in batch_detail
    assert_no_sequence_leak(batch_detail)

    # 暂停期间零揭盲、零幂等消耗
    for subject_id in ("s1", "s2"):
        progress = client.get(
            f"/experiments/{eid}/subjects/{subject_id}/progress"
        ).json()
        assert progress["next_period"] == 1
        assert progress["reveals"] == []

    # 恢复后原 request_id / batch_id 均为首次成功（暂停 409 不消耗幂等键）
    assert resume_experiment(client, eid, 1).status_code == 200
    ok = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    )
    assert ok.status_code == 200
    ok_batch = advance_batch(
        client, eid, [{"subject_id": "s2", "expected_period": 1}], "b1"
    )
    assert ok_batch.status_code == 200


def test_successful_unblinding_replays_first_result_during_pause(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]

    first = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    )
    assert first.status_code == 200
    first_body = first.json()

    first_batch = advance_batch(
        client, eid, [{"subject_id": "s2", "expected_period": 1}], "b1"
    )
    assert first_batch.status_code == 200
    first_batch_body = first_batch.json()

    assert pause_experiment(client, eid, "hold").status_code == 200

    # 已有幂等成功的重放仍返回首次结果，不受暂停影响
    replay = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    )
    assert replay.status_code == 200
    assert replay.json() == first_body

    batch_replay = advance_batch(
        client, eid, [{"subject_id": "s2", "expected_period": 1}], "b1"
    )
    assert batch_replay.status_code == 200
    assert batch_replay.json() == first_batch_body

    # 重放不产生新的揭盲记录
    progress = client.get(f"/experiments/{eid}/subjects/s1/progress").json()
    assert len(progress["reveals"]) == 1


def test_completion_remains_allowed_while_paused(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]
    assert client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    ).status_code == 200
    assert pause_experiment(client, eid, "hold").status_code == 200

    # 完成登记不是揭盲，暂停期间仍允许
    done = complete_period(client, eid, "s1", 1, "c1")
    assert done.status_code == 200

    # 但下一周期揭盲仍被门禁拦截
    blocked = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 2, "request_id": "a2"},
    )
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["reason"] == "experiment_paused"


def test_resume_when_already_active_returns_409(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]

    # 运行中以版本 0 恢复：状态已是 active
    resp = resume_experiment(client, eid, 0)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "already_active"
    assert detail["state_version"] == 0
    assert detail["current_status"] == "active"
    assert client.get(f"/experiments/{eid}").json()["state_version"] == 0


def test_resume_with_operation_id_is_idempotent(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]
    assert pause_experiment(client, eid, "p1").status_code == 200

    payload = {"expected_version": 1, "operation_id": "r1", "reason": "ok"}
    first = client.post(f"/experiments/{eid}/resume", json=payload)
    assert first.status_code == 200
    # 恢复后（active, v2）重放仍返回首次结果，而非 409
    for _ in range(3):
        replay = client.post(f"/experiments/{eid}/resume", json=payload)
        assert replay.status_code == 200
        assert replay.json() == first.json()
    assert client.get(f"/experiments/{eid}").json()["state_version"] == 2

    # 相同 operation_id 但期望版本不同 → 冲突
    diff = client.post(
        f"/experiments/{eid}/resume",
        json={"expected_version": 2, "operation_id": "r1"},
    )
    assert diff.status_code == 409
    assert diff.json()["detail"]["reason"] == "operation_id_conflict"


def test_state_events_listed_in_reverse_chronological_order(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]

    assert pause_experiment(client, eid, "p1", "hold-1").status_code == 200
    assert resume_experiment(client, eid, 1, "r1").status_code == 200
    assert pause_experiment(client, eid, "p2", "hold-2").status_code == 200

    resp = client.get(f"/experiments/{eid}/state-events")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["experiment_id"] == eid
    events = payload["events"]
    # 最新在前（按时间倒序）
    assert [(e["action"], e["state_version"], e["reason"]) for e in events] == [
        ("pause", 3, "hold-2"),
        ("resume", 2, None),
        ("pause", 1, "hold-1"),
    ]
    assert all("created_at" in e for e in events)
    assert events[0]["id"] > events[1]["id"] > events[2]["id"]
    assert_no_sequence_leak(payload)

    # limit / offset 分页
    page1 = client.get(f"/experiments/{eid}/state-events?limit=2").json()["events"]
    assert [e["operation_id"] for e in page1] == ["p2", "r1"]
    page2 = client.get(
        f"/experiments/{eid}/state-events?limit=2&offset=2"
    ).json()["events"]
    assert [e["operation_id"] for e in page2] == ["p1"]

    # 非法分页参数
    assert client.get(
        f"/experiments/{eid}/state-events?limit=0"
    ).status_code == 422
    assert client.get(
        f"/experiments/{eid}/state-events?limit=501"
    ).status_code == 422
    assert client.get(
        f"/experiments/{eid}/state-events?offset=-1"
    ).status_code == 422

    # 没有任何状态变更的实验返回空列表
    fresh, _ = create_experiment(
        client,
        conditions=["A", "B"],
        subject_ids=["t1", "t2"],
        seed=2,
    )
    empty = client.get(
        f"/experiments/{fresh['experiment_id']}/state-events"
    ).json()
    assert empty["events"] == []


def test_pause_resume_payload_validation(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]

    assert client.post(
        f"/experiments/{eid}/pause",
        json={"operation_id": "", "reason": "x"},
    ).status_code == 422
    assert client.post(
        f"/experiments/{eid}/pause",
        json={"operation_id": "p", "reason": ""},
    ).status_code == 422
    assert client.post(
        f"/experiments/{eid}/pause", json={"operation_id": "p"}
    ).status_code == 422

    # 恢复：expected_version 必填且非负
    assert client.post(
        f"/experiments/{eid}/resume", json={}
    ).status_code == 422
    assert client.post(
        f"/experiments/{eid}/resume", json={"expected_version": -1}
    ).status_code == 422
    # 空字符串的 operation_id / reason 不合法
    assert client.post(
        f"/experiments/{eid}/resume",
        json={"expected_version": 0, "operation_id": ""},
    ).status_code == 422


def test_pause_resume_events_unknown_experiment_return_404(
    client: TestClient,
) -> None:
    assert client.post(
        "/experiments/missing/pause",
        json={"operation_id": "p", "reason": "x"},
    ).status_code == 404
    assert client.post(
        "/experiments/missing/resume", json={"expected_version": 1}
    ).status_code == 404
    assert client.get(
        "/experiments/missing/state-events"
    ).status_code == 404


def test_concurrent_resume_has_single_winner(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "conc-resume.db"))
    with TestClient(app) as setup:
        created = setup.post(
            "/experiments",
            json={
                "conditions": ["A", "B"],
                "subject_ids": ["s1", "s2"],
                "seed": 31,
            },
        ).json()
        eid = created["experiment_id"]
        assert setup.post(
            f"/experiments/{eid}/pause",
            json={"operation_id": "p1", "reason": "hold"},
        ).status_code == 200

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    statuses: list[int] = []
    statuses_lock = threading.Lock()

    def worker(i: int) -> None:
        with TestClient(app) as local_client:
            barrier.wait()
            resp = local_client.post(
                f"/experiments/{eid}/resume",
                json={"expected_version": 1, "operation_id": f"r-{i}"},
            )
            with statuses_lock:
                statuses.append(resp.status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # BEGIN IMMEDIATE + 版本 compare-and-set：只有一次恢复成功
    assert statuses.count(200) == 1
    assert statuses.count(409) == n_threads - 1

    with TestClient(app) as verify:
        detail = verify.get(f"/experiments/{eid}").json()
        assert detail["status"] == "active"
        assert detail["state_version"] == 2
        events = verify.get(f"/experiments/{eid}/state-events").json()["events"]
    # 只有 pause + 一次 resume 两条审计记录
    assert [(e["action"], e["state_version"]) for e in events] == [
        ("resume", 2),
        ("pause", 1),
    ]


def test_legacy_database_defaults_to_active_version_zero(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "legacy-pause.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE experiments (
                id TEXT PRIMARY KEY,
                conditions TEXT NOT NULL,
                seed INTEGER NOT NULL,
                group_size INTEGER NOT NULL,
                num_subjects INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE assignments (
                experiment_id TEXT NOT NULL REFERENCES experiments(id),
                subject_id TEXT NOT NULL,
                sequence_index INTEGER NOT NULL,
                sequence TEXT NOT NULL,
                PRIMARY KEY (experiment_id, subject_id)
            );
            """
        )
        conn.execute(
            "INSERT INTO experiments "
            "(id, conditions, seed, group_size, num_subjects, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy-pause", '["A","B"]', 1, 2, 2, "2026-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO assignments "
            "(experiment_id, subject_id, sequence_index, sequence) "
            "VALUES (?, ?, ?, ?)",
            ("legacy-pause", "s1", 0, '["A","B"]'),
        )
        conn.execute(
            "INSERT INTO assignments "
            "(experiment_id, subject_id, sequence_index, sequence) "
            "VALUES (?, ?, ?, ?)",
            ("legacy-pause", "s2", 1, '["B","A"]'),
        )

    with TestClient(app) as migrated:
        detail = migrated.get("/experiments/legacy-pause").json()
        # 旧实验迁移后按“运行中、版本 0”处理
        assert detail["status"] == "active"
        assert detail["state_version"] == 0

        # 暂停 / 恢复完整链路在迁移库上可用
        assert migrated.post(
            "/experiments/legacy-pause/pause",
            json={"operation_id": "p1", "reason": "hold legacy"},
        ).status_code == 200
        blocked = migrated.post(
            "/experiments/legacy-pause/advance",
            json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
        )
        assert blocked.status_code == 409
        assert blocked.json()["detail"]["reason"] == "experiment_paused"

        assert migrated.post(
            "/experiments/legacy-pause/resume",
            json={"expected_version": 1},
        ).status_code == 200
        assert migrated.post(
            "/experiments/legacy-pause/advance",
            json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
        ).status_code == 200


# ---------- 批量状态切换（跨实验暂停 / 恢复） ----------

def state_batch(
    client: TestClient,
    batch_operation_id: str,
    operations: list[dict],
):
    return client.post(
        "/experiments/state-batch",
        json={
            "batch_operation_id": batch_operation_id,
            "operations": operations,
        },
    )


def transition_op(
    experiment_id: str,
    action: str,
    expected_version: int,
    operation_id: str,
    reason: str | None = None,
) -> dict:
    op = {
        "experiment_id": experiment_id,
        "action": action,
        "expected_version": expected_version,
        "operation_id": operation_id,
    }
    if reason is not None:
        op["reason"] = reason
    return op


def create_two_experiments(client: TestClient) -> tuple[str, str]:
    first, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    second, _ = create_experiment(
        client,
        conditions=["A", "B", "C"],
        subject_ids=[f"t{i}" for i in range(6)],
        seed=2,
    )
    return first["experiment_id"], second["experiment_id"]


def test_state_batch_pauses_multiple_experiments_in_one_transaction(
    client: TestClient,
) -> None:
    eid1, eid2 = create_two_experiments(client)

    operations = [
        transition_op(eid2, "pause", 0, "b-op-2", "hold-2"),
        transition_op(eid1, "pause", 0, "b-op-1", "hold-1"),
    ]
    resp = state_batch(client, "sb-1", operations)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["batch_operation_id"] == "sb-1"
    # 结果严格按输入顺序返回
    assert [r["experiment_id"] for r in body["results"]] == [eid2, eid1]
    assert [r["operation_id"] for r in body["results"]] == ["b-op-2", "b-op-1"]
    for result, reason in zip(body["results"], ["hold-2", "hold-1"]):
        assert result["action"] == "pause"
        assert result["from_status"] == "active"
        assert result["to_status"] == "paused"
        assert result["state_version"] == 1
        assert result["reason"] == reason
        assert "id" in result
        assert "created_at" in result
    assert_no_sequence_leak(body)

    # 两个实验都已暂停，审计各一条，揭盲被门禁拦截
    for eid, subject_id in ((eid1, "s1"), (eid2, "t0")):
        detail = client.get(f"/experiments/{eid}").json()
        assert detail["status"] == "paused"
        assert detail["state_version"] == 1
        events = client.get(f"/experiments/{eid}/state-events").json()["events"]
        assert len(events) == 1
        blocked = client.post(
            f"/experiments/{eid}/advance",
            json={
                "subject_id": subject_id,
                "expected_period": 1,
                "request_id": "a1",
            },
        )
        assert blocked.status_code == 409
        assert blocked.json()["detail"]["reason"] == "experiment_paused"


def test_state_batch_mixed_pause_and_resume(client: TestClient) -> None:
    eid1, eid2 = create_two_experiments(client)
    # eid2 先暂停到版本 1
    assert pause_experiment(client, eid2, "pre-pause", "hold").status_code == 200

    resp = state_batch(
        client,
        "sb-mixed",
        [
            transition_op(eid1, "pause", 0, "m-op-1"),
            transition_op(eid2, "resume", 1, "m-op-2", "review ok"),
        ],
    )
    assert resp.status_code == 200, resp.text
    results = {r["experiment_id"]: r for r in resp.json()["results"]}
    assert results[eid1]["action"] == "pause"
    assert results[eid1]["to_status"] == "paused"
    assert results[eid1]["state_version"] == 1
    assert results[eid1]["reason"] is None
    assert results[eid2]["action"] == "resume"
    assert results[eid2]["from_status"] == "paused"
    assert results[eid2]["to_status"] == "active"
    assert results[eid2]["state_version"] == 2
    assert results[eid2]["reason"] == "review ok"

    assert client.get(f"/experiments/{eid1}").json()["status"] == "paused"
    detail2 = client.get(f"/experiments/{eid2}").json()
    assert detail2["status"] == "active"
    assert detail2["state_version"] == 2
    # 审计按时间倒序：批量恢复在前，此前的暂停在后
    events = client.get(f"/experiments/{eid2}/state-events").json()["events"]
    assert [(e["action"], e["state_version"]) for e in events] == [
        ("resume", 2),
        ("pause", 1),
    ]

    # 恢复后 eid2 揭盲恢复正常
    ok = client.post(
        f"/experiments/{eid2}/advance",
        json={"subject_id": "t0", "expected_period": 1, "request_id": "a1"},
    )
    assert ok.status_code == 200


def test_state_batch_any_conflict_rolls_back_entire_batch(
    client: TestClient,
) -> None:
    eid1, eid2 = create_two_experiments(client)
    # eid2 已处于暂停（版本 1），批量中再次暂停必然冲突
    assert pause_experiment(client, eid2, "pre", "hold").status_code == 200

    resp = state_batch(
        client,
        "sb-rollback",
        [
            transition_op(eid1, "pause", 0, "rb-1"),
            transition_op(eid2, "pause", 1, "rb-2"),
        ],
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "batch_operation_rejected"
    assert detail["batch_operation_id"] == "sb-rollback"
    assert detail["failures"] == [
        {
            "experiment_id": eid2,
            "operation_id": "rb-2",
            "reason": "already_paused",
            "current_status": "paused",
            "state_version": 1,
            "expected_version": 1,
        }
    ]
    assert_no_sequence_leak(detail)

    # 整批回滚：eid1 未暂停、版本未变、无审计记录
    detail1 = client.get(f"/experiments/{eid1}").json()
    assert detail1["status"] == "active"
    assert detail1["state_version"] == 0
    assert client.get(f"/experiments/{eid1}/state-events").json()["events"] == []
    # eid2 保持暂停，仍只有最初一条审计
    events2 = client.get(f"/experiments/{eid2}/state-events").json()["events"]
    assert [e["operation_id"] for e in events2] == ["pre"]

    # 失败不消耗 batch_operation_id：修正操作列表后同键首次成功
    ok = state_batch(
        client, "sb-rollback", [transition_op(eid1, "pause", 0, "rb-1")]
    )
    assert ok.status_code == 200, ok.text
    assert client.get(f"/experiments/{eid1}").json()["status"] == "paused"


def test_state_batch_version_mismatch_rolls_back(client: TestClient) -> None:
    eid1, eid2 = create_two_experiments(client)

    resp = state_batch(
        client,
        "sb-version",
        [
            transition_op(eid1, "pause", 0, "v-1"),
            transition_op(eid2, "pause", 5, "v-2"),  # 期望版本过期
        ],
    )
    assert resp.status_code == 409
    (failure,) = resp.json()["detail"]["failures"]
    assert failure["experiment_id"] == eid2
    assert failure["operation_id"] == "v-2"
    assert failure["reason"] == "state_version_mismatch"
    assert failure["current_status"] == "active"
    assert failure["state_version"] == 0
    assert failure["expected_version"] == 5

    # 零写入：两个实验都保持 active / 版本 0
    for eid in (eid1, eid2):
        detail = client.get(f"/experiments/{eid}").json()
        assert detail["status"] == "active"
        assert detail["state_version"] == 0
        assert client.get(f"/experiments/{eid}/state-events").json()["events"] == []


def test_state_batch_collects_all_failures_in_input_order(
    client: TestClient,
) -> None:
    eid1, _ = create_two_experiments(client)

    resp = state_batch(
        client,
        "sb-failures",
        [
            transition_op("missing-exp", "pause", 0, "f-1"),
            transition_op(eid1, "resume", 0, "f-2"),  # 运行中不能恢复
        ],
    )
    assert resp.status_code == 409
    failures = resp.json()["detail"]["failures"]
    assert [f["experiment_id"] for f in failures] == ["missing-exp", eid1]
    assert [f["reason"] for f in failures] == [
        "experiment_not_found",
        "already_active",
    ]
    assert failures[0] == {
        "experiment_id": "missing-exp",
        "operation_id": "f-1",
        "reason": "experiment_not_found",
    }
    assert failures[1]["current_status"] == "active"
    assert failures[1]["state_version"] == 0

    # 零写入
    assert client.get(f"/experiments/{eid1}").json()["state_version"] == 0
    assert client.get(f"/experiments/{eid1}/state-events").json()["events"] == []


def test_state_batch_same_batch_operation_id_replays_first_result(
    client: TestClient,
) -> None:
    eid1, eid2 = create_two_experiments(client)
    operations = [
        transition_op(eid1, "pause", 0, "idem-op-1", "hold-1"),
        transition_op(eid2, "pause", 0, "idem-op-2", "hold-2"),
    ]

    first = state_batch(client, "sb-idem", operations)
    assert first.status_code == 200
    first_body = first.json()

    for _ in range(3):
        replay = state_batch(client, "sb-idem", operations)
        assert replay.status_code == 200
        assert replay.json() == first_body

    # 重放不重复递增版本、不重复写审计
    for eid in (eid1, eid2):
        assert client.get(f"/experiments/{eid}").json()["state_version"] == 1
        assert len(client.get(f"/experiments/{eid}/state-events").json()["events"]) == 1

    # 同一 batch_operation_id 绑定不同操作列表 → 409
    subset = state_batch(client, "sb-idem", operations[:1])
    assert subset.status_code == 409
    detail = subset.json()["detail"]
    assert detail["reason"] == "batch_operation_id_conflict"
    assert detail["batch_operation_id"] == "sb-idem"

    # 单项字段不同（如期望版本）也视为不同请求
    changed = [operations[0], transition_op(eid2, "pause", 1, "idem-op-2", "hold-2")]
    assert state_batch(client, "sb-idem", changed).status_code == 409

    # 冲突路径零写入：状态与审计保持不变
    for eid in (eid1, eid2):
        assert client.get(f"/experiments/{eid}").json()["state_version"] == 1
        assert len(client.get(f"/experiments/{eid}/state-events").json()["events"]) == 1


def test_state_batch_item_operation_id_conflict_rolls_back(
    client: TestClient,
) -> None:
    eid1, eid2 = create_two_experiments(client)
    # 单条暂停已占用 eid1 的 used-op
    assert pause_experiment(client, eid1, "used-op", "first hold").status_code == 200

    resp = state_batch(
        client,
        "sb-opconflict",
        [
            transition_op(eid2, "pause", 0, "fresh-op"),
            # operation_id 已被占用且原因不同 → 冲突
            transition_op(eid1, "pause", 0, "used-op", "different reason"),
        ],
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["failures"] == [
        {
            "experiment_id": eid1,
            "operation_id": "used-op",
            "reason": "operation_id_conflict",
        }
    ]

    # 整批回滚：eid2 未暂停、无审计
    detail2 = client.get(f"/experiments/{eid2}").json()
    assert detail2["status"] == "active"
    assert detail2["state_version"] == 0
    assert client.get(f"/experiments/{eid2}/state-events").json()["events"] == []


def test_state_batch_replays_matching_existing_operations(
    client: TestClient,
) -> None:
    eid1, eid2 = create_two_experiments(client)
    paused = pause_experiment(client, eid1, "shared-op", "same reason")
    assert paused.status_code == 200
    first_event = paused.json()

    # 新批次引用已存在且参数完全一致的操作：该项按幂等重放，其余项正常执行
    operations = [
        transition_op(eid1, "pause", 0, "shared-op", "same reason"),
        transition_op(eid2, "pause", 0, "new-op"),
    ]
    resp = state_batch(client, "sb-replay", operations)
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert results[0] == first_event  # 重放首次审计事件，不重复执行
    assert results[1]["operation_id"] == "new-op"
    assert results[1]["to_status"] == "paused"

    # 重放项不重复写审计、不重复递增版本；新项正常落库
    assert client.get(f"/experiments/{eid1}").json()["state_version"] == 1
    assert len(client.get(f"/experiments/{eid1}/state-events").json()["events"]) == 1
    assert client.get(f"/experiments/{eid2}").json()["status"] == "paused"

    # 该批次自身也可整体幂等重放
    replay = state_batch(client, "sb-replay", operations)
    assert replay.status_code == 200
    assert replay.json() == resp.json()


def test_state_batch_payload_validation(client: TestClient) -> None:
    eid1, _ = create_two_experiments(client)
    op = transition_op(eid1, "pause", 0, "v-op")

    # 空操作列表 / 空 batch_operation_id
    assert state_batch(client, "sb-v", []).status_code == 422
    assert state_batch(client, "", [op]).status_code == 422
    # 同一实验在同一批中重复出现
    assert state_batch(
        client, "sb-v", [op, transition_op(eid1, "resume", 0, "v-op-2")]
    ).status_code == 422
    # 非法动作 / 负版本 / 空 operation_id / 空 reason / 空 experiment_id
    assert state_batch(
        client, "sb-v", [transition_op(eid1, "stop", 0, "v-op")]
    ).status_code == 422
    assert state_batch(
        client, "sb-v", [transition_op(eid1, "pause", -1, "v-op")]
    ).status_code == 422
    assert state_batch(
        client, "sb-v", [transition_op(eid1, "pause", 0, "")]
    ).status_code == 422
    assert state_batch(
        client, "sb-v", [transition_op(eid1, "pause", 0, "v-op", "")]
    ).status_code == 422
    assert state_batch(
        client, "sb-v", [transition_op("", "pause", 0, "v-op")]
    ).status_code == 422
    # 超过 100 项（校验在入库前发生）
    too_many = [
        transition_op(f"e{i}", "pause", 0, f"v-op-{i}") for i in range(101)
    ]
    assert state_batch(client, "sb-v", too_many).status_code == 422


def test_concurrent_state_batches_have_single_winner(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "conc-state-batch.db"))
    with TestClient(app) as setup:
        created = [
            setup.post(
                "/experiments",
                json={
                    "conditions": ["A", "B"],
                    "subject_ids": [f"s{k}", f"t{k}"],
                    "seed": 40 + k,
                },
            ).json()
            for k in range(2)
        ]
    eids = [c["experiment_id"] for c in created]

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    statuses: list[int] = []
    statuses_lock = threading.Lock()

    def worker(i: int) -> None:
        with TestClient(app) as local_client:
            barrier.wait()
            resp = state_batch(
                local_client,
                f"sb-race-{i}",
                [
                    transition_op(eid, "pause", 0, f"race-{i}-{k}")
                    for k, eid in enumerate(eids)
                ],
            )
            with statuses_lock:
                statuses.append(resp.status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # BEGIN IMMEDIATE 串行化：只有一批通过全部校验，其余看到版本已变而 409
    assert statuses.count(200) == 1
    assert statuses.count(409) == n_threads - 1

    with TestClient(app) as verify:
        for eid in eids:
            detail = verify.get(f"/experiments/{eid}").json()
            assert detail["status"] == "paused"
            assert detail["state_version"] == 1
            events = verify.get(f"/experiments/{eid}/state-events").json()["events"]
            assert len(events) == 1


def test_legacy_database_supports_state_batch_after_migration(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "legacy-state-batch.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    # 模拟旧版本数据库：只有 experiments / assignments 两张表
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE experiments (
                id TEXT PRIMARY KEY,
                conditions TEXT NOT NULL,
                seed INTEGER NOT NULL,
                group_size INTEGER NOT NULL,
                num_subjects INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE assignments (
                experiment_id TEXT NOT NULL REFERENCES experiments(id),
                subject_id TEXT NOT NULL,
                sequence_index INTEGER NOT NULL,
                sequence TEXT NOT NULL,
                PRIMARY KEY (experiment_id, subject_id)
            );
            """
        )
        conn.execute(
            "INSERT INTO experiments "
            "(id, conditions, seed, group_size, num_subjects, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy-sb", '["A","B"]', 1, 2, 2, "2026-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO assignments "
            "(experiment_id, subject_id, sequence_index, sequence) "
            "VALUES (?, ?, ?, ?)",
            ("legacy-sb", "s1", 0, '["A","B"]'),
        )
        conn.execute(
            "INSERT INTO assignments "
            "(experiment_id, subject_id, sequence_index, sequence) "
            "VALUES (?, ?, ?, ?)",
            ("legacy-sb", "s2", 1, '["B","A"]'),
        )

    # 启动自动迁移后，批量状态切换在旧库上可用（旧实验按 active / 版本 0）
    with TestClient(app) as migrated:
        operations = [transition_op("legacy-sb", "pause", 0, "lg-1", "hold")]
        resp = state_batch(migrated, "sb-legacy", operations)
        assert resp.status_code == 200, resp.text
        assert resp.json()["results"][0]["state_version"] == 1
        detail = migrated.get("/experiments/legacy-sb").json()
        assert detail["status"] == "paused"
        assert detail["state_version"] == 1

        # 幂等重放返回首次结果
        replay = state_batch(migrated, "sb-legacy", operations)
        assert replay.status_code == 200
        assert replay.json() == resp.json()

        # 批量恢复同样可用
        resumed = state_batch(
            migrated,
            "sb-legacy-2",
            [transition_op("legacy-sb", "resume", 1, "lg-2")],
        )
        assert resumed.status_code == 200
        assert migrated.get("/experiments/legacy-sb").json()["status"] == "active"


# ---------- 实验级紧急揭盲（双人审批 / 15 分钟窗口） ----------

EMERGENCY_REASON = "subject reported serious adverse event"


def emergency_unblind(
    client: TestClient,
    experiment_id: str,
    *,
    subject_id: str,
    approver_id: str,
    operation_id: str,
    reason: str = EMERGENCY_REASON,
):
    return client.post(
        f"/experiments/{experiment_id}/emergency-unblind",
        json={
            "subject_id": subject_id,
            "approver_id": approver_id,
            "operation_id": operation_id,
            "reason": reason,
        },
    )


def emergency_events(
    client: TestClient, experiment_id: str, query: str = ""
):
    return client.get(
        f"/experiments/{experiment_id}/emergency-unblind-events{query}"
    )


def test_emergency_unblind_requires_two_distinct_approvers_within_window(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=5
    )
    eid = body["experiment_id"]

    # 首位审批人：开启 15 分钟授权窗口，不返回任何条件
    first = emergency_unblind(
        client, eid, subject_id="s1", approver_id="alice", operation_id="em-1"
    )
    assert first.status_code == 200, first.text
    pending = first.json()
    assert pending["status"] == "pending"
    assert pending["experiment_id"] == eid
    assert pending["subject_id"] == "s1"
    assert pending["operation_id"] == "em-1"
    assert pending["first_approver_id"] == "alice"
    assert pending["second_approver_id"] is None
    assert pending["reason"] == EMERGENCY_REASON
    assert pending["sequence"] is None
    assert pending["unblinded_at"] is None
    assert "window_opened_at" in pending and "window_expires_at" in pending
    opened_at = datetime.fromisoformat(pending["window_opened_at"])
    expires_at = datetime.fromisoformat(pending["window_expires_at"])
    assert expires_at - opened_at == timedelta(minutes=15)

    # 窗口内第二位不同审批人：一次性返回完整条件序列
    second = emergency_unblind(
        client, eid, subject_id="s1", approver_id="bob", operation_id="em-1"
    )
    assert second.status_code == 200, second.text
    completed = second.json()
    assert completed["status"] == "completed"
    assert completed["first_approver_id"] == "alice"
    assert completed["second_approver_id"] == "bob"
    assert completed["unblinded_at"] is not None
    assert len(completed["sequence"]) == 2
    assert set(completed["sequence"]) == {"A", "B"}
    assert completed["window_expires_at"] == pending["window_expires_at"]

    # 紧急揭盲不触碰常规揭盲进度 / 不写 unblindings
    progress = client.get(f"/experiments/{eid}/subjects/s1/progress").json()
    assert progress["next_period"] == 1
    assert progress["reveals"] == []

    # 紧急揭盲给出的序列与逐周期受控揭盲看到的条件完全一致
    regular_first = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    ).json()
    assert regular_first["condition"] == completed["sequence"][0]
    assert complete_period(client, eid, "s1", 1, "c1").status_code == 200
    regular_second = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 2, "request_id": "a2"},
    ).json()
    assert regular_second["condition"] == completed["sequence"][1]

    # 不同受试者的紧急揭盲操作互相独立
    other = emergency_unblind(
        client, eid, subject_id="s2", approver_id="carol", operation_id="em-2"
    )
    assert other.status_code == 200
    assert other.json()["status"] == "pending"
    other_done = emergency_unblind(
        client, eid, subject_id="s2", approver_id="dave", operation_id="em-2"
    )
    assert other_done.status_code == 200
    other_sequence = other_done.json()["sequence"]
    assert other_done.json()["status"] == "completed"
    assert len(other_sequence) == 2
    assert set(other_sequence) == {"A", "B"}


def test_emergency_same_approver_repeated_returns_409_and_does_not_leak(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=5
    )
    eid = body["experiment_id"]

    first = emergency_unblind(
        client, eid, subject_id="s1", approver_id="alice", operation_id="em-dup"
    )
    assert first.status_code == 200
    assert first.json()["sequence"] is None

    # 同一审批人重复（无论视为开窗重放还是自我二次批准）：409，零泄露
    repeated = emergency_unblind(
        client, eid, subject_id="s1", approver_id="alice", operation_id="em-dup"
    )
    assert repeated.status_code == 409
    detail = repeated.json()["detail"]
    assert detail["reason"] == "duplicate_approver"
    assert detail["operation_id"] == "em-dup"
    assert "sequence" not in detail
    assert "condition" not in detail
    assert_no_sequence_leak(detail)

    # 重复尝试不多写审计：仍只有一条 requested
    events = emergency_events(client, eid).json()["events"]
    assert [(e["event"], e["approver_id"]) for e in events] == [
        ("requested", "alice")
    ]

    # 拒绝自我批准不破坏窗口：不同的第二位审批人随后仍可完成揭盲
    second = emergency_unblind(
        client, eid, subject_id="s1", approver_id="bob", operation_id="em-dup"
    )
    assert second.status_code == 200
    assert second.json()["status"] == "completed"
    assert len(second.json()["sequence"]) == 2


def test_emergency_window_expiry_uses_server_time(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "emergency-window.db"))
    t0 = datetime(2026, 9, 22, 8, 0, 0, tzinfo=timezone.utc)

    class Clock:
        def __init__(self) -> None:
            self.now = t0

        def __call__(self) -> datetime:
            return self.now

    clock = Clock()
    monkeypatch.setattr(db, "utc_now", clock)

    with TestClient(app) as local:
        created = local.post(
            "/experiments",
            json={
                "conditions": ["A", "B"],
                "subject_ids": ["s1", "s2"],
                "seed": 1,
            },
        ).json()
        eid = created["experiment_id"]

        first = emergency_unblind(
            local, eid, subject_id="s1", approver_id="alice", operation_id="win-1"
        )
        assert first.status_code == 200
        window_expires_at = first.json()["window_expires_at"]

        # 窗口内（差 1 秒）第二位审批人完成
        clock.now = t0 + timedelta(minutes=14, seconds=59)
        ok = emergency_unblind(
            local, eid, subject_id="s1", approver_id="bob", operation_id="win-1"
        )
        assert ok.status_code == 200
        assert ok.json()["status"] == "completed"

        # 已完成的操作：窗口时间之后重放仍返回首次完整结果（幂等重放）
        clock.now = t0 + timedelta(hours=1)
        replay = emergency_unblind(
            local, eid, subject_id="s1", approver_id="bob", operation_id="win-1"
        )
        assert replay.status_code == 200
        assert replay.json()["sequence"] == ok.json()["sequence"]

        # 新操作在 t0 开窗：恰好在第 15 分钟边界判定为过期（now >= expires）
        clock.now = t0
        second_open = emergency_unblind(
            local, eid, subject_id="s2", approver_id="alice", operation_id="win-2"
        )
        assert second_open.status_code == 200
        assert second_open.json()["window_expires_at"] == (
            t0 + timedelta(minutes=15)
        ).isoformat()
        clock.now = t0 + timedelta(minutes=15)
        expired = emergency_unblind(
            local, eid, subject_id="s2", approver_id="bob", operation_id="win-2"
        )
        assert expired.status_code == 409
        detail = expired.json()["detail"]
        assert detail["reason"] == "window_expired"
        assert detail["window_expires_at"] == second_open.json()["window_expires_at"]
        assert "sequence" not in detail

        # 过期后连首位审批人本人重放也不能泄露 / 续期
        first_replay = emergency_unblind(
            local, eid, subject_id="s2", approver_id="alice", operation_id="win-2"
        )
        assert first_replay.status_code == 409
        assert first_replay.json()["detail"]["reason"] == "window_expired"

        # 过期操作不多写审计：win-2 只有 requested；状态保持 pending
        events = emergency_events(local, eid).json()["events"]
        win2 = [e for e in events if e["operation_id"] == "win-2"]
        assert [e["event"] for e in win2] == ["requested"]

        # 过期后必须以新的 operation_id 重新走双人审批
        clock.now = t0 + timedelta(minutes=16)
        reopened = emergency_unblind(
            local, eid, subject_id="s2", approver_id="alice", operation_id="win-3"
        )
        assert reopened.status_code == 200
        assert reopened.json()["status"] == "pending"
        approved = emergency_unblind(
            local, eid, subject_id="s2", approver_id="bob", operation_id="win-3"
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "completed"
        assert len(approved.json()["sequence"]) == 2


def test_emergency_unknown_experiment_or_subject_returns_404_and_opens_nothing(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=6
    )
    eid = body["experiment_id"]
    payload_args = dict(
        subject_id="ghost", approver_id="alice", operation_id="em-x"
    )

    missing_exp = emergency_unblind(client, "missing", **payload_args)
    assert missing_exp.status_code == 404
    missing_subject = emergency_unblind(client, eid, **payload_args)
    assert missing_subject.status_code == 404
    detail = missing_subject.json()["detail"]
    assert detail["subject_id"] == "ghost"
    assert "sequence" not in detail

    # 404 不开窗、不写审计
    assert emergency_events(client, eid).json()["events"] == []
    assert emergency_events(client, "missing").status_code == 404

    # 失败未消耗 operation_id：受试者修正后同键首次成功开窗
    ok = emergency_unblind(
        client, eid, subject_id="s1", approver_id="alice", operation_id="em-x"
    )
    assert ok.status_code == 200
    assert ok.json()["status"] == "pending"


def test_emergency_operation_id_conflict_different_subject_or_reason(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=4
    )
    eid = body["experiment_id"]
    assert (
        emergency_unblind(
            client, eid, subject_id="s1", approver_id="alice", operation_id="shared"
        ).status_code
        == 200
    )

    # 同一 operation_id 绑定不同受试者
    other_subject = emergency_unblind(
        client, eid, subject_id="s2", approver_id="alice", operation_id="shared"
    )
    assert other_subject.status_code == 409
    detail = other_subject.json()["detail"]
    assert detail["reason"] == "operation_id_conflict"
    assert detail["operation_id"] == "shared"
    assert "sequence" not in detail

    # 同一 operation_id 绑定不同原因（首位审批人本人也不行）
    other_reason = emergency_unblind(
        client,
        eid,
        subject_id="s1",
        approver_id="alice",
        operation_id="shared",
        reason="a totally different reason",
    )
    assert other_reason.status_code == 409
    assert other_reason.json()["detail"]["reason"] == "operation_id_conflict"

    # 冲突路径零泄露：s2 未开窗
    events = emergency_events(client, eid).json()["events"]
    assert [(e["subject_id"], e["event"]) for e in events] == [("s1", "requested")]


def test_emergency_completed_replays_to_approvers_and_denies_third_party(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B", "C"], subject_ids=[f"s{i}" for i in range(6)],
        seed=9,
    )
    eid = body["experiment_id"]
    emergency_unblind(
        client, eid, subject_id="s0", approver_id="alice", operation_id="em-idem"
    )
    completed = emergency_unblind(
        client, eid, subject_id="s0", approver_id="bob", operation_id="em-idem"
    )
    assert completed.status_code == 200
    completed_body = completed.json()
    assert len(completed_body["sequence"]) == 3

    # 两位原审批人重放均返回首次完整结果，且只产生一次结果
    for approver in ("bob", "alice", "bob"):
        replay = emergency_unblind(
            client,
            eid,
            subject_id="s0",
            approver_id=approver,
            operation_id="em-idem",
        )
        assert replay.status_code == 200
        assert replay.json() == completed_body

    events = emergency_events(client, eid).json()["events"]
    assert [e["event"] for e in events] == ["approved", "requested"]

    # 第三位审批人（或任何其他人）完成后再请求：409，响应不含序列
    intruder = emergency_unblind(
        client, eid, subject_id="s0", approver_id="carol", operation_id="em-idem"
    )
    assert intruder.status_code == 409
    intruder_detail = intruder.json()["detail"]
    assert intruder_detail["reason"] == "already_completed"
    assert "sequence" not in intruder_detail
    assert "condition" not in intruder_detail

    # 不额外写审计
    assert len(emergency_events(client, eid).json()["events"]) == 2


def test_emergency_audit_events_paginated_without_conditions(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]

    # 三个紧急揭盲操作，其中两个完成
    emergency_unblind(client, eid, subject_id="s1", approver_id="a1",
                     operation_id="op-1")
    emergency_unblind(client, eid, subject_id="s1", approver_id="b1",
                     operation_id="op-1")
    emergency_unblind(client, eid, subject_id="s2", approver_id="a2",
                     operation_id="op-2")
    emergency_unblind(client, eid, subject_id="s2", approver_id="b2",
                     operation_id="op-2")
    emergency_unblind(client, eid, subject_id="s1", approver_id="a3",
                     operation_id="op-3")

    resp = emergency_events(client, eid)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["experiment_id"] == eid
    events = payload["events"]
    # 时间倒序：最新的 approved 在前，最早的 requested 在后
    assert [(e["operation_id"], e["event"]) for e in events] == [
        ("op-3", "requested"),
        ("op-2", "approved"),
        ("op-2", "requested"),
        ("op-1", "approved"),
        ("op-1", "requested"),
    ]
    assert events[0]["id"] > events[-1]["id"]

    approved = next(
        e for e in events if e["event"] == "approved" and e["operation_id"] == "op-1"
    )
    assert approved["approver_id"] == "b1"
    assert approved["subject_id"] == "s1"
    assert approved["reason"] == EMERGENCY_REASON
    assert approved["window_opened_at"] <= approved["window_expires_at"]
    assert "created_at" in approved
    # 审计记录任何层级都不含条件 / 序列
    assert_no_sequence_leak(payload)
    assert all("condition" not in event for event in events)

    # limit / offset 分页
    page1 = emergency_events(client, eid, "?limit=2").json()["events"]
    assert [(e["operation_id"], e["event"]) for e in page1] == [
        ("op-3", "requested"),
        ("op-2", "approved"),
    ]
    page2 = emergency_events(client, eid, "?limit=2&offset=2").json()["events"]
    assert [(e["operation_id"], e["event"]) for e in page2] == [
        ("op-2", "requested"),
        ("op-1", "approved"),
    ]

    # 非法分页参数
    assert emergency_events(client, eid, "?limit=0").status_code == 422
    assert emergency_events(client, eid, "?limit=501").status_code == 422
    assert emergency_events(client, eid, "?offset=-1").status_code == 422

    # 从未紧急揭盲的实验返回空列表
    fresh, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["t1", "t2"], seed=2
    )
    empty = emergency_events(client, fresh["experiment_id"]).json()
    assert empty == {"experiment_id": fresh["experiment_id"], "events": []}


def test_emergency_payload_validation(client: TestClient) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=2
    )
    eid = body["experiment_id"]
    base = {
        "subject_id": "s1",
        "approver_id": "alice",
        "operation_id": "op",
        "reason": "SAE",
    }
    for missing in ("subject_id", "approver_id", "operation_id", "reason"):
        payload = dict(base)
        payload[missing] = ""
        resp = client.post(
            f"/experiments/{eid}/emergency-unblind", json=payload
        )
        assert resp.status_code == 422, missing

    resp = client.post(
        f"/experiments/{eid}/emergency-unblind",
        json={"subject_id": "s1", "approver_id": "alice", "operation_id": "op"},
    )
    assert resp.status_code == 422


def test_emergency_unblind_allowed_while_paused_without_affecting_progress(
    client: TestClient,
) -> None:
    body, _ = create_experiment(
        client, conditions=["A", "B"], subject_ids=["s1", "s2"], seed=1
    )
    eid = body["experiment_id"]
    assert pause_experiment(client, eid, "hold").status_code == 200

    # 紧急揭盲是例外通道：暂停期间仍可双人审批完成
    first = emergency_unblind(
        client, eid, subject_id="s1", approver_id="alice", operation_id="em-paused"
    )
    assert first.status_code == 200
    assert first.json()["status"] == "pending"
    second = emergency_unblind(
        client, eid, subject_id="s1", approver_id="bob", operation_id="em-paused"
    )
    assert second.status_code == 200
    assert second.json()["status"] == "completed"
    sequence = second.json()["sequence"]

    # 常规受控揭盲仍被暂停门禁拦截，且紧急揭盲未改动进度
    blocked = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    )
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["reason"] == "experiment_paused"
    progress = client.get(f"/experiments/{eid}/subjects/s1/progress").json()
    assert progress["next_period"] == 1
    assert progress["reveals"] == []

    # 恢复后常规揭盲看到的条件与紧急揭盲给出的序列一致
    assert resume_experiment(client, eid, 1).status_code == 200
    period1 = client.post(
        f"/experiments/{eid}/advance",
        json={"subject_id": "s1", "expected_period": 1, "request_id": "a1"},
    ).json()
    assert period1["condition"] == sequence[0]


def test_concurrent_emergency_second_approvals_have_single_winner(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "DATABASE_PATH", str(tmp_path / "conc-emergency.db")
    )
    with TestClient(app) as setup:
        created = setup.post(
            "/experiments",
            json={
                "conditions": ["A", "B"],
                "subject_ids": ["s1", "s2"],
                "seed": 41,
            },
        ).json()
        eid = created["experiment_id"]
        first = emergency_unblind(
            setup, eid, subject_id="s1", approver_id="alice",
            operation_id="em-race",
        )
        assert first.status_code == 200
        assert first.json()["status"] == "pending"

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    responses: list[dict] = []
    responses_lock = threading.Lock()

    def worker(i: int) -> None:
        with TestClient(app) as local_client:
            barrier.wait()
            resp = emergency_unblind(
                local_client,
                eid,
                subject_id="s1",
                approver_id=f"approver-{i}",
                operation_id="em-race",
            )
            with responses_lock:
                responses.append(
                    {"status": resp.status_code, "body": resp.json()}
                )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    winners = [r for r in responses if r["status"] == 200]
    losers = [r for r in responses if r["status"] == 409]
    assert len(winners) == 1
    assert len(losers) == n_threads - 1
    winner = winners[0]["body"]
    assert winner["status"] == "completed"
    assert winner["second_approver_id"] in {
        f"approver-{i}" for i in range(n_threads)
    }
    assert len(winner["sequence"]) == 2
    # 其余并发第二审批都被拒（完成后非原审批人），且响应无序列
    assert {r["body"]["detail"]["reason"] for r in losers} == {
        "already_completed"
    }
    assert all("sequence" not in r["body"]["detail"] for r in losers)

    with TestClient(app) as verify:
        events = emergency_events(verify, eid).json()["events"]
        assert [(e["event"], e["approver_id"]) for e in events] == [
            ("approved", winner["second_approver_id"]),
            ("requested", "alice"),
        ]
        # 胜出者之后重放始终返回同一份首次结果
        replay = emergency_unblind(
            verify,
            eid,
            subject_id="s1",
            approver_id=winner["second_approver_id"],
            operation_id="em-race",
        )
        assert replay.status_code == 200
        assert replay.json() == winner


def test_concurrent_distinct_approvers_open_and_complete_once(
    tmp_path, monkeypatch
) -> None:
    # 两位不同审批人携带同一 operation_id 并发首调：串行化后
    # 必然一人开窗（pending）、另一人在窗口内完成（completed），只产生一次结果
    monkeypatch.setenv(
        "DATABASE_PATH", str(tmp_path / "conc-emergency-open.db")
    )
    with TestClient(app) as setup:
        created = setup.post(
            "/experiments",
            json={
                "conditions": ["A", "B"],
                "subject_ids": ["s1", "s2"],
                "seed": 42,
            },
        ).json()
    eid = created["experiment_id"]

    n_threads = 2
    barrier = threading.Barrier(n_threads)
    responses: list[dict] = []
    responses_lock = threading.Lock()

    def worker(approver: str) -> None:
        with TestClient(app) as local_client:
            barrier.wait()
            resp = emergency_unblind(
                local_client,
                eid,
                subject_id="s1",
                approver_id=approver,
                operation_id="em-open-race",
            )
            with responses_lock:
                responses.append(
                    {"status": resp.status_code, "body": resp.json()}
                )

    threads = [
        threading.Thread(target=worker, args=(a,)) for a in ("alice", "bob")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(r["status"] for r in responses) == [200, 200]
    statuses = sorted(r["body"]["status"] for r in responses)
    assert statuses == ["completed", "pending"]
    completed = next(r["body"] for r in responses if r["body"]["status"] == "completed")
    assert len(completed["sequence"]) == 2

    with TestClient(app) as verify:
        events = emergency_events(verify, eid).json()["events"]
    assert [e["event"] for e in events] == ["approved", "requested"]


def test_legacy_database_supports_emergency_unblind_after_migration(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "legacy-emergency.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    # 模拟旧版本数据库：只有 experiments / assignments 两张表
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE experiments (
                id TEXT PRIMARY KEY,
                conditions TEXT NOT NULL,
                seed INTEGER NOT NULL,
                group_size INTEGER NOT NULL,
                num_subjects INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE assignments (
                experiment_id TEXT NOT NULL REFERENCES experiments(id),
                subject_id TEXT NOT NULL,
                sequence_index INTEGER NOT NULL,
                sequence TEXT NOT NULL,
                PRIMARY KEY (experiment_id, subject_id)
            );
            """
        )
        conn.execute(
            "INSERT INTO experiments "
            "(id, conditions, seed, group_size, num_subjects, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy-em", '["A","B"]', 1, 2, 2, "2026-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO assignments "
            "(experiment_id, subject_id, sequence_index, sequence) "
            "VALUES (?, ?, ?, ?)",
            ("legacy-em", "s1", 0, '["A","B"]'),
        )
        conn.execute(
            "INSERT INTO assignments "
            "(experiment_id, subject_id, sequence_index, sequence) "
            "VALUES (?, ?, ?, ?)",
            ("legacy-em", "s2", 1, '["B","A"]'),
        )

    # 启动自动建表迁移后，紧急揭盲双人审批与审计分页在旧库上可用
    with TestClient(app) as migrated:
        first = emergency_unblind(
            migrated,
            "legacy-em",
            subject_id="s1",
            approver_id="alice",
            operation_id="lg-em-1",
            reason="legacy SAE",
        )
        assert first.status_code == 200, first.text
        assert first.json()["status"] == "pending"

        done = emergency_unblind(
            migrated,
            "legacy-em",
            subject_id="s1",
            approver_id="bob",
            operation_id="lg-em-1",
            reason="legacy SAE",
        )
        assert done.status_code == 200
        assert done.json()["sequence"] == ["A", "B"]

        events = emergency_events(migrated, "legacy-em").json()["events"]
        assert [(e["event"], e["approver_id"], e["reason"]) for e in events] == [
            ("approved", "bob", "legacy SAE"),
            ("requested", "alice", "legacy SAE"),
        ]
        assert_no_sequence_leak(events)
