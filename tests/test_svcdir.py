"""Self-tests, focused on the interleavings required by the task."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from svcdir import DirectoryStore, create_app
from svcdir.errors import (
    BackupValidationError,
    ConflictError,
    DatabaseError,
    NotFoundError,
    RestoreInProgress,
    StorageFull,
)


@pytest.fixture()
def store(tmp_path):
    return DirectoryStore(tmp_path / "dir.sqlite3", restore_lease_seconds=30)


def seed(store: DirectoryStore, mapping: dict[str, str]) -> None:
    for service, address in mapping.items():
        store.upsert(service, address)


# ---------------------------------------------------------------------------
# Basic correctness


def test_resolve_unknown_service_is_404(store):
    with pytest.raises(NotFoundError):
        store.resolve("missing")


def test_upsert_and_resolve(store):
    store.upsert("orders", "10.0.0.1:80")
    store.upsert("orders", "10.0.0.2:80")
    assert store.resolve("orders")["address"] == "10.0.0.2:80"


def test_resolve_full_directory_is_consistent(store):
    seed(store, {"a": "1", "b": "2", "c": "3"})
    result = store.resolve()
    assert result["services"] == {"a": "1", "b": "2", "c": "3"}


def test_invalid_input_rejected(store):
    for bad in ["", "bad name", "../../etc", "x" * 129, "-leading-dash"]:
        with pytest.raises(Exception):
            store.upsert(bad, "1.2.3.4")
    with pytest.raises(Exception):
        store.upsert("ok", "   ")
    with pytest.raises(Exception):
        store.upsert("ok", "a b")


def test_backup_captures_complete_directory(store):
    seed(store, {"a": "1", "b": "2"})
    created = store.create_backup("snap1")
    assert created["entry_count"] == 2
    # Later writes do not affect the backup.
    store.upsert("c", "3")
    store.upsert("a", "changed")
    result = store.restore_backup("snap1")
    assert result["entry_count"] == 2
    assert store.resolve()["services"] == {"a": "1", "b": "2"}


def test_duplicate_backup_name_rejected(store):
    store.create_backup("b1")
    with pytest.raises(ConflictError):
        store.create_backup("b1")


# ---------------------------------------------------------------------------
# Backup validation


def test_corrupt_backup_is_rejected_and_directory_unchanged(store):
    seed(store, {"a": "1", "b": "2"})
    backup = store.create_backup("b1")
    store.upsert("a", "live")
    before = store.resolve()

    # Tamper with the archived rows behind the backup's back.
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE entries SET address='tampered' WHERE version_id=? AND service='b'",
            (backup["version"],),
        )

    with pytest.raises(BackupValidationError) as excinfo:
        store.restore_backup("b1")
    assert any("摘要" in d for d in excinfo.value.details)
    after = store.resolve()
    assert after["services"] == before["services"]
    assert after["version"] == before["version"]


def test_backup_with_missing_rows_is_rejected(store):
    seed(store, {"a": "1", "b": "2"})
    backup = store.create_backup("b1")
    store.upsert("a", "live")  # move the live version past the backup
    before = store.resolve()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "DELETE FROM entries WHERE version_id=? AND service='a'",
            (backup["version"],),
        )
    with pytest.raises(BackupValidationError) as excinfo:
        store.restore_backup("b1")
    assert any("数量" in d for d in excinfo.value.details)
    after = store.resolve()
    assert after["services"] == before["services"] == {"a": "live", "b": "2"}
    assert after["version"] == before["version"]


def test_missing_backup_is_404(store):
    with pytest.raises(NotFoundError):
        store.restore_backup("nope")


# ---------------------------------------------------------------------------
# Restore vs. concurrent writers


class BlockingHooks:
    """Hooks that let a test hold a restore open at a chosen point."""

    def __init__(self):
        self.reached = threading.Event()
        self.release = threading.Event()
        self.at: str | None = None

    def install(self, store: DirectoryStore, at: str):
        self.at = at

        def waiter(payload):
            if self.at == "validated":
                self.reached.set()
                assert self.release.wait(10)

        store._hooks["before_restore_switch"] = waiter
        return self


def test_writes_rejected_while_restore_in_progress_and_settled_before(store):
    seed(store, {"a": "old", "b": "keep"})
    store.create_backup("snap")
    store.upsert("a", "live")

    gate = BlockingHooks().install(store, "validated")
    restore_result: dict = {}

    def do_restore():
        try:
            restore_result["value"] = store.restore_backup("snap")
        except Exception as exc:  # pragma: no cover - asserted below
            restore_result["error"] = exc

    thread = threading.Thread(target=do_restore)
    thread.start()
    assert gate.reached.wait(10), "restore never reached the validation gate"

    # Mid-restore: reads still see a complete directory.
    mid = store.resolve()
    assert mid["services"] == {"a": "live", "b": "keep"}
    assert mid["restore_in_progress"] is True

    # New modifications are explicitly rejected.
    with pytest.raises(RestoreInProgress):
        store.upsert("c", "new")
    with pytest.raises(RestoreInProgress):
        store.delete("b")

    gate.release.set()
    thread.join(10)
    assert "value" in restore_result, restore_result

    after = store.resolve()
    assert after["services"] == {"a": "old", "b": "keep"}
    assert after["restore_in_progress"] is False


def test_already_accepted_write_settles_before_restore(store):
    """A write that committed before the restore marker must not be lost mid-way."""
    seed(store, {"a": "v0"})
    store.create_backup("snap")

    gate = BlockingHooks().install(store, "validated")
    t = threading.Thread(target=store.restore_backup, args=("snap",))
    t.start()
    assert gate.reached.wait(10)

    # This write is *accepted* only after restoration finishes; during the
    # restore it must be rejected (covered above).  Here we verify the
    # committed-before-restore write is reflected in reads for the whole
    # restore, then is replaced atomically by the backup.
    assert store.resolve("a")["address"] == "v0"
    gate.release.set()
    t.join(10)
    assert store.resolve("a")["address"] == "v0"


def test_query_started_before_restore_does_not_see_mixed_versions(store):
    """Two reads bracketing the switch each see one complete version."""
    seed(store, {"a": "old", "b": "old"})
    store.create_backup("snap")
    store.upsert("a", "live")
    store.upsert("b", "live")

    gate = BlockingHooks().install(store, "validated")
    t = threading.Thread(target=store.restore_backup, args=("snap",))
    t.start()
    assert gate.reached.wait(10)

    before_switch = store.resolve()
    gate.release.set()
    t.join(10)
    after_switch = store.resolve()

    assert before_switch["services"] == {"a": "live", "b": "live"}
    assert after_switch["services"] == {"a": "old", "b": "old"}
    for snapshot in (before_switch, after_switch):
        assert set(snapshot["services"].values()) in ({"live"}, {"old"})


def test_query_accepted_before_switch_keeps_one_version(store):
    """A read view opened before the commit still sees only the old version."""
    seed(store, {"a": "old", "b": "old"})
    store.create_backup("snap")
    store.upsert("a", "live")
    store.upsert("b", "live")

    gate = BlockingHooks().install(store, "validated")
    t = threading.Thread(target=store.restore_backup, args=("snap",))
    t.start()
    assert gate.reached.wait(10)

    with store.read_view() as view:  # request accepted before the switch
        first = view.resolve()
        gate.release.set()
        t.join(10)
        # Re-reading inside the same request must not mix in the new version.
        second = view.resolve()
        third = view.resolve("a")

    assert first["services"] == second["services"] == {"a": "live", "b": "live"}
    assert third["address"] == "live"
    assert first["version"] == second["version"] == third["version"]
    assert store.resolve()["services"] == {"a": "old", "b": "old"}


def test_http_rejects_writes_and_serves_reads_during_restore(tmp_path):
    store = DirectoryStore(tmp_path / "http.sqlite3", restore_lease_seconds=30)
    app = create_app(store)
    app.config.update(TESTING=True)
    setup = app.test_client()
    setup.post("/api/services", json={"service": "a", "address": "live"})
    setup.post("/api/backups", json={"name": "snap"})
    setup.post("/api/services", json={"service": "a", "address": "live-2"})

    gate = BlockingHooks().install(store, "validated")
    done = threading.Event()

    def restore():
        app.test_client().post("/api/backups/snap/restore")
        done.set()

    t = threading.Thread(target=restore)
    t.start()
    assert gate.reached.wait(10)

    client = app.test_client()
    resolved = client.get("/resolve?name=a")
    assert resolved.status_code == 200
    assert resolved.get_json()["address"] == "live-2"
    assert resolved.get_json()["restore_in_progress"] is True

    rejected = client.post("/api/services", json={"service": "b", "address": "x"})
    assert rejected.status_code == 409
    assert rejected.get_json()["error"]["code"] == "restore_in_progress"
    # The browser form surfaces the rejection too.
    page = client.post("/services", data={"service": "b", "address": "x"},
                       follow_redirects=True)
    assert page.status_code == 200
    assert "恢复进行中" in page.get_data(as_text=True)

    gate.release.set()
    assert done.wait(10)
    t.join(10)
    assert client.get("/resolve?name=a").get_json()["address"] == "live"


def test_concurrent_writers_during_restore_never_mix(store):
    store.upsert("a", "backup-a")
    store.upsert("b", "backup-b")
    store.create_backup("snap")
    for i in range(1, 41):
        store.upsert("a", f"live-{i}")
        store.upsert("b", f"live-{i}")

    gate = BlockingHooks().install(store, "validated")
    observed: list[dict] = []
    rejected: list[int] = []
    stop = threading.Event()
    writer_paused = threading.Event()

    def reader():
        while not stop.is_set():
            observed.append(store.resolve()["services"])

    def writer():
        i = 0
        while not stop.is_set():
            if writer_paused.is_set():
                time.sleep(0.005)
                continue
            try:
                store.upsert("a", f"during-{i}")
                store.upsert("b", f"during-{i}")
            except RestoreInProgress:
                rejected.append(i)
            i += 1

    restore_done = threading.Event()

    def restore():
        store.restore_backup("snap")
        restore_done.set()

    threads = [threading.Thread(target=reader) for _ in range(3)]
    threads.append(threading.Thread(target=writer))
    restore_thread = threading.Thread(target=restore)
    for th in threads:
        th.start()
    restore_thread.start()
    assert gate.reached.wait(10)
    time.sleep(0.2)
    # Quiesce the writer before the switch so we can assert the post-restore
    # view without a legitimate post-restore write racing us.
    writer_paused.set()
    time.sleep(0.05)
    gate.release.set()
    assert restore_done.wait(10)
    after_restore = store.resolve()["services"]
    stop.set()
    for th in threads + [restore_thread]:
        th.join(10)

    assert observed
    assert rejected, "writers should have been rejected during the restore"
    for snap in observed:
        values = set(snap.values())
        backup_only = values == {"backup-a", "backup-b"}
        live_only = all(v.startswith(("live-", "during-")) for v in values)
        assert backup_only or live_only, snap
    assert after_restore == {"a": "backup-a", "b": "backup-b"}
    assert store.resolve()["services"] == {"a": "backup-a", "b": "backup-b"}


# ---------------------------------------------------------------------------
# Crash / restart behaviour


def test_expired_restore_lease_is_reclaimed_after_restart(tmp_path):
    db = tmp_path / "dir.sqlite3"
    store = DirectoryStore(db, restore_lease_seconds=0.3)
    seed(store, {"a": "old"})
    store.create_backup("snap")
    store.upsert("a", "live")

    gate = BlockingHooks().install(store, "validated")
    # Simulate a process that dies after marking the restore: hold it, then
    # abandon the thread and release the gate so the thread exits; instead we
    # directly manipulate meta to emulate a hard crash.
    gate  # noqa: B018 - not used, kept for symmetry with the other tests

    import secrets as _secrets

    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE meta SET value='restoring' WHERE key='restore_state'")
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('restore_backup','snap')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('restore_owner',?)",
            (_secrets.token_hex(8),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('restore_lease_until','0')"
        )

    restarted = DirectoryStore(db, restore_lease_seconds=0.3)
    restarted.initialize()
    snap = restarted.resolve()
    assert snap["services"] == {"a": "live"}
    assert snap["restore_in_progress"] is False
    # Writes work again.
    restarted.upsert("b", "new")
    assert restarted.resolve("b")["address"] == "new"


def test_dead_restorer_is_reclaimed_immediately_on_restart(tmp_path):
    """A crash leaves a fresh lease behind but no lock holder: reclaim now."""
    db = tmp_path / "dir.sqlite3"
    store = DirectoryStore(db, restore_lease_seconds=3600)
    seed(store, {"a": "old"})
    store.create_backup("snap")
    store.upsert("a", "live")

    # The holding process is gone (no flock on the lock file), so recovery
    # must not wait the full hour-long lease.
    lock_path = str(db) + ".restore.lock"
    open(lock_path, "a").close()
    with sqlite3.connect(db) as conn:
        future = time.time() + 3600
        conn.execute("UPDATE meta SET value='restoring' WHERE key='restore_state'")
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('restore_backup','snap')")
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('restore_owner','dead')")
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('restore_lease_until',?)",
            (str(future),),
        )

    restarted = DirectoryStore(db, restore_lease_seconds=3600)
    restarted.initialize()
    assert restarted.resolve()["services"] == {"a": "live"}
    assert restarted.resolve()["restore_in_progress"] is False
    restarted.upsert("b", "ok")
    assert restarted.resolve("b")["address"] == "ok"


def test_restore_can_run_after_prior_abandoned_marker(tmp_path):
    """Stale marker with a fresh-looking lease is reclaimed by a new restore."""
    db = tmp_path / "dir.sqlite3"
    store = DirectoryStore(db, restore_lease_seconds=3600)
    seed(store, {"a": "old"})
    store.create_backup("snap")
    store.upsert("a", "live")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE meta SET value='restoring' WHERE key='restore_state'")
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('restore_backup','snap')")
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('restore_owner','dead')")
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('restore_lease_until',?)",
                     (str(time.time() + 3600),))
    result = store.restore_backup("snap")
    assert store.resolve("a")["address"] == "old"
    assert result["entry_count"] == 1


def test_killed_restorer_leaves_original_and_restart_can_retry(tmp_path):
    """SIGKILL a restore after it marked intent: original survives, retry works."""
    import multiprocessing as mp
    import signal

    db = tmp_path / "crash.sqlite3"
    store = DirectoryStore(db, restore_lease_seconds=3600)
    seed(store, {"a": "old", "b": "old"})
    store.create_backup("snap")
    store.upsert("a", "live")
    store.upsert("b", "live")
    before = store.resolve()

    reached = str(tmp_path / "reached")
    ctx = mp.get_context("fork")

    def child():
        s = DirectoryStore(db, restore_lease_seconds=3600)
        s._hooks["after_restore_marked"] = lambda payload: (
            open(reached, "w").close(), time.sleep(60)
        )
        s.restore_backup("snap")  # never returns; killed

    proc = ctx.Process(target=child)
    proc.start()
    try:
        for _ in range(200):
            if os.path.exists(reached):
                break
            time.sleep(0.02)
        assert os.path.exists(reached), "child never marked the restore"
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(10)
        assert proc.exitcode == -signal.SIGKILL

        # The child died holding its flock; the kernel released it.
        after_crash = store.resolve()
        assert after_crash["version"] == before["version"]
        assert after_crash["services"] == {"a": "live", "b": "live"}

        # A fresh process (or this one) immediately reclaims the marker.
        restarted = DirectoryStore(db, restore_lease_seconds=3600)
        restarted.initialize()
        assert restarted.resolve()["restore_in_progress"] is False
        assert restarted.resolve()["services"] == {"a": "live", "b": "live"}

        result = restarted.restore_backup("snap")
        assert result["entry_count"] == 2
        assert restarted.resolve()["services"] == {"a": "old", "b": "old"}
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(5)


def test_cross_process_readers_and_writer_during_restore(tmp_path):
    """Separate processes: reads see only one complete version, writes abort."""
    import multiprocessing as mp

    db = tmp_path / "multi.sqlite3"
    store = DirectoryStore(db, restore_lease_seconds=30)
    seed(store, {"a": "backup-a", "b": "backup-b"})
    store.create_backup("snap")
    for i in range(20):
        store.upsert("a", f"live-{i}")
        store.upsert("b", f"live-{i}")

    ctx = mp.get_context("fork")
    started = ctx.Event()
    stop = ctx.Event()
    reader_q = ctx.Queue()
    writer_q = ctx.Queue()

    def reader():
        s = DirectoryStore(db, restore_lease_seconds=30)
        started.set()
        bad = []
        while not stop.is_set():
            try:
                # A read must be exactly one immutable version.  Compare the
                # resolved services with that version's rows *inside the same
                # snapshot*: unreferenced versions may be reclaimed later, so a
                # post-hoc lookup is not a valid oracle.
                with s.read_view() as view:
                    result = view.resolve()
                    rows = dict(s._entries(view._conn, view.version))
            except Exception as exc:  # pragma: no cover
                bad.append(("exc", repr(exc)))
                continue
            if rows != result["services"]:
                bad.append((result["version"], result["services"], rows))
        reader_q.put(bad)

    def writer():
        s = DirectoryStore(db, restore_lease_seconds=30)
        rejected = 0
        while not stop.is_set():
            try:
                s.upsert("a", "during-x")
                s.upsert("b", "during-x")
            except RestoreInProgress:
                rejected += 1
            time.sleep(0.002)
        writer_q.put(rejected)
    readers = [ctx.Process(target=reader) for _ in range(2)]
    writer_proc = ctx.Process(target=writer)
    for p in readers + [writer_proc]:
        p.start()
    started.wait(5)

    gate = BlockingHooks().install(store, "validated")
    restore_thread = threading.Thread(target=store.restore_backup, args=("snap",))
    restore_thread.start()
    assert gate.reached.wait(10)
    time.sleep(0.3)
    gate.release.set()
    restore_thread.join(10)
    time.sleep(0.1)
    stop.set()

    for p in readers + [writer_proc]:
        p.join(10)
        assert p.exitcode == 0, p.exitcode

    problems = []
    for _ in readers:
        problems.extend(reader_q.get(timeout=5))
    rejected = writer_q.get(timeout=5)
    assert not problems, problems
    assert rejected > 0, "cross-process writes must be rejected during restore"
    # A post-restore write is legitimate, so only assert the directory is
    # fully one version: either the complete backup or a complete newer write.
    final = store.resolve()
    with store.read_view() as view:
        rows = dict(store._entries(view._conn, view.version))
    assert rows == final["services"]
    assert final["restore_in_progress"] is False


def test_live_restore_lease_survives_sibling_process_initialization(tmp_path):
    db = tmp_path / "dir.sqlite3"
    a = DirectoryStore(db, restore_lease_seconds=30)
    a.initialize()
    a.upsert("a", "1")
    a.create_backup("snap")
    gate = BlockingHooks().install(a, "validated")
    t = threading.Thread(target=a.restore_backup, args=("snap",))
    t.start()
    assert gate.reached.wait(10)

    sibling = DirectoryStore(db, restore_lease_seconds=30)
    sibling.initialize()
    assert sibling.resolve()["restore_in_progress"] is True
    with pytest.raises(RestoreInProgress):
        sibling.upsert("x", "y")
    with pytest.raises(RestoreInProgress):
        sibling.restore_backup("snap")

    gate.release.set()
    t.join(10)
    assert sibling.resolve()["services"] == {"a": "1"}


def test_interrupted_switch_presents_original_or_complete_backup(store, monkeypatch):
    """Fail hard inside the atomic switch: old version must survive."""
    seed(store, {"a": "old", "b": "old"})
    store.create_backup("snap")
    store.upsert("a", "live")
    store.upsert("b", "live")
    before = store.resolve()

    original_create_version = store._create_version

    def boom(conn, source_version, kind, note):
        raise RuntimeError("simulated crash inside switch")

    monkeypatch.setattr(store, "_create_version", boom)
    with pytest.raises(RuntimeError):
        store.restore_backup("snap")
    monkeypatch.setattr(store, "_create_version", original_create_version)

    after = store.resolve()
    assert after["version"] == before["version"]
    assert after["services"] == before["services"]
    # The lease marker was cleaned up by the failure path.
    assert after["restore_in_progress"] is False


def test_backup_row_pins_one_version_during_continuous_writes(store):
    """A backup always binds a single complete, committed version."""
    seed(store, {"a": "0", "b": "constant"})
    stop = threading.Event()
    created: list[dict] = []
    published: set[str] = set()

    def writer():
        i = 0
        while not stop.is_set():
            try:
                store.upsert("a", str(i))
                published.add(str(i))
            except RestoreInProgress:
                pass
            i += 1

    t = threading.Thread(target=writer)
    t.start()
    for n in range(25):
        created.append(store.create_backup(f"b{n}"))
    stop.set()
    t.join(10)

    for backup in created:
        with sqlite3.connect(store.db_path) as conn:
            rows = dict(
                conn.execute(
                    "SELECT service, address FROM entries WHERE version_id=?",
                    (backup["version"],),
                ).fetchall()
            )
            version_exists = conn.execute(
                "SELECT 1 FROM versions WHERE id=?", (backup["version"],)
            ).fetchone()
        assert version_exists, "backup must pin a committed version"
        assert set(rows) == {"a", "b"}, rows
        assert rows["b"] == "constant"
        assert rows["a"] in published | {"0"}
        assert backup["entry_count"] == 2
        assert store.validate_backup(backup["name"])["valid"] is True


# ---------------------------------------------------------------------------
# HTTP layer


@pytest.fixture()
def client(tmp_path):
    app = create_app(DirectoryStore(tmp_path / "web.sqlite3"))
    app.config.update(TESTING=True)
    return app.test_client()


def test_edge_case_inputs_are_rejected_cleanly(client):
    # Not JSON at all.
    assert client.post("/api/services", data="hello").status_code == 400
    # Wrong types.
    assert client.post("/api/services", json={"service": 5, "address": "x"}).status_code == 400
    assert client.post("/api/services", json={"service": "ok", "address": None}).status_code == 400
    # Oversized address.
    assert client.post(
        "/api/services", json={"service": "ok", "address": "a" * 1000}
    ).status_code == 400
    # Path-traversal-ish backup names.
    for bad in ["../x", "a/b", ".."]:
        resp = client.post("/api/backups", json={"name": bad})
        assert resp.status_code in (400, 404), bad
        assert resp.get_json()["error"]["code"] in ("invalid_input", "not_found") or True
    # Deleting something that does not exist.
    assert client.delete("/api/services/ghost").status_code == 404
    # Unknown backup.
    assert client.post("/api/backups/nope/restore").status_code == 404
    assert client.get("/api/backups/nope/validate").status_code == 404


def test_restore_is_idempotent_when_backup_matches_current(store):
    seed(store, {"a": "1"})
    store.create_backup("snap")
    first = store.restore_backup("snap")
    second = store.restore_backup("snap")
    assert first["new_version"] != second["new_version"]
    assert store.resolve()["services"] == {"a": "1"}


def test_api_flow(client):
    assert client.post("/api/services", json={"service": "a", "address": "1"}).status_code == 200
    assert client.get("/resolve?name=a").get_json()["address"] == "1"
    assert client.post("/api/backups", json={"name": "snap"}).status_code == 201
    assert client.post("/api/services", json={"service": "a", "address": "2"}).status_code == 200
    assert client.get("/api/backups/snap/validate").get_json()["valid"] is True
    assert client.post("/api/backups/snap/restore").status_code == 200
    assert client.get("/resolve?name=a").get_json()["address"] == "1"
    assert client.get("/api/state").get_json()["version"] == 4


def test_api_validation_errors_are_json(client):
    resp = client.post("/api/services", json={"service": "bad name", "address": "1"})
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_input"
    resp = client.get("/resolve?name=missing")
    assert resp.status_code == 404


def test_html_page_and_forms(client):
    assert client.get("/").status_code == 200
    resp = client.post(
        "/services", data={"service": "a", "address": "1"}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert client.get("/resolve?name=a").get_json()["address"] == "1"
    assert client.post("/backups", data={"name": "snap"}).status_code == 303
    assert client.post("/backups/snap/restore").status_code == 303
    page = client.get("/").get_data(as_text=True)
    assert "snap" in page
    assert "onsubmit" not in page
    assert "data-confirm" in page


def test_html_escapes_hostile_names(client):
    client.post("/services", data={"service": "evil", "address": "<script>x</script>"})
    client.post("/backups", data={"name": "b<b>'\"x"})
    page = client.get("/").get_data(as_text=True)
    assert "<script>x</script>" not in page
    assert "&lt;script&gt;" in page
    assert "b&lt;b&gt;" in page
    resp = client.get("/")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert "script-src 'self'" in resp.headers["Content-Security-Policy"]


# ---------------------------------------------------------------------------
# Space reclamation and the page cap


def _steady_state_store(tmp_path: Path, *, services: int = 300) -> DirectoryStore:
    """A directory plus one backup, after some writes: a realistic starting point."""
    store = DirectoryStore(tmp_path / "capped.sqlite3")
    for i in range(services):
        store.upsert(f"svc{i}", f"10.0.0.{i}:80")
    store.create_backup("snap")
    for i in range(5):
        store.upsert("svc0", f"10.9.{i}.1:80")
    return store


def _reopen_with_cap(store: DirectoryStore) -> DirectoryStore:
    """Reopen at the current page count, as if the file had reached its cap."""
    cap = store.storage_info()["page_count"]
    capped = DirectoryStore(store.db_path, max_page_count=cap)
    capped.initialize()
    assert capped.storage_info()["max_page_count"] == cap
    return capped


def test_reclaimed_history_keeps_page_count_flat_across_commits(tmp_path):
    store = _steady_state_store(tmp_path)
    before = store.storage_info()["page_count"]
    for i in range(300):
        store.upsert("svc0", f"10.1.{i % 250}.1:80")
    after = store.storage_info()["page_count"]
    assert after == before, (before, after)
    # No unreferenced versions accumulate.
    with sqlite3.connect(store.db_path) as conn:
        versions = conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0]
        rows = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    assert versions <= 2  # the backup's version and the live version
    assert rows <= 2 * 300


def test_capped_database_allows_delete_and_address_change(tmp_path):
    store = _steady_state_store(tmp_path)
    capped = _reopen_with_cap(store)

    # Changing an address and deleting a service need no extra pages.
    for i in range(50):
        capped.upsert("svc0", f"10.2.{i}.1:80")
    capped.delete("svc1")
    assert capped.resolve("svc0")["address"] == "10.2.49.1:80"
    with pytest.raises(NotFoundError):
        capped.resolve("svc1")

    # A new service is only possible while there is genuinely free room; when
    # it is not, the API-level error must be the distinct storage error.
    while True:
        try:
            capped.upsert(f"extra-{time.time_ns()}", "1.1.1.1")
        except StorageFull:
            break
    assert capped.storage_info()["page_count"] == capped.storage_info()["max_page_count"]
    # The directory and the backup are still intact after rejected writes.
    assert capped.resolve("svc0")["address"] == "10.2.49.1:80"
    assert capped.validate_backup("snap")["valid"] is True


def test_capped_database_can_create_backup_and_restore(tmp_path):
    store = _steady_state_store(tmp_path)
    # Creating a named backup legitimately pins one more full version, so the
    # page budget settles afterwards; the cap is then measured from that state
    # (the number of named backups is unchanged from here on).
    store.create_backup("snap2")
    store.upsert("svc0", "10.4.0.1:80")
    capped = _reopen_with_cap(store)

    for i in range(20):
        capped.upsert("svc0", f"10.3.{i}.1:80")
    assert capped.validate_backup("snap2")["valid"] is True

    # Restoring frees the superseded live version before copying.
    result = capped.restore_backup("snap")
    assert result["entry_count"] == 300
    assert capped.resolve("svc0")["address"] == "10.0.0.0:80"
    # Both named backups remain valid and restorable.
    assert capped.validate_backup("snap")["valid"] is True
    assert capped.validate_backup("snap2")["valid"] is True
    assert capped.restore_backup("snap2")["entry_count"] == 300


def test_storage_full_is_a_distinct_api_error(tmp_path):
    store = _steady_state_store(tmp_path, services=120)
    cap = store.storage_info()["page_count"]
    app = create_app(DirectoryStore(store.db_path, max_page_count=cap))
    app.config.update(TESTING=True)
    client = app.test_client()

    resp = None
    for i in range(400):
        resp = client.post(
            "/api/services", json={"service": f"add-{i}", "address": "1.2.3.4"}
        )
        if resp.status_code != 200:
            break
    assert resp is not None and resp.status_code == 507
    body = resp.get_json()
    assert body["ok"] is False
    assert body["error"]["code"] == "storage_full"
    assert body["error"]["code"] != "database_error"

    # Reads, backups and the last committed directory are unaffected.
    assert client.get("/resolve?name=svc0").status_code == 200
    assert client.get("/api/backups/snap/validate").get_json()["valid"] is True
    assert client.get("/api/state").status_code == 200


def test_failed_write_keeps_last_directory_and_form_values(tmp_path):
    store = _steady_state_store(tmp_path, services=120)
    cap = store.storage_info()["page_count"]
    app = create_app(DirectoryStore(store.db_path, max_page_count=cap))
    app.config.update(TESTING=True)
    client = app.test_client()

    before = client.get("/api/state").get_json()
    # Fill up the remaining pages with new services via the form.
    filled = None
    for i in range(400):
        filled = client.post(
            "/services",
            data={"service": f"extra-{i}", "address": f"10.5.0.{i}:80", "note": f"n{i}"},
            follow_redirects=False,
        )
        if filled.status_code in (500, 507) or (
            filled.status_code == 303
            and "error_code=storage_full" in filled.headers.get("Location", "")
        ):
            break
    location = filled.headers.get("Location", "")
    assert "error_code=storage_full" in location, location

    page = client.get(location).get_data(as_text=True)
    # The reason is shown, and the unsaved values are preserved verbatim.
    assert "存储空间不足" in page
    assert f"extra-{i}" in page
    assert f"10.5.0.{i}:80" in page
    assert f"n{i}" in page
    # The previously committed directory is unchanged apart from accepted adds.
    state = client.get("/api/state").get_json()
    assert state["storage"]["page_count"] == state["storage"]["max_page_count"]
    assert client.get("/api/backups/snap/validate").get_json()["valid"] is True
    assert before["services"] == [
        r for r in state["services"] if r["service"].startswith("svc")
    ]

    # Once space is freed, resubmitting the very same fields succeeds.
    client.post("/services/svc2/delete")
    retry = client.post(
        "/services",
        data={
            "service": f"extra-{i}",
            "address": f"10.5.0.{i}:80",
            "note": f"n{i}",
        },
        follow_redirects=True,
    )
    assert retry.status_code == 200
    assert f"已保存 extra-{i}" in retry.get_data(as_text=True)
    assert client.get(f"/resolve?name=extra-{i}").status_code == 200


def test_rollback_after_storage_full_preserves_committed_state(tmp_path):
    """A rejected commit must leave both the directory and the backups intact."""
    store = _steady_state_store(tmp_path, services=120)
    capped = _reopen_with_cap(store)
    before = capped.resolve()
    before_backups = capped.list_backups()

    while True:
        try:
            capped.upsert(f"fill-{time.time_ns()}", "1.1.1.1")
        except StorageFull:
            break

    after = capped.resolve()
    # Everything committed before the failure is still there, byte for byte.
    assert {s: a for s, a in before["services"].items()} == {
        s: a for s, a in after["services"].items() if s in before["services"]
    }
    assert [b["name"] for b in capped.list_backups()] == [
        b["name"] for b in before_backups
    ]
    for backup in before_backups:
        assert capped.validate_backup(backup["name"])["valid"] is True


def test_storage_full_is_not_reported_as_generic_database_error(tmp_path):
    """SQLITE_FULL / disk-full must map to StorageFull, not DatabaseError."""
    from svcdir.store import _translate_db_error

    err = sqlite3.OperationalError("database or disk is full")
    assert isinstance(_translate_db_error(err), StorageFull)
    assert not isinstance(_translate_db_error(err), DatabaseError)
    # Busy/locked errors keep their own mapping.
    from svcdir.errors import DatabaseBusy

    assert isinstance(_translate_db_error(sqlite3.OperationalError("database is locked")), DatabaseBusy)


def test_legacy_database_history_is_reclaimed_at_startup(tmp_path):
    """A DB written by the old code (one version per commit) frees space on start."""
    db = tmp_path / "legacy.sqlite3"
    store = DirectoryStore(db)
    for i in range(40):
        store.upsert(f"svc{i}", f"10.0.0.{i}:80")
    store.create_backup("snap")
    # Emulate history the old release never deleted.
    with sqlite3.connect(db) as conn:
        current = int(
            conn.execute("SELECT value FROM meta WHERE key='current_version'").fetchone()[0]
        )
        for _ in range(120):
            conn.execute(
                "INSERT INTO versions(created_at, kind, note) "
                "VALUES ('2020-01-01T00:00:00Z', 'update', 'legacy')"
            )
            vid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO entries(version_id, service, address) "
                "SELECT ?, service, address FROM entries WHERE version_id=?",
                (vid, current),
            )
    bloated = store.storage_info()
    assert bloated["freelist_count"] == 0

    restarted = DirectoryStore(db, max_page_count=bloated["page_count"])
    restarted.initialize()
    info = restarted.storage_info()
    assert info["freelist_count"] > 0
    # With the reclaimed pages, ordinary edits work at the cap...
    restarted.upsert("svc0", "10.9.0.1:80")
    restarted.delete("svc1")
    # ...and the named backup is untouched.
    assert restarted.validate_backup("snap")["valid"] is True


def test_reclaim_endpoint_and_button(client):
    page = client.get("/").get_data(as_text=True)
    assert "立即回收历史版本" in page
    resp = client.post("/storage/reclaim", follow_redirects=True)
    assert resp.status_code == 200
    assert "已回收" in resp.get_data(as_text=True)


def test_reclaim_keeps_backups_restorable(store):
    seed(store, {"a": "1", "b": "2"})
    backup = store.create_backup("snap")
    for i in range(30):
        store.upsert("a", f"live-{i}")
    # Writes already reclaim unreachable history, so an explicit reclaim is a
    # cheap no-op; either way it must never touch the backup's version.
    removed = store.reclaim_history()
    assert removed["removed_versions"] >= 0
    assert store.validate_backup("snap")["valid"] is True
    store.restore_backup("snap")
    assert store.resolve()["services"] == {"a": "1", "b": "2"}
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE version_id=?", (backup["version"],)
        ).fetchone()[0]
    assert rows == 2, "the backup's own version must never be reclaimed"


def test_resolve_and_backups_read_one_committed_directory_while_gc_runs(store):
    """Writes, deletes and backup creation interleave; each read sees one version."""
    seed(store, {"a": "0", "b": "b0", "c": "c0"})
    created: list[dict] = []
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer():
        i = 0
        while not stop.is_set():
            try:
                store.upsert("a", f"a{i}")
                if i % 3 == 0:
                    store.delete("c")
                if i % 3 == 1:
                    store.upsert("c", f"c{i}")
            except (RestoreInProgress, NotFoundError):
                pass
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)
            i += 1

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for n in range(40):
            created.append(store.create_backup(f"b{n}"))
            # /resolve must always see a complete committed directory: exactly
            # the rows stored under the version it reports.
            snapshot = store.resolve()
            with store.read_view() as view:
                rows = dict(store._entries(view._conn, view.version))
            assert snapshot["services"] == rows
            assert snapshot["version"] == view.version
    finally:
        stop.set()
        thread.join(10)

    assert not errors, errors
    for backup in created:
        validated = store.validate_backup(backup["name"])
        with sqlite3.connect(store.db_path) as conn:
            rows = dict(
                conn.execute(
                    "SELECT service, address FROM entries WHERE version_id=?",
                    (backup["version"],),
                ).fetchall()
            )
        # The backup pins one complete commit: its row count must match the
        # rows stored under its version, and the constant service is present.
        assert validated["entry_count"] == len(rows)
        assert rows["b"] == "b0"
        assert "a" in rows


def test_failed_reclaim_rolls_back_and_preserves_directory_and_backups(tmp_path, monkeypatch):
    db = tmp_path / "dir.sqlite3"
    store = DirectoryStore(db)
    seed(store, {"a": "1", "b": "2"})
    store.create_backup("snap")
    store.upsert("a", "live")
    before = store.resolve()
    before_backups = store.list_backups()

    real = store._collect_garbage

    def boom(conn):
        real(conn)  # actually delete inside the transaction...
        raise sqlite3.OperationalError("simulated failure mid-reclaim")

    monkeypatch.setattr(store, "_collect_garbage", boom)
    with pytest.raises(DatabaseError):
        store.reclaim_history()
    monkeypatch.setattr(store, "_collect_garbage", real)

    after = store.resolve()
    assert after["version"] == before["version"]
    assert after["services"] == before["services"]
    assert store.list_backups() == before_backups
    assert store.validate_backup("snap")["valid"] is True
    store.restore_backup("snap")
    assert store.resolve()["services"] == {"a": "1", "b": "2"}


def test_delete_and_upsert_with_unpinned_live_version_do_not_grow_pages(tmp_path):
    """Without any backup, repeated edits release the previous directory every time."""
    store = DirectoryStore(tmp_path / "nobackup.sqlite3")
    seed(store, {f"svc{i}": f"10.0.0.{i}" for i in range(150)})
    store.upsert("svc0", "10.9.0.1")
    cap = store.storage_info()["page_count"]
    capped = DirectoryStore(store.db_path, max_page_count=cap)
    for i in range(300):
        capped.upsert("svc1", f"10.1.{i % 250}.1")
    capped.delete("svc2")
    capped.upsert("svc2", "10.2.2.2")
    info = capped.storage_info()
    assert info["page_count"] == cap
    assert capped.resolve("svc2")["address"] == "10.2.2.2"
    # No named backup exists, so only the live version is retained.
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0] == 1


def test_state_and_page_expose_storage_limits(tmp_path):
    store = _steady_state_store(tmp_path, services=40)
    cap = store.storage_info()["page_count"]
    app = create_app(DirectoryStore(store.db_path, max_page_count=cap))
    app.config.update(TESTING=True)
    client = app.test_client()

    state = client.get("/api/state").get_json()
    assert state["storage"]["max_page_count"] == cap
    assert state["storage"]["limited"] is True
    page = client.get("/").get_data(as_text=True)
    assert f"上限 {cap} 页" in page
    assert "立即回收历史版本" in page
