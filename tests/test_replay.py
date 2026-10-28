"""Showcase / Replay mode: the recorder's choice of landings, the server's replay API and the page's no-load guarantees."""
import csv
import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from src.app import server
from tools import record_replays as RR

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "src" / "app" / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------- the recorder's choice
def _table():
    """A small fake evaluation table: 4 levels x 130 seeds, success gets rarer with the level."""
    rows = []
    for lv in range(4):
        for seed in range(130):
            ok = seed % (lv + 2) != 1 or lv == 0
            rows.append(dict(controller="c", level=str(lv), seed=str(seed), success="1" if ok else "0", failure_reason="none" if ok else ("too_fast" if seed % 2 else "off_pad"),
                             touchdown_speed_mps=str(1.0 + 0.01 * seed + 0.1 * lv), pad_offset_m=str(1.0 + 0.05 * ((seed * 7) % 40)), wind_scale_mps=str(5 * lv), wind_mean_mps="1.0"))
    return rows


def test_plan_starts_with_the_featured_landing_and_covers_every_level():
    p = RR.plan(_table(), per_level=2, failures=3)
    assert (int(p[0]["level"]), int(p[0]["seed"])) == RR.FEATURED
    ok = [r for r in p if r["success"] == "1"]
    assert {int(r["level"]) for r in ok} == {0, 1, 2, 3}
    assert len({(r["level"], r["seed"]) for r in p}) == len(p)                      # no landing twice
    assert sum(r["success"] == "0" for r in p) == 3


def test_plan_never_puts_two_failures_in_a_row_or_a_failure_first():
    for per_level in (1, 2, 3):
        p = RR.plan(_table(), per_level=per_level, failures=3)
        flags = [r["success"] == "0" for r in p]
        assert not flags[0]
        assert not any(a and b for a, b in zip(flags, flags[1:])), flags


def test_typical_picks_the_most_median_rows():
    rows = [dict(seed=str(i), touchdown_speed_mps=str(v), pad_offset_m="1.0") for i, v in enumerate([1.0, 2.0, 2.1, 2.2, 9.0])]
    assert [int(r["seed"]) for r in RR.typical(rows, 1)] == [2]


# ---------------------------------------------------------------- the server's replay API
def _frame(i):
    return dict(x=100.0 - i, y=50.0 - i, a=0.0, legs=[[0, 0, 0, 0], [0, 0, 0, 0]], m=0.0, s=0.0, fuel=0.5, t={})


def _library(tmp_path, n=2):
    eps = []
    for k in range(1, n + 1):
        name = f"ep_{k:02d}_L{k}_s{k}.json"
        (tmp_path / name).write_text(json.dumps(dict(id=k, env=0, timesteps=0, level=k, dt=0.05, frames=[_frame(i) for i in range(5)], result=dict(end="rest", success=True, speed=1.5, dist=1.0))))
        eps.append(dict(file=name, level=k, seed=k, success=True, reason="none", speed=1.5, offset_m=1.0, wind_scale_mps=5 * k, wind_mean_mps=1.0, checkpoint="x", featured=False))
    (tmp_path / "index.json").write_text(json.dumps(dict(version=1, checkpoint="x", episodes=eps)))
    return eps


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REPLAY_DIR", tmp_path)
    monkeypatch.setattr(server, "PORT", 0)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    port = srv.server_address[1]
    monkeypatch.setattr(server, "PORT", port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", tmp_path
    srv.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def test_replay_index_and_episode_are_served(api):
    base, tmp = api
    eps = _library(tmp)
    code, body = _get(base + "/api/replay/index")
    assert code == 200 and [e["file"] for e in json.loads(body)["episodes"]] == [e["file"] for e in eps]
    code, body = _get(base + "/api/replay/episode?file=" + eps[1]["file"])
    ep = json.loads(body)
    assert code == 200 and ep["id"] == 2 and len(ep["frames"]) == 5


def test_replay_without_a_library_is_an_empty_list_not_an_error(api):
    base, _ = api
    code, body = _get(base + "/api/replay/index")
    assert code == 200 and json.loads(body)["episodes"] == []
    code, body = _get(base + "/api/replay/facts")
    assert code == 200 and json.loads(body) == {}


def test_replay_facts_are_served(api):
    base, tmp = api
    (tmp / "facts.json").write_text(json.dumps(dict(version=1, rules=dict(speed_mps=3.0))))
    code, body = _get(base + "/api/replay/facts")
    assert code == 200 and json.loads(body)["rules"]["speed_mps"] == 3.0


def test_replay_episode_only_serves_files_named_in_the_index(api):
    base, tmp = api
    _library(tmp)
    (tmp / "secret.json").write_text("{}")
    for bad in ("secret.json", "../index.json", "..%2F..%2FREADME.md", "index.json", ""):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + "/api/replay/episode?file=" + bad)
        assert e.value.code == 404


def test_replay_api_keeps_the_host_check(api):
    base, tmp = api
    _library(tmp)
    req = urllib.request.Request(base + "/api/replay/index", headers={"Host": "evil.example"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 403


# ---------------------------------------------------------------- the page
def test_showcase_and_its_recorder_are_interfaces_next_to_simple_and_advanced():
    assert "const SHOW = new URLSearchParams(location.search).has('showcase')" in PAGE
    seg = re.search(r'<div class="seg" id="modeseg">(.*?)</div>', PAGE, re.S).group(1)
    modes = ["simple", "advanced", "showcase", "recorder"]
    assert re.findall(r'data-mode="(\w+)"', seg) == modes                                                     # the switch under the picture
    assert re.findall(r'data-mode="(\w+)"', re.search(r'id="judgeSeg".*?</div>', PAGE, re.S).group(0)) == modes     # the same switch in the showcase itself
    assert re.findall(r'data-mode="(\w+)"', re.search(r'id="recSeg".*?</div>', PAGE, re.S).group(0)) == modes      # and on the recorder screen
    assert "Showcase Recorder</button>" in seg and "if (m === 'recorder') { location.search = '?recorder=1'; return; }" in PAGE
    assert "if (m === 'showcase') { location.search = '?showcase=1'; return; }" in PAGE
    assert 'id="showcap"' in PAGE and 'id="judge"' in PAGE and 'id="jNow"' in PAGE and 'id="jSlide"' in PAGE


def test_showcase_leaves_through_the_switch_without_changing_the_saved_mode():
    handler = PAGE[PAGE.index("$('judgeSeg').onclick"):]
    handler = handler[:handler.index("judgeSlide(0);")]
    assert "b.dataset.mode === 'showcase'" in handler and "spacexperiment.mode" in handler and "location.href = '/'" in handler
    assert "setItem('spacexperiment.mode', 'showcase')" not in PAGE            # a bookmarked / is never turned into the showcase by a saved setting


def test_showcase_slides_cover_the_questions_a_reviewer_asks():
    names = re.search(r"names: \[(.*?)\]", PAGE).group(1)
    for n in ("Results", "Touchdown", "Failures", "Training", "Task", "Vehicle", "Network", "Method", "Limits"):
        assert f"'{n}'" in names and re.search(rf"^  {n}\(\) \{{", PAGE, re.M), n
    for must in ("95 % interval", "Reliability is not proven", "Simulation only", "SHA-256", "validation", "seeds 0"):
        assert must in PAGE, must


def test_showcase_starts_no_polling_and_no_training_calls():
    start = PAGE.index("if (SHOW) { mkSlots(1)")
    boot = PAGE[start:PAGE.index("requestAnimationFrame(loop);", start)]
    assert "showcaseInit()" in boot
    live, other = boot.split("else {")
    assert "setInterval" not in live and "refreshInfo" not in live and "pollTrain" not in live       # the showcase branch polls nothing
    for line in PAGE.splitlines():                                                                    # every other timer that talks to the server is switched off in showcase and recorder
        if re.search(r"setInterval\(.*(live_ids|refreshRunNames|updateBrainText)", line) and "SHB.timers" not in line:
            assert "!LITE" in line, line


def test_showcase_caps_frame_rate_and_stops_when_the_tab_is_hidden():
    loop = next(l for l in PAGE.splitlines() if l.startswith("function loop(now)"))
    assert "document.hidden" in loop and "SHC.FPS" in loop and loop.index("SHC.FPS") < loop.index("update(dt)")
    m = re.search(r"FPS: (\d+)", PAGE)
    assert m and 10 <= int(m.group(1)) <= 30


def test_showcase_library_in_the_repo_is_a_real_recording():
    idx = json.loads((ROOT / "replays" / "index.json").read_text())
    eps = idx["episodes"]
    assert len(eps) >= 8 and {e["level"] for e in eps} == {0, 1, 2, 3}
    assert any(e["level"] == 1 and e["seed"] == 122 and e["success"] for e in eps)
    assert any(not e["success"] for e in eps) and sum(e["success"] for e in eps) > sum(not e["success"] for e in eps)
    for e in eps:
        ep = json.loads((ROOT / "replays" / e["file"]).read_text())
        assert len(ep["frames"]) == e["frames"] > 100 and ep["result"]["success"] == e["success"] and "act" not in ep
        assert all(k in ep["frames"][0] for k in ("x", "y", "a", "legs", "m", "s", "fuel", "t"))
        assert abs(ep["result"]["speed"] - e["speed"]) < 1e-9


def test_showcase_facts_in_the_repo_match_the_evaluation_table():
    facts = json.loads((ROOT / "replays" / "facts.json").read_text())
    summary = {(r["controller"], int(r["level"])): r for r in csv.DictReader(open(ROOT / "paper_assets" / "retrain" / "summary.csv"))}
    for key, block in facts["results"].items():
        for L in block["levels"]:
            row = summary[(key, L["level"])]
            assert L["successes"] == int(row["successes"]) and L["n"] == int(row["n"]) and abs(L["rate"] - float(row["success_rate"])) < 1e-9
            assert L["lo"] < L["rate"] <= L["hi"] + 1e-9 and L["hi"] <= 1.0 + 1e-9
            assert sum(L["fails"].values()) == L["n"] - L["successes"]                                  # every failure has exactly one reason
    c = facts["controller"]
    assert c["name"] == "rt1_final" and len(c["sha256"]) == 64 and c["hidden"] == [256, 256] and c["inputs"] == 18 and c["outputs"] == 5
    assert [l["wind_scale_mps"] for l in facts["levels"]] == [0.0, 5.0, 10.0, 15.0]
    assert facts["rules"]["speed_mps"] == 3.0 and facts["rules"]["offset_m"] == 10.0 and abs(facts["rules"]["tilt_rad"] - 0.15) < 1e-9
    assert len(facts["training_curve"]) == 9 and sum(c["chosen"] for c in facts["training_curve"]) == 1


def test_every_recorded_landing_carries_its_measured_numbers():
    idx = json.loads((ROOT / "replays" / "index.json").read_text())
    table = {(r["level"], r["seed"]): r for r in csv.DictReader(open(ROOT / "paper_assets" / "retrain" / "episodes.csv")) if r["controller"] == "sx_retrained"}
    for e in idx["episodes"]:
        st, row = e["stats"], table[(str(e["level"]), str(e["seed"]))]
        assert abs(st["touchdown"]["speed_mps"] - float(row["touchdown_speed_mps"])) < 0.01                      # the same landing as in the evaluation table
        assert abs(st["touchdown"]["offset_m"] - float(row["pad_offset_m"])) < 0.01
        assert abs(st["fuel"]["left_kg"] - float(row["fuel_remaining_kg"])) <= 1
        assert st["wind"]["scale_mps"] == float(row["wind_scale_mps"]) and 0 <= st["rank"]["slower_than_pct"] <= 100 and st["rank"]["n"] == 300
        assert st["start"]["height_m"] > 2000 and st["peak_load_g"] > 1 and st["max_q_kpa"] > 1
        c, x = st["control"], st["extremes"]
        assert 40 <= c["throttle_mean_pct"] <= c["throttle_peak_pct"] <= 100 and 0 < c["gimbal_max_deg"] <= 5.01 and 0 <= c["fin_max_deg"] <= 25.01       # inside the vehicle's own limits
        assert c["rcs_a_pulses"] >= 0 and 0 <= c["rcs_gas_left_pct"] <= 100 and 100 < x["fall_speed_mps"] < 400 and x["tilt_deg"] > 0 and x["spin_dps"] > 0
        if st["burn"]:
            assert 0 < st["burn"]["start_s"] < st["flight_time_s"] and st["burn"]["start_alt_m"] < st["start"]["height_m"]


# ---------------------------------------------------------------- the Showcase Recorder (browser button for tools/record_replays.py)
from src.app import jobs


class FakeProc:
    def __init__(self, code=None):
        self.code, self.pid = code, 0

    def poll(self):
        return self.code

    @property
    def returncode(self):
        return self.code


@pytest.fixture
def rec(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, "REC", dict(proc=None, started=None, args=None, stopped=False))
    monkeypatch.setattr(jobs, "REC_OUT", tmp_path / "rec.out")
    monkeypatch.setattr(jobs, "_popen", lambda argv, out, env_extra=None: (calls.append(argv), FakeProc())[1])
    monkeypatch.setattr(jobs, "_kill_tree", lambda p: setattr(p, "code", 1))
    return calls


def test_recorder_runs_the_same_command_a_person_would_type(rec):
    code, res = jobs.replay_record_start({"per_level": 3, "failures": 1})
    assert code == 200 and rec[0][1] == str(jobs.REPLAY_SCRIPT) and rec[0][2:] == ["--per-level", "3", "--failures", "1"]
    assert jobs.replay_record_start({})[0] == 409                                                         # one recording at a time
    jobs.REC["proc"] = FakeProc(0)
    code, _ = jobs.replay_record_start({"facts_only": True})
    assert code == 200 and rec[1][2:] == ["--facts-only"]


def test_recorder_refuses_values_outside_its_range(rec):
    for bad in ({"per_level": 0}, {"per_level": 9}, {"failures": 4}, {"failures": -1}, {"per_level": "many"}):
        with pytest.raises(ValueError):
            jobs.replay_record_start(bad)
    assert rec == [] and jobs.replay_record_status() == {"state": "idle"}


def test_recorder_status_follows_the_scripts_own_lines(rec):
    jobs.replay_record_start({})
    assert jobs.replay_record_status()["state"] == "running" and jobs.replay_record_status()["done"] == 0
    jobs.REC_OUT.write_text("warnings.warn('x')\nplan: 11 landings of rt1_final\n 1/11  level 1 seed 122  success   2.088 m/s  0.738 m  471 frames\n 4/11  level 2 seed 125  success   1.71 m/s\n")
    st = jobs.replay_record_status()
    assert (st["done"], st["total"]) == (4, 11) and all("warn" not in l for l in st["log"])
    jobs.REC["proc"] = FakeProc(0)
    assert jobs.replay_record_status()["state"] == "done"
    jobs.REC["proc"] = FakeProc(1)
    assert jobs.replay_record_status()["state"] == "error" and jobs.replay_record_status()["code"] == 1


def test_recorder_stop_cancels_and_says_so(rec):
    jobs.replay_record_start({})
    st = jobs.replay_record_stop()
    assert st["state"] == "stopped"
    assert jobs.replay_record_stop()["state"] == "stopped"                                                # stopping again changes nothing


def test_recorder_endpoints_answer_over_http(api, rec):
    base, _ = api
    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    assert json.loads(_get(base + "/api/replay/record/status")[1]) == {"state": "idle"}
    code, res = post("/api/replay/record/start", {"per_level": 99})
    assert code == 400 and "per_level" in res["errors"][0]
    assert post("/api/replay/record/start", {"per_level": 2, "failures": 3})[0] == 200
    assert post("/api/replay/record/start", {})[0] == 409
    assert post("/api/replay/record/stop", {})[1]["state"] == "stopped"


def test_a_failed_recording_leaves_the_saved_library_alone_and_a_finished_one_replaces_it(tmp_path):
    out, tmp = tmp_path / "replays", tmp_path / "replays.tmp"
    out.mkdir(); tmp.mkdir()
    (out / "ep_01_old.json").write_text("old"); (out / "index.json").write_text("old"); (out / "facts.json").write_text("old")
    (tmp / "ep_01_new.json").write_text("new"); (tmp / "index.json").write_text("new"); (tmp / "facts.json").write_text("new")
    assert (out / "index.json").read_text() == "old"                                                       # nothing is moved until commit_library runs
    RR.commit_library(tmp, out)
    assert sorted(p.name for p in out.iterdir()) == ["ep_01_new.json", "facts.json", "index.json"] and (out / "index.json").read_text() == "new" and not tmp.exists()


def test_recorder_screen_is_a_plain_page_that_polls_only_while_recording():
    assert "if (RECP) recorderInit();" in PAGE and "if (!RECP) requestAnimationFrame(loop);" in PAGE
    assert "const LITE = SHOW || RECP;" in PAGE
    for line in PAGE.splitlines():
        if re.search(r"setInterval\(.*(live_ids|refreshRunNames|updateBrainText)", line) and "SHB.timers" not in line:
            assert "!LITE" in line, line
    poll = PAGE[PAGE.index("async function recorderPoll()"):PAGE.index("function recorderInit()")]
    assert poll.count("setTimeout(recorderPoll") == 2 and "if (run) RC.timer = setTimeout(recorderPoll, 700);" in poll       # one retry when the server is busy, one while a recording runs
    assert "setInterval" not in PAGE[PAGE.index("const RC = {"):PAGE.index("let camT = 0")]


def test_showcase_keeps_the_usual_page_below_the_picture_and_wakes_it_only_when_scrolled_to():
    css = PAGE[:PAGE.index("</style>")]
    assert not re.search(r"body\.showcase\s+\.below[^{]*\{[^}]*display:\s*none", css)                 # the usual tabs are not removed
    fn = PAGE[PAGE.index("function showcaseBelow(on)"):PAGE.index("function showcaseWatch(now)")]
    assert "window.scrollY > window.innerHeight * 0.4" in fn and "buildForm()" in fn and "refreshInfo()" in fn and "SHB.timers.forEach(clearInterval)" in fn
    assert "if (SHOW) window.addEventListener('scroll'" in fn
    boot = PAGE[PAGE.index("else if (SHOW) { mkSlots(1)"):PAGE.index("if (!RECP) requestAnimationFrame(loop);")]
    assert "buildForm" not in boot.split("else {")[0] and "refreshInfo" not in boot.split("else {")[0]        # nothing of it runs at the start of the showcase
    assert "if (!SHOW) resetView(info.n_envs)" in PAGE and "&& !SHOW) resetView(info.n_envs)" in PAGE         # the live run's size never rebuilds the replay's slot


def test_showcase_does_not_draw_while_scrolled_away_and_the_switch_below_leaves_it():
    loop = next(l for l in PAGE.splitlines() if l.startswith("function loop(now)"))
    assert "window.scrollY >= window.innerHeight" in loop and loop.index("window.scrollY") < loop.index("update(dt)")
    seg = PAGE[PAGE.index("document.querySelectorAll('#modeseg button').forEach(b => {"):]
    seg = seg[:seg.index("setMode(b.dataset.mode);")]
    assert "SHOW && (b.dataset.mode === 'simple' || b.dataset.mode === 'advanced')" in seg and "location.href = '/'" in seg
    assert ".screen.idle .jhead .seg" not in PAGE                                                               # the switch in the data column no longer fades away
    assert "scroll down for the usual tabs" in PAGE


def test_showcase_slides_are_filled_and_the_landing_card_keeps_its_height():
    fn = PAGE[PAGE.index("function judgeSlide(k)"):PAGE.index("function judgeTick(now)")]
    assert "getBoundingClientRect().height" in fn and "lo = 0.7, hi = 1.18" in fn and "fits(hi)" in fn           # short slides grow to fill the column, long ones shrink to fit it
    assert "if (!fits(lo)) lo *= 0.96" in fn                                                                    # a table that wraps suddenly never leaves a slide clipped
    now = PAGE[PAGE.index("function judgeNow(phase)"):PAGE.index("// ---- the slides")]
    assert "visibility:hidden" in now and now.count("visibility:hidden") >= 2                                    # rank and rule ticks are reserved before touchdown, so the card does not jump
    for row in ("'Control'", "'Extremes'", "'Rank'"):
        assert row in now, row
    vehicle = PAGE[PAGE.index("  Vehicle() {"):PAGE.index("  Network() {")]
    assert "derived" in vehicle and "Not validated against flight data" in vehicle                                # computed numbers are labelled as such
    network = PAGE[PAGE.index("  Network() {"):PAGE.index("  Method() {")]
    assert "IN_NAMES.slice(0, C.inputs)" in network and "signed square roots" in network


def test_showcase_column_can_be_docked_beside_the_usual_tabs_and_back():
    assert '<div class="lower" id="lower">\n<div class="below">' in PAGE and '<div id="dockSlot"></div>' in PAGE and 'id="jDock"' in PAGE
    fn = PAGE[PAGE.index("function showcaseDock("):PAGE.index("function judgeInit()")]
    assert "$('dockSlot').appendChild($('judge'))" in fn and "document.querySelector('.screen').appendChild($('judge'))" in fn      # the same column moves, it is not built twice
    assert "spacexperiment.showdock" in fn and "if (JC.inited) judgeSlide(JC.slide)" in fn                                  # remembered, and the slide is fitted to its new height
    assert "Move beside the tabs" in fn and "Back beside the picture" in fn
    css = PAGE[:PAGE.index("</style>")]
    assert "body.showcase.docked .lower .below" in css and "position:sticky" in css and "body.showcase.docked .screen { display:flex; }" in css
    assert "setInterval(() => judgeTick(performance.now()), 1000)" in PAGE[PAGE.index("function showcaseBelow(on)"):PAGE.index("function showcaseWatch(now)")]      # the slides keep turning while you read them down there
    assert "localStorage.getItem('spacexperiment.showdock') === '1') showcaseDock(true, false)" in PAGE                         # a docked showcase starts docked
