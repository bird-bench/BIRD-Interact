"""Generate HTML report from evaluation results."""

import json
import html
import sys
from pathlib import Path


def _badge(passed, label=None):
    if passed is True:
        return f'<span class="badge pass">{label or "PASS"}</span>'
    elif passed is False:
        return f'<span class="badge fail">{label or "FAIL"}</span>'
    return f'<span class="badge skip">{label or "SKIP"}</span>'


def _esc(text):
    return html.escape(str(text))


def _build_timeline(r):
    """Build a unified timeline from ADK events, merging thinking + tool calls + responses."""
    events = r.get("adk_events", [])
    if not events:
        return None  # fallback to tool_trajectory

    timeline = []
    for e in events:
        etype = e.get("type")
        if etype == "user_message":
            timeline.append({
                "kind": "user_msg",
                "text": e.get("message", ""),
            })
        elif etype == "adk_event":
            content = e.get("content", {})
            role = content.get("role", "")
            parts = content.get("parts", [])
            for part in parts:
                ptype = part.get("type", "")
                if ptype == "text" and role == "model":
                    text = part.get("text", "").strip()
                    if text:
                        timeline.append({"kind": "thinking", "text": text})
                elif ptype == "function_call":
                    timeline.append({
                        "kind": "tool_call",
                        "name": part.get("name", "?"),
                        "args": part.get("args", {}),
                    })
                elif ptype == "function_response":
                    resp = part.get("response", "")
                    if isinstance(resp, str) and len(resp) > 2000:
                        resp = resp[:2000] + "..."
                    timeline.append({
                        "kind": "tool_response",
                        "name": part.get("name", "?"),
                        "response": resp,
                    })
            if e.get("final") and role == "model":
                texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
                final_text = "\n".join(t for t in texts if t.strip())
                if final_text:
                    timeline.append({"kind": "final", "text": final_text})
    return timeline


def _build_tool_trajectory_html(r):
    """Fallback: render from tool_trajectory when no ADK events."""
    traj = r.get("tool_trajectory", [])
    tools = [t for t in traj if t.get("type") == "tool"]
    rows = []
    for i, t in enumerate(tools):
        name = t.get("tool", "?")
        cost = t.get("cost", "")
        budget_after = t.get("budget_after", "")
        args = t.get("args", {})
        result_text = str(t.get("result", ""))
        cls = "tool-submit" if name == "submit_sql" else "tool-ask" if name == "ask_user" else ""
        cost_html = f'<span class="tag cost">cost={cost}</span>' if cost != "" else ""
        budget_html = f'<span class="tag budget">budget={budget_after}</span>' if budget_after != "" else ""
        args_html = ""
        if args:
            args_str = json.dumps(args, ensure_ascii=False)
            if len(args_str) > 500:
                args_str = args_str[:500] + "..."
            args_html = f'<pre class="args">{_esc(args_str)}</pre>'
        rows.append(f"""
        <div class="ev tool-call {cls}">
            <div class="ev-header">
                <span class="ev-icon">🔧</span>
                <span class="ev-label">{_esc(name)}</span>
                {cost_html} {budget_html}
            </div>
            {args_html}
            <pre class="ev-body">{_esc(result_text[:4000])}</pre>
        </div>""")
    return "\n".join(rows) if rows else '<p class="empty">No tool calls</p>'


def _render_ev(kind, **kw):
    """Render a single event element."""
    if kind == "user_msg":
        return f"""<div class="ev user-msg">
            <div class="ev-header"><span class="ev-icon">👤</span><span class="ev-label">User Message</span></div>
            <pre class="ev-body">{_esc(kw["text"])}</pre></div>"""
    if kind == "thinking":
        return f"""<div class="ev thinking">
            <div class="ev-header"><span class="ev-icon">💭</span><span class="ev-label">Thinking</span></div>
            <pre class="ev-body">{_esc(kw["text"])}</pre></div>"""
    if kind == "tool_call":
        name = kw["name"]
        cls = "tool-submit" if name == "submit_sql" else "tool-ask" if name == "ask_user" else ""
        cost_html = f'<span class="tag cost">cost={kw["cost"]}</span>' if kw.get("cost") != "" else ""
        budget_html = f'<span class="tag budget">budget={kw["budget"]}</span>' if kw.get("budget") != "" else ""
        args_str = kw.get("args_str", "{}")
        return f"""<div class="ev tool-call {cls}">
            <div class="ev-header"><span class="ev-icon">🔧</span><span class="ev-label">{_esc(name)}</span>
                {cost_html} {budget_html}</div>
            <pre class="args">{_esc(args_str)}</pre></div>"""
    if kind == "tool_response":
        return f"""<div class="ev tool-response">
            <div class="ev-header"><span class="ev-icon">📋</span><span class="ev-label">{_esc(kw["name"])} response</span></div>
            <pre class="ev-body">{_esc(kw["text"])}</pre></div>"""
    if kind == "final":
        return f"""<div class="ev final-response">
            <div class="ev-header"><span class="ev-icon">✅</span><span class="ev-label">Final Response</span></div>
            <pre class="ev-body">{_esc(kw["text"])}</pre></div>"""
    return ""


def _build_timeline_html(timeline, traj_costs):
    """Render unified timeline as HTML, split by LLM step."""
    # Group into steps: each step starts with a "thinking" event (= one LLM call)
    # user_msg and final are standalone
    steps = []
    current_step = []
    tool_idx = 0

    def _flush():
        nonlocal current_step
        if current_step:
            steps.append(current_step)
            current_step = []

    for item in timeline:
        kind = item["kind"]
        if kind == "user_msg":
            _flush()
            steps.append([("user_msg", {"text": item["text"]})])
        elif kind == "thinking":
            _flush()
            current_step.append(("thinking", {"text": item["text"]}))
        elif kind == "tool_call":
            name = item["name"]
            args = item.get("args", {})
            cost_info = traj_costs.get(tool_idx, {})
            args_str = json.dumps(args, ensure_ascii=False)
            if len(args_str) > 2000:
                args_str = args_str[:2000] + "..."
            current_step.append(("tool_call", {
                "name": name,
                "args_str": args_str,
                "cost": cost_info.get("cost", ""),
                "budget": cost_info.get("budget_after", ""),
            }))
        elif kind == "tool_response":
            resp = item.get("response", "")
            if isinstance(resp, str):
                try:
                    parsed = json.loads(resp)
                    resp = parsed.get("result", resp)
                except (json.JSONDecodeError, AttributeError):
                    pass
            resp_str = str(resp)
            if len(resp_str) > 4000:
                resp_str = resp_str[:4000] + "..."
            current_step.append(("tool_response", {"name": item["name"], "text": resp_str}))
            tool_idx += 1
        elif kind == "final":
            _flush()
            steps.append([("final", {"text": item["text"]})])
    _flush()

    # Render steps
    rows = []
    step_num = 0
    for step in steps:
        if len(step) == 1 and step[0][0] in ("user_msg", "final"):
            # Standalone — no step wrapper
            rows.append(_render_ev(step[0][0], **step[0][1]))
        else:
            step_num += 1
            # Summarize: list tool names
            tool_names = [kw["name"] for k, kw in step if k == "tool_call"]
            summary = ", ".join(tool_names) if tool_names else "reasoning"
            inner = "\n".join(_render_ev(k, **kw) for k, kw in step)
            rows.append(f"""<div class="turn">
                <div class="turn-header" onclick="this.parentElement.classList.toggle('collapsed')">
                    <span class="turn-label">Step {step_num}</span>
                    <span class="turn-summary">{_esc(summary)}</span>
                    <span class="turn-toggle">▾</span>
                </div>
                <div class="turn-body">{inner}</div>
            </div>""")

    return "\n".join(rows) if rows else '<p class="empty">No events</p>'


def generate_html(result_path: str, output_path: str = None):
    with open(result_path) as f:
        data = json.load(f)

    mode = data.get("mode", "unknown")
    metrics = data.get("metrics", {})
    results = data.get("results", [])
    output_path = output_path or result_path.replace(".json", ".html")

    task_cards = []
    for r in results:
        tid = r.get("task_id", "?")
        p1 = r.get("phase1_passed")
        p2 = r.get("phase2_passed")
        reward = r.get("total_reward", 0)
        elapsed = r.get("elapsed_seconds", 0)
        budget_used = r.get("budget_used", "")
        has_fu = r.get("has_follow_up", False)

        # Build cost map from tool_trajectory
        traj = r.get("tool_trajectory", [])
        tools = [t for t in traj if t.get("type") == "tool"]
        traj_costs = {}
        for i, t in enumerate(tools):
            traj_costs[i] = {"cost": t.get("cost", ""), "budget_after": t.get("budget_after", "")}

        # Try timeline from ADK events, fallback to tool_trajectory
        timeline = _build_timeline(r)
        if timeline:
            events_html = _build_timeline_html(timeline, traj_costs)
        else:
            events_html = _build_tool_trajectory_html(r)

        # Dialogue
        dlg = r.get("dialogue_history", [])
        dlg_rows = []
        for d in dlg:
            role = d.get("role", "?")
            content = str(d.get("content", ""))
            role_cls = "dlg-agent" if role == "agent" else "dlg-user"
            dlg_rows.append(f"""
            <div class="dlg-entry {role_cls}">
                <span class="dlg-role">{_esc(role)}</span>
                <div class="dlg-content">{_esc(content[:1000])}</div>
            </div>""")

        budget_info = f' | Budget used: {budget_used}' if budget_used != "" else ""
        fu_badge = ' <span class="badge fu">Follow-up</span>' if has_fu else ""
        tool_count = len(tools)

        task_cards.append(f"""
        <div class="task-card" id="task-{_esc(tid)}">
            <div class="task-header" onclick="this.parentElement.classList.toggle('expanded')">
                <span class="task-id">{_esc(tid)}</span>
                {_badge(p1, "P1")} {_badge(p2, "P2")}{fu_badge}
                <span class="reward">Reward: {reward:.2f}</span>
                <span class="meta">{tool_count} tools | {elapsed:.1f}s{budget_info}</span>
                <span class="expand-icon">▸</span>
            </div>
            <div class="task-body">
                <div class="tab-bar">
                    <button class="tab active" onclick="switchTab(this, 'timeline')">Timeline</button>
                    <button class="tab" onclick="switchTab(this, 'dialogue')">Dialogue ({len(dlg)})</button>
                </div>
                <div class="tab-content timeline active">
                    {events_html}
                </div>
                <div class="tab-content dialogue">
                    {''.join(dlg_rows) if dlg_rows else '<p class="empty">No dialogue</p>'}
                </div>
            </div>
        </div>""")

    n = metrics.get("total_tasks", len(results))
    p1_count = metrics.get("phase1_count", 0)
    p2_count = metrics.get("phase2_count", 0)
    avg_reward = metrics.get("average_reward", 0)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BIRD-Interact — {_esc(mode)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 20px; max-width: 1100px; margin: 0 auto; }}
.header {{ background: #1a1a2e; color: #fff; padding: 24px 32px; border-radius: 12px; margin-bottom: 20px; }}
.header h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 12px; }}
.metrics {{ display: flex; gap: 32px; flex-wrap: wrap; }}
.metric {{ text-align: center; }}
.metric .value {{ font-size: 28px; font-weight: 700; color: #e94560; }}
.metric .label {{ font-size: 12px; opacity: 0.7; margin-top: 2px; }}
.controls {{ margin-bottom: 16px; display: flex; gap: 8px; }}
.controls button {{ padding: 6px 14px; border: 1px solid #ddd; border-radius: 6px; background: #fff; cursor: pointer; font-size: 13px; }}
.controls button:hover {{ background: #eee; }}
.controls button.active {{ background: #1a1a2e; color: #fff; border-color: #1a1a2e; }}
.task-card {{ background: #fff; border-radius: 10px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden; }}
.task-header {{ padding: 12px 20px; display: flex; align-items: center; gap: 10px; cursor: pointer; user-select: none; }}
.task-header:hover {{ background: #fafafa; }}
.task-id {{ font-weight: 600; font-size: 14px; min-width: 160px; }}
.badge {{ padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
.badge.pass {{ background: #d4edda; color: #155724; }}
.badge.fail {{ background: #f8d7da; color: #721c24; }}
.badge.skip {{ background: #e2e3e5; color: #383d41; }}
.badge.fu {{ background: #cce5ff; color: #004085; }}
.reward {{ font-size: 13px; color: #666; }}
.meta {{ font-size: 12px; color: #999; margin-left: auto; }}
.expand-icon {{ font-size: 14px; color: #999; transition: transform 0.2s; }}
.task-card.expanded .expand-icon {{ transform: rotate(90deg); }}
.task-body {{ display: none; padding: 0 20px 20px; }}
.task-card.expanded .task-body {{ display: block; }}
.tab-bar {{ display: flex; gap: 4px; margin-top: 12px; border-bottom: 2px solid #eee; }}
.tab {{ padding: 6px 16px; border: none; background: none; cursor: pointer; font-size: 13px; color: #888; border-bottom: 2px solid transparent; margin-bottom: -2px; }}
.tab.active {{ color: #1a1a2e; border-bottom-color: #1a1a2e; font-weight: 600; }}
.tab-content {{ display: none; padding-top: 12px; }}
.tab-content.active {{ display: block; }}
.ev {{ margin-bottom: 10px; border-left: 3px solid #ddd; padding: 8px 12px; border-radius: 0 6px 6px 0; }}
.ev.thinking {{ border-left-color: #a29bfe; background: #f8f7ff; }}
.ev.tool-call {{ border-left-color: #6c5ce7; background: #fafafa; }}
.ev.tool-call.tool-submit {{ border-left-color: #e94560; }}
.ev.tool-call.tool-ask {{ border-left-color: #0984e3; }}
.ev.tool-response {{ border-left-color: #00b894; background: #f0faf7; }}
.ev.user-msg {{ border-left-color: #fdcb6e; background: #fffdf5; }}
.ev.final-response {{ border-left-color: #00b894; background: #f0fff4; }}
.ev-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
.ev-icon {{ font-size: 14px; }}
.ev-label {{ font-weight: 600; font-size: 13px; color: #555; }}
.tag {{ font-size: 11px; padding: 1px 6px; border-radius: 3px; }}
.tag.cost {{ color: #e94560; background: #fff0f3; }}
.tag.budget {{ color: #0984e3; background: #f0f7ff; }}
.ev-body {{ font-size: 12px; background: rgba(0,0,0,0.03); padding: 8px 10px; border-radius: 4px; white-space: pre-wrap; word-break: break-word; max-height: 250px; overflow-y: auto; line-height: 1.5; }}
.args {{ font-size: 11px; color: #6c5ce7; background: #f3f0ff; padding: 4px 8px; border-radius: 4px; white-space: pre-wrap; word-break: break-word; margin-bottom: 4px; max-height: 120px; overflow-y: auto; }}
.dlg-entry {{ margin-bottom: 8px; padding: 10px 14px; border-radius: 8px; }}
.dlg-entry.dlg-agent {{ background: #f0f7ff; }}
.dlg-entry.dlg-user {{ background: #f0fff4; }}
.dlg-role {{ font-size: 11px; font-weight: 700; text-transform: uppercase; color: #888; }}
.dlg-content {{ font-size: 13px; margin-top: 4px; white-space: pre-wrap; word-break: break-word; line-height: 1.5; }}
.turn {{ border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }}
.turn-header {{ padding: 8px 14px; background: #f8f9fa; cursor: pointer; display: flex; align-items: center; gap: 8px; user-select: none; }}
.turn-header:hover {{ background: #eef; }}
.turn-label {{ font-size: 12px; font-weight: 700; color: #555; text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap; }}
.turn-summary {{ font-size: 11px; color: #888; font-family: monospace; }}
.turn-toggle {{ font-size: 12px; color: #999; margin-left: auto; transition: transform 0.2s; }}
.turn.collapsed .turn-toggle {{ transform: rotate(-90deg); }}
.turn-body {{ padding: 8px; }}
.turn.collapsed .turn-body {{ display: none; }}
.empty {{ color: #999; font-style: italic; font-size: 13px; }}
</style>
</head>
<body>
<div class="header">
    <h1>BIRD-Interact — {_esc(mode)}</h1>
    <div class="metrics">
        <div class="metric"><div class="value">{n}</div><div class="label">Tasks</div></div>
        <div class="metric"><div class="value">{avg_reward:.3f}</div><div class="label">Avg Reward</div></div>
        <div class="metric"><div class="value">{p1_count}/{n}</div><div class="label">Phase 1</div></div>
        <div class="metric"><div class="value">{p2_count}/{n}</div><div class="label">Phase 2</div></div>
    </div>
</div>
<div class="controls">
    <button onclick="filterTasks('all')" class="active" id="btn-all">All</button>
    <button onclick="filterTasks('pass')" id="btn-pass">Pass</button>
    <button onclick="filterTasks('fail')" id="btn-fail">Fail</button>
    <button onclick="expandAll()">Expand All</button>
    <button onclick="collapseAll()">Collapse All</button>
</div>
<div id="tasks">
{''.join(task_cards)}
</div>
<script>
function filterTasks(f) {{
    document.querySelectorAll('.controls button').forEach(b => b.classList.remove('active'));
    document.getElementById('btn-' + f).classList.add('active');
    document.querySelectorAll('.task-card').forEach(card => {{
        const badges = card.querySelectorAll('.badge');
        const hasPass = Array.from(badges).some(b => b.classList.contains('pass'));
        if (f === 'all') card.style.display = '';
        else if (f === 'pass') card.style.display = hasPass ? '' : 'none';
        else card.style.display = hasPass ? 'none' : '';
    }});
}}
function expandAll() {{ document.querySelectorAll('.task-card').forEach(c => c.classList.add('expanded')); }}
function collapseAll() {{ document.querySelectorAll('.task-card').forEach(c => c.classList.remove('expanded')); }}
function switchTab(btn, tabName) {{
    const card = btn.closest('.task-body');
    card.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    card.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    card.querySelector('.tab-content.' + tabName).classList.add('active');
}}
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(page)
    print(f"Report: {output_path}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        generate_html(path)
