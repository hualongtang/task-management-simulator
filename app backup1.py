import re
import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
from model import run_simulation, df as default_df


def parse_event_segments(event_log):
    """Parse `state.event_log` entries into timeline segments.

    Each START begins a new segment for the named task; PREEMPT or COMPLETE ends the most
    recent open segment for that task. Returns a list of segments with start/finish times.
    """
    segments = []
    open_segments = []  # list of dicts with keys task_id,type,start
    prefix_re = re.compile(r"^\[t=(?P<t>[0-9.]+)\] ?(?P<rest>.*)")
    for entry in event_log:
        m = prefix_re.match(entry)
        if not m:
            continue
        t = float(m.group("t"))
        rest = m.group("rest")
        # START lines: START <task_id> type=<TYPE> ...
        if rest.startswith("START "):
            # extract task_id and type
            parts = rest.split()
            task_id = parts[1]
            type_part = [p for p in parts if p.startswith("type=")]
            typ = type_part[0].split("=", 1)[1] if type_part else None
            open_segments.append({"task_id": task_id, "type": typ, "start": t})
        elif rest.startswith("PREEMPT ") or rest.startswith("COMPLETE "):
            parts = rest.split()
            task_id = parts[1]
            # find last open segment for this task
            for i in range(len(open_segments) - 1, -1, -1):
                if open_segments[i]["task_id"] == task_id and "finish" not in open_segments[i]:
                    seg = open_segments.pop(i)
                    seg["finish"] = t
                    seg["event"] = "PREEMPT" if rest.startswith("PREEMPT") else "COMPLETE"
                    segments.append(seg)
                    break
    # Close any remaining open segments by leaving finish as None
    # (these may correspond to tasks still running at sim end)
    for seg in open_segments:
        seg_copy = seg.copy()
        seg_copy["finish"] = None
        seg_copy["event"] = "OPEN"
        segments.append(seg_copy)
    # sort segments by start
    segments.sort(key=lambda x: x["start"])
    return segments


def segments_to_df(segments, tasks_df):
    df = pd.DataFrame(segments)
    if df.empty:
        return df
    # fill missing finish with NaN
    df["finish"] = df["finish"].astype(float)
    # annotate with task-level flags from tasks_df (is_committed, is_residual, scheduled_start)
    if not tasks_df.empty:
        lookup = tasks_df.set_index("task_id")["is_committed"].to_dict()
        res_lookup = tasks_df.set_index("task_id")["is_residual"].to_dict()
        sched_lookup = tasks_df.set_index("task_id")["scheduled_start"].to_dict()
        df["is_committed"] = df["task_id"].map(lookup).fillna(False)
        df["is_residual"] = df["task_id"].map(res_lookup).fillna(False)
        df["scheduled_start"] = df["task_id"].map(sched_lookup)
    else:
        df["is_committed"] = False
        df["is_residual"] = False
        df["scheduled_start"] = None
    return df


def to_bool(v):
    """Robustly coerce various representations to a Python bool.

    Handles booleans, numeric 0/1, and common strings like 'True'/'False'.
    Returns False for NaN/None/unknown values.
    """
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    try:
        if isinstance(v, (int, float)):
            return v != 0
    except Exception:
        pass
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "t", "yes", "y")
    return False




EVENT_RE = re.compile(r"^\[t=(?P<t>[0-9.]+)\]\s*(?P<msg>.*)")

def build_snapshots(state, tasks_df):
    snapshots = []
    current_task = None
    committed_tasks_at_t = {}
    running_tasks = set()
    completed_ids = set()
    residual_tasks = set()

    for idx, entry in enumerate(state.event_log):
        m = EVENT_RE.match(entry)
        if not m:
            continue

        current_time = float(m.group("t"))
        msg = m.group("msg")

        # =========================
        # 1️⃣ APPLY EVENT EFFECTS
        # =========================
        if msg.startswith("START"):
            tid = msg.split()[1]
            current_task = tid
            running_tasks.add(tid)
            residual_tasks.discard(tid)

        elif msg.startswith("PREEMPT"):
            tid = msg.split()[1]
            running_tasks.discard(tid)
            residual_tasks.add(tid)
            if tid == current_task:
                current_task = None

        elif msg.startswith("COMPLETE"):
            tid = msg.split()[1]
            running_tasks.discard(tid)
            completed_ids.add(tid)
            committed_tasks_at_t.pop(tid, None)
            if tid == current_task:
                current_task = None

        elif re.search(r"\bcommit\b", msg, re.IGNORECASE):
            m2 = re.search(
                r"commit\s+(\S+).*?sched[@=]([0-9.]+)",
                msg,
                re.IGNORECASE,
            )
            if m2:
                committed_tasks_at_t[m2.group(1)] = float(m2.group(2))

        # =========================
        # 2️⃣ BUILD QUEUES (CONSISTENT)
        # =========================
        has_arrived = tasks_df["arrival"] <= current_time
        is_not_completed = ~tasks_df["task_id"].isin(completed_ids)

        # ---- OIT queue ----
        oit_queue_with_status = []
        active_oit = tasks_df[
            (tasks_df["type"] == "OIT") & has_arrived & is_not_completed
        ]
        for _, task in active_oit.iterrows():
            tid = task["task_id"]
            if tid in running_tasks:
                status = "Running"
            elif tid in residual_tasks:
                status = "Residual"
            elif tid in committed_tasks_at_t:
                status = f"Committed @ {committed_tasks_at_t[tid]:.1f}s"
            else:
                status = "Pending"
            oit_queue_with_status.append((tid, status))

        # ---- NOIT queue ----
        noit_queue_with_status = []
        active_noit = tasks_df[
            (tasks_df["type"] == "NOIT") & has_arrived & is_not_completed
        ]
        for _, task in active_noit.iterrows():
            tid = task["task_id"]
            if tid in running_tasks:
                status = "Running"
            elif tid in residual_tasks:
                status = "Residual"
            else:
                status = "Pending"
            noit_queue_with_status.append((tid, status))

        # =========================
        # 3️⃣ SNAPSHOT
        # =========================
        snapshots.append({
            "step": idx,
            "t": current_time,
            "event": msg,
            "current_task": current_task,
            "oit_queue": oit_queue_with_status,
            "noit_queue": noit_queue_with_status,
            "running_queue": list(running_tasks),
            "committed": committed_tasks_at_t.copy(),
        })

    return snapshots




st.set_page_config(page_title="Task Simulator Dashboard", layout="wide")

st.title("Task Management/Prioritizaiton Simulator")

with st.sidebar:
    st.header("Simulation Controls")
    commit_window = st.number_input("Commit window (s)", value=30.0, step=1.0)
    max_time = st.number_input("Max simulation time (s)", value=200.0, step=10.0)
    st.markdown("--")
    st.markdown("Edit tasks (rows) then press **Run simulation**.")

st.header("Tasks — edit then Run")

# Safe default: use the example df from model.py if available, otherwise a tiny default
try:
    display_df = default_df.copy()
except Exception:
    display_df = pd.DataFrame(
        [
            (0, "OIT1", "OIT", 20, 0, 50, None),
            (5, "NOIT1", "NOIT", 6, 5, None, 1),
        ],
        columns=["arrival_time", "task_id", "type", "duration", "EST", "deadline", "priority"],
    )



    
# Provide a data editor for interactive edits
# Map legacy `use_container_width` semantics to the newer `width` API:
# - use_container_width=True  -> width='stretch'
# - use_container_width=False -> width='content'
edited = st.data_editor(display_df, num_rows="dynamic", width='stretch')

run = st.button("Run simulation")

if run:
    # sanitize types
    df_in = edited.copy()
    df_in["arrival_time"] = pd.to_numeric(df_in["arrival_time"], errors="coerce").fillna(0).astype(float)
    df_in["duration"] = pd.to_numeric(df_in["duration"], errors="coerce").fillna(0).astype(float)
    df_in["EST"] = pd.to_numeric(df_in["EST"], errors="coerce").fillna(0).astype(float)
    df_in["deadline"] = pd.to_numeric(df_in["deadline"], errors="coerce")
    df_in["priority"] = pd.to_numeric(df_in["priority"], errors="coerce")

    with st.spinner("Running simulation..."):
        state, tasks_df, events_df = run_simulation(df_in.sort_values("arrival_time").reset_index(drop=True), commit_window=commit_window, max_time=max_time)

    st.success(f"Simulation finished at t={state.now:.1f}")

    # Parse event segments for Gantt plotting
    segments = parse_event_segments(state.event_log)
    # Use segments_to_df to include is_committed and is_residual from the start
    seg_df = segments_to_df(segments, tasks_df)

    # Mark per-segment residual flags
    if not seg_df.empty and "task_id" in seg_df.columns:
        seg_df = seg_df.sort_values(["task_id", "start"]).reset_index(drop=True)
        seg_df["is_residual"] = seg_df.groupby("task_id").cumcount().apply(lambda x: x > 0)

    # STORE EVERYTHING ONCE
    st.session_state.seg_df = seg_df.copy()
    st.session_state.tasks_df = tasks_df
    st.session_state.snapshots = build_snapshots(state, tasks_df)
    st.session_state.step = 0
    st.session_state.df_in = df_in  # <-- STORE THE INPUT DATAFRAME
    st.session_state.state = state


# The report generation should be outside the `if run:` block
# so it re-renders when the user interacts with widgets like the slider.
if "snapshots" in st.session_state:
    snaps = st.session_state.snapshots

    if "snapshots" in st.session_state:
        snaps = st.session_state.snapshots
        max_step = len(snaps) - 1

        col1, col2, col3 = st.columns([1, 6, 1])

        with col1:
            if st.button("⏮ Prev"):
                st.session_state.step = max(0, st.session_state.step - 1)

        with col3:
            if st.button("Next ⏭"):
                st.session_state.step = min(max_step, st.session_state.step + 1)

        st.session_state.step = st.slider(
            "Event step",
            0,
            max_step,
            st.session_state.step,
        )

    snap = snaps[st.session_state.step]



    state = st.session_state.state
    st.subheader("⏱ Current State")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Time & Event**")
        st.metric("Time", f"{snap['t']:.1f} s")
        st.info(snap["event"])

    with c2:
        st.markdown("**Operator Status**")
        if snap["current_task"]:
            st.success(f"Executing: **{snap['current_task']}**")
        else:
            st.warning("Operator idle")

    st.subheader("📋Queues & Plan")

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        st.markdown("**Running Task**")
        if snap["running_queue"]:
            for task_id in snap["running_queue"]:
                st.write("•", task_id)
        else:
            st.write("_empty_")


    with q2:
        st.markdown("**NOIT Queue**")
        if snap["noit_queue"]:
            for task_id, status in sorted(snap["noit_queue"]):
                st.write(f"• {task_id} ({status})")
        else:
            st.write("_empty_")
   
    with q3:
        st.markdown("**OIT Queue**")
        if snap["oit_queue"]:
            for task_id, status in sorted(snap["oit_queue"]):
                st.write(f"• {task_id} ({status})")
        else:
            st.write("_empty_")

    with q4:
        st.markdown("**Committed Plan**")
        if snap["committed"]:
            # Sort by scheduled start time for a more intuitive display
            sorted_committed = sorted(snap["committed"].items(), key=lambda item: item[1])
            for task_id, sched_start in sorted_committed:
                st.write(f"• {task_id} @ {sched_start:.1f}s")
        else:
            st.write("_empty_")










    now_t = snap["t"]

    # show only segments that have started
    seg_df_step = st.session_state.seg_df[
        st.session_state.seg_df["start"] <= now_t
    ].copy()

    # clip unfinished segments to now
    seg_df_step["finish"] = seg_df_step["finish"].fillna(now_t)
    seg_df_step.loc[seg_df_step["finish"] > now_t, "finish"] = now_t


    
    gantt_seg_df = seg_df_step
    gantt_tasks_df = st.session_state.tasks_df






    # Mark per-segment residual flags: for a task, any segment after the first is a resumed (residual) segment
    if not gantt_seg_df.empty and "task_id" in gantt_seg_df.columns:
        gantt_seg_df = gantt_seg_df.sort_values(["task_id", "start"]).reset_index(drop=True)
        gantt_seg_df["is_residual"] = gantt_seg_df.groupby("task_id").cumcount().apply(lambda x: x > 0)

    # If event parsing produced nothing, fall back to using the simulator's task_log
    if gantt_seg_df.empty:
        if not gantt_tasks_df.empty:
            # tasks_df contains start/finish per task for completed tasks; use those as segments
            gantt_seg_df = gantt_tasks_df.rename(columns={"start": "start", "finish": "finish", "type": "type"})[
                ["task_id", "type", "start", "finish", "is_committed", "is_residual", "scheduled_start", "arrival"]
            ].copy()
            # mark these segments as COMPLETE (they came from finished tasks)
            gantt_seg_df["event"] = "COMPLETE"
            # fill missing start using scheduled_start or arrival
            if "start" in gantt_seg_df.columns:
                gantt_seg_df["start"] = gantt_seg_df["start"].fillna(gantt_seg_df.get("scheduled_start")).fillna(gantt_seg_df.get("arrival"))
        else:
            # No finished tasks — build approximate segments from the input df_in (arrival/EST + duration)
            if 'df_in' in st.session_state and not st.session_state.df_in.empty:
                gantt_seg_df = st.session_state.df_in.copy()
                gantt_seg_df = gantt_seg_df.rename(columns={"arrival_time": "arrival", "EST": "EST", "type": "type"})
                gantt_seg_df["start"] = gantt_seg_df.get("EST").fillna(gantt_seg_df.get("arrival"))
                gantt_seg_df["finish"] = gantt_seg_df["start"] + gantt_seg_df.get("duration", 0)
                gantt_seg_df["is_committed"] = False
                gantt_seg_df["is_residual"] = False
                gantt_seg_df["scheduled_start"] = None
                gantt_seg_df = gantt_seg_df[["task_id", "type", "start", "finish", "is_committed", "is_residual", "scheduled_start"]]
                gantt_seg_df["event"] = "ESTIMATED"

    if not gantt_seg_df.empty:
        # fill finish NULLs with sim end time
        gantt_seg_df["finish"] = gantt_seg_df["finish"].fillna(state.now)

        # if seg_df doesn't already include task-level flags, merge them
        if not {"is_committed", "is_residual", "scheduled_start"}.issubset(gantt_seg_df.columns):
            gantt_seg_df = gantt_seg_df.merge(gantt_tasks_df[["task_id", "is_committed", "is_residual", "scheduled_start"]].drop_duplicates(subset=["task_id"], keep="last"), on="task_id", how="left")
        # Ensure per-segment residual flags reflect resumed segments (override any task-level is_residual)
        if "task_id" in gantt_seg_df.columns:
            gantt_seg_df = gantt_seg_df.sort_values(["task_id", "start"]).reset_index(drop=True)
            gantt_seg_df["is_residual"] = gantt_seg_df.groupby("task_id").cumcount().apply(lambda x: x > 0)
            # Drop any _x or _y suffixed columns from the merge (we only want per-segment is_residual)
            gantt_seg_df = gantt_seg_df.drop(columns=[c for c in gantt_seg_df.columns if c.endswith("_x") or c.endswith("_y")], errors="ignore")

        # Ensure a plotting 'type' column exists (merge may have produced type_x / type_y)
        if "type" not in gantt_seg_df.columns:
            if "type_x" in gantt_seg_df.columns:
                gantt_seg_df["type"] = gantt_seg_df["type_x"]
            elif "type_y" in gantt_seg_df.columns:
                gantt_seg_df["type"] = gantt_seg_df["type_y"]
            else:
                gantt_seg_df["type"] = "UNKNOWN"

        # Normalize type values to uppercase to avoid casing mismatches (e.g., 'oit' vs 'OIT')
        gantt_seg_df["type"] = gantt_seg_df["type"].astype(str).fillna("UNKNOWN").str.upper()

        
        
        
        # ==============Gantt plot — shape-based renderer for precise control ==============

        # Ensure numeric types for plotting
        gantt_seg_df["start"] = pd.to_numeric(gantt_seg_df["start"], errors="coerce")
        gantt_seg_df["finish"] = pd.to_numeric(gantt_seg_df["finish"], errors="coerce")
        gantt_seg_df["duration"] = gantt_seg_df["finish"] - gantt_seg_df["start"]

        # Coerce residual/committed flags to booleans so plotting follows seg_df values exactly
        if "is_residual" in gantt_seg_df.columns:
            gantt_seg_df["is_residual"] = gantt_seg_df["is_residual"].apply(to_bool)
        else:
            gantt_seg_df["is_residual"] = False
        if "is_committed" in gantt_seg_df.columns:
            gantt_seg_df["is_committed"] = gantt_seg_df["is_committed"].apply(to_bool)
        else:
            gantt_seg_df["is_committed"] = False

        fig = go.Figure()

        # determine unique tasks and y positions (reverse for top-down ordering)
        tasks = list(dict.fromkeys(gantt_seg_df["task_id"]))
        tasks_rev = list(reversed(tasks))
        y_pos = {task: i for i, task in enumerate(tasks_rev)}

        # color map for types (refined palette)
        # OIT: calmer blue, NOIT: warm orange
        type_colors = {"OIT": "#2b83ba", "NOIT": "#fdae61"}

        exec_h = 0.5  # height for executed bars
        commit_h = 0.12  # very thin height for committed bars
        commit_gap = 0.2  # vertical gap between executed bar and committed thin block

        # Add executed segments (solid bars). Residual coloring strictly follows seg_df['is_residual']
        for _, row in gantt_seg_df.iterrows():
            tid = row.get("task_id")
            if pd.isna(tid):
                continue
            y = y_pos.get(tid, 0)
            x0 = float(row.get("start", 0))
            x1 = float(row.get("finish", x0))
            if x1 < x0:
                x1 = x0
            ttype = str(row.get("type")).upper()
            base_color = type_colors.get(ttype, "gray")
            is_preempt = str(row.get("event", "")).upper() == "PREEMPT"
            # Use the DataFrame's boolean value for residual and committed — do not infer here
            is_residual = bool(row.get("is_residual"))
            is_committed_exec = bool(row.get("is_committed"))

            # Determine fill and line based on residual status first so the column controls appearance
            if is_residual:
                if ttype == "OIT":
                    fillcolor = "rgba(43,131,186,0.35)"
                    line_color = "#2b83ba"
                elif ttype == "NOIT":
                    fillcolor = "rgba(253,174,97,0.28)"
                    line_color = "#fdae61"
                else:
                    fillcolor = "rgba(180,180,180,0.5)"
                    line_color = "black"
                line_dash = "solid"
                line_width = 1
            else:
                # normal executed segment color
                fillcolor = base_color
                if ttype == "OIT":
                    line_color = "#236a95"
                elif ttype == "NOIT":
                    line_color = "#e07d2f"
                #line_color = "black"
                line_dash = "solid"
                line_width = 1

            # Do not visually distinguish committed vs uncommitted executed segments;
            # use a single opacity for all executed segments so appearance is consistent.
            exec_opacity = 0.85
            shape = dict(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y - exec_h / 2,
                y1=y + exec_h / 2,
                fillcolor=fillcolor,
                opacity=exec_opacity,
                line=dict(color=line_color, width=line_width, dash=line_dash),
                layer="above",
            )
            fig.add_shape(shape)

            # Overlay a thicker dashed red border for preempted segments to make them stand out
            # (preempt status is shown as a red dashed border on top of whatever fill is chosen)
            if is_preempt:
                border_shape = dict(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=y - exec_h / 2,
                    y1=y + exec_h / 2,
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(color="red", width=1, dash="dash"),
                    layer="above",
                )
                fig.add_shape(border_shape)

            # If this executed segment is committed, add an inner border to indicate committed execution
            # committed executed segments are already visually emphasized by opacity and committed blocks;
            # no inner border is drawn here to keep the visual simple


        # Add committed scheduled blocks on top (thin bars). Use scheduled_start and duration from seg_df or tasks_df/input
        gantt_committed_tasks = {}
        # gather scheduled_start per task
        for _, row in gantt_seg_df.iterrows():
            if row.get("is_committed") and pd.notna(row.get("scheduled_start")):
                gantt_committed_tasks[row.get("task_id")] = float(row.get("scheduled_start"))

        # fallback: also check tasks_df for scheduled_start
        if not gantt_tasks_df.empty:
            # Use .loc to avoid SettingWithCopyWarning and ensure we operate on the correct data
            committed_from_tasks_df = gantt_tasks_df.loc[gantt_tasks_df["scheduled_start"].notna()]
            for _, r in committed_from_tasks_df.iterrows():
                gantt_committed_tasks.setdefault(r["task_id"], float(r["scheduled_start"]))

        # determine durations for committed tasks (prefer input df if available)
        input_durations = {}
        if 'df_in' in st.session_state and not st.session_state.df_in.empty:
            input_durations = st.session_state.df_in.set_index('task_id')['duration'].to_dict()
        # also record executed durations from seg_df grouped by task (span)
        exec_durs = gantt_seg_df.groupby("task_id")["finish"].max() - gantt_seg_df.groupby("task_id")["start"].min()

        for tid, sched in gantt_committed_tasks.items():
            dur = None
            if tid in input_durations:
                dur = input_durations[tid]
            elif tid in exec_durs.index and pd.notna(exec_durs.loc[tid]):
                dur = float(exec_durs.loc[tid])
            else:
                # try to infer from seg_df rows for the task
                rows = gantt_seg_df[gantt_seg_df["task_id"] == tid]
                if not rows.empty:
                    dur = float((rows["finish"] - rows["start"]).sum())
            if dur is None:
                continue
            y = y_pos.get(tid, 0)
            rows = gantt_seg_df[gantt_seg_df["task_id"] == tid]
            color = type_colors.get(rows.iloc[0].get("type") if not rows.empty else None, "#888888")
            # draw committed block as a very thin filled rectangle using a semi-transparent type color
            # Use a distinct color (purple) for committed planned blocks so they stand out
            committed_fill = {"OIT": "rgba(102,45,145,0.9)", "NOIT": "rgba(102,45,145,0.9)"}
            fill = committed_fill.get(rows.iloc[0].get("type") if not rows.empty else None, "rgba(102,45,145,0.9)")
            shape = dict(
                type="rect",
                x0=sched,
                x1=sched + dur,
                y0=y + exec_h / 2 + commit_gap,
                y1=y + exec_h / 2 + commit_gap + commit_h,
                fillcolor=fill,
                opacity=1.0,
                line=dict(color="rgba(0,0,0,0)", width=0),
                layer="above",
            )
            fig.add_shape(shape)

        # Add legend traces (samples)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=type_colors.get("OIT"), symbol="square"), name="OIT"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=type_colors.get("NOIT"), symbol="square"), name="NOIT"))
        # residual sample (lighter tint, match the residual tint used above)
        residual_tints = {"OIT": "rgba(43,131,186,0.35)", "NOIT": "rgba(253,174,97,0.28)"}
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=residual_tints.get("OIT")), name="Residual segment"))
        # preempted sample: dashed red border
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="red", dash="dash", width=3), name="Preempted segment"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="black", width=1), name="Executed segment"))
        # committed legend uses the same purple color as the committed fill
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="rgba(102,45,145,0.9)", width=6), name="Committed"))

        # Configure axes and layout
        n_tasks = len(tasks)
        height = max(300, 40 * n_tasks)
        fig.update_layout(height=height, showlegend=True, margin=dict(l=120, r=40, t=40, b=40))
        # y ticks
        tickvals = [y_pos[t] for t in tasks_rev]
        ticktext = tasks_rev
        fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, autorange="reversed")
        fig.update_xaxes(title_text="Time (s)")

        # Add commit window vertical shaded region (from commit_start to commit_end)
        # try:
        #     if getattr(state, "commit_end", None) and state.commit_end > 0:
        #         commit_end = float(state.commit_end)
        #         commit_start = max(0.0, commit_end - float(state.commit_window_length))
        #         # draw a translucent rectangle spanning the commit window across all tasks
        #         fig.add_shape(dict(
        #             type="rect",
        #             x0=commit_start,
        #             x1=commit_end,
        #             y0=-1,
        #             y1=len(tasks_rev) + 1,
        #             fillcolor="rgba(200,200,200,0.12)",
        #             line=dict(width=0),
        #             layer="below",
        #         ))
        #         # add border lines at window edges
        #         fig.add_vline(x=commit_start, line_dash="dash", line_color="gray", opacity=0.6)
        #         fig.add_vline(x=commit_end, line_dash="dash", line_color="gray", opacity=0.6)
        # except Exception:
        #     pass




        # --------------------------------------------------
        # ADD COMMIT-WINDOW VERTICAL LINES (ALIGNED TO LEFT)
        # --------------------------------------------------
        cw = commit_window
        max_t = now_t

        # forces the x-axis NOT to pad, so x=0 aligns to the left edge
        fig.update_xaxes(range=[0, max_t], constrain='domain')

        # draw vertical lines at 0, cw, 2*cw, ... <= max_t
        for t in [i * cw for i in range(int(math.floor(max_t / cw)) + 1)]:
            fig.add_shape(
                type="line",
                x0=t, x1=t,
                y0=-1, y1=len(tasks),   # covers all task rows
                line=dict(color="black", width=1, dash="dot"),
                layer="below"
            )

        # optional: annotate the commit-window boundaries
        for t in [i * cw for i in range(int(math.floor(max_t / cw)) + 1)]:
            fig.add_annotation(
                x=t, y=len(tasks),
                text=f"CW {t:.0f}",
                showarrow=False,
                yshift=10,
                font=dict(size=10)
            )



        fig.add_vline(
            x=now_t,
            line_color="red",
            line_width=2,
            )



        # Use `width='stretch'` to match previous `use_container_width=True` behavior
        # and pass Plotly configuration via `config=` (keyword args are deprecated).
        plotly_config = {"responsive": True}
        # st.plotly_chart(fig, config=plotly_config, width='stretch')
        st.plotly_chart(
            fig,
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True
            }
        )

    else:
        st.info("No segments available to plot — check that the simulation produced start/finish times or events.")

    # Show segments dataframe before the task log
    if "seg_df" in st.session_state and not st.session_state.seg_df.empty:
        st.subheader("Task segment log")
        # Display the full segment log from session state with the desired columns
        display_seg_df = st.session_state.seg_df.sort_values("start").reset_index(drop=True)
        cols_to_show = ["task_id", "type", "start", "finish", "event", "is_committed", "is_residual"]
        existing_cols = [col for col in cols_to_show if col in display_seg_df.columns]
        st.dataframe(display_seg_df[existing_cols])

    # Show logs and allow downloads
    st.subheader("Task execution log (completed tasks)")
    st.dataframe(st.session_state.tasks_df)
    
    st.subheader("Event log")
    st.dataframe(pd.DataFrame({"event": st.session_state.state.event_log}))
    

    
