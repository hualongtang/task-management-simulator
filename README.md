# Task Management / Prioritization Simulator

An interactive **Streamlit dashboard** for simulating task scheduling and prioritization with support for:
- **OIT (Operator-Independent Tasks)** and **NOIT (Operator-Independent Tasks)**
- **Preemption** and **residual** task handling
- **MILP-based commit windows** for guaranteed task scheduling
- **Interactive Gantt timeline** visualization with residual segments, preemption overlays, and committed task blocks
- **Multi-strategy comparison** (Deadline-first, Shortest-first, FCFS, etc.)

## Features

- **Task Editor**: Edit task parameters (duration, EST, deadline, priority, arrival time) interactively
- **Event-Driven Simulation**: Accurate simulation clock advancing through arrivals, finishes, commit-window boundaries, and committed scheduled starts
- **Gantt Chart**: Shape-based Plotly visualization showing:
  - Executed segments (solid bars colored by type)
  - Residual/resumed segments (lighter tint)
  - Preempted segments (dashed red overlay)
  - Committed scheduled blocks (thin purple bars above executed segments)
  - Commit-window shading (light gray background)
- **Logs & Downloads**: View and export task execution logs, event logs, and segment dataframes as CSV
- **Strategy Comparison**: Run multiple prioritization strategies, compare KPIs (on-time rate, preemptions, utilization, avg turnaround), and inspect per-strategy details

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then:
1. Edit task parameters in the **Tasks** table (or leave default)
2. Click **Run simulation**
3. View the Gantt chart, segment log, task execution log, and event log
4. (Optional) Use the **Strategy Comparison** sidebar to compare different prioritization strategies

## Files

- `model.py` — Core simulation engine (Task/SystemState dataclasses, event-driven loop, MILP commit runner, reactive task selection)
- `app.py` — Streamlit dashboard (interactive editor, simulation runner, Gantt renderer, logs, strategy comparison)
- `requirements.txt` — Python package dependencies

## Example

The default example in `model.py` runs a 46-second simulation with:
- OIT1, OIT2, OIT3 (operator-independent tasks)
- NOIT1, NOIT2, NOIT3 (non-operator-independent, high-priority tasks)

Run the example standalone:
```bash
python3 model.py
```

Output shows event log with START / PREEMPT / COMPLETE events and final task execution summary.

## Simulation Logic

The simulator uses an **event-driven clock** that advances on:
1. Task arrivals (tasks enter the system)
2. Task completions (operator finishes a task)
3. Commit-window boundaries (MILP re-planning window triggers)
4. Committed scheduled starts (wakes the clock to start pre-planned tasks)

Tasks are **reactively selected** by the operator based on:
1. Resume residual NOITs (highest priority: task being re-started after preemption)
2. Start waiting NOITs (lower priority: enqueued NOITs waiting for operator)
3. Resume residual OITs (pre-empted OITs needing to resume)
4. Start committed OITs (tasks committed by MILP at their scheduled_start)
5. Start ready OITs (tasks ready by EST, not yet committed)

## License

MIT
