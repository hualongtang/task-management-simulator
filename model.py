# Second retry: fix committed scheduling and avoid unnecessary repeated MILP runs when nothing to do.
from dataclasses import dataclass, field
from typing import Optional, List
import pandas as pd
import heapq

@dataclass
class Task:
    task_id: str
    type: str
    duration: float
    EST: float
    deadline: Optional[float]
    arrival_time: float
    priority: Optional[int] = None
    start_time: Optional[float] = None   # actual start
    finish_time: Optional[float] = None  # actual finish
    scheduled_start: Optional[float] = None  # plan assigned by MILP (if any)
    remaining: Optional[float] = None
    is_committed: bool = False
    is_residual: bool = False
    original_task_id: Optional[str] = None
    def __post_init__(self):
        if self.remaining is None:
            self.remaining = float(self.duration)

@dataclass
class SystemState:
    now: float = 0.0
    commit_end: float = 0.0
    commit_window_length: float = 30.0
    current_task: Optional[Task] = None
    current_task_finish: Optional[float] = None
    oit_queue: List[Task] = field(default_factory=list)
    waiting_noit_queue: List = field(default_factory=list)
    residual_noit_stack: List[Task] = field(default_factory=list)
    finished_tasks: List[Task] = field(default_factory=list)
    event_log: List[str] = field(default_factory=list)
    task_log: List[dict] = field(default_factory=list)
    def log(self, msg: str):
        ts = f"[t={self.now:.1f}] "
        self.event_log.append(ts + msg)

def push_waiting_noit(state: SystemState, task: Task):
    heapq.heappush(state.waiting_noit_queue, (task.priority if task.priority is not None else 9999, task.arrival_time, task))

def pop_waiting_noit(state: SystemState) -> Optional[Task]:
    if not state.waiting_noit_queue:
        return None
    return heapq.heappop(state.waiting_noit_queue)[2]

def start_task(state: SystemState, task: Task):
    if task.start_time is None:
        task.start_time = state.now
    state.current_task = task
    state.current_task_finish = state.now + task.remaining
    state.log(f"START {task.task_id} type={task.type} rem={task.remaining:.1f} finish{state.current_task_finish:.1f}")

def complete_current_task(state: SystemState):
    t = state.current_task
    if t is None:
        return
    t.finish_time = state.now
    t.remaining = 0.0
    state.finished_tasks.append(t)
    state.log(f"COMPLETE {t.task_id} type={t.type}")
    state.task_log.append({
        "task_id": t.task_id, "type": t.type, "start": t.start_time, "finish": t.finish_time,
        "is_committed": t.is_committed, "is_residual": t.is_residual, "arrival": t.arrival_time,
        "scheduled_start": t.scheduled_start
    })
    state.current_task = None
    state.current_task_finish = None

def preempt_current_for_new_noit(state: SystemState, new_noit: Task):
    ct = state.current_task
    if ct is None:
        start_task(state, new_noit)
        return
    should_preempt = False
    if ct.type == "OIT":
        should_preempt = True
    elif ct.type == "NOIT":
        cur_p = ct.priority if ct.priority is not None else 9999
        new_p = new_noit.priority if new_noit.priority is not None else 9999
        if new_p < cur_p:
            should_preempt = True
    if should_preempt:
        remaining = max(0.0, state.current_task_finish - state.now)
        ct.remaining = remaining
        ct.is_residual = True
        state.log(f"PREEMPT {ct.task_id} rem={ct.remaining:.1f}")
        if ct.type == "NOIT":
            state.residual_noit_stack.append(ct)
        else:
            state.oit_queue.insert(0, ct)
        state.current_task = None
        state.current_task_finish = None
        start_task(state, new_noit)
    else:
        state.log(f"ENQUEUE NOIT {new_noit.task_id} prio={new_noit.priority}")
        push_waiting_noit(state, new_noit)

def handle_arrival(state: SystemState, task: Task):
    state.log(f"ARRIVAL {task.task_id} type={task.type} EST={task.EST} dur={task.duration} prio={task.priority}")
    if task.type == "OIT":
        state.oit_queue.append(task)
    else:
        if state.current_task is None:
            start_task(state, task)
        else:
            preempt_current_for_new_noit(state, task)

def reactive_handler(state: SystemState):
    if state.current_task is not None:
        return
    if state.residual_noit_stack:
        t = state.residual_noit_stack.pop()
        state.log(f"RH: resume residual NOIT {t.task_id}")
        start_task(state, t)
        return
    next_noit = pop_waiting_noit(state)
    if next_noit:
        state.log(f"RH: start waiting NOIT {next_noit.task_id}")
        start_task(state, next_noit)
        return
    for i,t in enumerate(state.oit_queue):
        if t.is_residual:
            state.oit_queue.pop(i)
            state.log(f"RH: resume residual OIT {t.task_id}")
            start_task(state, t)
            return
    # committed tasks: start if scheduled_start <= now and not started
    for t in state.oit_queue:
        if t.is_committed and (t.scheduled_start is not None) and (t.scheduled_start <= state.now) and (t.start_time is None):
            state.oit_queue.remove(t)
            state.log(f"RH: start committed OIT {t.task_id} scheduled@{t.scheduled_start:.1f}")
            start_task(state, t)
            return
    for i,t in enumerate(state.oit_queue):
        if (not t.is_committed) and (t.EST <= state.now):
            state.oit_queue.pop(i)
            state.log(f"RH: start ready OIT {t.task_id}")
            start_task(state, t)
            return
    state.log("RH: idle")

def run_milp_and_freeze(state: SystemState):
    commit_start = state.now
    commit_end_new = commit_start + state.commit_window_length
    # only run if there are OITs to consider (or waiting NOIT residuals to be aware of)
    if not state.oit_queue and not state.waiting_noit_queue and not state.residual_noit_stack:
        state.log("MILP run skipped (no OITs or NOITs to plan)")
        state.commit_end = commit_end_new
        return []
    state.log(f"MILP run -> new window [{commit_start:.1f},{commit_end_new:.1f})")
    op_available = commit_start
    if state.current_task is not None and state.current_task_finish is not None and state.current_task_finish > commit_start:
        op_available = state.current_task_finish
    candidates = sorted([t for t in state.oit_queue if not t.is_committed],
                        key=lambda x: (x.EST if x.EST is not None else 0, x.deadline if x.deadline is not None else 1e9, x.arrival_time))
    time_ptr = op_available
    scheduled = []
    for t in candidates:
        earliest = max(time_ptr, t.EST)
        finish = earliest + t.remaining
        if finish <= commit_end_new:
            t.scheduled_start = earliest
            t.is_committed = True
            time_ptr = finish
            scheduled.append(t)
            state.log(f"MILP commit {t.task_id} sched@{earliest:.1f} finish{finish:.1f}")
        else:
            state.log(f"MILP leave {t.task_id} uncommitted (earliest {earliest:.1f} finish {finish:.1f})")
    state.commit_end = commit_end_new
    return scheduled

def run_simulation(df_tasks: pd.DataFrame, commit_window=30.0, max_time=200.0):
    """
    Main simulation loop.
    - Loads task definitions
    - Uses an event-driven clock to process arrivals, finishes, and commit-window boundaries
    - Calls MILP only at commit-window boundaries
    - Uses reactive_handler for real-time task selection
    """

    # ------------------------------------------------------------
    # 1. Build Task objects from dataframe
    # ------------------------------------------------------------
    tasks = []
    for _, row in df_tasks.iterrows():
        tasks.append(Task(
            task_id=row['task_id'],
            type=row['type'],
            duration=float(row['duration']),
            EST=float(row['EST']) if pd.notna(row['EST']) else 0.0,
            deadline=float(row['deadline']) if pd.notna(row['deadline']) else None,
            arrival_time=float(row['arrival_time']),
            priority=int(row['priority']) if pd.notna(row['priority']) else None
        ))

    # ------------------------------------------------------------
    # 2. Build arrival event heap
    # ------------------------------------------------------------
    arrival_heap = [(t.arrival_time, i, t) for i, t in enumerate(tasks)]
    heapq.heapify(arrival_heap)

    # ------------------------------------------------------------
    # 3. Initialize system state
    # ------------------------------------------------------------
    state = SystemState(
        now=0.0,
        commit_end=0,
        commit_window_length=commit_window
    )

    # NOTE: We removed the unnecessary reactive_handler(state) at t=0.
    # At time 0, queues are empty — reactive handler would do nothing.
    # The first real invocation happens when the first task arrives.

    # ------------------------------------------------------------
    # 4. Main simulation loop
    # ------------------------------------------------------------
    steps = 0
    while True:
        steps += 1
        if steps > 20000:
            state.log("Simulation aborted: too many steps")
            break

        # Determine next event time: arrival, finish, commit boundary, or next scheduled start
        # Initialize defensively so NameError cannot occur if something unexpected
        # happens earlier in the loop (e.g., an early continue/return).
        next_arrival = float('inf')
        next_finish = float('inf')
        next_commit = float('inf')
        next_scheduled = float('inf')

        if arrival_heap:
            try:
                next_arrival = arrival_heap[0][0]
            except Exception:
                next_arrival = float('inf')

        if state.current_task_finish is not None:
            try:
                next_finish = state.current_task_finish
            except Exception:
                next_finish = float('inf')

        if state.commit_end is not None:
            try:
                next_commit = state.commit_end
            except Exception:
                next_commit = float('inf')

        # compute next scheduled start from committed tasks that haven't started
        try:
            committed_candidates = [
                t.scheduled_start for t in state.oit_queue
                if getattr(t, 'is_committed', False) and getattr(t, 'scheduled_start', None) is not None and (getattr(t, 'start_time', None) is None)
            ]
            if committed_candidates:
                # only consider scheduled starts strictly after now
                next_scheduled = min([float(s) for s in committed_candidates if float(s) > state.now] or [float('inf')])
        except Exception:
            next_scheduled = float('inf')

        next_time = min(next_arrival, next_finish, next_commit, next_scheduled, max_time)

        # Termination conditions
        if next_time == float('inf') or next_time > max_time:
            state.now = min(next_time, max_time)
            break
        if next_time < state.now - 1e-9:
            state.log("Time went backwards! abort")
            break

        # Advance time
        state.now = next_time

        # ------------------------------------------------------------
        # Process arrivals
        # ------------------------------------------------------------
        while arrival_heap and arrival_heap[0][0] <= state.now + 1e-9:
            _, _, task = heapq.heappop(arrival_heap)
            handle_arrival(state, task)

        # ------------------------------------------------------------
        # Process task completion
        # ------------------------------------------------------------
        if (
            state.current_task is not None and
            abs(state.current_task_finish - state.now) < 1e-9
        ):
            complete_current_task(state)
            reactive_handler(state)

        # ------------------------------------------------------------
        # Commit window boundary → run MILP
        # ------------------------------------------------------------
        if abs(state.commit_end - state.now) < 1e-9:

            # If literally nothing remains, simulation ends
            if (
                not arrival_heap and not state.oit_queue and
                not state.waiting_noit_queue and not state.residual_noit_stack and
                state.current_task is None
            ):
                state.log("No tasks left at commit boundary -> terminating")
                break

            # Run MILP to produce new committed tasks for this window
            run_milp_and_freeze(state)

            # After freezing, if operator is idle, decide next task
            if state.current_task is None:
                reactive_handler(state)

        # If we advanced specifically to a committed task's scheduled start
        # (i.e. next_time == next_scheduled), ensure the reactive handler
        # runs so the committed OIT actually starts. Without this the
        # clock could advance to the scheduled time but nothing would
        # start because reactive selection wasn't invoked for that wake.
        if state.current_task is None:
            try:
                # next_scheduled is defined earlier in the loop
                if next_scheduled is not None and next_scheduled != float('inf'):
                    if abs(state.now - next_scheduled) < 1e-9 or state.now >= next_scheduled:
                        reactive_handler(state)
            except NameError:
                # defensive: if next_scheduled isn't in scope, skip
                pass

        # ------------------------------------------------------------
        # Terminate if nothing remains to happen
        # ------------------------------------------------------------
        if (
            not arrival_heap and state.current_task is None and
            not state.oit_queue and not state.waiting_noit_queue and
            not state.residual_noit_stack
        ):
            state.log("No pending events -> terminating")
            break

    # ------------------------------------------------------------
    # If sim ends while a task is still running → finish it
    # ------------------------------------------------------------
    if state.current_task is not None:
        state.now = state.current_task_finish
        complete_current_task(state)

    # ------------------------------------------------------------
    # Export logs
    # ------------------------------------------------------------
    tasks_df = pd.DataFrame(state.task_log)
    events_df = pd.DataFrame({"event": state.event_log})

    return state, tasks_df, events_df




# Run on provided data
# (arrival_time, task_id, type, duration, EST, deadline, priority)
data = [
    (0,   "OIT1",  "OIT",  20,  0,  50, None),
    (5,   "NOIT1", "NOIT", 6,   5,  None, 1),
    (7,   "NOIT2", "NOIT", 4,   7,  None, 2),
    (12,  "OIT2",  "OIT",  8,  20, 30, None),
    (22,  "NOIT3", "NOIT", 3,  22, None, 1),
    (35,  "OIT3",  "OIT",  5,  40, 60, None),
]
df = pd.DataFrame(data, columns=[
    "arrival_time","task_id","type","duration","EST","deadline","priority"
]).sort_values("arrival_time").reset_index(drop=True)

if __name__ == "__main__":
    state, tasks_df, events_df = run_simulation(df, commit_window=30.0, max_time=200.0)

    # import caas_jupyter_tools as cjt
    # cjt.display_dataframe_to_user("Task execution log", tasks_df)
    # cjt.display_dataframe_to_user("Event log", events_df)

    print("SIM DONE time", state.now)
    print("Events (first 50):")
    for e in state.event_log[:80]:
        print(e)














# # plots
# import matplotlib.pyplot as plt
# import pandas as pd

# # Reconstruct tasks from the user's tasks_df example
# data = [
#     ("NOIT1","NOIT",5,11),
#     ("NOIT2","NOIT",11,15),
#     ("NOIT3","NOIT",22,25),
#     ("OIT1","OIT",0,5),
#     ("OIT1","OIT",15,22),
#     ("OIT1","OIT",25,33),
#     ("OIT2","OIT",33,41),
#     ("OIT3","OIT",41,46)
# ]

# df = pd.DataFrame(data, columns=["task_id","type","start","finish"])

# # Assign numeric y positions
# unique_tasks = df["task_id"].unique()
# y_pos = {task:i for i,task in enumerate(unique_tasks)}

# # Plot
# plt.figure(figsize=(12,6))
# for _, row in df.iterrows():
#     y = y_pos[row["task_id"]]
#     plt.barh(
#         y, 
#         row["finish"] - row["start"], 
#         left=row["start"],
#         edgecolor="black"
#     )
#     plt.text(
#         (row["start"] + row["finish"]) / 2,
#         y,
#         row["task_id"],
#         va="center",
#         ha="center"
#     )

# plt.yticks(list(y_pos.values()), list(y_pos.keys()))
# plt.xlabel("Time (s)")
# plt.title("Gantt Chart of Task Execution")
# plt.tight_layout()
# plt.show()
