import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

nurses = pd.read_csv('/home/claude/nurse_scenarios/nurse_profiles.csv')

N = list(nurses.nurse_id)
n_idx = {n: i for i, n in enumerate(N)}
n_nurses = len(N)
n_days = 91          # 13 weeks
shifts = ['Day', 'Evening', 'Night']
s_idx = {s: i for i, s in enumerate(shifts)}
n_shift_types = 3
n_vars = n_nurses * n_days * n_shift_types

def vi(n, d, s):
    return n_idx[n] * n_days * n_shift_types + (d - 1) * n_shift_types + s_idx[s]

role = dict(zip(nurses.nurse_id, nurses.role))
night_eligible = dict(zip(nurses.nurse_id, nurses.night_eligible == 'Yes'))
contracted_hrs = dict(zip(nurses.nurse_id, nurses.contracted_weekly_hours))

RN_NURSES = [n for n in N if role[n] == 'RN']
TOTAL_MIN = {'Day': 8, 'Evening': 5, 'Night': 3}
RN_MIN = {'Day': 5, 'Evening': 3, 'Night': 2}
FORBIDDEN_PAIRS = [('Evening', 'Day'), ('Night', 'Day'), ('Night', 'Evening')]
WEEKS = list(range(1, 14))  # 13 weeks
def days_in_week(w):
    return list(range((w - 1) * 7 + 1, w * 7 + 1))
SHIFT_HOURS = 8.0
TOL = 0.15

QUARTER_STARTS = {
    1: dt.date(2024, 12, 30),
    2: dt.date(2025, 3, 31),
    3: dt.date(2025, 6, 30),
    4: dt.date(2025, 9, 29),
}

def check_quarter(qid):
    start = QUARTER_STARTS[qid]
    vacation = pd.read_csv(f'/home/claude/nurse_scenarios/vacation_days_scenario{qid}.csv')
    vacation['vacation_date'] = pd.to_datetime(vacation['vacation_date']).dt.date
    vac_by_date = vacation.groupby('vacation_date')['nurse_id'].apply(set).to_dict()

    rows, lbs, ubs = [], [], []
    def add(cd, lb, ub):
        rows.append(cd); lbs.append(lb); ubs.append(ub)

    for n in N:
        for d in range(1, n_days + 1):
            add({vi(n, d, s): 1.0 for s in shifts}, 0, 1)

    for d in range(1, n_days + 1):
        for s in shifts:
            add({vi(n, d, s): 1.0 for n in N}, TOTAL_MIN[s], n_nurses)
            add({vi(n, d, s): 1.0 for n in RN_NURSES}, RN_MIN[s], len(RN_NURSES))

    for n in N:
        if not night_eligible[n]:
            for d in range(1, n_days + 1):
                add({vi(n, d, 'Night'): 1.0}, 0, 0)

    for d_obj, nset in vac_by_date.items():
        d = (d_obj - start).days + 1
        if 1 <= d <= n_days:
            for n in nset:
                for s in shifts:
                    add({vi(n, d, s): 1.0}, 0, 0)

    for n in N:
        for d in range(1, n_days):
            for (s1, s2) in FORBIDDEN_PAIRS:
                add({vi(n, d, s1): 1.0, vi(n, d + 1, s2): 1.0}, 0, 1)

    for n in N:
        for w in WEEKS:
            cd = {vi(n, d, s): 1.0 for d in days_in_week(w) for s in shifts}
            add(cd, 0, 6)

    for n in N:
        for w in WEEKS:
            cd = {vi(n, d, s): SHIFT_HOURS for d in days_in_week(w) for s in shifts}
            add(cd, 0, 48)

    for n in N:
        cd = {vi(n, d, s): SHIFT_HOURS for d in range(1, n_days + 1) for s in shifts}
        target = contracted_hrs[n] * len(WEEKS)
        add(cd, target * (1 - TOL), target * (1 + TOL))

    row_idx, col_idx, data = [], [], []
    for i, cd in enumerate(rows):
        for col, val in cd.items():
            row_idx.append(i); col_idx.append(col); data.append(val)
    A = csr_matrix((data, (row_idx, col_idx)), shape=(len(rows), n_vars))
    lb_arr = np.array(lbs, dtype=float); ub_arr = np.array(ubs, dtype=float)
    constraint = LinearConstraint(A, lb_arr, ub_arr)
    bounds = Bounds(0, 1)
    integrality = np.ones(n_vars)
    c = np.zeros(n_vars)

    res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds,
               options={'time_limit': 180})
    status = "FEASIBLE" if res.status == 0 else f"INFEASIBLE (status {res.status})"
    print(f"Quarter {qid} ({start} + 91 days): {status}")
    return res.status == 0

results = {}
for qid in [1, 2, 3, 4]:
    results[qid] = check_quarter(qid)

print()
print("Summary:", results)
