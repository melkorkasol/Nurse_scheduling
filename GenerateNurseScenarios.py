"""
Multi-scenario nurse profile + vacation generator
----------------------------------------------------
Ward: general medical/surgical, 30 nurses (18 RN / 12 SoSu)

Same 30 nurses (roles, contracts, night eligibility, seniority, shift
preference) are used across all scenarios -- these are staffing
attributes that don't change quarter to quarter in real life.

What DOES change per scenario is the vacation pattern, since different
14-week windows through the year fall in different Danish holiday
seasons. Four consecutive 14-week scenarios are generated, covering
56 weeks total (~1.08 years) since 52 doesn't divide evenly by 14:

  Scenario 1: ISO weeks 1-14,  2025 (2024-12-30 -> 2025-04-06) - winter/vinterferie
  Scenario 2: ISO weeks 15-28, 2025 (2025-04-07 -> 2025-07-13) - Easter + early summer
  Scenario 3: ISO weeks 29-42, 2025 (2025-07-14 -> 2025-10-19) - peak summer + efterarsferie
  Scenario 4: ISO weeks 43-56, 2025/26 (2025-10-20 -> 2026-01-25) - Christmas/New Year

Output: nurse_profiles.csv (shared), vacation_days_scenario{1-4}.csv,
and scenarios_manifest.json for driving a dropdown in the website.
"""

import datetime as dt
import json
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

N_NURSES = 30
N_RN = 18
N_SOSU = 12

SCENARIO_START = dt.date(2024, 12, 30)  # Monday, ISO week 1 of 2025
WEEKS_PER_SCENARIO = 13
N_SCENARIOS = 4

OUTPUT_DIR = "."
# ---------------------------------------------------------------------------
# 1. Shared nurse population — generated once, reused across all scenarios
#    (identical logic/parameters to the original single-scenario script)
# ---------------------------------------------------------------------------
nurse_ids = [f"N{str(i+1).zfill(3)}" for i in range(N_NURSES)]
roles = np.array(["RN"] * N_RN + ["SoSu"] * N_SOSU)
rng.shuffle(roles)

contract_levels = np.array([100, 80, 70, 50, 30])
contract_probs = np.array([0.45, 0.20, 0.15, 0.15, 0.05])
contract_pct = rng.choice(contract_levels, size=N_NURSES, p=contract_probs)

base_night_prob_by_contract = {100: 0.80, 80: 0.70, 70: 0.60, 50: 0.35, 30: 0.15}
night_eligible = []
for role, pct in zip(roles, contract_pct):
    p = base_night_prob_by_contract[pct]
    if role == "SoSu":
        p = min(p + 0.10, 0.95)
    night_eligible.append(rng.random() < p)
night_eligible = np.array(night_eligible)

years_experience = np.round(rng.beta(2, 5, size=N_NURSES) * 25, 1)

pref_levels = np.array(["No preference", "Day", "Evening", "Night"])
pref_probs = np.array([0.45, 0.30, 0.15, 0.10])
shift_preference = []
for elig in night_eligible:
    pref = rng.choice(pref_levels, p=pref_probs)
    if pref == "Night" and not elig:
        pref = "Day"
    shift_preference.append(pref)
shift_preference = np.array(shift_preference)

nurses = pd.DataFrame({
    "nurse_id": nurse_ids,
    "role": roles,
    "contract_pct": contract_pct,
    "night_eligible": night_eligible,
    "years_experience": years_experience,
    "shift_preference": shift_preference,
})
nurses["contracted_weekly_hours"] = round(37 * nurses["contract_pct"] / 100, 1)

# ---------------------------------------------------------------------------
# 6b. Per-nurse hourly wage — driven by role + years_experience, not a flat
#     role-only rate.
#     SOURCED (RN): DSR grundløn-by-anciennitet (real løntrin steps, before
#     tillæg/pension) — nyuddannet 25,382.67 kr/mo, 8 yrs 29,333 kr/mo,
#     10 yrs 30,738.99 kr/mo. Piecewise-linear between these; beyond 10
#     years the 8-10yr slope is extrapolated (ASSUMPTION -- no further
#     public løntrin anchor was found for higher seniority).
#     DERIVED (SoSu): no equivalent pure-grundløn anciennitet table was
#     found; a "+tillæg" SoSu overenskomst figure exists but mixing that
#     basis with RN's tillæg-free curve produced an unrealistic crossover
#     (SoSu overtaking RN pay at high seniority) when checked numerically.
#     Instead, SoSu wage = RN wage at the same experience x 0.780, the
#     ratio between the two roles' DST-sourced average base-pay figures
#     (188/241) -- this keeps role ordering correct at every experience
#     level while still being anciennitet-sensitive.
STD_MONTH_HOURS = 160.33  # DST's standard month, used consistently throughout
RN_ANCHORS = [(0, 25382.67), (8, 29333.0), (10, 30738.99)]  # (years, kr/month)
SOSU_RATIO = 188.0 / 241.0

def rn_hourly_wage(years):
    y0, m0 = RN_ANCHORS[0]
    y1, m1 = RN_ANCHORS[1]
    y2, m2 = RN_ANCHORS[2]
    if years <= y1:
        monthly = m0 + (m1 - m0) / (y1 - y0) * years
    elif years <= y2:
        monthly = m1 + (m2 - m1) / (y2 - y1) * (years - y1)
    else:
        slope = (m2 - m1) / (y2 - y1)
        monthly = m2 + slope * (years - y2)
    return monthly / STD_MONTH_HOURS

def hourly_wage(role, years):
    rn_wage = rn_hourly_wage(years)
    return round(rn_wage if role == "RN" else rn_wage * SOSU_RATIO, 2)

nurses["hourly_wage_dkk"] = [hourly_wage(r, y) for r, y in zip(nurses["role"], nurses["years_experience"])]
nurses_out = nurses.copy()
nurses_out["night_eligible"] = nurses_out["night_eligible"].map({True: "Yes", False: "No"})

# ---------------------------------------------------------------------------
# 2. Scenario windows (computed, guaranteed consecutive) + seasonal
#    vacation "boost windows" (ASSUMPTION: illustrative, not sourced —
#    modeled loosely on typical Danish school-holiday timing)
# ---------------------------------------------------------------------------
def add_weeks(d, weeks):
    return d + dt.timedelta(weeks=weeks)

scenarios = []
for i in range(N_SCENARIOS):
    start = add_weeks(SCENARIO_START, i * WEEKS_PER_SCENARIO)
    end = add_weeks(start, WEEKS_PER_SCENARIO) - dt.timedelta(days=1)
    scenarios.append({"index": i + 1, "start": start, "end": end})

# label + boost windows per scenario, expressed as (start_date, end_date, multiplier)
scenario_meta = {
    1: {
        "label": "Q1 (Winter — vinterferie & flu season)",
        "boosts": [(dt.date(2025, 2, 10), dt.date(2025, 2, 16), 3.5)],
    },
    2: {
        "label": "Q2 (Spring — Easter & pre-summer ramp-up)",
        "boosts": [
            (dt.date(2025, 4, 14), dt.date(2025, 4, 21), 3.0),   # Easter week 2025
            (dt.date(2025, 6, 16), dt.date(2025, 6, 29), 2.0),   # pre-summer ramp-up, now within Q2's window (ends Jun 29)
        ],
    },
    3: {
        "label": "Q3 (Summer — peak sommerferie)",
        "boosts": [
            (dt.date(2025, 7, 14), dt.date(2025, 8, 10), 4.0),   # peak summer holiday
        ],
    },
    4: {
        "label": "Q4 (Autumn/Winter — efterarsferie & Christmas)",
        "boosts": [
            (dt.date(2025, 10, 13), dt.date(2025, 10, 19), 3.0), # efterarsferie, ISO week 42 -- now falls in Q4's window
            (dt.date(2025, 12, 22), dt.date(2025, 12, 28), 4.0), # Christmas, trimmed to end within Q4 (which ends Dec 28)
        ],
    },
}

BASELINE_P = 0.03
TARGET_DAYS_RANGE = (5, 11)  # 5-10 inclusive, per scenario

manifest = {
    "shared_nurse_profiles": "nurse_profiles.csv",
    "scenarios": [],
}

for sc in scenarios:
    idx = sc["index"]
    meta = scenario_meta[idx]
    all_dates = [sc["start"] + dt.timedelta(days=d) for d in range((sc["end"] - sc["start"]).days + 1)]

    scenario_rng = np.random.default_rng(SEED + idx)  # distinct but reproducible per scenario

    vacation_rows = []
    for nid in nurse_ids:
        target_days = scenario_rng.integers(*TARGET_DAYS_RANGE)
        date_probs = []
        for d in all_dates:
            p = BASELINE_P
            for (bstart, bend, mult) in meta["boosts"]:
                if bstart <= d <= bend:
                    p *= mult
                    break
            date_probs.append(p)
        date_probs = np.array(date_probs)
        date_probs = date_probs / date_probs.sum()

        chosen = scenario_rng.choice(len(all_dates), size=min(target_days, len(all_dates)),
                                      replace=False, p=date_probs)
        for i in chosen:
            vacation_rows.append({"nurse_id": nid, "vacation_date": all_dates[i]})

    vacation = pd.DataFrame(vacation_rows).sort_values(["nurse_id", "vacation_date"]).reset_index(drop=True)

    vac_filename = f"vacation_days_scenario{idx}.csv"
    vacation.to_csv(f"{OUTPUT_DIR}\\{vac_filename}", index=False)

    manifest["scenarios"].append({
        "id": idx,
        "label": f"{meta['label']} — Weeks {(idx-1)*WEEKS_PER_SCENARIO+1}-{idx*WEEKS_PER_SCENARIO}",
        "start_date": sc["start"].isoformat(),
        "end_date": sc["end"].isoformat(),
        "vacation_file": vac_filename,
    })

    boosted_days = sum(1 for row in vacation.itertuples()
                        if any(b[0] <= row.vacation_date <= b[1] for b in meta["boosts"]))
    print(f"Scenario {idx} ({meta['label']}): {sc['start']} -> {sc['end']}, "
          f"{len(vacation)} vacation-days total, {boosted_days} in boosted windows "
          f"({100*boosted_days/len(vacation):.1f}%)")

# ---------------------------------------------------------------------------
# 3. Save shared nurse profiles + manifest
# ---------------------------------------------------------------------------
nurses_out.to_csv(f"{OUTPUT_DIR}\\nurse_profiles.csv", index=False)
with open(f"{OUTPUT_DIR}\\scenarios_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("\nShared nurse population:", len(nurses_out), "nurses")
print(nurses_out["role"].value_counts().to_dict())
print("\nManifest written with", len(manifest["scenarios"]), "scenarios.")
