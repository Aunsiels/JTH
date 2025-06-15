# JTH v1.0 – Job Tracking History

*A time-resolved, two-sided dataset for research on cold-start and temporal dynamics in job recommendation*

---

## 1  Overview

JTH contains five years (2018-01 → 2025-04) of real-world recruitment data collected by professional head-hunters in France.
It links **38 424 candidates**, **6 199 vacancies** and **45 546 multi-stage application traces**, each time-stamped at **day** granularity.
All records are pseudonymised and released for **non-commercial research** under **CC BY-NC 4.0**.

* **Home page / code**   [https://github.com/Aunsiels/JTH](https://github.com/Aunsiels/JTH)
* **Download (zip, XXX MB)**   `XXX`   (DOI coming)
* **Contact**   [julien.romero@telecom-sudparis.eu](mailto:julien.romero@telecom-sudparis.eu)

---

## 2  File list

| file             | rows   | size   | description                                           |
| ---------------- | ------ | ------ | ----------------------------------------------------- |
| `candidates.csv` | 38 424 | 11 MB  | static profile of each applicant                      |
| `jobs.csv`       | 6 199  | 1.3 MB | static profile of each vacancy                        |
| `history.csv`    | 45 546 | 2.3 MB | full, ordered timeline of every candidate–job process |

All files are **UTF-8**, comma-separated, with one header row and Unix line endings.

---

## 3  Schema

### 3.1 `candidates.csv`

| column                | type             | notes                                                       |
| --------------------- | ---------------- | ----------------------------------------------------------- |
| `candidate_id`        | `str`            | deterministic hash                                          |
| `create_date`         | `YYYY-MM-DD`     | profile creation                                            |
| `skills`              | `str`            | semicolon list, `_rare_skill_` sentinel for infrequent items |
| `expertise_area`      | `str`            | semicolon list, hand-assigned                               |
| `job_category`        | `str`            | semicolon list, recruiter-assigned                          |
| `years_experience`    | `float` \| `NaN` | tenure at creation                                          |
| `actual_salary`       | `float` €        | µ-aggregated + Laplace noise, negatives clipped to 0        |
| `actual_daily_salary` | `float` €        | same as above                                               |
| `contract_type`       | `str`            | semicolon list (e.g. *Permanent;Freelance*)                 |
| `zipcode`             | `str`            | two-digit French **département** code                       |
| `source`              | `str`            | acquisition channel (e.g. `LinkedIn`, `Website`)            |
| `languages`           | `str`            | semicolon list                                              |
| `hobbies`             | `str`            | semicolon list, `_rare_hobby_` sentinel                     |
| `sex`                 | `str`            | `M`/`F`/`Unknown`, inferred by LLM                                 |

### 3.2 `jobs.csv`

Columns mirror the candidate schema where meaningful; additional field:
`business_sector` – automatic tag from job description.

### 3.3 `history.csv`

Each row = one candidate–job pair.

| column                                                | type                  | meaning                         |
| ----------------------------------------------------- | --------------------- | ------------------------------- |
| `candidate_id` / `job_id`                             | `str`                 | foreign keys                    |
| `spontaneous_application_date` … `4th_interview_date` | `YYYY-MM-DD` \| `NaN` | chronological funnel steps      |
| `job_offer_proposed_date` / `job_offer_accepted_date` | `YYYY-MM-DD` \| `NaN` | final outcome                   |
| `end_of_process_date`                                 | `YYYY-MM-DD`          | last recorded action            |
| `last_stage_reached`                                  | `str`                 | label of deepest non-null stage |

All dates are in UTC, anonymised with Laplace noise (± ≤ 2 days, order preserved).

---

## 4  Missing-value rates

```
               candidates   jobs
creation_date        7 %     0 %
skills               5 %     0 %
expertise_area       8 %    30 %
job_category        27 %    30 %
years_experience    83 %    89 %
salary              94 %    34 %
daily_rate          98 %    68 %
contract_type       15 %     0 %
zipcode             41 %    68 %
```

*Empty cells* in CSVs are rendered as empty strings → `NaN` when read with `pandas`.

---

## 5  Quick start

### 5.1  Quick start (code)

```text
# 1.  Generate time-aware train / test splits (binary labels)
python code/utils/binary_splitter.py data/history.csv data/binary

# 2.  Run every baseline (models live in code/baseline/)
bash scripts/run_all_baselines.sh

# 3.  Evaluate all saved predictions → results.csv
bash scripts/evaluate_all.sh
```

*After step 3 you will find `results.csv` in the project root, containing MRR, NDCG\@K, P\@K, etc. for both directions.*

---

### 5.2  Quick start (example Python join)

Download the dataset and save it in the `data` directory.

```python
import pandas as pd

cand = pd.read_csv("data/candidates.csv", parse_dates=["create_date"])
jobs = pd.read_csv("data/jobs.csv", parse_dates=["create_date"])
hist = pd.read_csv("data/history.csv", parse_dates=[
    "spontaneous_application_date", "shortlist_date",
    "qualification_date", "resume_sent_to_company_date",
    "1st_interview_date", "2nd_interview_date",
    "3rd_interview_date", "4th_interview_date",
    "job_offer_proposed_date", "job_offer_accepted_date",
    "end_of_process_date"
])

# join candidate → job interactions
df = hist.merge(cand, on="candidate_id", how="left") \
         .merge(jobs, on="job_id", suffixes=("_cand", "_job"))
```

Reproducible baselines and evaluation scripts reside in `/code`.
Run **`./scripts/run_all.sh`** to reproduce paper tables.

---

## 6  Recommended tasks

* Cold-start ranking for **both** directions (job→candidate, candidate→job)
* Funnel-aware learning with stage weighting
* Heterogeneous-graph embedding over skills / contract types / sectors
* Temporal survival analysis (time-to-offer, vacancy expiration)

---

## 7  Licensing & permitted use

CC BY-NC 4.0 — research and teaching only.
Commercial exploitation or any attempt to re-identify individuals or companies is strictly forbidden.

---

## 8  Citation

```bibtex
@misc{XXX2025jth,
  title  = {JTH: A Dataset for Evaluating Cold-Start and Temporal Dynamics in Job Recommendation},
  author = {Millet, Yann and Behar, {\'E}ric and Romero, Julien},
  year   = {2025},
  note   = {Dataset v1.0, June 2025}
}
```

(DOI forthcoming.)

---

## 9  Change log

| version | date    | notes                                                           |
| ------- | ------- | --------------------------------------------------------------- |
| 1.0     | 2025-06 | initial public release; negative salaries clipped, README added |

Contributions and issues are welcome via GitHub pull requests or the **Issues** tracker.
