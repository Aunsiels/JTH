# JTH v1.0 – Job Tracking History

*A time-resolved, two-sided dataset for research on cold-start and temporal dynamics in job recommendation*

---

## 1  Overview

JTH contains five years (2018-01 → 2025-04) of real-world recruitment data collected by professional head-hunters in France.
It links **37,554 candidates**, **6,011 vacancies** and **42,288 multi-stage application traces**, each time-stamped at **day** granularity.
All records are pseudonymised and released for **non-commercial research** under **CC BY-NC 4.0**.

* **Download (zip, 24 MB)**   `XXX`   (DOI coming)

---

## 2  File list

| file             | rows   | size   | description                                           |
| ---------------- | ------ | ------ | ----------------------------------------------------- |
| `candidates.csv` | 37,554 | 19 MB  | static profile of each applicant                      |
| `jobs.csv`       | 6,011  | 2.3 MB | static profile of each vacancy                        |
| `history.csv`    | 42,288 | 2.7 MB | full, ordered timeline of every candidate–job process |

All files are **UTF-8**, comma-separated, with one header row and Unix line endings.

---

## 3  Schema

### 3.1 `candidates.csv`

| column                              | type              | notes                                                                 |
| ----------------------------------- | ----------------- | --------------------------------------------------------------------- |
| `candidate_id`                      | `str`             | deterministic hash                                                    |
| `create_date`                       | `YYYY-MM-DD`      | profile creation date                                                 |
| `skills`                            | `str`             | semicolon list of manual skills; `_rare_skill_` sentinel for infrequent ones |
| `expertise_area`                   | `str`             | manual area(s) of expertise; semicolon list                          |
| `job_category`                      | `str`             | manual job category(ies); semicolon list                             |
| `years_experience`                 | `str` or `NaN`    | binned: `0-2 years`, `3-6 years`, `+6 years`                          |
| `actual_salary`                     | `float` €         | manually entered; µ-aggregated + Laplace noise                       |
| `actual_daily_salary`              | `float` €         | same as above                                                         |
| `contract_type`                     | `str`             | semicolon list (e.g. `Permanent;Freelance`)                          |
| `zipcode`                           | `str`             | French département code (2-digit)                                     |
| `source`                            | `str`             | acquisition channel, anonymized                                      |

**LLM-Inferred Fields:**

| column                              | type              | notes                                                                 |
| ----------------------------------- | ----------------- | --------------------------------------------------------------------- |
| `llm_sex`                            | `str`             | inferred sex (`Male`, `Female`)                                       |
| `llm_nationality`                   | `str`             | inferred nationality; `_rare_nationality_` for infrequent items       |
| `llm_languages_spoken`             | `str`             | inferred list                                                         |
| `llm_highest_diploma`              | `str`             | e.g. `Bachelor`, `Master`, `PhD`                                      |
| `llm_from_top_university`          | `bool`            | top institution indicator                                              |
| `llm_multiple_degrees`             | `bool`            | whether has multiple diplomas                                         |
| `llm_ongoing_education`            | `bool`            | whether still studying                                                 |
| `llm_years_of_work_experience`     | `str`             | binned years: e.g. `2-5`, `10-15`                                      |
| `llm_number_of_previous_positions` | `int`             | count of prior jobs                                                    |
| `llm_number_of_unique_employers`   | `int`             | count of distinct employers                                            |
| `llm_management_experience`        | `bool`            | management role experience                                             |
| `llm_years_of_management_roles`    | `str`             | binned, e.g. `0-2`, `10-15`                                            |
| `llm_industry_domains`             | `str`             | semicolon list of industry sectors                                     |
| `llm_entrepreneurial_experience`   | `bool`            | entrepreneurship indicator                                             |
| `llm_startup_experience`           | `bool`            | startup background indicator                                           |
| `llm_large_company_experience`     | `bool`            | large company background indicator                                     |
| `llm_freelance_experience`         | `bool`            | freelance experience                                                   |
| `llm_contract_experience`          | `bool`            | contract job experience                                                |
| `llm_international_work_experience`| `bool`            | worked internationally                                                 |
| `llm_remote_work_experience`       | `bool`            | remote work history                                                    |
| `llm_average_job_duration_months`  | `str`             | binned duration (e.g. `12-24`)                                         |
| `llm_number_of_career_gaps`        | `int`             | number of gaps                                                         |
| `llm_leadership_experience`        | `bool`            | leadership role experience                                             |
| `llm_hard_skills`                  | `str`             | inferred hard skills list                                              |
| `llm_soft_skills`                  | `str`             | inferred soft skills list                                              |
| `llm_programming_languages`        | `str`             | inferred programming languages                                         |
| `llm_tools_technologies`           | `str`             | inferred tools used                                                    |
| `llm_certifications`               | `str`             | inferred certifications list                                           |
| `llm_volunteer_experience`         | `bool`            | volunteer activity                                                     |
| `llm_has_publications`             | `bool`            | publication record                                                     |
| `llm_has_patents`                  | `bool`            | patent record                                                          |
| `llm_has_portfolio_or_github_or_website` | `bool`       | visible portfolio presence                                             |
| `llm_has_linkedin`                 | `bool`            | contains LinkedIn link                                                 |
| `llm_participated_in_competitions` | `bool`            | e.g. Kaggle, hackathons                                                |
| `llm_hobbies`                      | `str`             | inferred hobbies list                                                  |
| `llm_expertise_area`               | `str`             | inferred area(s) of expertise                                          |
| `llm_job_category`                 | `str`             | inferred job category(ies)                                             |
| `llm_estimated_salary_eur_per_year`| `float` €         | estimated annual salary                                                |
| `llm_client_facing_role`           | `bool`            | customer-facing role experience                                        |
| `llm_values`                       | `str`             | inferred personal values                                               |
| `llm_seniority_level`              | `str`             | `Intern`, `Junior`, `Mid`, `Senior`                                   |
| `llm_age_bucket`                   | `str`             | e.g. `30-40`, `40-50`, `>=70`                                          |
| `llm_graduation_recency`          | `str`             | time since last graduation, binned                                     |


### 3.2 `jobs.csv`

| column                             | type              | notes                                                              |
| ---------------------------------- | ----------------- | ------------------------------------------------------------------ |
| `job_id`                           | `str`             | deterministic hash                                                 |
| `create_date`                      | `YYYY-MM-DD`      | job creation date                                                  |
| `expertise_area`                  | `str`             | manual expertise area(s); semicolon list                          |
| `job_category`                     | `str`             | manual job category(ies); semicolon list                          |
| `skills`                           | `str`             | manual skills list; `_rare_skill_` sentinel                        |
| `contract_type`                    | `str`             | e.g. `Permanent`, `Freelance`                                     |
| `years_experience`                | `str` or `NaN`    | binned: `0-2`, `3-6`, `+6 years`                                   |
| `zipcode`                          | `str`             | département code; may be empty                                     |
| `salary`                           | `float` €         | manually entered                                                   |
| `daily_rate`                       | `float` €         | manually entered                                                   |
| `source`                           | `str`             | anonymized source code                                             |

**LLM-Inferred Fields:**

| column                             | type              | notes                                                              |
| ---------------------------------- | ----------------- | ------------------------------------------------------------------ |
| `llm_remote_possible`              | `bool`            | remote option                                                       |
| `llm_industry_domains`            | `str`             | sector list                                                         |
| `llm_company_values`              | `str`             | inferred company values                                             |
| `llm_seniority_level`             | `str`             | e.g. `Mid`, `Senior`                                                |
| `llm_salary_estimation_eur_per_year` | `float` €       | inferred                                                            |
| `llm_required_languages_spoken`   | `str`             | required spoken languages                                           |
| `llm_required_lowest_diploma`     | `str`             | e.g. `Bachelor`, `PhD`                                              |
| `llm_required_years_of_work_experience` | `int`        | required years                                                      |
| `llm_required_management_experience` | `bool`          | boolean                                                             |
| `llm_is_startup`                  | `bool`            | startup flag                                                        |
| `llm_is_large_company`            | `bool`            | large company flag                                                  |
| `llm_required_freelance_experience` | `bool`           | boolean                                                             |
| `llm_required_contract_experience` | `bool`           | boolean                                                             |
| `llm_required_international_work_experience` | `bool`   | boolean                                                             |
| `llm_required_leadership_experience` | `bool`         | boolean                                                             |
| `llm_hard_skills`                 | `str`             | required hard skills                                                |
| `llm_soft_skills`                 | `str`             | required soft skills                                                |
| `llm_programming_languages`       | `str`             | required programming languages                                      |
| `llm_tools_technologies`          | `str`             | required tools                                                      |
| `llm_certifications`              | `str`             | required certifications                                             |
| `llm_expertise_area`             | `str`             | inferred areas of expertise                                         |
| `llm_job_category`               | `str`             | inferred job category(ies)                                          |
| `llm_client_facing_role`          | `bool`            | client-facing role indicator                                        |


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

### 4 Regression features

Computed features for each candidate–job pair.

| column                       | type         | notes                                              |
| ----------------------------| ------------ | -------------------------------------------------- |
| `skill_coverage`            | `float`      | manual skill match ratio                           |
| `hard_skill_coverage`       | `float`      | LLM-inferred hard skills match                     |
| `soft_skill_coverage`       | `float`      | LLM-inferred soft skills match                     |
| `prog_language_coverage`    | `float`      | LLM-inferred programming language match            |
| `tool_coverage`             | `float`      | tool/tech match                                    |
| `certification_coverage`    | `float`      | match on certifications                            |
| `expertise_area_coverage`   | `float`      | manual coverage match                              |
| `job_category_coverage`     | `float`      | manual coverage match                              |
| `expertise_area_llm_coverage` | `float`    | LLM-based coverage                                 |
| `job_category_llm_coverage` | `float`      | LLM-based coverage                                 |
| `industry_coverage`         | `float`      | match in industry domains                          |
| `values_coverage`           | `float`      | personal/company values match                      |
| `language_coverage`         | `float`      | spoken language match                              |
| `contract_match`            | `bool`       | exact contract compatibility                       |
| `remote_match`              | `bool`       | remote condition match                             |
| `freelance_match`           | `bool`       | freelance compatibility                            |
| `with_contract_match`       | `bool`       | any contract match                                 |
| `years_bucket_match`        | `bool`       | manual experience years match                      |
| `management_match`          | `bool`       | experience vs requirement                          |
| `leadership_match`          | `bool`       | leadership alignment                               |
| `startup_match`             | `bool`       | startup compatibility                              |
| `large_company_match`       | `bool`       | large company experience match                     |
| `international_exp_match`   | `bool`       | international work experience                      |
| `client_role_match`         | `bool`       | client-facing role match                           |
| `seniority_gap`             | `int`        | numeric difference in seniority levels             |
| `diploma_match`             | `bool`       | diploma requirement met                            |
| `salary_gap`                | `float` €    | difference between entered salaries                |
| `daily_gap`                 | `float` €    | difference between daily rates                     |
| `est_salary_gap`            | `float` €    | inferred salary gap                                |
| `same_zip`                  | `bool`       | same département                                   |
| `work_years_match`          | `bool`       | match on inferred years of work                    |
| `top_university`            | `bool`       | candidate from elite school                        |
| `multi_degree`              | `bool`       | candidate has several degrees                      |
| `ongoing_study`             | `bool`       | still studying                                     |
| `entrepreneurial_exp`       | `bool`       | entrepreneurial experience                         |
| `volunteer`                 | `bool`       | volunteer activities present                        |
| `has_publications`          | `bool`       | candidate has published                            |
| `has_patents`               | `bool`       | candidate holds patents                            |
| `visible_portfolio`         | `bool`       | portfolio or GitHub link present                   |
| `has_linkedin`              | `bool`       | LinkedIn link in CV                                |
| `entered_competitions`      | `bool`       | participated in competitions                       |
| `n_prev_positions`          | `int`        | number of prior roles                              |
| `n_unique_employers`        | `int`        | count of unique employers                          |
| `years_mgmt_roles`          | `int`        | years in management                                |
| `avg_job_duration`          | `float`      | months                                              |
| `n_career_gaps`             | `int`        | number of gaps                                     |
| `entity_delta_days`         | `int`        | job - candidate creation                           |
| `rec_delta_days`            | `int`        | recommendation date - job                          |
| `existed_before`            | `bool`       | candidate existed before recommendation            |


---

## 5  Missing-value rates

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

## 6  Quick start

### 6.1  Quick start (code)

```text
# 1.  Generate time-aware train / test splits (binary labels)
python code/utils/binary_splitter.py data/history.csv data/binary

# 2.  Run every baseline (models live in code/baseline/)
bash scripts/run_all_baselines.sh data results

# 3.  Evaluate all saved predictions → results.csv
bash scripts/evaluate_all.sh data results
```

*After step 3 you will find `results.csv` in the result directory, containing MRR, NDCG\@K, P\@K, etc. for both directions.*

---

### 6.2  Quick start (example Python join)

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
Run **`./scripts/run_all_baselines.sh`** and **`./scripts/evaluate_all.sh`** to reproduce paper tables.

---

## 7  Recommended tasks

* Cold-start ranking for **both** directions (job→candidate, candidate→job)
* Funnel-aware learning with stage weighting
* Heterogeneous-graph embedding over skills / contract types / sectors
* Temporal survival analysis (time-to-offer, vacancy expiration)

---

## 8  Licensing & permitted use

CC BY-NC 4.0 — research and teaching only.
Commercial exploitation or any attempt to re-identify individuals or companies is strictly forbidden.

---

## 9  Citation

```bibtex
@misc{XXX2025jth,
  title  = {JTH: A Dataset for Evaluating Cold-Start and Temporal Dynamics in Job Recommendation},
  author = {Anon.},
  year   = {2025},
  note   = {Dataset v1.0, June 2025}
}
```

(DOI forthcoming.)

---

## 10  Change log

| version | date    | notes                                                           |
| ------- | ------- | --------------------------------------------------------------- |
| 1.0     | 2025-08 | initial public release; negative salaries clipped, README added |

Contributions and issues are welcome via GitHub pull requests or the **Issues** tracker.
