# JTH v1.1 – Job Tracking History

*A time-resolved, two-sided dataset for research on cold-start and temporal dynamics in job recommendation*

---

## 1 Overview

JTH contains seven years (2018-03 → 2025-04) of real-world recruitment data curated by professional head-hunters in France.
It links **37,554 candidates**, **6,011 vacancies** and **42,288 multi-stage application traces**, encompassing **78,195 specific recruiting events** each time-stamped at **day** granularity. 

This dataset serves as the official artifact for the **RecSys 2026** paper: *JTH: A Dataset for Evaluating Cold-Start and Temporal Dynamics in Job Recommendation*. 

All records are rigorously pseudonymised (incorporating k=5 anonymity for quasi-identifiers, clipping extreme numerical outliers at the 95th percentile, and applying a 5-day Laplace noise to timestamps) and released for **non-commercial research** under **CC BY-NC 4.0**.

* **Dataset Download (Zenodo)**: [https://zenodo.org/records/20020466](https://zenodo.org/records/20020466)
* **Code Repository**: [https://github.com/Aunsiels/JTH](https://github.com/Aunsiels/JTH)
* **Paper DOI**: [https://doi.org/10.1145/3773078.3831846](https://doi.org/10.1145/3773078.3831846)

---

## 2 File list

| file             | rows   | size   | description                                           |
| ---------------- | ------ | ------ | ----------------------------------------------------- |
| `candidates.csv` | 37,554 | 19 MB  | static profile of each applicant                      |
| `jobs.csv`       | 6,011  | 2.3 MB | static profile of each vacancy                        |
| `history.csv`    | 42,288 | 2.7 MB | full, ordered timeline of every candidate–job process |

All files are **UTF-8**, comma-separated, with one header row and Unix line endings.

---

## 3 Schema

### 3.1 `candidates.csv`

| column                              | type              | notes                                                                 |
| ----------------------------------- | ----------------- | --------------------------------------------------------------------- |
| `candidate_id`                      | `str`             | deterministic hash                                                    |
| `create_date`                       | `YYYY-MM-DD`      | profile creation date                                                 |
| `skills`                            | `str`             | semicolon list of manual skills; `_rare_skill_` sentinel for infrequent ones |
| `expertise_area`                    | `str`             | manual area(s) of expertise; semicolon list                           |
| `job_category`                      | `str`             | manual job category(ies); semicolon list                               |
| `years_experience`                  | `str` or `NaN`    | binned: `0-2 years`, `3-6 years`, `+6 years`                          |
| `actual_salary`                     | `float` €         | manually entered; µ-aggregated + Laplace noise                        |
| `actual_daily_salary`               | `float` €         | same as above                                                         |
| `contract_type`                     | `str`             | semicolon list (e.g. `Permanent;Freelance`)                           |
| `zipcode`                           | `str`             | French département code (2-digit, coarsened)                          |
| `source`                            | `str`             | acquisition channel, anonymized                                       |

**LLM-Inferred Fields (Extracted using Llama 3.3 80B):**

| column                              | type              | notes                                                                 |
| ----------------------------------- | ----------------- | --------------------------------------------------------------------- |
| `llm_sex`                           | `str`             | inferred sex (`Male`, `Female`)                                       |
| `llm_nationality`                   | `str`             | inferred nationality; `_rare_nationality_` for infrequent items       |
| `llm_languages_spoken`              | `str`             | inferred list                                                         |
| `llm_highest_diploma`               | `str`             | e.g. `Bachelor`, `Master`, `PhD`                                      |
| `llm_from_top_university`           | `bool`            | top institution indicator                                              |
| `llm_multiple_degrees`              | `bool`            | whether has multiple diplomas                                         |
| `llm_ongoing_education`             | `bool`            | whether still studying                                                 |
| `llm_years_of_work_experience`      | `str`             | binned years: e.g. `2-5`, `10-15`                                      |
| `llm_number_of_previous_positions`  | `int`             | count of prior jobs                                                   |
| `llm_number_of_unique_employers`    | `int`             | count of distinct employers                                            |
| `llm_management_experience`         | `bool`            | management role experience                                             |
| `llm_years_of_management_roles`     | `str`             | binned, e.g. `0-2`, `10-15`                                            |
| `llm_industry_domains`              | `str`             | semicolon list of industry sectors                                     |
| `llm_entrepreneurial_experience`    | `bool`            | entrepreneurship indicator                                             |
| `llm_startup_experience`            | `bool`            | startup background indicator                                           |
| `llm_large_company_experience`      | `bool`            | large company background indicator                                     |
| `llm_freelance_experience`          | `bool`            | freelance experience                                                   |
| `llm_contract_experience`           | `bool`            | contract job experience                                                |
| `llm_international_work_experience` | `bool`            | worked internationally                                                 |
| `llm_remote_work_experience`        | `bool`            | remote work history                                                    |
| `llm_average_job_duration_months`   | `str`             | binned duration (e.g. `12-24`)                                         |
| `llm_number_of_career_gaps`         | `int`             | number of gaps                                                         |
| `llm_leadership_experience`         | `bool`            | leadership role experience                                             |
| `llm_hard_skills`                   | `str`             | inferred hard skills list                                              |
| `llm_soft_skills`                   | `str`             | inferred soft skills list                                              |
| `llm_programming_languages`         | `str`             | inferred programming languages                                         |
| `llm_tools_technologies`            | `str`             | inferred tools used                                                    |
| `llm_certifications`                | `str`             | inferred certifications list                                           |
| `llm_volunteer_experience`          | `bool`            | volunteer activity                                                     |
| `llm_has_publications`              | `bool`            | publication record                                                     |
| `llm_has_patents`                   | `bool`            | patent record                                                          |
| `llm_has_portfolio_or_github_or_website` | `bool`        | visible portfolio presence                                             |
| `llm_has_linkedin`                  | `bool`            | contains LinkedIn link                                                 |
| `llm_participated_in_competitions`  | `bool`            | e.g. Kaggle, hackathons                                                |
| `llm_hobbies`                       | `str`             | inferred hobbies list                                                  |
| `llm_expertise_area`                | `str`             | inferred area(s) of expertise                                          |
| `llm_job_category`                  | `str`             | inferred job category(ies)                                             |
| `llm_estimated_salary_eur_per_year` | `float` €         | estimated annual salary                                                |
| `llm_client_facing_role`            | `bool`            | customer-facing role experience                                        |
| `llm_values`                        | `str`             | inferred personal values                                               |
| `llm_seniority_level`               | `str`             | `Intern`, `Junior`, `Mid`, `Senior`                                    |
| `llm_age_bucket`                    | `str`             | e.g. `30-40`, `40-50`, `>=70`                                          |
| `llm_graduation_recency`            | `str`             | time since last graduation, binned                                     |

### 3.2 `jobs.csv`

| column                              | type              | notes                                                              |
| ---------------------------------- | ----------------- | ------------------------------------------------------------------ |
| `job_id`                           | `str`             | deterministic hash                                                 |
| `create_date`                      | `YYYY-MM-DD`      | job creation date                                                  |
| `expertise_area`                   | `str`             | manual expertise area(s); semicolon list                           |
| `job_category`                     | `str`             | manual job category(ies); semicolon list                           |
| `skills`                           | `str`             | manual skills list; `_rare_skill_` sentinel                        |
| `contract_type`                    | `str`             | e.g. `Permanent`, `Freelance`                                      |
| `years_experience`                 | `str` or `NaN`    | binned: `0-2`, `3-6`, `+6 years`                                   |
| `zipcode`                          | `str`             | département code; may be empty                                     |
| `salary`                           | `float` €         | manually entered                                                   |
| `daily_rate`                       | `float` €         | manually entered                                                   |
| `source`                           | `str`             | anonymized source code                                             |

**LLM-Inferred Fields (Extracted using Llama 3.3 80B):**

| column                              | type              | notes                                                              |
| ---------------------------------- | ----------------- | ------------------------------------------------------------------ |
| `llm_remote_possible`              | `bool`            | remote option                                                      |
| `llm_industry_domains`             | `str`             | sector list                                                        |
| `llm_company_values`               | `str`             | inferred company values                                            |
| `llm_seniority_level`              | `str`             | e.g. `Mid`, `Senior`                                               |
| `llm_salary_estimation_eur_per_year` | `float` €       | inferred                                                           |
| `llm_required_languages_spoken`    | `str`             | required spoken languages                                          |
| `llm_required_lowest_diploma`      | `str`             | e.g. `Bachelor`, `PhD`                                             |
| `llm_required_years_of_work_experience` | `int`        | required years                                                     |
| `llm_required_management_experience` | `bool`          | boolean                                                            |
| `llm_is_startup`                   | `bool`            | startup flag                                                       |
| `llm_is_large_company`             | `bool`            | large company flag                                                 |
| `llm_required_freelance_experience` | `bool`           | boolean                                                            |
| `llm_required_contract_experience` | `bool`            | boolean                                                            |
| `llm_required_international_work_experience` | `bool`  | boolean                                                            |
| `llm_required_leadership_experience` | `bool`          | boolean                                                            |
| `llm_hard_skills`                  | `str`             | required hard skills                                               |
| `llm_soft_skills`                  | `str`             | required soft skills                                               |
| `llm_programming_languages`        | `str`             | required programming languages                                     |
| `llm_tools_technologies`           | `str`             | required tools                                                     |
| `llm_certifications`               | `str`             | required certifications                                            |
| `llm_expertise_area`               | `str`             | inferred areas of expertise                                        |
| `llm_job_category`                 | `str`             | inferred job category(ies)                                         |
| `llm_client_facing_role`           | `bool`            | client-facing role indicator                                       |

### 3.3 `history.csv`

Each row = one candidate–job pair.

| column                                                | type                  | meaning                         |
| ----------------------------------------------------- | --------------------- | ------------------------------- |
| `candidate_id` / `job_id`                             | `str`                 | foreign keys                    |
| `spontaneous_application_date` … `4th_interview_date` | `YYYY-MM-DD` \| `NaN` | chronological funnel steps (11 stages) |
| `job_offer_proposed_date` / `job_offer_accepted_date` | `YYYY-MM-DD` \| `NaN` | final outcome                   |
| `end_of_process_date`                                 | `YYYY-MM-DD`          | last recorded action            |
| `last_stage_reached`                                  | `str`                 | label of deepest non-null stage |

*All dates are in UTC, anonymised with a Laplace noise scale of 5 days (monotonic sorting ensures order is preserved for recruitment funnels).*

---

## 4 Regression features

Computed features for each candidate–job pair generated for baseline classifiers. 

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
| `volunteer`                 | `bool`       | volunteer activities present                       |
| `has_publications`          | `bool`       | candidate has published                            |
| `has_patents`               | `bool`       | candidate holds patents                            |
| `visible_portfolio`         | `bool`       | portfolio or GitHub link present                   |
| `has_linkedin`              | `bool`       | LinkedIn link in CV                                |
| `entered_competitions`      | `bool`       | participated in competitions                       |
| `n_prev_positions`          | `int`        | number of prior roles                              |
| `n_unique_employers`        | `int`        | count of unique employers                          |
| `years_mgmt_roles`          | `int`        | years in management                                |
| `avg_job_duration`          | `float`      | months                                             |
| `n_career_gaps`             | `int`        | number of gaps                                     |
| `entity_delta_days`         | `int`        | job - candidate creation                           |
| `rec_delta_days`            | `int`        | recommendation date - job                          |
| `existed_before`            | `bool`       | candidate existed before recommendation            |

---

## 5 Missing-value rates

*High missingness in human-entered fields underscores the sparsity in real-world platforms and the need for robust cold-start handling and LLM derivation.*


```

```
           candidates   jobs

```

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

## 6 Quick start

### 6.1 Quick start (code)

```text
# 1. Generate time-aware train / test splits (binary labels)
python code/utils/binary_splitter.py data/history.csv data/binary

# 2. Run every baseline (models live in code/baseline/)
bash scripts/run_all_baselines.sh data results

# 3. Evaluate all saved predictions → results.csv
bash scripts/evaluate_all.sh data results

```

*After step 3 you will find `results.csv` in the result directory, containing MRR, NDCG@K, P@K, etc. for both directions.*

### 6.2 Quick start (example Python join)

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

## 7 Results

The evaluation highlights the directional asymmetry between jobs and candidates, the primacy of fine-grained time features, and the necessary synergy of signals. See the paper for ablation discussions.

### 7.1 Candidates to Jobs

| Baseline | MRR | P@1 | P@5 | P@10 | R@1 | R@5 | R@10 | F1@1 | F1@5 | F1@10 | NDCG@1 | NDCG@5 | NDCG@10 | MAP@1 | MAP@5 | MAP@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 |
| Jaccard | 0.019 | 0.006 | 0.005 | 0.004 | 0.006 | 0.024 | 0.038 | 0.006 | 0.008 | 0.007 | 0.006 | 0.015 | 0.019 | 0.006 | 0.012 | 0.013 |
| -human feat. | 0.010 | 0.004 | 0.002 | 0.002 | 0.004 | 0.011 | 0.018 | 0.004 | 0.004 | 0.003 | 0.004 | 0.007 | 0.009 | 0.004 | 0.006 | 0.007 |
| -LLM feat. | 0.032 | 0.012 | 0.009 | 0.007 | 0.012 | 0.045 | 0.075 | 0.012 | 0.015 | 0.014 | 0.012 | 0.028 | 0.038 | 0.012 | 0.023 | 0.027 |
| Popularity | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 | 0.001 |
| Pop. + recency | 0.086 | 0.031 | 0.022 | 0.018 | 0.031 | 0.112 | 0.182 | 0.031 | 0.037 | 0.033 | 0.031 | 0.071 | 0.094 | 0.031 | 0.058 | 0.067 |
| Temporal | 0.036 | 0.014 | 0.008 | 0.006 | 0.014 | 0.039 | 0.062 | 0.014 | 0.013 | 0.011 | 0.014 | 0.027 | 0.034 | 0.014 | 0.023 | 0.026 |
| Past temporal | 0.057 | 0.008 | 0.013 | 0.012 | 0.008 | 0.063 | 0.117 | 0.008 | 0.021 | 0.021 | 0.008 | 0.035 | 0.052 | 0.008 | 0.026 | 0.033 |
| User-based CF | 0.058 | 0.035 | 0.016 | 0.010 | 0.035 | 0.080 | 0.096 | 0.035 | 0.027 | 0.018 | 0.035 | 0.058 | 0.064 | 0.035 | 0.051 | 0.053 |
| MF CF | 0.033 | 0.021 | 0.009 | 0.005 | 0.021 | 0.043 | 0.053 | 0.021 | 0.014 | 0.010 | 0.021 | 0.032 | 0.036 | 0.021 | 0.029 | 0.030 |
| CF + recency | 0.119 | 0.076 | 0.028 | 0.017 | 0.076 | 0.140 | 0.170 | 0.076 | 0.047 | 0.031 | 0.076 | 0.110 | 0.120 | 0.076 | 0.100 | 0.104 |
| Logistic reg. | **0.283** | **0.177** | **0.079** | **0.050** | **0.177** | **0.393** | **0.503** | **0.177** | **0.131** | **0.092** | **0.177** | **0.289** | **0.325** | **0.177** | **0.255** | **0.269** |
| -LLM feat. | **0.276** | **0.176** | **0.076** | 0.048 | **0.176** | **0.379** | 0.477 | **0.176** | **0.126** | 0.087 | **0.176** | **0.281** | **0.313** | **0.176** | **0.249** | **0.262** |
| -LLM/time feat. | 0.025 | 0.008 | 0.006 | 0.006 | 0.008 | 0.031 | 0.057 | 0.008 | 0.010 | 0.010 | 0.008 | 0.019 | 0.027 | 0.008 | 0.015 | 0.019 |
| -time feat. | 0.031 | 0.010 | 0.008 | 0.007 | 0.010 | 0.042 | 0.067 | 0.010 | 0.014 | 0.012 | 0.010 | 0.026 | 0.034 | 0.010 | 0.021 | 0.024 |
| -human/time feat. | 0.009 | 0.002 | 0.002 | 0.002 | 0.002 | 0.009 | 0.018 | 0.002 | 0.003 | 0.003 | 0.002 | 0.005 | 0.008 | 0.002 | 0.004 | 0.005 |
| -human feat. | 0.146 | 0.061 | 0.042 | 0.032 | 0.061 | 0.209 | 0.316 | 0.061 | 0.070 | 0.057 | 0.061 | 0.137 | 0.171 | 0.061 | 0.113 | 0.127 |
| Random forest | 0.256 | 0.144 | **0.076** | **0.049** | 0.144 | 0.378 | **0.491** | 0.144 | **0.126** | **0.089** | 0.144 | 0.265 | 0.301 | 0.144 | 0.227 | 0.242 |
| -time feat. | 0.026 | 0.007 | 0.006 | 0.006 | 0.007 | 0.030 | 0.055 | 0.007 | 0.010 | 0.010 | 0.007 | 0.019 | 0.027 | 0.007 | 0.015 | 0.018 |
| Past temporal (raw data) | 0.129 | 0.039 | 0.039 | 0.034 | 0.039 | 0.196 | 0.342 | 0.039 | 0.065 | 0.062 | 0.039 | 0.117 | 0.164 | 0.039 | 0.091 | 0.110 |
| CF + recency (raw data) | 0.073 | 0.034 | 0.018 | 0.012 | 0.034 | 0.092 | 0.121 | 0.034 | 0.031 | 0.022 | 0.034 | 0.063 | 0.073 | 0.034 | 0.054 | 0.058 |
| Logistic reg. (raw data) | 0.323 | 0.221 | 0.087 | 0.053 | 0.221 | 0.436 | 0.532 | 0.221 | 0.145 | 0.097 | 0.221 | 0.333 | 0.363 | 0.221 | 0.298 | 0.311 |

### 7.2 Jobs to Candidates

| Baseline | MRR | P@1 | P@5 | P@10 | R@1 | R@5 | R@10 | F1@1 | F1@5 | F1@10 | NDCG@1 | NDCG@5 | NDCG@10 | MAP@1 | MAP@5 | MAP@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Jaccard | 0.005 | 0.002 | 0.001 | 0.001 | 0.002 | 0.005 | 0.008 | 0.002 | 0.002 | 0.002 | 0.002 | 0.003 | 0.004 | 0.002 | 0.003 | 0.003 |
| -human feat. | 0.003 | 0.001 | 0.000 | 0.000 | 0.001 | 0.002 | 0.003 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 | 0.001 | 0.001 | 0.002 |
| -LLM feat. | 0.010 | 0.002 | 0.002 | 0.002 | 0.002 | 0.012 | 0.022 | 0.002 | 0.004 | 0.004 | 0.002 | 0.007 | 0.010 | 0.002 | 0.005 | 0.007 |
| Popularity | 0.006 | 0.003 | 0.002 | 0.001 | 0.003 | 0.009 | 0.012 | 0.003 | 0.003 | 0.002 | 0.003 | 0.006 | 0.007 | 0.003 | 0.005 | 0.006 |
| Pop. + recency | 0.015 | 0.008 | 0.004 | 0.003 | 0.008 | 0.019 | 0.027 | 0.008 | 0.006 | 0.005 | 0.008 | 0.014 | 0.016 | 0.008 | 0.012 | 0.013 |
| Temporal | 0.010 | 0.003 | 0.003 | 0.002 | 0.003 | 0.013 | 0.019 | 0.003 | 0.004 | 0.003 | 0.003 | 0.008 | 0.010 | 0.003 | 0.006 | 0.007 |
| Past temporal | 0.019 | 0.000 | 0.001 | 0.002 | 0.000 | 0.003 | 0.023 | 0.000 | 0.001 | 0.004 | 0.000 | 0.001 | 0.007 | 0.000 | 0.001 | 0.003 |
| User-based CF | 0.011 | 0.004 | 0.003 | 0.003 | 0.004 | 0.014 | 0.027 | 0.004 | 0.005 | 0.005 | 0.004 | 0.009 | 0.013 | 0.004 | 0.007 | 0.009 |
| MF CF | 0.024 | 0.017 | 0.006 | 0.003 | 0.017 | 0.029 | 0.034 | 0.017 | 0.010 | 0.006 | 0.017 | 0.023 | 0.025 | 0.017 | 0.022 | 0.022 |
| CF + recency | 0.009 | 0.002 | 0.002 | 0.002 | 0.002 | 0.009 | 0.018 | 0.002 | 0.003 | 0.003 | 0.002 | 0.006 | 0.009 | 0.002 | 0.004 | 0.006 |
| Logistic reg. | 0.038 | 0.016 | 0.010 | 0.008 | 0.016 | 0.051 | 0.081 | 0.016 | 0.017 | 0.015 | 0.016 | 0.034 | 0.044 | 0.016 | 0.029 | 0.032 |
| -LLM feat. | **0.064** | **0.034** | **0.017** | **0.012** | **0.034** | **0.086** | **0.124** | **0.034** | **0.029** | **0.022** | **0.034** | **0.061** | **0.073** | **0.034** | **0.052** | **0.057** |
| -LLM/time feat. | 0.012 | 0.004 | 0.002 | 0.002 | 0.004 | 0.012 | 0.025 | 0.004 | 0.004 | 0.004 | 0.004 | 0.008 | 0.012 | 0.004 | 0.007 | 0.008 |
| -time feat. | 0.010 | 0.003 | 0.003 | 0.002 | 0.003 | 0.013 | 0.023 | 0.003 | 0.004 | 0.004 | 0.003 | 0.008 | 0.011 | 0.003 | 0.006 | 0.007 |
| -human/time feat. | 0.003 | 0.000 | 0.000 | 0.001 | 0.000 | 0.002 | 0.005 | 0.000 | 0.001 | 0.001 | 0.000 | 0.001 | 0.002 | 0.000 | 0.001 | 0.001 |
| -human feat. | 0.021 | 0.009 | 0.005 | 0.004 | 0.009 | 0.027 | 0.041 | 0.009 | 0.009 | 0.007 | 0.009 | 0.018 | 0.022 | 0.009 | 0.015 | 0.016 |
| Random Forest | **0.078** | **0.033** | **0.023** | **0.017** | **0.033** | **0.114** | **0.169** | **0.033** | **0.038** | **0.031** | **0.033** | **0.074** | **0.091** | **0.033** | **0.060** | **0.068** |
| -time feat. | 0.008 | 0.002 | 0.001 | 0.002 | 0.002 | 0.007 | 0.015 | 0.002 | 0.002 | 0.003 | 0.002 | 0.005 | 0.007 | 0.002 | 0.004 | 0.005 |
| Past temporal -Anon | 0.124 | 0.043 | 0.037 | 0.032 | 0.043 | 0.184 | 0.320 | 0.043 | 0.061 | 0.058 | 0.043 | 0.114 | 0.157 | 0.043 | 0.091 | 0.108 |
| CF + recency -Anon. | 0.008 | 0.002 | 0.001 | 0.001 | 0.002 | 0.007 | 0.011 | 0.002 | 0.002 | 0.002 | 0.002 | 0.005 | 0.006 | 0.002 | 0.004 | 0.004 |
| Logistic reg. -Anon. | 0.085 | 0.054 | 0.023 | 0.014 | 0.054 | 0.113 | 0.143 | 0.054 | 0.038 | 0.026 | 0.054 | 0.085 | 0.095 | 0.054 | 0.076 | 0.080 |

---

## 8 Recommended Future Tasks

JTH establishes a foundation for several advanced methodological paradigms in Recommender Systems:

1. **Funnel-Aware and Multi-Task Learning:** Leverage detailed, timestamped application trajectories to optimize for deep-funnel success.
2. **Heterogeneous Graph Networks and Explainability:** Connect bipartite candidate-job interactions via shared attribute nodes (e.g., skills, locations, contract types).
3. **LLM Integration and Semantic Matching:** Evaluate models that rely on granular semantic alignment utilizing both human-curated and Llama-extracted features.
4. **Survival Modeling and Temporal Dynamics:** Predict the likelihood of a vacancy expiring unfilled and address temporal biases from bursty recruiter behavior.

---

## 9 Licensing & permitted use

**CC BY-NC 4.0** — non-commercial research and teaching only.
Commercial exploitation or any attempt to re-identify individuals or companies is strictly forbidden.

---

## 10 Citation

If you use this dataset in your research, please cite our RecSys '26 paper:

```bibtex
@inproceedings{romero2026jth,
  title = {JTH: A Dataset for Evaluating Cold-Start and Temporal Dynamics in Job Recommendation},
  author = {Romero, Julien and Millet, Yann and Behar, Eric},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems (RecSys '26)},
  year = {2026},
  publisher = {ACM},
  doi = {10.1145/3773078.3831846},
  url = {[https://doi.org/10.1145/3773078.3831846](https://doi.org/10.1145/3773078.3831846)}
}

```

---

## 11 Change log

| version | date | notes |
| --- | --- | --- |
| 1.1 | 2026-07 | Camera-ready release for RecSys '26. Updated metadata/citations. |
| 1.0 | 2025-08 | initial public release; negative salaries clipped, README added |

Contributions and issues are welcome via GitHub pull requests or the **Issues** tracker.
