**JTH v1.0 (June 2025) — Dataset Card**

---

### • Overview

JTH (Job Tracking History) is a real-world corpus for research on time-aware job recommendation.
It links **37,554 candidates**, **6,011 vacancies** and **42,288 interaction trajectories** collected by professional recruiters in France (2018 – 2025).  Each entity carries rich, English-language attributes; every event in the hiring funnel is time-stamped at day granularity.  Data are pseudonymised and licensed for **non-commercial research** under **CC BY-NC 4.0**.

---

### • Files & Sizes (CSV UTF-8)

| file             | rows   | size   | description                                           |
| ---------------- | ------ | ------ | ----------------------------------------------------- |
| `candidates.csv` | 37,554 | 19 MB  | static profile of each applicant                      |
| `jobs.csv`       | 6,011  | 2.3 MB | static profile of each vacancy                        |
| `history.csv`    | 42,288 | 2.7 MB | full, ordered timeline of every candidate–job process |

---

### Schema

#### `candidates.csv`

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


#### `jobs.csv`

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


#### `history.csv`

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

### • Missing-Value Rates

| attribute (examples) | candidates | jobs |
| -------------------- | ---------- | ---- |
| `create_date`        | 7 %        | 0 %  |
| `skills`             | 5 %        | 0 %  |
| `expertise_area`     | 8 %        | 30 % |
| `job_category`       | 27 %       | 30 % |
| `years_experience`   | 83 %       | 89 % |
| `salary`             | 94 %       | 34 % |
| `daily_rate`         | 98 %       | 68 % |
| `contract_type`      | 15 %       | 0 %  |
| `zipcode`            | 41 %       | 68 % |

Negative salaries created by noise have been clipped to zero.

---

### • Recommended Uses

* Time-aware or funnel-aware recommender systems
* Cold-start evaluation for both sides of a two-sided market
* Heterogeneous-graph and skill-ontology modelling (e.g. ESCO)
* Recruitment-funnel analytics (time-to-hire, stage attrition)

---

### • Limitations

* **Geographic scope**: predominantly France; results may not generalise elsewhere.
* **Demographics**: no explicit age/ethnicity; `sex` is LLM-inferred and may be noisy.
* **Missing fields**: many salaries and experience values are absent; users must handle sparsity.
* **Recruiter bias**: interactions reflect human screening choices and may embed subjective preferences.
* **Licensing**: research-only; commercial use or de-anonymisation is forbidden.

---

### • Ethical & Privacy Notes

Identifiers are deterministic digests; free-text was removed.
Salary/daily-rate fields were micro-aggregated and Laplace-noised; dates were jittered and order-preserving.
Dataset meets k = 5 anonymity on {zipcode, experience bucket, contract type}, but residual disclosure risk cannot be eliminated.

---

### • Citation

> *BibTeX placeholder — to be released with DOI XXX*

---

### • Resources

* **Dataset download (ZIP, 24 MB):** XXX (DOI XXX)
