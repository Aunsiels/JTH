**JTH v1.0 (June 2025) — Dataset Card**

---

### • Overview

JTH (Job Tracking History) is a real-world corpus for research on time-aware job recommendation.
It links **38 424 candidates**, **6 199 vacancies** and **45 546 interaction trajectories** collected by professional recruiters in France (2018 – 2025).  Each entity carries rich, English-language attributes; every event in the hiring funnel is time-stamped at day granularity.  Data are pseudonymised and licensed for **non-commercial research** under **CC BY-NC 4.0**.

---

### • Files & Sizes (CSV UTF-8)

| file             | rows   | size   | key columns                            |
| ---------------- | ------ | ------ | -------------------------------------- |
| `candidates.csv` | 38 424 | 11 MB  | `candidate_id`, profile fields         |
| `jobs.csv`       | 6 199  | 1.3 MB | `job_id`, vacancy fields               |
| `history.csv`    | 45 546 | 2.3 MB | `candidate_id`, `job_id`, funnel dates |

---

### • Column Schema (abridged)

*Candidates* – `candidate_id`, `create_date`, `skills` (semicolon list, `_rare_skill_` sentinel), `expertise_area`, `job_category`, `years_experience`, `actual_salary`, `actual_daily_salary`, `contract_type`, `zipcode` (two-digit French département code), `source`, `languages`, `hobbies` (`_rare_hobby_` sentinel), `sex` (LLM-inferred).

*Jobs* – `job_id`, `create_date`, `job_category`, `skills`, `contract_type`, `expertise_area`, `years_experience`, `zipcode`, `salary`, `daily_rate`, `source`, `business_sector`.

*History* – `candidate_id`, `job_id`, `spontaneous_application_date`, `shortlist_date`, `qualification_date`, `resume_sent_to_company_date`, `1st_interview_date` … `4th_interview_date`, `job_offer_proposed_date`, `job_offer_accepted_date`, `end_of_process_date`, `last_stage_reached`.

Full field descriptions are in the repo README.

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

* **Code & documentation:** [https://github.com/Aunsiels/JTH](https://github.com/Aunsiels/JTH)
* **Dataset download (ZIP, 15 MB):** XXX (DOI XXX)

---

### • Maintainer

[julien.romero@telecom-sudparis.eu](mailto:julien.romero@telecom-sudparis.eu)
