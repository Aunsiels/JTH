import ollama

from models import ResumeExtraction, SexEnum, NationalityEnum, LanguageLevelEnum, EthnicityEnum, DiplomaEnum, \
    IndustryDomainEnum, ExpertiseAreaEnum, JobCategoryEnum, CompanyValuesEnum, LanguageEnum, SeniorityLevelEnum

fields_description = """
Extract the following fields precisely:
- sex: inferred or explicit, using SexEnum
- birth_year: integer, inferred or explicit
- nationality: explicit, using NationalityEnum
- languages_spoken: list of objects {name: LanguageEnum, level: LanguageLevelEnum}
- ethnicity: inferred or explicit, using EthnicityEnum
- highest_diploma: highest degree, using DiplomaEnum
- degrees: list of previous education {university: str, start_year: int, end_year: int, degree: DiplomaEnum}
- last_graduation_date: int, inferred or explicit
- from_top_university: boolean
- multiple_degrees: boolean
- ongoing_education: boolean
- years_of_work_experience: int, inferred
- number_of_previous_positions: int
- previous_jobs: list of {title: str, start_year: int, end_year: int, company: str}  (**in English**)
- number_of_unique_employers: int
- management_experience: boolean
- years_of_management_roles: int
- industry_domains: list of IndustryDomainEnum, at least one
- entrepreneurial_experience: boolean
- startup_experience: boolean
- large_company_experience: boolean
- freelance_experience: boolean
- contract_experience: boolean
- international_work_experience: boolean
- remote_work_experience: boolean
- average_job_duration_months: int
- number_of_career_gaps: int
- leadership_experience: boolean
- hard_skills: list of str (in English, at least one)
- soft_skills: list of str (in English, at least one)
- programming_languages: list of str
- tools_technologies: list of str
- certifications: list of str (in English)
- volunteer_experience: boolean
- number_of_publications: int
- number_of_patents: int
- has_portfolio_or_github_or_website: boolean
- has_linkedin: boolean
- participated_in_competitions: boolean
- hobbies: list of str (in English)
- expertise_area: inferred, using List[ExpertiseAreaEnum], at least one, at most 5
- job_category: using List[JobCategoryEnum], at least one, at most 5
- client_facing_role: boolean
- values: list of CompanyValuesEnum, at most 5
- seniority_level: SeniorityLevelEnum
"""

prompt = f"""
We are in July, 2025.

You are a professional recruiter analyzing resumes to extract structured information. The resume may be in French, but your answer must be **in English** (translate if necessary).

You will output your answer strictly as a JSON file matching the structure of the `ResumeExtraction` Pydantic model described below.
For each field, provide the value if explicitly available or infer as best as possible when not explicit. For fields like ethnicity and sex, acknowledge these are biased fields, but you must still guess them as well as possible using any hints (name, location, photo if present, language, context).

Use the following enumerations when relevant:
- SexEnum: {list(e.value for e in SexEnum)}
- NationalityEnum: {list(e.value for e in NationalityEnum)}
- LanguageLevelEnum: {list(e.value for e in LanguageLevelEnum)}
- EthnicityEnum: {list(e.value for e in EthnicityEnum)}
- DiplomaEnum: {list(e.value for e in DiplomaEnum)}
- IndustryDomainEnum: {list(e.value for e in IndustryDomainEnum)}
- ExpertiseAreaEnum: {list(e.value for e in ExpertiseAreaEnum)}
- JobCategoryEnum: {list(e.value for e in JobCategoryEnum)}
- CompanyValuesEnum: {list(e.value for e in CompanyValuesEnum)}
- LanguageEnum: {list(e.value for e in LanguageEnum)}
- SeniorityLevelEnum: {list(e.value for e in SeniorityLevelEnum)}

{fields_description}

Return boolean values (true/false) for boolean fields when possible.
Return strings in English.
Return null if the field cannot be inferred at all or found.

Be as exhaustive as you can. Be careful, some spaces and new lines might be missing. Correct it as best as you can (e.g.: postgresdocker -> postgres docker).

[RESUME]

```
[HERE]
```

[IMPORTANT] **Every field must be translated to English, French is not allowed from now**
"""

short_prompt = f"""
We are in July, 2025.

You are a professional recruiter analyzing resumes to extract structured information. The resume may be in French, but your answer must be **in English** (translate if necessary).

You will output your answer strictly as a JSON file matching the structure of the `ShortResumeExtraction` Pydantic model described below.
For each field, provide the value if explicitly available or infer as best as possible when not explicit. For fields like ethnicity and sex, acknowledge these are biased fields, but you must still guess them as well as possible using any hints (name, location, photo if present, language, context).

Use the following enumerations when relevant:
- SexEnum: {list(e.value for e in SexEnum)}
- NationalityEnum: {list(e.value for e in NationalityEnum)}
- EthnicityEnum: {list(e.value for e in EthnicityEnum)}

Extract the following fields precisely:
- sex: inferred or explicit, using SexEnum
- nationality: explicit, using NationalityEnum
- ethnicity: inferred or explicit, using EthnicityEnum

Return boolean values (true/false) for boolean fields when possible.
Return strings in English.
Return null if the field cannot be inferred at all or found.

Be as exhaustive as you can.

[RESUME]

```
[HERE]
```

[IMPORTANT] **Every field must be translated to English, French is not allowed from now**
"""


def parse_resume(resume, stream=False, short=False):
    if not short:
        local_prompt = prompt.replace("[HERE]", resume)
    else:
        local_prompt = short_prompt.replace("[HERE]", resume)
    if stream:
        stream = ollama.chat(
            model='llama3.3',
            messages=[{'role': 'user', 'content': local_prompt}],
            format=ResumeExtraction.model_json_schema(),
            stream=True,
            options={"num_ctx": 4096,
                     "temperature": 0}
        )

        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
    resp = ollama.generate(model='llama3.3',
                           prompt=local_prompt,
                           format=ResumeExtraction.model_json_schema(),
                           options={"num_ctx": 4096,
                                    "temperature": 0}
                           )
    return resp.response


if __name__ == '__main__':
    print(prompt)
