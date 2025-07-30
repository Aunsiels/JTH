import ollama

from models import IndustryDomainEnum, CompanyValuesEnum, SeniorityLevelEnum, DiplomaEnum, ExpertiseAreaEnum, \
    JobCategoryEnum, LanguageEnum, JobDescriptionExtraction

prompt = f"""
You are a professional recruiter analyzing job descriptions to extract structured information. The job description may be in French, but your answer must be **in English** (translate if necessary).

You will output your answer strictly as a JSON file matching the structure of the `JobDescriptionExtraction` Pydantic model.
For each field, provide the value if explicitly available or infer as best as possible when not explicit.

Use the following enumerations when relevant:
- IndustryDomainEnum: {list(e.value for e in IndustryDomainEnum)}
- CompanyValuesEnum: {list(e.value for e in CompanyValuesEnum)}
- SeniorityLevelEnum: {list(e.value for e in SeniorityLevelEnum)}
- DiplomaEnum: {list(e.value for e in DiplomaEnum)}
- ExpertiseAreaEnum: {list(e.value for e in ExpertiseAreaEnum)}
- JobCategoryEnum: {list(e.value for e in JobCategoryEnum)}
- LanguageEnum: {list(e.value for e in LanguageEnum)}

The fields to extract with their types are:
- title: str (in English)
- company: str
- location: str
- remote_possible: bool
- industry_domains: List[IndustryDomainEnum], at least one
- company_values: List[CompanyValuesEnum]
- seniority_level: SeniorityLevelEnum
- salary_estimation_eur_per_year: int
- required_languages_spoken: List[Language] (object with name: LanguageEnum, level: LanguageLevelEnum)
- required_lowest_diploma: DiplomaEnum
- required_years_of_work_experience: int
- required_management_experience: bool
- is_startup: bool
- is_large_company: bool
- required_freelance_experience: bool
- required_contract_experience: bool
- required_international_work_experience: bool
- required_leadership_experience: bool
- hard_skills: List[str] (in English), at least one
- soft_skills: List[str] (in English), at least one
- programming_languages: List[str]
- tools_technologies: List[str]
- certifications: List[str] (in English)
- expertise_area: List[ExpertiseAreaEnum], at least one, at most 5
- job_category: List[JobCategoryEnum], at least one, at most 5
- client_facing_role: bool

Return boolean values (true/false) for boolean fields when possible.
Return strings in English.
Return null if the field cannot be inferred or found.

Be as exhaustive as you can.

[DESCRIPTION]

```
[HERE]
```

[IMPORTANT] **Every field must be translated to English, French is not allowed from now**

"""


def parse_description(description):
    local_prompt = prompt.replace("[HERE]", description)
    resp = ollama.generate(model='llama3.3',
                           prompt=local_prompt,
                           format=JobDescriptionExtraction.model_json_schema(),
                           options={"num_ctx": 4096,
                                    "temperature": 0}
                           )
    return resp.response


if __name__ == '__main__':
    print(prompt)
