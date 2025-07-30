from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class SexEnum(Enum):
    male = 'Male'
    female = 'Female'


class NationalityEnum(Enum):
    french = 'French'
    american = 'American'
    british = 'British'
    german = 'German'
    italian = 'Italian'
    spanish = 'Spanish'
    portuguese = 'Portuguese'
    dutch = 'Dutch'
    belgian = 'Belgian'
    swiss = 'Swiss'
    canadian = 'Canadian'
    australian = 'Australian'
    chinese = 'Chinese'
    japanese = 'Japanese'
    korean = 'Korean'
    indian = 'Indian'
    brazilian = 'Brazilian'
    mexican = 'Mexican'
    argentinian = 'Argentinian'
    russian = 'Russian'
    ukrainian = 'Ukrainian'
    polish = 'Polish'
    swedish = 'Swedish'
    norwegian = 'Norwegian'
    danish = 'Danish'
    finnish = 'Finnish'
    egyptian = 'Egyptian'
    moroccan = 'Moroccan'
    tunisian = 'Tunisian'
    algerian = 'Algerian'
    turkish = 'Turkish'
    lebanese = 'Lebanese'
    saudi = 'Saudi'
    emirati = 'Emirati'
    south_african = 'South African'
    nigerian = 'Nigerian'
    kenyan = 'Kenyan'
    singaporean = 'Singaporean'
    malaysian = 'Malaysian'
    indonesian = 'Indonesian'
    thai = 'Thai'
    vietnamese = 'Vietnamese'
    filipino = 'Filipino'
    new_zealander = 'New Zealander'
    irish = 'Irish'
    scottish = 'Scottish'
    welsh = 'Welsh'
    greek = 'Greek'
    austrian = 'Austrian'
    czech = 'Czech'
    slovak = 'Slovak'
    hungarian = 'Hungarian'
    romanian = 'Romanian'
    bulgarian = 'Bulgarian'
    croatian = 'Croatian'
    serbian = 'Serbian'
    slovenian = 'Slovenian'
    israeli = 'Israeli'
    palestinian = 'Palestinian'
    syrian = 'Syrian'
    jordanian = 'Jordanian'
    iraqi = 'Iraqi'
    iranian = 'Iranian'
    pakistani = 'Pakistani'
    bangladeshi = 'Bangladeshi'
    sri_lankan = 'Sri Lankan'


class LanguageLevelEnum(Enum):
    beginner = 'Beginner'
    intermediate = 'Intermediate'
    advanced = 'Advanced'


class EthnicityEnum(Enum):
    asian = 'Asian'
    east_asian = 'East Asian'
    south_asian = 'South Asian'
    southeast_asian = 'Southeast Asian'
    black_african = 'Black African'
    african_american = 'African American'
    white_european = 'White European'
    hispanic_latino = 'Hispanic or Latino'
    middle_eastern = 'Middle Eastern'
    indigenous = 'Indigenous'
    pacific_islander = 'Pacific Islander'
    mixed = 'Mixed'
    north_african = 'North African'
    jewish = 'Jewish'
    other = 'Other'


class DiplomaEnum(Enum):
    phd = 'PhD'
    master = 'Master'
    bachelor = 'Bachelor'
    high_school = 'High School'
    mba = 'MBA'


class IndustryDomainEnum(Enum):
    finance = 'Finance'
    banking = 'Banking'
    insurance = 'Insurance'
    healthcare = 'Healthcare'
    pharmaceuticals = 'Pharmaceuticals'
    biotechnology = 'Biotechnology'
    education = 'Education'
    technology = 'Technology'
    information_technology = 'Information Technology'
    software = 'Software'
    hardware = 'Hardware'
    telecommunications = 'Telecommunications'
    manufacturing = 'Manufacturing'
    automotive = 'Automotive'
    aerospace = 'Aerospace'
    defense = 'Defense'
    construction = 'Construction'
    real_estate = 'Real Estate'
    energy = 'Energy'
    oil_and_gas = 'Oil and Gas'
    renewable_energy = 'Renewable Energy'
    retail = 'Retail'
    e_commerce = 'E-Commerce'
    logistics = 'Logistics'
    transportation = 'Transportation'
    travel = 'Travel'
    hospitality = 'Hospitality'
    food_and_beverage = 'Food and Beverage'
    agriculture = 'Agriculture'
    media = 'Media'
    entertainment = 'Entertainment'
    legal = 'Legal'
    consulting = 'Consulting'
    government = 'Government'
    non_profit = 'Non-Profit'
    environmental_services = 'Environmental Services'
    marketing = 'Marketing'
    advertising = 'Advertising'
    public_relations = 'Public Relations'
    human_resources = 'Human Resources'
    management = 'Management'
    security = 'Security'
    cybersecurity = 'Cybersecurity'
    mining = 'Mining'
    sports = 'Sports'
    arts_and_culture = 'Arts and Culture'
    supply_chain = 'Supply Chain'
    research = 'Research'
    scientific_services = 'Scientific Services'
    customer_service = 'Customer Service'
    printing_and_publishing = 'Printing and Publishing'


class ExpertiseAreaEnum(Enum):
    ea0 = 'Development'
    ea1 = 'Recruitment'
    ea2 = 'Training'
    ea3 = 'Electronics'
    ea4 = 'Management'
    ea5 = 'Mobile'
    ea6 = 'Cloud & Infrastructure'
    ea7 = 'Data'
    ea8 = 'Functional'
    ea9 = 'Product & Design'
    ea10 = 'Sales & Digital Marketing'
    ea11 = 'QA'


class JobCategoryEnum(Enum):
    jc0 = 'Digital Design'
    jc1 = 'SoC Engineer'
    jc2 = 'Chief Data Officer (CDO)'
    jc3 = 'Trainer'
    jc4 = 'CPO/Head of Product'
    jc5 = 'Network Administrator'
    jc6 = 'Digital Project Manager'
    jc7 = 'Cybersecurity Specialist'
    jc8 = 'Infrastructure Architect'
    jc9 = 'VP Engineering'
    jc10 = 'Mobile Developer'
    jc11 = 'Materials Engineer'
    jc12 = 'Data Analyst'
    jc13 = 'Technical Project Manager'
    jc14 = 'Game Developer'
    jc15 = 'Energy Manager'
    jc16 = 'Technical Referent'
    jc17 = 'Sales'
    jc18 = 'Firmware Engineer'
    jc19 = 'System Administrator'
    jc20 = 'Technical Lead'
    jc21 = 'Cybersecurity Engineer'
    jc22 = 'Software Engineer'
    jc23 = 'Lead QA'
    jc24 = 'Product Manager'
    jc25 = 'Power Electronics Engineer'
    jc26 = 'DevOps Engineer'
    jc27 = 'IT Director'
    jc28 = 'Technician'
    jc29 = 'Digital Marketing'
    jc30 = 'Mechanical Engineer'
    jc31 = 'Machine Learning Engineer'
    jc32 = 'FPGA Engineer'
    jc33 = 'SRE (Site Reliability Engineer)'
    jc34 = 'Business Analyst'
    jc35 = 'Service Delivery Manager'
    jc36 = 'Frontend Developer'
    jc37 = 'Embedded Linux Engineer'
    jc38 = 'Agile Coach'
    jc39 = 'Integrator (HTML/CSS)'
    jc40 = 'Talent Acquisition Manager'
    jc41 = 'Technical Services Manager'
    jc42 = 'Embedded Systems Developer'
    jc43 = 'Product Owner'
    jc44 = 'Data Engineer'
    jc45 = 'Analog Design Engineer'
    jc46 = 'Embedded System Engineer'
    jc47 = 'Full Stack Developer'
    jc48 = 'QA Tester'
    jc49 = 'Tech Recruiter'
    jc50 = 'Infrastructure Manager'
    jc51 = 'Application Engineer'
    jc52 = 'CTO'
    jc53 = 'MOA Project Manager'
    jc54 = 'System Engineer'
    jc55 = 'Scrum Master'
    jc56 = 'Network Engineer'
    jc57 = 'CRM Consultant'
    jc58 = 'ASIC Engineer'
    jc59 = 'Product Designer'
    jc60 = 'Database Administrator'
    jc61 = 'IC Verification Engineer'
    jc62 = 'Layout Engineer'
    jc63 = 'PCB Designer'
    jc64 = 'Electronics Manager'
    jc65 = 'Technical Support'
    jc66 = 'Graphic Developer'
    jc67 = 'Optronics Engineer'
    jc68 = 'Software Manager'
    jc69 = 'QA / Test Engineer'
    jc70 = 'Data Scientist'
    jc71 = 'Incident Manager'
    jc72 = 'Engineering Manager'
    jc73 = 'SAP Developer'
    jc74 = 'IT Architect'
    jc75 = 'Backend Developer'
    jc76 = 'Hardware Engineer'
    jc77 = 'CSM'
    jc78 = 'DFT Engineer (Design for Test)'
    jc79 = 'Electronics Engineer'
    jc80 = 'Support Technician'
    jc81 = 'Production Engineer'
    jc82 = 'AI Engineer'
    jc83 = 'Automation Engineer'
    jc84 = 'Project Manager'
    jc85 = 'Field Application Engineer'
    jc86 = 'UX UI Designer'
    jc87 = 'Energy Engineer'


class LanguageEnum(Enum):
    english = 'English'
    french = 'French'
    spanish = 'Spanish'
    german = 'German'
    italian = 'Italian'
    portuguese = 'Portuguese'
    dutch = 'Dutch'
    russian = 'Russian'
    chinese = 'Chinese'
    japanese = 'Japanese'
    korean = 'Korean'
    arabic = 'Arabic'
    hindi = 'Hindi'
    bengali = 'Bengali'
    urdu = 'Urdu'
    turkish = 'Turkish'
    vietnamese = 'Vietnamese'
    thai = 'Thai'
    swedish = 'Swedish'
    norwegian = 'Norwegian'
    danish = 'Danish'
    finnish = 'Finnish'
    polish = 'Polish'
    greek = 'Greek'
    czech = 'Czech'
    slovak = 'Slovak'
    hungarian = 'Hungarian'
    romanian = 'Romanian'
    bulgarian = 'Bulgarian'
    serbian = 'Serbian'
    croatian = 'Croatian'
    ukrainian = 'Ukrainian'
    hebrew = 'Hebrew'
    indonesian = 'Indonesian'
    malay = 'Malay'
    filipino = 'Filipino'
    swahili = 'Swahili'
    persian = 'Persian'
    tamil = 'Tamil'
    telugu = 'Telugu'
    marathi = 'Marathi'
    punjabi = 'Punjabi'
    gujarati = 'Gujarati'
    sinhala = 'Sinhala'
    lao = 'Lao'
    khmer = 'Khmer'
    burmese = 'Burmese'


class Language(BaseModel):
    name: LanguageEnum
    level: LanguageLevelEnum


class PreviousEducation(BaseModel):
    university: str
    start_year: int
    end_year: int
    degree: DiplomaEnum


class PreviousJob(BaseModel):
    title: str
    start_year: int
    end_year: int
    company: str


class Skill(BaseModel):
    skill: str = Field(..., max_length=100)


class CompanyValuesEnum(Enum):
    integrity = 'Integrity'
    innovation = 'Innovation'
    customer_focus = 'Customer Focus'
    teamwork = 'Teamwork'
    excellence = 'Excellence'
    accountability = 'Accountability'
    diversity = 'Diversity'
    inclusion = 'Inclusion'
    sustainability = 'Sustainability'
    respect = 'Respect'
    transparency = 'Transparency'
    agility = 'Agility'
    collaboration = 'Collaboration'
    quality = 'Quality'
    trust = 'Trust'
    social_responsibility = 'Social Responsibility'
    passion = 'Passion'
    courage = 'Courage'
    adaptability = 'Adaptability'
    empathy = 'Empathy'
    learning = 'Learning'


class SeniorityLevelEnum(Enum):
    intern = 'Intern'
    junior = 'Junior'
    mid = 'Mid'
    senior = 'Senior'
    lead = 'Lead'


class ResumeExtraction(BaseModel):
    sex: SexEnum
    birth_year: Optional[int]
    nationality: Optional[NationalityEnum]
    languages_spoken: List[Language] = Field(..., max_length=10)
    ethnicity: EthnicityEnum
    highest_diploma: DiplomaEnum
    degrees: List[PreviousEducation] = Field(..., max_length=10)
    last_graduation_date: int
    from_top_university: bool
    multiple_degrees: bool
    ongoing_education: bool
    years_of_work_experience: int
    number_of_previous_positions: int
    previous_jobs: List[PreviousJob] = Field(..., max_length=10)
    number_of_unique_employers: int
    management_experience: bool
    years_of_management_roles: int
    industry_domains: List[IndustryDomainEnum] = Field(..., max_length=5)
    entrepreneurial_experience: bool
    startup_experience: bool
    large_company_experience: bool
    freelance_experience: bool
    contract_experience: bool
    international_work_experience: bool
    remote_work_experience: bool
    average_job_duration_months: int
    number_of_career_gaps: int
    leadership_experience: bool
    hard_skills: List[Skill] = Field(..., max_length=20)
    soft_skills: List[Skill] = Field(..., max_length=20)
    programming_languages: List[Skill] = Field(..., max_length=20)
    tools_technologies: List[Skill] = Field(..., max_length=20)
    certifications: List[Skill] = Field(..., max_length=10)
    volunteer_experience: bool
    number_of_publications: int
    number_of_patents: int
    has_portfolio_or_github_or_website: bool
    has_linkedin: bool
    participated_in_competitions: bool
    hobbies: List[Skill] = Field(..., max_length=10)
    expertise_area: List[ExpertiseAreaEnum] = Field(..., max_length=5)
    job_category: List[JobCategoryEnum] = Field(..., max_length=5)
    estimated_salary_eur_per_year: int
    client_facing_role: bool
    values: List[CompanyValuesEnum] = Field(..., max_length=5)
    seniority_level: SeniorityLevelEnum


class JobDescriptionExtraction(BaseModel):
    title: str
    company: str
    location: str
    remote_possible: bool
    industry_domains: List[IndustryDomainEnum] = Field(..., max_length=5)
    company_values: List[CompanyValuesEnum] = Field(..., max_length=5)
    seniority_level: SeniorityLevelEnum
    salary_estimation_eur_per_year: int
    required_languages_spoken: List[Language] = Field(..., max_length=5)
    required_lowest_diploma: DiplomaEnum
    required_years_of_work_experience: int
    required_management_experience: bool
    is_startup: bool
    is_large_company: bool
    required_freelance_experience: bool
    required_contract_experience: bool
    required_international_work_experience: bool
    required_leadership_experience: bool
    hard_skills: List[Skill] = Field(..., max_length=20)
    soft_skills: List[Skill] = Field(..., max_length=20)
    programming_languages: List[Skill] = Field(..., max_length=20)
    tools_technologies: List[Skill] = Field(..., max_length=20)
    certifications: List[Skill] = Field(..., max_length=10)
    expertise_area: List[ExpertiseAreaEnum] = Field(..., max_length=5)
    job_category:  List[JobCategoryEnum] = Field(..., max_length=5)
    client_facing_role: bool
