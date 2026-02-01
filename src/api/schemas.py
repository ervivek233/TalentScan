from pydantic import BaseModel

class ResumeText(BaseModel):
    text: str

class ResumeJDInput(BaseModel):
    resume_text: str
    jd_text: str
    
class ResumeScoreOutput(BaseModel):
    score: float