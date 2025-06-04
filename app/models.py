from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ExpertCreate(BaseModel):
    name: str = Field(..., description="Expert's full name")
    # email: Optional[str] = Field(None, description="Expert's email address")
    cv_text: str = Field(..., description="CV content as text")
    remote_experience: bool = Field(True, description="Has remote work experience")
    max_tasks: int = Field(3, description="Maximum number of tasks expert can handle")

class ExpertResponse(BaseModel):
    id: int
    name: str
    # email: Optional[str]
    skills: List[str]
    experience_years: int
    education: List[str]
    remote_experience: bool
    max_tasks: int
    created_at: datetime

class TaskCreate(BaseModel):
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Detailed task description")
    required_skills: List[str] = Field([], description="List of required skills")
    remote_allowed: bool = Field(True, description="Whether remote work is allowed")
    priority: int = Field(1, ge=1, le=5, description="Task priority (1-5)")

class TaskResponse(BaseModel):
    id: int
    title: str
    description: str
    required_skills: List[str]
    remote_allowed: bool
    priority: int
    created_at: datetime

class MatchResult(BaseModel):
    expert_id: int
    expert_name: str
    # expert_email: Optional[str]
    task_id: int
    task_title: str
    text_similarity: float = Field(..., ge=0, le=1)
    skill_overlap: float = Field(..., ge=0, le=1)
    combined_score: float = Field(..., ge=0, le=1)
    matching_skills: List[str]
    experience_years: int

class MatchRequest(BaseModel):
    task_description: str = Field(..., description="Task description to match against")
    required_skills: Optional[List[str]] = Field([], description="Required skills for the task")
    top_k: int = Field(5, ge=1, le=20, description="Number of top matches to return")
    remote_required: bool = Field(False, description="Whether remote work capability is required")

class SystemStats(BaseModel):
    total_experts: int
    total_tasks: int
    total_matches: int
    avg_similarity_score: float
    top_skills: List[str]

class SkillAnalysis(BaseModel):
    skill: str
    frequency: int
    experts_count: int 