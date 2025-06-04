from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging
import traceback
from datetime import datetime
import nltk
import sqlite3
import json

from app.models import (
    ExpertCreate, ExpertResponse, TaskCreate, TaskResponse,
    MatchResult, MatchRequest, SystemStats, SkillAnalysis
)
from app.matcher import ExpertMatcher
from app.database import DatabaseManager


nltk.download('punkt_tab')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Expert Matching System API",
    description="AI-powered system for matching experts to tasks using NLP and machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize matcher
matcher = ExpertMatcher()

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Expert Matching System API")
    
    # Test database connection
    try:
        stats = matcher.db.get_system_stats()
        logger.info(f"Database connected. Current stats: {stats}")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Expert Matching System API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        stats = matcher.db.get_system_stats()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "experts_count": str(stats['total_experts']),
            "tasks_count": str(stats['total_tasks'])
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# Expert Management Endpoints

@app.post("/experts/", response_model=Dict[str, Any])
async def create_expert(expert: ExpertCreate):
    """Create a new expert from CV text"""
    try:
        logger.info(f"Creating expert: {expert.name}")
        
        expert_id = matcher.add_expert_from_cv(
            name=expert.name,
            # email=expert.email,
            cv_text=expert.cv_text,
            remote_experience=expert.remote_experience,
            max_tasks=expert.max_tasks
        )
        
        # Get the created expert
        created_expert = matcher.db.get_expert(expert_id)
        
        return {
            "message": "Expert created successfully",
            "expert_id": expert_id,
            "expert": created_expert
        }
        
    except Exception as e:
        logger.error(f"Error creating expert: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to create expert: {str(e)}")

def read_file_content(file_path: str) -> str:
    """Read file content with multiple encoding attempts"""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
    
    # If all encodings fail, try binary read and decode with error handling
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            # Try to decode with error handling
            return content.decode('utf-8', errors='replace')
    except Exception as e:
        logger.error(f"Failed to read file {file_path} with any encoding: {str(e)}")
        raise

@app.post("/upload-cv")
async def upload_cv(
    file: UploadFile = File(...),
    name: str = Form(...),
    remote_experience: bool = Form(True),
    max_tasks: int = Form(3)
):
    try:
        # Read file content with encoding handling
        content = await file.read()
        try:
            cv_text = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                cv_text = content.decode('latin1')
            except UnicodeDecodeError:
                cv_text = content.decode('utf-8', errors='replace')
        
        # Process the CV
        processed_data = matcher.processor.process_cv(cv_text, name)
        
        # Add expert to database
        expert_id = matcher.add_expert_from_cv(
            name=name,
            cv_text=cv_text,
            remote_experience=remote_experience,
            max_tasks=max_tasks
        )
        
        return {"message": "CV processed successfully", "expert_id": expert_id}
        
    except Exception as e:
        logger.error(f"Error processing CV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experts/", response_model=List[Dict[str, Any]])
async def get_all_experts():
    """Get all experts"""
    try:
        experts = matcher.db.get_all_experts()
        return experts
    except Exception as e:
        logger.error(f"Error fetching experts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch experts: {str(e)}")

@app.get("/experts/{expert_id}", response_model=Dict[str, Any])
async def get_expert(expert_id: int):
    """Get a specific expert by ID"""
    try:
        expert = matcher.db.get_expert(expert_id)
        if not expert:
            raise HTTPException(status_code=404, detail="Expert not found")
        return expert
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching expert {expert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch expert: {str(e)}")

@app.delete("/experts/{expert_id}", response_model=Dict[str, str])
async def delete_expert(expert_id: int):
    """Delete an expert"""
    try:
        success = matcher.db.delete_expert(expert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Expert not found")
        
        # Clear matcher cache
        matcher._clear_cache()
        
        return {"message": "Expert deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting expert {expert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete expert: {str(e)}")

@app.post("/experts/{expert_id}/skills", response_model=Dict[str, Any])
async def add_expert_skills(expert_id: int, skills: List[str]):
    """Add skills to an expert"""
    try:
        # Get the expert
        expert = matcher.db.get_expert(expert_id)
        if not expert:
            raise HTTPException(status_code=404, detail="Expert not found")
        
        # Get current skills
        current_skills = expert.get('skills', [])
        
        # Add new skills, avoiding duplicates
        updated_skills = list(set(current_skills + skills))
        
        # Update expert in database
        conn = sqlite3.connect(matcher.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE experts 
            SET skills = ?
            WHERE id = ?
        """, (json.dumps(updated_skills), expert_id))
        
        conn.commit()
        conn.close()
        
        # Clear matcher cache to ensure fresh data
        matcher._clear_cache()
        
        return {
            "message": "Skills added successfully",
            "expert_id": expert_id,
            "updated_skills": updated_skills
        }
        
    except Exception as e:
        logger.error(f"Error adding skills: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add skills: {str(e)}")

# Task Management Endpoints

@app.post("/tasks/", response_model=Dict[str, Any])
async def create_task(task: TaskCreate):
    """Create a new task"""
    try:
        task_id = matcher.add_task(
            title=task.title,
            description=task.description,
            required_skills=task.required_skills,
            remote_allowed=task.remote_allowed,
            priority=task.priority
        )
        
        created_task = matcher.db.get_task(task_id)
        
        return {
            "message": "Task created successfully",
            "task_id": task_id,
            "task": created_task
        }
        
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

@app.get("/tasks/", response_model=List[Dict[str, Any]])
async def get_all_tasks():
    """Get all tasks"""
    try:
        tasks = matcher.db.get_all_tasks()
        return tasks
    except Exception as e:
        logger.error(f"Error fetching tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch tasks: {str(e)}")

@app.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task(task_id: int):
    """Get a specific task by ID"""
    try:
        task = matcher.db.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch task: {str(e)}")

@app.delete("/tasks/{task_id}", response_model=Dict[str, str])
async def delete_task(task_id: int):
    """Delete a task"""
    try:
        success = matcher.db.delete_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"message": "Task deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")

# Matching Endpoints

@app.post("/match/find", response_model=List[Dict[str, Any]])
async def find_matches(match_request: MatchRequest):
    """Find expert matches for a task description"""
    try:
        matches = matcher.find_best_matches(
            task_description=match_request.task_description,
            required_skills=match_request.required_skills,
            remote_required=match_request.remote_required,
            top_k=match_request.top_k
        )
        
        return matches
        
    except Exception as e:
        logger.error(f"Error finding matches: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to find matches: {str(e)}")

@app.post("/match/task/{task_id}", response_model=List[Dict[str, Any]])
async def match_existing_task(task_id: int, top_k: int = Query(5, ge=1, le=20)):
    """Find matches for an existing task"""
    try:
        matches = matcher.find_matches_for_existing_task(task_id, top_k)
        if not matches:
            # Check if task exists
            task = matcher.db.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            return []  # Task exists but no matches found
        
        return matches
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error matching task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to match task: {str(e)}")

@app.post("/match/bulk", response_model=Dict[str, Any])
async def bulk_match_all_tasks(background_tasks: BackgroundTasks):
    """Run matching for all tasks (background process)"""
    try:
        def run_bulk_matching():
            return matcher.bulk_match_all_tasks()
        
        # Run in background
        background_tasks.add_task(run_bulk_matching)
        
        return {
            "message": "Bulk matching started in background",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error starting bulk matching: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start bulk matching: {str(e)}")

@app.get("/matches/", response_model=List[Dict[str, Any]])
async def get_all_matches(limit: int = Query(100, ge=1, le=1000)):
    """Get all matches with optional limit"""
    try:
        matches = matcher.db.get_matches(limit)
        return matches
    except Exception as e:
        logger.error(f"Error fetching matches: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch matches: {str(e)}")

# Analytics and Statistics Endpoints

@app.get("/stats/", response_model=Dict[str, Any])
async def get_system_statistics():
    """Get comprehensive system statistics"""
    try:
        # Get basic stats first
        try:
            stats = matcher.db.get_system_stats()
        except Exception as e:
            logger.error(f"Error getting basic stats: {e}")
            stats = {
                'total_experts': 0,
                'total_tasks': 0,
                'total_matches': 0,
                'avg_similarity_score': 0.0,
                'top_skills': []
            }
        
        # Get workload stats
        try:
            workload = matcher.get_expert_workload()
        except Exception as e:
            logger.error(f"Error getting workload stats: {e}")
            workload = {}
        
        # Get skill analysis
        try:
            skill_demand = matcher.get_skill_demand_analysis()
        except Exception as e:
            logger.error(f"Error getting skill demand: {e}")
            skill_demand = {}
        
        try:
            skill_supply = matcher.get_skill_supply_analysis()
        except Exception as e:
            logger.error(f"Error getting skill supply: {e}")
            skill_supply = {}
        
        # Calculate additional metrics
        try:
            available_experts = sum(1 for w in workload.values() if w.get('availability') == 'Available')
            busy_experts = sum(1 for w in workload.values() if w.get('availability') == 'Fully Loaded')
        except Exception as e:
            logger.error(f"Error calculating expert availability: {e}")
            available_experts = 0
            busy_experts = 0
        
        # Calculate average experience
        try:
            avg_experience = 0
            if stats['total_experts'] > 0:
                experts = matcher.db.get_all_experts()
                total_exp = sum(expert.get('experience_years', 0) for expert in experts)
                avg_experience = round(total_exp / len(experts), 1)
        except Exception as e:
            logger.error(f"Error calculating average experience: {e}")
            avg_experience = 0
        
        # Get top skills safely
        try:
            top_demand_skills = list(skill_demand.keys())[:5]
            top_supply_skills = list(skill_supply.keys())[:5]
        except Exception as e:
            logger.error(f"Error getting top skills: {e}")
            top_demand_skills = []
            top_supply_skills = []
        
        # Analyze skill gaps safely
        try:
            skill_gaps = matcher._analyze_skill_gaps(skill_demand, skill_supply)
        except Exception as e:
            logger.error(f"Error analyzing skill gaps: {e}")
            skill_gaps = []
        
        return {
            'basic_stats': stats,
            'expert_availability': {
                'available': available_experts,
                'busy': busy_experts,
                'total': stats['total_experts']
            },
            'average_experience_years': avg_experience,
            'top_demand_skills': top_demand_skills,
            'top_supply_skills': top_supply_skills,
            'skill_gap_analysis': skill_gaps
        }
        
    except Exception as e:
        logger.error(f"Error in get_system_statistics: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system statistics: {str(e)}"
        )

@app.get("/stats/workload", response_model=Dict[int, Dict[str, Any]])
async def get_expert_workload():
    """Get expert workload information"""
    try:
        # Get workload from matcher instead of database manager
        workload = matcher.get_expert_workload()
        if not workload:
            return {}
        return workload
    except Exception as e:
        logger.error(f"Error fetching workload: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch workload: {str(e)}"
        )

@app.get("/stats/skills/demand", response_model=Dict[str, int])
async def get_skill_demand():
    """Get skill demand analysis"""
    try:
        demand = matcher.get_skill_demand_analysis()
        return demand
    except Exception as e:
        logger.error(f"Error fetching skill demand: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch skill demand: {str(e)}")

@app.get("/stats/skills/supply", response_model=Dict[str, int])
async def get_skill_supply():
    """Get skill supply analysis"""
    try:
        supply = matcher.get_skill_supply_analysis()
        return supply
    except Exception as e:
        logger.error(f"Error fetching skill supply: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch skill supply: {str(e)}")

# Utility Endpoints

@app.post("/utils/process-cv", response_model=Dict[str, Any])
async def process_cv_text(cv_text: str = Form(...)):
    """Process CV text and extract information (for testing)"""
    try:
        processed = matcher.processor.process_cv(cv_text)
        return {
            "message": "CV processed successfully",
            "extracted_data": processed
        }
    except Exception as e:
        logger.error(f"Error processing CV: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CV: {str(e)}")

@app.get("/utils/skills/extract")
async def extract_skills_from_text(text: str = Query(...)):
    """Extract skills from text (utility endpoint)"""
    try:
        skills = matcher.processor.extract_skills(text)
        return {
            "text": text,
            "extracted_skills": skills
        }
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract skills: {str(e)}")

# Data Management Endpoints

@app.post("/data/seed")
async def seed_database():
    """Seed database with sample data from data directory"""
    try:
        import os
        from pathlib import Path
        
        # Ensure data directory exists
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        added_experts = 0
        added_tasks = 0
        
        # Check if data directory is empty
        if not any(data_dir.glob("*.txt")):
            logger.warning("No .txt files found in data directory. Please add CV and JD files.")
            return {
                "message": "No data files found",
                "experts_added": 0,
                "tasks_added": 0,
                "matches_created": {"tasks_processed": 0, "total_matches": 0}
            }
        
        # Load CV files for experts
        for cv_file in data_dir.glob("*.txt"):
            # Check if file is a CV file based on naming patterns
            if any(pattern in cv_file.name for pattern in [
                "CV for Research Assistants - Lab Instructors",
                "CV for Lab Instructor",
                "CV for Research Assistant",
                "CV for for Research Assistant"
            ]):
                try:
                    with open(cv_file, 'r', encoding='utf-8') as f:
                        cv_content = f.read()
                    
                    # Extract name from filename using the specific patterns
                    name = cv_file.stem
                    # Remove common suffixes
                    name = name.replace(" - CV for Research Assistants - Lab Instructors", "")
                    name = name.replace(" - CV for Lab Instructor", "")
                    name = name.replace(" - CV for Research Assistant", "")
                    name = name.replace(" - CV for for Research Assistant", "")
                    name = name.replace(" (1)", "")  # Remove any numbering
                    
                    # Clean up the name
                    name = name.strip()
                    
                    expert_id = matcher.add_expert_from_cv(
                        name=name,
                        # email=None,
                        cv_text=cv_content,
                        remote_experience=True,
                        max_tasks=3
                    )
                    added_experts += 1
                    logger.info(f"Added expert from {cv_file.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {cv_file.name}: {e}")
        
        # Load job descriptions and create tasks
        for jd_file in data_dir.glob("*.txt"):
            if "JD" in jd_file.name:
                try:
                    # Try different encodings
                    jd_content = None
                    for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
                        try:
                            with open(jd_file, 'r', encoding=encoding) as f:
                                jd_content = f.read()
                            break  # If successful, break the loop
                        except UnicodeDecodeError:
                            continue
                    
                    if jd_content is None:
                        # If all encodings fail, try with error handling
                        with open(jd_file, 'r', encoding='utf-8', errors='replace') as f:
                            jd_content = f.read()
                    
                    # Extract title from filename
                    title = jd_file.stem.replace("JD ", "")
                    
                    # Extract required skills from job description
                    required_skills = matcher.processor.extract_skills(jd_content)
                    all_skills = []
                    for category_skills in required_skills.values():
                        all_skills.extend(category_skills)
                    
                    # Create task from job description
                    task_id = matcher.add_task(
                        title=title,
                        description=jd_content,
                        required_skills=all_skills,
                        remote_allowed=True,
                        priority=3  # Default priority
                    )
                    added_tasks += 1
                    logger.info(f"Added task from {jd_file.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {jd_file.name}: {e}")
        
        # Run initial matching
        matches = matcher.bulk_match_all_tasks()
        
        return {
            "message": "Database seeded successfully",
            "experts_added": added_experts,
            "tasks_added": added_tasks,
            "matches_created": matches
        }
        
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to seed database: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)