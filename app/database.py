import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import logging
import traceback
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "data/experts.db"):
        self.db_path = db_path
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create experts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                skills TEXT,  -- JSON string
                experience_years INTEGER DEFAULT 0,
                education TEXT,  -- JSON string
                remote_experience BOOLEAN DEFAULT 1,
                max_tasks INTEGER DEFAULT 3,
                raw_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                required_skills TEXT,  -- JSON string
                remote_allowed BOOLEAN DEFAULT 1,
                priority INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create matches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expert_id INTEGER,
                task_id INTEGER,
                text_similarity REAL,
                skill_overlap REAL,
                combined_score REAL,
                matching_skills TEXT,  -- JSON string
                matched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (expert_id) REFERENCES experts (id),
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_experts_name ON experts(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_title ON tasks(title)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_score ON matches(combined_score)")
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def add_expert(self, name: str, skills: List[str],
                  experience_years: int, education: Dict,
                  remote_experience: bool = True, max_tasks: int = 3,
                  raw_text: str = None) -> int:
        """Add a new expert to the database"""
        try:
            # Ensure experience_years is an integer
            experience_years = int(experience_years) if experience_years is not None else 0
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO experts (
                    name, skills, experience_years, education,
                    remote_experience, max_tasks, raw_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                name,
                json.dumps(skills),
                experience_years,
                json.dumps(education),
                remote_experience,
                max_tasks,
                raw_text
            ))
            
            expert_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return expert_id
            
        except Exception as e:
            logger.error(f"Error adding expert: {e}")
            raise
    
    def get_expert(self, expert_id: int) -> Optional[Dict]:
        """Get a specific expert by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM experts WHERE id = ?", (expert_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_expert_dict(row)
        return None
    
    def get_all_experts(self) -> List[Dict]:
        """Get all experts from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM experts ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_expert_dict(row) for row in rows]
    
    def add_task(self, title: str, description: str, required_skills: List[str],
                 remote_allowed: bool, priority: int) -> int:
        """Add a new task to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO tasks (title, description, required_skills, remote_allowed, priority)
            VALUES (?, ?, ?, ?, ?)
        """, (title, description, json.dumps(required_skills), remote_allowed, priority))
        
        task_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Added task: {title} with ID: {task_id}")
        return task_id
    
    def get_task(self, task_id: int) -> Optional[Dict]:
        """Get a specific task by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_task_dict(row)
        return None
    
    def get_all_tasks(self) -> List[Dict]:
        """Get all tasks from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM tasks ORDER BY priority DESC, created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_task_dict(row) for row in rows]
    
    def save_matches(self, matches: List[Dict]) -> None:
        """Save matching results to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing matches for the same expert-task pairs
        for match in matches:
            cursor.execute("""
                DELETE FROM matches 
                WHERE expert_id = ? AND task_id = ?
            """, (match['expert_id'], match.get('task_id', 0)))
        
        # Insert new matches
        for match in matches:
            cursor.execute("""
                INSERT INTO matches (expert_id, task_id, text_similarity, skill_overlap, 
                                   combined_score, matching_skills)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                match['expert_id'], match.get('task_id', 0), 
                match['text_similarity'], match['skill_overlap'],
                match['combined_score'], json.dumps(match['matching_skills'])
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(matches)} matches to database")
    
    def get_matches(self, limit: int = 100) -> List[Dict]:
        """Get all matches from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT m.*, e.name as expert_name,
                   t.title as task_title, t.description as task_description
            FROM matches m
            LEFT JOIN experts e ON m.expert_id = e.id
            LEFT JOIN tasks t ON m.task_id = t.id
            ORDER BY m.combined_score DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        matches = []
        for row in rows:
            try:
                match_dict = {
                    'id': row[0],
                    'expert_id': row[1],
                    'task_id': row[2],
                    'text_similarity': row[3],
                    'skill_overlap': row[4],
                    'combined_score': row[5],
                    'matching_skills': json.loads(row[6]) if row[6] else [],
                    'matched_at': row[7],
                    'expert_name': row[8] if len(row) > 8 else None,
                    'task_title': row[9] if len(row) > 9 else None,
                    'task_description': row[10] if len(row) > 10 else None
                }
                matches.append(match_dict)
            except Exception as e:
                logger.error(f"Error processing match row: {e}")
                logger.error(f"Row data: {row}")
                continue
        
        return matches
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total counts with safe defaults
            try:
                cursor.execute("SELECT COUNT(*) FROM experts")
                total_experts = cursor.fetchone()[0] or 0
            except Exception:
                total_experts = 0
            
            try:
                cursor.execute("SELECT COUNT(*) FROM tasks")
                total_tasks = cursor.fetchone()[0] or 0
            except Exception:
                total_tasks = 0
            
            try:
                cursor.execute("SELECT COUNT(*) FROM matches")
                total_matches = cursor.fetchone()[0] or 0
            except Exception:
                total_matches = 0
            
            # Get average similarity score with safe default
            try:
                cursor.execute("SELECT AVG(combined_score) FROM matches")
                avg_score = cursor.fetchone()[0]
                avg_similarity_score = round(float(avg_score or 0), 2)
            except Exception:
                avg_similarity_score = 0.0
            
            # Get top skills with safe default
            try:
                cursor.execute("""
                    SELECT skill, COUNT(*) as count
                    FROM (
                        SELECT json_extract(value, '$') as skill
                        FROM experts, json_each(experts.skills)
                    )
                    GROUP BY skill
                    ORDER BY count DESC
                    LIMIT 5
                """)
                top_skills = [row[0] for row in cursor.fetchall()] or []
            except Exception:
                top_skills = []
            
            conn.close()
            
            return {
                'total_experts': total_experts,
                'total_tasks': total_tasks,
                'total_matches': total_matches,
                'avg_similarity_score': avg_similarity_score,
                'top_skills': top_skills
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                'total_experts': 0,
                'total_tasks': 0,
                'total_matches': 0,
                'avg_similarity_score': 0.0,
                'top_skills': []
            }
    
    def delete_expert(self, expert_id: int) -> bool:
        """Delete an expert and related matches"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete related matches first
        cursor.execute("DELETE FROM matches WHERE expert_id = ?", (expert_id,))
        
        # Delete expert
        cursor.execute("DELETE FROM experts WHERE id = ?", (expert_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if deleted:
            logger.info(f"Deleted expert with ID: {expert_id}")
        
        return deleted
    
    def delete_task(self, task_id: int) -> bool:
        """Delete a task and related matches"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete related matches first
        cursor.execute("DELETE FROM matches WHERE task_id = ?", (task_id,))
        
        # Delete task
        cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if deleted:
            logger.info(f"Deleted task with ID: {task_id}")
        
        return deleted
    
    def _row_to_expert_dict(self, row) -> Dict:
        """Convert database row to expert dictionary"""
        try:
            # Ensure we have all required fields
            if len(row) < 9:  # Changed from 10 to 9 to match schema
                logger.error(f"Expert row has insufficient columns: {row}")
                raise ValueError("Expert row has insufficient columns")

            # Handle education field which could be string or int
            education = []
            if row[5]:  # If education field is not None
                if isinstance(row[5], str):
                    try:
                        education = json.loads(row[5])
                    except json.JSONDecodeError:
                        education = [row[5]]  # If not valid JSON, treat as single string
                elif isinstance(row[5], (int, float)):
                    education = [str(row[5])]  # Convert number to string
                else:
                    education = []

            # Handle skills field
            skills = []
            if row[2]:  # If skills field is not None
                if isinstance(row[2], str):
                    try:
                        skills = json.loads(row[2])
                    except json.JSONDecodeError:
                        skills = [row[2]]  # If not valid JSON, treat as single string
                elif isinstance(row[2], (int, float)):
                    skills = [str(row[2])]  # Convert number to string
                else:
                    skills = []

            return {
                'id': row[0],
                'name': row[1],
                'skills': skills,
                'experience_years': row[3] if len(row) > 3 else 0,
                'education': education,
                'remote_experience': bool(row[5]) if len(row) > 5 else True,
                'max_tasks': row[6] if len(row) > 6 else 3,
                'raw_text': row[7] if len(row) > 7 else None,
                'created_at': row[8] if len(row) > 8 else None
            }
        except Exception as e:
            logger.error(f"Error converting row to expert dict: {e}")
            logger.error(f"Row data: {row}")
            raise
    
    def _row_to_task_dict(self, row) -> Dict:
        """Convert database row to task dictionary"""
        try:
            # Handle required_skills field which could be string or int
            required_skills = []
            if row[3]:  # If required_skills field is not None
                if isinstance(row[3], str):
                    try:
                        required_skills = json.loads(row[3])
                    except json.JSONDecodeError:
                        required_skills = [row[3]]  # If not valid JSON, treat as single string
                elif isinstance(row[3], (int, float)):
                    required_skills = [str(row[3])]  # Convert number to string
                else:
                    required_skills = []

            # Ensure we have all required fields
            if len(row) < 7:
                logger.error(f"Task row has insufficient columns: {row}")
                raise ValueError("Task row has insufficient columns")

            return {
                'id': row[0],
                'title': row[1],
                'description': row[2],
                'required_skills': required_skills,
                'remote_allowed': bool(row[4]),
                'priority': row[5] if len(row) > 5 else 1,  # Default to 1 if missing
                'created_at': row[6] if len(row) > 6 else None
            }
        except Exception as e:
            logger.error(f"Error converting row to task dict: {e}")
            logger.error(f"Row data: {row}")
            raise

    def get_expert_workload(self) -> Dict[int, Dict[str, Any]]:
        """Get workload information for all experts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all experts
            cursor.execute("SELECT id, name, max_tasks FROM experts")
            experts = cursor.fetchall() or []
            
            workload = {}
            for expert_id, name, max_tasks in experts:
                try:
                    # Get current task count
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM matches 
                        WHERE expert_id = ?
                    """, (expert_id,))
                    current_tasks = cursor.fetchone()[0] or 0
                    
                    # Calculate availability
                    if current_tasks >= max_tasks:
                        availability = 'Fully Loaded'
                    elif current_tasks > 0:
                        availability = 'Partially Available'
                    else:
                        availability = 'Available'
                    
                    # Get assigned tasks with combined_score
                    cursor.execute("""
                        SELECT t.id, t.title, m.combined_score
                        FROM matches m
                        JOIN tasks t ON m.task_id = t.id
                        WHERE m.expert_id = ?
                        ORDER BY m.combined_score DESC
                    """, (expert_id,))
                    
                    assigned_tasks = []
                    for task_row in cursor.fetchall() or []:
                        try:
                            if len(task_row) >= 3:  # Ensure we have all required fields
                                assigned_tasks.append({
                                    'task_id': task_row[0],
                                    'title': task_row[1],
                                    'similarity_score': round(float(task_row[2] or 0), 2)
                                })
                        except (IndexError, TypeError, ValueError) as e:
                            logger.error(f"Error processing task row for expert {expert_id}: {e}")
                            continue
                    
                    workload[expert_id] = {
                        'name': name,
                        'max_tasks': max_tasks,
                        'current_tasks': current_tasks,
                        'availability': availability,
                        'assigned_tasks': assigned_tasks
                    }
                    
                except Exception as e:
                    logger.error(f"Error getting workload for expert {expert_id}: {e}")
                    workload[expert_id] = {
                        'name': name,
                        'max_tasks': max_tasks,
                        'current_tasks': 0,
                        'availability': 'Unknown',
                        'assigned_tasks': []
                    }
            
            conn.close()
            return workload
            
        except Exception as e:
            logger.error(f"Error getting expert workload: {e}")
            logger.error(traceback.format_exc())
            return {} 