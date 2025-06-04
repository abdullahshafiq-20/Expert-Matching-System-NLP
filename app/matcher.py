import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from app.database import DatabaseManager
from app.processor import CVProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpertMatcher:
    def __init__(self, db_path: str = "data/experts.db"):
        self.db = DatabaseManager(db_path)
        self.processor = CVProcessor()
        
        # Initialize ML models
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        self.lda_model = LatentDirichletAllocation(
            n_components=10,
            random_state=42,
            max_iter=50,
            learning_method='batch'
        )
        
        # Cached data for faster processing
        self._expert_vectors = None
        self._expert_ids = []
        self._topic_distributions = {}
        self._is_fitted = False
    
    def add_expert_from_cv(self, name: str, cv_text: str,
                          remote_experience: bool = True,
                          max_tasks: int = 3) -> int:
        """Add a new expert from CV text"""
        try:
            # Process CV using NLP
            processed_data = self.processor.process_cv(cv_text, name)
            
            # Flatten skills dictionary into a single list
            all_skills = []
            for category_skills in processed_data['skills'].values():
                all_skills.extend(category_skills)
            
            # Ensure experience_years is an integer
            experience_years = int(processed_data['experience']['total_years'])
            
            # Add to database
            expert_id = self.db.add_expert(
                name=processed_data['name'],
                skills=all_skills,
                experience_years=experience_years,
                education=processed_data['education'],
                remote_experience=remote_experience,
                max_tasks=max_tasks,
                raw_text=processed_data['raw_text']
            )
            
            # Clear cache to ensure fresh data
            self._clear_cache()
            
            return expert_id
            
        except Exception as e:
            logger.error(f"Error adding expert from CV: {e}")
            raise
    
    def add_task(self, title: str, description: str, required_skills: List[str],
                 remote_allowed: bool = True, priority: int = 1) -> int:
        """Add a new task to the system"""
        task_id = self.db.add_task(title, description, required_skills, remote_allowed, priority)
        logger.info(f"Added task: {title} with ID {task_id}")
        return task_id
    
    def _clear_cache(self):
        """Clear cached models and data"""
        self._expert_vectors = None
        self._expert_ids = []
        self._topic_distributions = {}
        self._is_fitted = False
    
    def _fit_models(self, force_refit: bool = False):
        """Fit TF-IDF and LDA models on expert data"""
        if self._is_fitted and not force_refit:
            return
        
        experts = self.db.get_all_experts()
        if not experts:
            logger.warning("No experts found for model fitting")
            return
        
        logger.info(f"Fitting models on {len(experts)} experts")
        
        # Prepare text data
        expert_texts = []
        self._expert_ids = []
        
        for expert in experts:
            # Combine all text data for the expert
            text_parts = [expert['raw_text']]
            if expert['skills']:
                text_parts.append(' '.join(expert['skills']))
            
            combined_text = ' '.join(text_parts)
            expert_texts.append(combined_text)
            self._expert_ids.append(expert['id'])
        
        try:
            # Fit TF-IDF vectorizer
            self._expert_vectors = self.tfidf_vectorizer.fit_transform(expert_texts)
            
            # Fit LDA model for topic modeling
            self.lda_model.fit(self._expert_vectors)
            topic_distributions = self.lda_model.transform(self._expert_vectors)
            
            # Cache topic distributions
            for i, expert_id in enumerate(self._expert_ids):
                self._topic_distributions[expert_id] = topic_distributions[i]
            
            self._is_fitted = True
            logger.info("Models fitted successfully")
            
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            self._is_fitted = False
    
    def calculate_text_similarity(self, task_description: str) -> List[Tuple[int, float]]:
        """Calculate text similarity between task and experts"""
        if not self._is_fitted:
            self._fit_models()
        
        if self._expert_vectors is None:
            return []
        
        try:
            # Transform task description to TF-IDF vector
            task_vector = self.tfidf_vectorizer.transform([task_description])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(task_vector, self._expert_vectors).flatten()
            
            # Return expert IDs with their similarity scores
            results = list(zip(self._expert_ids, similarities))
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Text similarity calculation failed: {e}")
            return []
    
    def calculate_skill_overlap(self, expert_skills: List[str], 
                               required_skills: List[str]) -> float:
        """Calculate skill overlap using Jaccard similarity"""
        if not expert_skills or not required_skills:
            return 0.0
        
        # Convert to lowercase for comparison
        expert_set = set([skill.lower() for skill in expert_skills])
        required_set = set([skill.lower() for skill in required_skills])
        
        intersection = len(expert_set.intersection(required_set))
        union = len(expert_set.union(required_set))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_topic_similarity(self, task_description: str, 
                                  expert_id: int) -> float:
        """Calculate topic-based similarity using LDA"""
        if not self._is_fitted or expert_id not in self._topic_distributions:
            return 0.0
        
        try:
            # Get task topic distribution
            task_vector = self.tfidf_vectorizer.transform([task_description])
            task_topics = self.lda_model.transform(task_vector)[0]
            
            # Get expert topic distribution
            expert_topics = self._topic_distributions[expert_id]
            
            # Calculate cosine similarity between topic distributions
            similarity = cosine_similarity([task_topics], [expert_topics])[0][0]
            return similarity
            
        except Exception as e:
            logger.error(f"Topic similarity calculation failed: {e}")
            return 0.0
    
    def calculate_experience_score(self, expert_experience: int, 
                                  task_complexity: int = 3) -> float:
        """Calculate experience score based on task complexity"""
        if expert_experience <= 0:
            return 0.1
        
        # Simple scoring: more experience is better, but with diminishing returns
        optimal_experience = task_complexity * 2  # e.g., 6 years for complexity 3
        
        if expert_experience >= optimal_experience:
            return 1.0
        else:
            return expert_experience / optimal_experience
    
    def check_remote_compatibility(self, expert: Dict, task: Dict) -> bool:
        """Check if expert can work remotely if required"""
        if not task.get('remote_allowed', True):
            return True  # No remote requirement
        
        return expert.get('remote_experience', False)
    
    def find_best_matches(self, task_description: str, required_skills: List[str] = None,
                         remote_required: bool = False, top_k: int = 5,
                         task_complexity: int = 3) -> List[Dict]:
        """Find the best expert matches for a given task"""
        
        # Ensure models are fitted
        self._fit_models()
        
        # Get all experts
        experts = self.db.get_all_experts()
        if not experts:
            logger.warning("No experts found in database")
            return []
        
        logger.info(f"Finding matches for task among {len(experts)} experts")
        
        # Calculate text similarities
        text_similarities = dict(self.calculate_text_similarity(task_description))
        
        matches = []
        
        for expert in experts:
            expert_id = expert['id']
            
            # Check remote compatibility
            if remote_required and not expert.get('remote_experience', False):
                continue
            
            # Calculate various similarity scores
            text_sim = text_similarities.get(expert_id, 0.0)
            topic_sim = self.calculate_topic_similarity(task_description, expert_id)
            
            # Calculate skill overlap
            skill_overlap = 0.0
            matching_skills = []
            if required_skills:
                skill_overlap = self.calculate_skill_overlap(expert['skills'], required_skills)
                expert_skills_lower = [s.lower() for s in expert['skills']]
                required_skills_lower = [s.lower() for s in required_skills]
                matching_skills = [s for s in expert['skills'] 
                                 if s.lower() in required_skills_lower]
            
            # Calculate experience score
            exp_score = self.calculate_experience_score(expert['experience_years'], task_complexity)
            
            # Calculate combined score with weights
            weights = {
                'text': 0.3,
                'topic': 0.2,
                'skill': 0.3,
                'experience': 0.2
            }
            
            combined_score = (
                weights['text'] * text_sim +
                weights['topic'] * topic_sim +
                weights['skill'] * skill_overlap +
                weights['experience'] * exp_score
            )
            
            match_result = {
                'expert_id': expert_id,
                'expert_name': expert['name'],
                # 'expert_email': expert['email'],
                'text_similarity': round(text_sim, 3),
                'topic_similarity': round(topic_sim, 3),
                'skill_overlap': round(skill_overlap, 3),
                'experience_score': round(exp_score, 3),
                'combined_score': round(combined_score, 3),
                'matching_skills': matching_skills,
                'experience_years': expert['experience_years'],
                'remote_capable': expert.get('remote_experience', False),
                'max_tasks': expert.get('max_tasks', 3)
            }
            
            matches.append(match_result)
        
        # Sort by combined score and return top k
        matches.sort(key=lambda x: x['combined_score'], reverse=True)
        top_matches = matches[:top_k]
        
        logger.info(f"Found {len(top_matches)} top matches")
        return top_matches
    
    def find_matches_for_existing_task(self, task_id: int, top_k: int = 5) -> List[Dict]:
        """Find matches for an existing task in the database"""
        task = self.db.get_task(task_id)
        if not task:
            logger.error(f"Task with ID {task_id} not found")
            return []
        
        matches = self.find_best_matches(
            task_description=task['description'],
            required_skills=task['required_skills'],
            remote_required=not task['remote_allowed'],
            top_k=top_k,
            task_complexity=task['priority']
        )
        
        # Add task information to matches
        for match in matches:
            match['task_id'] = task_id
            match['task_title'] = task['title']
            match['task_description'] = task['description']
        
        # Save matches to database
        self.db.save_matches(matches)
        
        return matches
    
    def bulk_match_all_tasks(self) -> Dict[str, int]:
        """Find matches for all tasks in the database"""
        tasks = self.db.get_all_tasks()
        if not tasks:
            logger.warning("No tasks found for bulk matching")
            return {'tasks_processed': 0, 'total_matches': 0}
        
        total_matches = 0
        
        for task in tasks:
            matches = self.find_matches_for_existing_task(task['id'])
            total_matches += len(matches)
            logger.info(f"Found {len(matches)} matches for task: {task['title']}")
        
        return {
            'tasks_processed': len(tasks),
            'total_matches': total_matches
        }
    
    def get_expert_workload(self) -> Dict[int, Dict]:
        """Get current workload for each expert"""
        experts = self.db.get_all_experts()
        matches = self.db.get_matches()
        
        workload = defaultdict(lambda: {'assigned_tasks': 0, 'max_tasks': 3, 'availability': 'Available'})
        
        # Initialize with expert data
        for expert in experts:
            workload[expert['id']].update({
                'expert_name': expert['name'],
                'max_tasks': expert['max_tasks'],
                'assigned_tasks': 0
            })
        
        # Count assigned tasks from matches
        for match in matches:
            expert_id = match['expert_id']
            if expert_id in workload:
                workload[expert_id]['assigned_tasks'] += 1
        
        # Calculate availability
        for expert_id, data in workload.items():
            assigned = data['assigned_tasks']
            max_tasks = data['max_tasks']
            
            if assigned >= max_tasks:
                data['availability'] = 'Fully Loaded'
            elif assigned >= max_tasks * 0.8:
                data['availability'] = 'Nearly Full'
            else:
                data['availability'] = 'Available'
            
            data['load_percentage'] = round((assigned / max_tasks) * 100, 1)
        
        return dict(workload)
    
    def get_skill_demand_analysis(self) -> Dict[str, int]:
        """Analyze which skills are most in demand"""
        tasks = self.db.get_all_tasks()
        skill_demand = Counter()
        
        for task in tasks:
            for skill in task.get('required_skills', []):
                skill_demand[skill.lower()] += 1
        
        return dict(skill_demand.most_common(20))
    
    def get_skill_supply_analysis(self) -> Dict[str, int]:
        """Analyze which skills are most available among experts"""
        experts = self.db.get_all_experts()
        skill_supply = Counter()
        
        for expert in experts:
            for skill in expert.get('skills', []):
                skill_supply[skill.lower()] += 1
        
        return dict(skill_supply.most_common(20))
    
    def get_matching_statistics(self) -> Dict:
        """Get comprehensive matching statistics"""
        stats = self.db.get_system_stats()
        workload = self.get_expert_workload()
        skill_demand = self.get_skill_demand_analysis()
        skill_supply = self.get_skill_supply_analysis()
        
        # Calculate additional metrics
        available_experts = sum(1 for w in workload.values() if w['availability'] == 'Available')
        busy_experts = sum(1 for w in workload.values() if w['availability'] == 'Fully Loaded')
        
        avg_experience = 0
        if stats['total_experts'] > 0:
            experts = self.db.get_all_experts()
            total_exp = sum(expert['experience_years'] for expert in experts)
            avg_experience = round(total_exp / len(experts), 1)
        
        return {
            'basic_stats': stats,
            'expert_availability': {
                'available': available_experts,
                'busy': busy_experts,
                'total': stats['total_experts']
            },
            'average_experience_years': avg_experience,
            'top_demand_skills': list(skill_demand.keys())[:5],
            'top_supply_skills': list(skill_supply.keys())[:5],
            'skill_gap_analysis': self._analyze_skill_gaps(skill_demand, skill_supply)
        }
    
    def _analyze_skill_gaps(self, demand: Dict, supply: Dict) -> List[Dict]:
        """Analyze gaps between skill demand and supply"""
        all_skills = set(demand.keys()) | set(supply.keys())
        gaps = []
        
        for skill in all_skills:
            demand_count = demand.get(skill, 0)
            supply_count = supply.get(skill, 0)
            
            if demand_count > 0:
                gap_ratio = supply_count / demand_count if demand_count > 0 else float('inf')
                gaps.append({
                    'skill': skill,
                    'demand': demand_count,
                    'supply': supply_count,
                    'gap_ratio': round(gap_ratio, 2),
                    'status': 'Oversupplied' if gap_ratio > 2 else 'Undersupplied' if gap_ratio < 0.5 else 'Balanced'
                })
        
        return sorted(gaps, key=lambda x: x['gap_ratio'])[:10] 