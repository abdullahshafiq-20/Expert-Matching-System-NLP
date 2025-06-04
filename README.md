# Expert Matching System

An intelligent system for matching experts with tasks based on expertise, skills, and constraints. This implementation is based on the paper "Expertise Matching via Constraint-based Optimization" by Tang et al.

## Overview

The Expert Matching System is designed to solve the complex problem of matching experts with tasks while considering multiple constraints and optimization criteria. The system uses advanced NLP techniques and machine learning to ensure optimal matches based on expertise, skills, experience, and various constraints.

## Key Features

- **Multi-dimensional Matching**: Combines text similarity, topic modeling, skill overlap, and experience scoring
- **Constraint-based Optimization**: Considers various constraints like workload balance, remote work capability, and task complexity
- **Real-time Feedback**: Supports online adjustment of matches based on user feedback
- **Comprehensive Analysis**: Provides skill demand/supply analysis and workload statistics
- **Scalable Architecture**: Built with performance and scalability in mind

## Technical Implementation

### Core Components

1. **ExpertMatcher Class**
   - Handles the core matching logic
   - Implements multiple similarity metrics
   - Manages expert and task data

2. **Matching Algorithms**
   - Text similarity using TF-IDF and cosine similarity
   - Topic modeling using LDA (Latent Dirichlet Allocation)
   - Skill overlap calculation using Jaccard similarity
   - Experience scoring based on task complexity

3. **Constraint Handling**
   - Workload balance
   - Remote work compatibility
   - Task complexity matching
   - Maximum task limits per expert

### Matching Process

1. **Expert Profile Creation**
   - CV text processing
   - Skill extraction
   - Experience calculation
   - Topic distribution generation

2. **Task Analysis**
   - Required skills identification
   - Complexity assessment
   - Remote work requirements

3. **Match Generation**
   - Combined scoring using multiple factors
   - Constraint satisfaction
   - Top-k match selection

## API Endpoints

- `POST /experts/`: Create new expert profile
- `POST /tasks/`: Add new task
- `GET /matches/{task_id}`: Get matches for specific task
- `GET /workload`: Get expert workload statistics
- `GET /skills/analysis`: Get skill demand/supply analysis

## Usage Example

```python
# Initialize the matcher
matcher = ExpertMatcher()

# Add an expert
expert_id = matcher.add_expert_from_cv(
    name="John Doe",
    cv_text="...",
    remote_experience=True,
    max_tasks=3
)

# Add a task
task_id = matcher.add_task(
    title="Python Development",
    description="...",
    required_skills=["Python", "Django", "REST APIs"],
    remote_allowed=True,
    priority=2
)

# Find matches
matches = matcher.find_matches_for_existing_task(task_id, top_k=5)
```

## Performance Metrics

The system evaluates matches based on:
- Text similarity score
- Topic similarity score
- Skill overlap percentage
- Experience score
- Combined weighted score

## Future Improvements

1. Enhanced NLP capabilities for better text understanding
2. More sophisticated constraint handling
3. Real-time learning from user feedback
4. Integration with external expert databases
5. Advanced workload optimization

## References

- Tang, W., Tang, J., & Tan, C. (2010). Expertise Matching via Constraint-based Optimization. IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology.
