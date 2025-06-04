import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Any
import time
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Expert Matching System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for API calls
def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None):
    """Make API request with error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=data)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API server. Please ensure the FastAPI server is running on http://localhost:8000")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

def check_api_health():
    """Check if API is running"""
    health_data = make_api_request("/health")
    if health_data and health_data.get("status") == "healthy":
        return True
    return False

def display_expert_card(expert: Dict):
    """Display expert information in a card format"""
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.subheader(f"👤 {expert['name']}")
            if expert.get('email'):
                st.write(f"📧 {expert['email']}")
            st.write(f"💼 Experience: {expert['experience_years']} years")
            
        with col2:
            if expert.get('skills'):
                st.write("🛠️ **Skills:**")
                skills_text = ", ".join(expert['skills'][:8])  # Show first 8 skills
                if len(expert['skills']) > 8:
                    skills_text += f" (+{len(expert['skills']) - 8} more)"
                st.write(skills_text)
            
            if expert.get('education'):
                st.write(f"🎓 Education: {', '.join(expert['education'])}")
        
        with col3:
            remote_status = "🌐 Remote" if expert.get('remote_experience', False) else "🏢 On-site"
            st.write(remote_status)
            st.write(f"📊 Max Tasks: {expert.get('max_tasks', 3)}")

def display_match_results(matches: List[Dict], use_expanders: bool = True):
    """Display matching results in a formatted way"""
    if not matches:
        st.info("No matches found.")
        return
    
    st.subheader(f"🎯 Top {len(matches)} Matches Found")
    
    for i, match in enumerate(matches, 1):
        if use_expanders:
            with st.expander(f"#{i} {match['expert_name']} - Score: {match['combined_score']:.3f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Expert:** {match['expert_name']}")
                    if match.get('expert_email'):
                        st.write(f"**Email:** {match['expert_email']}")
                    if match.get('experience_years'):
                        st.write(f"**Experience:** {match['experience_years']} years")
                    
                    # Score breakdown
                    st.write("**Score Breakdown:**")
                    st.write(f"• Text Similarity: {match.get('text_similarity', 0):.3f}")
                    st.write(f"• Skill Overlap: {match.get('skill_overlap', 0):.3f}")
                    st.write(f"• Combined Score: {match.get('combined_score', 0):.3f}")
                
                with col2:
                    if match.get('matching_skills'):
                        st.write("**Matching Skills:**")
                        for skill in match['matching_skills']:
                            st.write(f"✅ {skill}")
                    
                    remote_status = "🌐 Remote capable" if match.get('remote_capable', False) else "🏢 On-site only"
                    st.write(f"**Work Style:** {remote_status}")
        else:
            # Use containers with borders when expanders can't be used
            with st.container():
                st.markdown(f"**#{i} {match['expert_name']} - Score: {match.get('combined_score', 0):.3f}**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Expert:** {match['expert_name']}")
                    if match.get('expert_email'):
                        st.write(f"**Email:** {match['expert_email']}")
                    if match.get('experience_years'):
                        st.write(f"**Experience:** {match['experience_years']} years")
                    
                    # Score breakdown
                    st.write("**Score Breakdown:**")
                    st.write(f"• Text Similarity: {match.get('text_similarity', 0):.3f}")
                    st.write(f"• Skill Overlap: {match.get('skill_overlap', 0):.3f}")
                    st.write(f"• Combined Score: {match.get('combined_score', 0):.3f}")
                
                with col2:
                    if match.get('matching_skills'):
                        st.write("**Matching Skills:**")
                        for skill in match['matching_skills']:
                            st.write(f"✅ {skill}")
                    
                    remote_status = "🌐 Remote capable" if match.get('remote_capable', False) else "🏢 On-site only"
                    st.write(f"**Work Style:** {remote_status}")
                
                st.markdown("---")  # Add separator between matches

# Main App Navigation
def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Expert Matching System</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("🚨 API server is not responding. Please start the FastAPI server first.")
        st.code("uvicorn app.main:app --reload", language="bash")
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "👥 Experts", "📋 Tasks", "🎯 Matching", "📊 Analytics", "⚙️ Admin"]
    )
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "👥 Experts":
        show_experts_page()
    elif page == "📋 Tasks":
        show_tasks_page()
    elif page == "🎯 Matching":
        show_matching_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "⚙️ Admin":
        show_admin_page()

def show_dashboard():
    """Display main dashboard with system overview"""
    st.header("📊 System Overview")
    
    # Get system statistics
    stats_data = make_api_request("/stats/")
    if not stats_data:
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{stats_data['basic_stats']['total_experts']}</h3>
            <p>Total Experts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{stats_data['basic_stats']['total_tasks']}</h3>
            <p>Total Tasks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{stats_data['basic_stats']['total_matches']}</h3>
            <p>Total Matches</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{stats_data['basic_stats']['avg_similarity_score']}</h3>
            <p>Avg Match Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Expert availability chart
    st.subheader("👥 Expert Availability")
    availability = stats_data['expert_availability']
    
    fig_availability = go.Figure(data=[
        go.Bar(x=['Available', 'Busy', 'Total'], 
               y=[availability['available'], availability['busy'], availability['total']],
               marker_color=['green', 'red', 'blue'])
    ])
    fig_availability.update_layout(title="Expert Availability Status", height=400)
    st.plotly_chart(fig_availability, use_container_width=True)
    
    # Top skills
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Top Skills in Demand")
        if stats_data.get('top_demand_skills'):
            for i, skill in enumerate(stats_data['top_demand_skills'], 1):
                st.write(f"{i}. {skill}")
        else:
            st.info("No demand data available")
    
    with col2:
        st.subheader("💼 Top Skills Available")
        if stats_data.get('top_supply_skills'):
            for i, skill in enumerate(stats_data['top_supply_skills'], 1):
                st.write(f"{i}. {skill}")
        else:
            st.info("No supply data available")

def show_experts_page():
    """Display experts management page"""
    st.header("👥 Expert Management")
    
    tab1, tab2, tab3 = st.tabs(["📋 View Experts", "➕ Add Expert", "📤 Upload CV"])
    
    with tab1:
        st.subheader("📋 All Experts")
        
        # Get all experts
        experts = make_api_request("/experts/")
        if experts:
            st.write(f"Total experts: {len(experts)}")
            
            for expert in experts:
                with st.expander(f"👤 {expert['name']} (ID: {expert['id']})"):
                    display_expert_card(expert)
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button(f"🗑️ Delete", key=f"delete_{expert['id']}"):
                            if make_api_request(f"/experts/{expert['id']}", method="DELETE"):
                                st.success("Expert deleted successfully!")
                                st.rerun()
        else:
            st.info("No experts found. Add some experts to get started!")
    
    with tab2:
        st.subheader("➕ Add New Expert")
        
        with st.form("add_expert_form"):
            name = st.text_input("Expert Name*", placeholder="Enter full name")
            email = st.text_input("Email", placeholder="expert@example.com")
            cv_text = st.text_area("CV Text*", height=300, placeholder="Paste the expert's CV content here...")
            
            col1, col2 = st.columns(2)
            with col1:
                remote_experience = st.checkbox("Has Remote Experience", value=True)
            with col2:
                max_tasks = st.number_input("Max Concurrent Tasks", min_value=1, max_value=10, value=3)
            
            submitted = st.form_submit_button("➕ Add Expert")
            
            if submitted:
                if name and cv_text:
                    expert_data = {
                        "name": name,
                        "email": email if email else None,
                        "cv_text": cv_text,
                        "remote_experience": remote_experience,
                        "max_tasks": max_tasks
                    }
                    
                    result = make_api_request("/experts/", method="POST", data=expert_data)
                    if result:
                        st.success(f"✅ Expert '{name}' added successfully!")
                        st.json(result)
                else:
                    st.error("Please fill in required fields (Name and CV Text)")
    
    with tab3:
        st.subheader("📤 Upload CV File")
        
        with st.form("upload_cv_form"):
            uploaded_file = st.file_uploader("Choose CV file", type=['txt'], help="Upload a text file containing the CV")
            name = st.text_input("Expert Name*", placeholder="Enter full name")
            email = st.text_input("Email", placeholder="expert@example.com")
            
            col1, col2 = st.columns(2)
            with col1:
                remote_experience = st.checkbox("Has Remote Experience", value=True, key="upload_remote")
            with col2:
                max_tasks = st.number_input("Max Concurrent Tasks", min_value=1, max_value=10, value=3, key="upload_max_tasks")
            
            submitted = st.form_submit_button("📤 Upload CV")
            
            if submitted:
                if uploaded_file and name:
                    # Prepare form data for file upload
                    files = {"file": uploaded_file}
                    form_data = {
                        "name": name,
                        "email": email if email else "",
                        "remote_experience": remote_experience,
                        "max_tasks": max_tasks
                    }
                    
                    result = make_api_request("/experts/upload-cv/", method="POST", data=form_data, files=files)
                    if result:
                        st.success(f"✅ CV uploaded and expert '{name}' created successfully!")
                        st.json(result)
                else:
                    st.error("Please upload a file and enter the expert's name")

def show_tasks_page():
    """Display tasks management page"""
    st.header("📋 Task Management")
    
    tab1, tab2 = st.tabs(["📋 View Tasks", "➕ Add Task"])
    
    with tab1:
        st.subheader("📋 All Tasks")
        
        tasks = make_api_request("/tasks/")
        if tasks:
            st.write(f"Total tasks: {len(tasks)}")
            
            for task in tasks:
                with st.expander(f"📋 {task['title']} (Priority: {task['priority']})"):
                    st.write(f"**Description:** {task['description']}")
                    
                    if task.get('required_skills'):
                        st.write("**Required Skills:**")
                        skills_cols = st.columns(min(len(task['required_skills']), 4))
                        for i, skill in enumerate(task['required_skills']):
                            with skills_cols[i % 4]:
                                st.write(f"• {skill}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        remote_text = "🌐 Remote allowed" if task.get('remote_allowed', True) else "🏢 On-site only"
                        st.write(remote_text)
                    with col2:
                        st.write(f"⭐ Priority: {task['priority']}/5")
                    with col3:
                        if st.button(f"🎯 Find Matches", key=f"match_{task['id']}"):
                            st.session_state.selected_task_id = task['id']
                            st.rerun()
                    
                    # Show matches if task is selected
                    if st.session_state.get('selected_task_id') == task['id']:
                        with st.spinner("Finding matches..."):
                            matches = make_api_request(f"/match/task/{task['id']}", method="POST")
                            if matches:
                                display_match_results(matches, use_expanders=False)
        else:
            st.info("No tasks found. Add some tasks to get started!")
    
    with tab2:
        st.subheader("➕ Add New Task")
        
        with st.form("add_task_form"):
            title = st.text_input("Task Title*", placeholder="Enter task title")
            description = st.text_area("Task Description*", height=200, placeholder="Detailed description of the task...")
            
            # Skills input
            skills_input = st.text_input("Required Skills", placeholder="Enter skills separated by commas (e.g., Python, Machine Learning, Data Analysis)")
            required_skills = [skill.strip() for skill in skills_input.split(',') if skill.strip()] if skills_input else []
            
            col1, col2 = st.columns(2)
            with col1:
                remote_allowed = st.checkbox("Remote Work Allowed", value=True)
            with col2:
                priority = st.slider("Priority Level", min_value=1, max_value=5, value=3)
            
            submitted = st.form_submit_button("➕ Add Task")
            
            if submitted:
                if title and description:
                    task_data = {
                        "title": title,
                        "description": description,
                        "required_skills": required_skills,
                        "remote_allowed": remote_allowed,
                        "priority": priority
                    }
                    
                    result = make_api_request("/tasks/", method="POST", data=task_data)
                    if result:
                        st.success(f"✅ Task '{title}' added successfully!")
                        st.json(result)
                else:
                    st.error("Please fill in required fields (Title and Description)")

def show_matching_page():
    """Display matching functionality page"""
    st.header("🎯 Expert-Task Matching")
    
    tab1, tab2 = st.tabs(["🔍 Find Matches", "📊 Bulk Matching"])
    
    with tab1:
        st.subheader("🔍 Find Expert Matches")
        
        with st.form("matching_form"):
            st.write("**Describe the task or project:**")
            task_description = st.text_area("Task Description", height=150, 
                                          placeholder="Describe what needs to be done, technologies required, project goals, etc.")
            
            col1, col2 = st.columns(2)
            with col1:
                skills_input = st.text_input("Required Skills", 
                                           placeholder="Python, Machine Learning, Web Development")
                required_skills = [skill.strip() for skill in skills_input.split(',') if skill.strip()] if skills_input else []
            
            with col2:
                top_k = st.number_input("Number of matches to show", min_value=1, max_value=20, value=5)
            
            col3, col4 = st.columns(2)
            with col3:
                remote_required = st.checkbox("Remote work required")
            
            submitted = st.form_submit_button("🎯 Find Matches")
            
            if submitted and task_description:
                match_data = {
                    "task_description": task_description,
                    "required_skills": required_skills,
                    "remote_required": remote_required,
                    "top_k": top_k
                }
                
                with st.spinner("🔍 Finding the best expert matches..."):
                    matches = make_api_request("/match/find", method="POST", data=match_data)
                    if matches:
                        display_match_results(matches)
                    else:
                        st.warning("No matches found or error occurred.")
    
    with tab2:
        st.subheader("📊 Bulk Matching")
        st.write("Run matching algorithms for all existing tasks in the system.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🚀 Run Bulk Matching"):
                # Initialize session state for tracking bulk matching status
                if 'bulk_matching_status' not in st.session_state:
                    st.session_state.bulk_matching_status = None
                
                # Start bulk matching
                result = make_api_request("/match/bulk", method="POST")
                if result:
                    st.session_state.bulk_matching_status = "processing"
                    st.success("✅ Bulk matching started!")
                    
                    # Show loading state
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Poll for results
                    max_attempts = 30  # Maximum number of polling attempts
                    for attempt in range(max_attempts):
                        # Update progress bar
                        progress = min((attempt + 1) / max_attempts, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing... {int(progress * 100)}%")
                        
                        # Check matches endpoint for results
                        matches = make_api_request("/matches/")
                        if matches:
                            st.session_state.bulk_matching_status = "completed"
                            progress_bar.progress(1.0)
                            status_text.text("✅ Processing complete!")
                            
                            # Display results
                            st.subheader("📊 Matching Results")
                            st.write(f"Total matches found: {len(matches)}")
                            
                            # Group matches by task
                            task_matches = {}
                            for match in matches:
                                task_id = match.get('task_id')
                                if task_id not in task_matches:
                                    task_matches[task_id] = []
                                task_matches[task_id].append(match)
                            
                            # Display matches for each task
                            for task_id, task_matches_list in task_matches.items():
                                task = make_api_request(f"/tasks/{task_id}")
                                if task:
                                    with st.expander(f"📋 Task: {task['title']} ({len(task_matches_list)} matches)"):
                                        display_match_results(task_matches_list, use_expanders=False)
                            
                            break
                        
                        time.sleep(1)  # Wait before next poll
                    
                    if st.session_state.bulk_matching_status != "completed":
                        st.warning("⚠️ Bulk matching is taking longer than expected. Please check the results later.")
        
        with col2:
            st.info("💡 This will find matches for all tasks in the database and save them for quick access.")

def show_analytics_page():
    """Display analytics and insights page"""
    st.header("📊 Analytics & Insights")
    
    # Get analytics data
    stats_data = make_api_request("/stats/")
    workload_data = make_api_request("/stats/workload")
    skill_demand = make_api_request("/stats/skills/demand")
    skill_supply = make_api_request("/stats/skills/supply")
    
    if not all([stats_data, workload_data, skill_demand, skill_supply]):
        st.error("Failed to load analytics data")
        return
    
    # Expert workload analysis
    st.subheader("👥 Expert Workload Analysis")
    
    if workload_data:
        workload_df = pd.DataFrame.from_dict(workload_data, orient='index').reset_index()
        workload_df.columns = ['Expert ID', 'Expert Name', 'Max Tasks', 'Assigned Tasks', 'Availability', 'load_percentage']
        
        fig_workload = px.bar(workload_df, x='Expert Name', y='load_percentage', 
                             color='Availability', title="Expert Workload Distribution")
        st.plotly_chart(fig_workload, use_container_width=True)
        
        st.dataframe(workload_df, use_container_width=True)
    
    # Skill analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Skill Demand")
        if skill_demand:
            demand_df = pd.DataFrame(list(skill_demand.items()), columns=['Skill', 'Demand'])
            demand_df = demand_df.head(10)  # Top 10
            
            fig_demand = px.bar(demand_df, x='Demand', y='Skill', orientation='h',
                               title="Top 10 Skills in Demand")
            st.plotly_chart(fig_demand, use_container_width=True)
    
    with col2:
        st.subheader("💼 Skill Supply")
        if skill_supply:
            supply_df = pd.DataFrame(list(skill_supply.items()), columns=['Skill', 'Supply'])
            supply_df = supply_df.head(10)  # Top 10
            
            fig_supply = px.bar(supply_df, x='Supply', y='Skill', orientation='h',
                               title="Top 10 Available Skills")
            st.plotly_chart(fig_supply, use_container_width=True)
    
    # Skill gap analysis
    st.subheader("⚖️ Skill Gap Analysis")
    if stats_data.get('skill_gap_analysis'):
        gap_data = stats_data['skill_gap_analysis']
        gap_df = pd.DataFrame(gap_data)
        
        fig_gap = px.scatter(gap_df, x='demand', y='supply', hover_name='skill',
                            color='status', size='gap_ratio', 
                            title="Skill Supply vs Demand Analysis")
        st.plotly_chart(fig_gap, use_container_width=True)
        
        st.dataframe(gap_df, use_container_width=True)

def show_admin_page():
    """Display admin controls and system management options"""
    st.header("⚙️ Admin Controls")
    
    # Create two columns for different admin functions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Database Management")
        
        # Add seed database button
        if st.button("🌱 Seed Database", help="Load sample data from the data directory"):
            with st.spinner("Seeding database with sample data..."):
                response = make_api_request("/data/seed", method="POST")
                if response:
                    st.success(f"""
                    Database seeded successfully!
                    - Added {response['experts_added']} experts
                    - Added {response['tasks_added']} tasks
                    - Created {response['matches_created']['total_matches']} matches
                    """)
                else:
                    st.error("Failed to seed database. Please check the server logs.")
        
        # Add system health check
        st.subheader("🔍 System Health")
        health_data = make_api_request("/health")
        if health_data:
            st.success(f"""
            System Status: {health_data['status']}
            - Database: {health_data['database']}
            - Experts: {health_data['experts_count']}
            - Tasks: {health_data['tasks_count']}
            """)
    
    with col2:
        st.subheader("📊 System Statistics")
        stats = make_api_request("/stats/")
        if stats:
            st.write("**Expert Availability:**")
            st.write(f"- Available: {stats['expert_availability']['available']}")
            st.write(f"- Busy: {stats['expert_availability']['busy']}")
            st.write(f"- Total: {stats['expert_availability']['total']}")
            
            st.write("**Average Experience:**")
            st.write(f"- {stats['average_experience_years']} years")
            
            st.write("**Top Skills in Demand:**")
            for skill in stats['top_demand_skills']:
                st.write(f"- {skill}")
    
    # Add a section for system maintenance
    st.subheader("🔧 System Maintenance")
    if st.button("🔄 Clear Cache", help="Clear the matcher cache to force re-indexing"):
        st.info("Cache cleared. The system will rebuild indexes on next matching operation.")
    
    # Add a section for data export
    st.subheader("📤 Data Export")
    if st.button("📥 Export System Data", help="Export current system data as JSON"):
        experts = make_api_request("/experts/")
        tasks = make_api_request("/tasks/")
        matches = make_api_request("/matches/")
        
        if experts and tasks and matches:
            export_data = {
                "experts": experts,
                "tasks": tasks,
                "matches": matches,
                "export_date": datetime.now().isoformat()
            }
            
            # Create download button
            st.download_button(
                label="⬇️ Download System Data",
                data=json.dumps(export_data, indent=2),
                file_name=f"system_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Initialize session state
if 'selected_task_id' not in st.session_state:
    st.session_state.selected_task_id = None

# Run the app
if __name__ == "__main__":
    main() 