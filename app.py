# Edexcel A-Level Physics Question Generator
# Enhanced Version with Modern UI, User Authentication, and Advanced Features

import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import datetime
import uuid
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import hashlib
import time
from io import BytesIO
import os
from pathlib import Path
import re
import logging

# Import from src directory
from src import pplx
from src import styles
from src.utils import ensure_log_directory

# Set up logging for the main application
log_dir = ensure_log_directory()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PhysicsApp")

# ===============================
# Configuration and Setup
# ===============================

# Set page configuration
st.set_page_config(
    page_title="A-Level Physics Question Generator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'token_count' not in st.session_state:
    st.session_state.token_count = 0
if 'question_history' not in st.session_state:
    st.session_state.question_history = []
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = {}
if 'current_question_id' not in st.session_state:
    st.session_state.current_question_id = None
if 'users' not in st.session_state:
    # In a real app, this would come from a database
    st.session_state.users = {
        'student1': {
            'password': hashlib.sha256('password1'.encode()).hexdigest(),
            'name': 'Student One',
            'preferred_topics': [],
            'completed_questions': 0,
            'topic_counts': {}
        },
        'teacher1': {
            'password': hashlib.sha256('password2'.encode()).hexdigest(),
            'name': 'Teacher One',
            'preferred_topics': [],
            'completed_questions': 0,
            'topic_counts': {}
        },
        'admin': {
            'password': hashlib.sha256('123'.encode()).hexdigest(),
            'name': 'Administrator',
            'role': 'admin',
            'preferred_topics': [],
            'completed_questions': 0,
            'topic_counts': {}
        }
    }
if 'pinecone_data' not in st.session_state:
    st.session_state.pinecone_data = []
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

# ===============================
# Helper Functions
# ===============================

def apply_theme():
    """Apply the appropriate theme based on dark mode setting"""
    if st.session_state.dark_mode:
        # Dark theme
        st.markdown("""
        <style>
            .main {background-color: #121212; color: #FFFFFF;}
            .stApp {background-color: #121212;}
            .css-1d391kg {background-color: #1E1E1E;}
            .stSidebar {background-color: #1E1E1E;}
            .stButton>button {background-color: #3B82F6; color: white;}
            .main-header {font-size: 2.5rem; color: #3B82F6;}
            .sub-header {font-size: 1.5rem; color: #60A5FA;}
            .card {background-color: #2D2D2D; border-radius: 10px; padding: 20px; margin-bottom: 20px;}
            .feedback-button {padding: 5px 15px; margin-right: 10px; border-radius: 5px; border: none; cursor: pointer;}
            .positive {background-color: #10B981; color: white;}
            .neutral {background-color: #6B7280; color: white;}
            .negative {background-color: #EF4444; color: white;}
            .token-counter {background-color: #374151; padding: 10px; border-radius: 5px; margin-top: 10px;}
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light theme
        st.markdown("""
        <style>
            .main {background-color: #FFFFFF; color: #1F2937;}
            .stApp {background-color: #FFFFFF;}
            .css-1d391kg {background-color: #F3F4F6;}
            .stSidebar {background-color: #F3F4F6;}
            .stButton>button {background-color: #2563EB; color: white;}
            .main-header {font-size: 2.5rem; color: #1E3A8A;}
            .sub-header {font-size: 1.5rem; color: #3B82F6;}
            .card {background-color: #F9FAFB; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);}
            .feedback-button {padding: 5px 15px; margin-right: 10px; border-radius: 5px; border: none; cursor: pointer;}
            .positive {background-color: #10B981; color: white;}
            .neutral {background-color: #6B7280; color: white;}
            .negative {background-color: #EF4444; color: white;}
            .token-counter {background-color: #E5E7EB; padding: 10px; border-radius: 5px; margin-top: 10px;}
        </style>
        """, unsafe_allow_html=True)

def simulate_pinecone_storage(question_data):
    """
    Simulate storing question data in Pinecone
    In a real app, this would use the Pinecone API
    """
    # Just append to session state for now
    st.session_state.pinecone_data.append(question_data)
    
    # Update user stats
    if st.session_state.authenticated:
        username = st.session_state.username
        st.session_state.users[username]['completed_questions'] += 1
        
        # Track topic preference
        topic = question_data['params']['topic']
        if 'topic_counts' not in st.session_state.users[username]:
            st.session_state.users[username]['topic_counts'] = {}
        
        if topic in st.session_state.users[username]['topic_counts']:
            st.session_state.users[username]['topic_counts'][topic] += 1
        else:
            st.session_state.users[username]['topic_counts'][topic] = 1

def retrieve_user_history(username):
    """
    Retrieve question history for a specific user
    In a real app, this would query Pinecone
    """
    # In a real app, you would filter by username in the database query
    return [q for q in st.session_state.pinecone_data if q.get('user') == username]

def call_n8n_api(params):
    """
    This function would normally call the n8n API, but for now it returns placeholder content
    and simulates token usage
    """
    # Simulate API call delay
    time.sleep(1)
    
    # Simulate token usage
    tokens_used = 150 + (50 if params["difficulty"] == "Hard" else 0) + (100 if params["include_diagram"] else 0)
    st.session_state.token_count += tokens_used
    
    # Placeholder response based on parameters
    topic_content = {
        "Mechanics - Motion": "A car accelerates uniformly from rest to 30 m/s in 6.0 seconds.",
        "Mechanics - Forces": "A 2.0 kg mass is suspended from a spring with spring constant 50 N/m.",
        "Electricity - Circuits": "A circuit contains a 12V battery and three resistors in series: 4Ω, 6Ω, and 10Ω.",
        "Electricity - Fields": "A point charge of +2.0 μC is placed at the origin of a coordinate system.",
        "Waves - Properties": "A sound wave travels through air with a frequency of 440 Hz and wavelength of 0.78 m.",
        "Waves - Optics": "Light passes from air (n=1.0) into glass (n=1.5) at an angle of incidence of 30°.",
        "Nuclear Physics": "A radioactive sample initially contains 8.0 × 10^20 atoms of a radioisotope with half-life 5.0 days.",
        "Thermodynamics": "An ideal gas undergoes an isothermal expansion, doubling its volume while maintaining a temperature of 300K.",
        "Particle Physics": "A proton and an antiproton annihilate, producing two gamma ray photons.",
        "Magnetic Fields": "A straight wire carrying a current of 5.0 A is placed perpendicular to a uniform magnetic field of 0.2 T."
    }
    
    difficulty_modifier = {
        "Easy": "Calculate the final velocity of the car.",
        "Medium": "Calculate the acceleration and the distance traveled during this time.",
        "Hard": "If the car continues with the same acceleration for another 4.0 seconds, calculate the total distance traveled from the start and the average velocity over the entire 10.0 second interval."
    }
    
    # Create a response structure
    question_id = str(uuid.uuid4())
    response = {
        "id": question_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "params": params,
        "question": f"{topic_content.get(params['topic'], 'A physics problem related to the selected topic.')} {difficulty_modifier.get(params['difficulty'], '')}",
        "hints": ["Consider the relevant equations of motion.", 
                 "Start by identifying the known and unknown variables.", 
                 "Remember to check your units for consistency."] if params["include_hints"] else [],
        "diagram_data": {"type": get_diagram_type(params["topic"])} if params["include_diagram"] else None,
        "tokens_used": tokens_used,
        "user": st.session_state.username if st.session_state.authenticated else "guest"
    }
    
    # Store in session state history
    st.session_state.question_history.append(response)
    st.session_state.current_question_id = question_id
    
    # Simulate storing in Pinecone
    simulate_pinecone_storage(response)
    
    return response

def get_diagram_type(topic):
    """Determine diagram type based on topic"""
    topic_diagram_map = {
        "Mechanics - Motion": "motion_graph",
        "Mechanics - Forces": "force_diagram",
        "Electricity - Circuits": "circuit_diagram",
        "Electricity - Fields": "field_lines",
        "Waves - Properties": "wave_diagram",
        "Waves - Optics": "ray_diagram",
        "Nuclear Physics": "decay_diagram",
        "Thermodynamics": "pv_diagram",
        "Particle Physics": "feynman_diagram",
        "Magnetic Fields": "magnetic_field"
    }
    return topic_diagram_map.get(topic, "generic_diagram")

def generate_interactive_diagram(diagram_data):
    """Generate interactive diagrams based on the topic"""
    if not diagram_data:
        return None
    
    diagram_type = diagram_data["type"]
    
    if diagram_type == "motion_graph":
        # Generate an interactive velocity-time graph
        t = np.linspace(0, 10, 100)
        v = 5 * t  # Simple v = at model
        
        fig = px.line(x=t, y=v, labels={"x": "Time (s)", "y": "Velocity (m/s)"})
        fig.update_layout(
            title="Velocity-Time Graph (Interactive)",
            hovermode="closest",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        # Add area under curve to show distance
        fig.add_trace(go.Scatter(
            x=t, y=v,
            fill='tozeroy',
            fillcolor='rgba(0, 176, 246, 0.2)',
            line=dict(color='rgba(0, 176, 246, 0.8)'),
            name='Area = Distance'
        ))
        
        return fig
    
    elif diagram_type == "circuit_diagram":
        # Create a simple circuit diagram using plotly
        fig = go.Figure()
        
        # Create a circuit layout
        fig.update_layout(
            title="Circuit Diagram",
            xaxis=dict(range=[0, 10], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[0, 5], showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True
        )
        
        # Add circuit elements as annotations
        fig.add_annotation(x=1, y=2.5, text="12V", showarrow=False)
        fig.add_annotation(x=3, y=1, text="4Ω", showarrow=False)
        fig.add_annotation(x=6, y=1, text="6Ω", showarrow=False)
        fig.add_annotation(x=9, y=1, text="10Ω", showarrow=False)
        
        # Add traces for interactive elements
        fig.add_trace(go.Scatter(
            x=[1, 1], y=[1, 4],
            mode='lines',
            line=dict(color='black', width=3),
            name='Battery'
        ))
        
        # Add resistors
        for i, pos in enumerate([3, 6, 9]):
            fig.add_trace(go.Scatter(
                x=[pos-1, pos+1], y=[1, 1],
                mode='lines',
                line=dict(color=['red', 'green', 'blue'][i], width=3),
                name=f'Resistor {i+1}'
            ))
        
        # Connect components
        fig.add_trace(go.Scatter(
            x=[1, 2, 4, 5, 7, 8, 10, 10, 1],
            y=[1, 1, 1, 1, 1, 1, 1, 4, 4],
            mode='lines',
            line=dict(color='black', width=2),
            name='Wires'
        ))
        
        return fig
    
    elif diagram_type == "force_diagram":
        # Create a force diagram
        fig = go.Figure()
        
        # Set up the figure
        fig.update_layout(
            title="Force Diagram - Spring System",
            xaxis=dict(range=[-2, 2], showgrid=True),
            yaxis=dict(range=[-2, 2], showgrid=True),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Add a mass
        fig.add_shape(
            type="circle",
            x0=-0.5, y0=-0.5, x1=0.5, y1=0.5,
            line=dict(color="blue", width=2),
            fillcolor="lightblue",
        )
        
        # Add spring
        x_spring = np.linspace(-1.5, -0.5, 20)
        y_spring = 0.2 * np.sin(np.linspace(0, 10*np.pi, 20))
        
        fig.add_trace(go.Scatter(
            x=x_spring, y=y_spring,
            mode='lines',
            line=dict(color='red', width=3),
            name='Spring'
        ))
        
        # Add force arrows
        fig.add_annotation(
            x=0, y=0.7,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor="green"
        )
        
        fig.add_annotation(
            x=0, y=-0.7,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor="orange"
        )
        
        # Add labels
        fig.add_annotation(x=0, y=1, text="Fg = mg", showarrow=False)
        fig.add_annotation(x=0, y=-1, text="Fs = kx", showarrow=False)
        
        return fig
    
    # Default case - return a basic diagram
    fig = go.Figure()
    fig.update_layout(
        title="Generic Physics Diagram",
        xaxis=dict(range=[0, 10], showgrid=True),
        yaxis=dict(range=[0, 5], showgrid=True),
        height=400
    )
    return fig

def record_feedback(question_id, rating):
    """Record user feedback for a question"""
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = {}
    
    st.session_state.feedback_data[question_id] = {
        'rating': rating,
        'timestamp': datetime.datetime.now().isoformat(),
        'user': st.session_state.username if st.session_state.authenticated else "guest"
    }
    
    st.success(f"Thank you for your {rating} feedback!")

def export_questions_as_json():
    """Export question history as a downloadable JSON file"""
    if not st.session_state.question_history:
        st.warning("No questions to export.")
        return None
    
    # Filter questions for the current user if authenticated
    if st.session_state.authenticated:
        user_questions = [q for q in st.session_state.question_history 
                         if q.get('user') == st.session_state.username]
    else:
        user_questions = st.session_state.question_history
    
    if not user_questions:
        st.warning("No questions to export for your account.")
        return None
    
    # Create JSON string
    json_str = json.dumps(user_questions, indent=4)
    
    # Return as downloadable file
    return json_str

def authenticate_user(username, password):
    """Authenticate a user with username and password"""
    if username in st.session_state.users:
        stored_password = st.session_state.users[username]['password']
        if stored_password == hashlib.sha256(password.encode()).hexdigest():
            st.session_state.authenticated = True
            st.session_state.username = username
            
            # Set admin flag if user is an admin
            if 'role' in st.session_state.users[username] and st.session_state.users[username]['role'] == 'admin':
                st.session_state.is_admin = True
            else:
                st.session_state.is_admin = False
                
            return True
    return False

def register_user(username, password, name):
    """Register a new user"""
    if username in st.session_state.users:
        return False, "Username already exists"
    
    # In a real app, you would store this in a database
    st.session_state.users[username] = {
        'password': hashlib.sha256(password.encode()).hexdigest(),
        'name': name,
        'preferred_topics': [],
        'completed_questions': 0,
        'topic_counts': {}
    }
    return True, "Registration successful!"

def logout_user():
    """Log out the current user"""
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.is_admin = False
    st.success("You have been logged out successfully.")

def get_topic_statistics(username):
    """Get topic statistics for a user"""
    if username not in st.session_state.users or 'topic_counts' not in st.session_state.users[username]:
        return {}
    
    return st.session_state.users[username]['topic_counts']

def generate_perplexity_question(params):
    """
    Generate a physics question using the Perplexity API
    
    Args:
        params: Dictionary containing question parameters
        
    Returns:
        Dictionary containing the generated question and metadata
    """
    # Get the current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths to guide and examples files
    guide_path = current_dir / "examples/guide.txt"
    examples_path = current_dir / "examples.docx"
    
    # Process guide and examples
    guide_content = pplx.process_guide(guide_path)
    examples_content = pplx.process_examples(examples_path)
    
    # Extract parameters
    topic = params.get("topic", "")
    difficulty = params.get("difficulty", "Medium")
    question_type = params.get("question_type", "calculation")
    
    # Call the Perplexity API function
    response_text = pplx.generate_physics_question(
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
        guide_content=guide_content,
        examples_content=examples_content
    )
    
    if not response_text:
        return {
            "error": "Failed to generate question using Perplexity API"
        }
    
    # Create a response structure similar to the existing one
    question_id = str(uuid.uuid4())
    response = {
        "id": question_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "params": params,
        "question": response_text,
        "tokens_used": 0,  # We don't have this info from Perplexity
        "user": st.session_state.username if st.session_state.authenticated else "guest",
        "source": "perplexity"
    }
    
    # Store in session state history
    st.session_state.question_history.append(response)
    st.session_state.current_question_id = question_id
    
    # Simulate storing in Pinecone
    simulate_pinecone_storage(response)
    
    return response

def format_latex_content(content):
    """
    Format content to properly render LaTeX expressions in Streamlit
    
    Args:
        content: String containing LaTeX expressions
        
    Returns:
        Formatted string with proper LaTeX delimiters
    """
    # Replace LaTeX expressions enclosed in \( \) with $ $ for inline math
    content = content.replace('\\(', '$').replace('\\)', '$')
    
    # Replace LaTeX expressions enclosed in \[ \] with $$ $$ for display math
    content = content.replace('\\[', '$$').replace('\\]', '$$')
    
    # Handle cases where LaTeX is already in the format R = \frac{...} without delimiters
    # This is more complex and might require regex for perfect handling
    
    return content

# Improve the extract_thinking_sections function
def extract_thinking_sections(content):
    """Extract thinking sections from the response"""
    # Try different possible tag formats
    thinking_patterns = [
        r'<thinking>(.*?)</thinking>',
        r'<think>(.*?)</think>',
        r'\[thinking\](.*?)\[/thinking\]',
        r'\*\*Thinking:\*\*(.*?)(?=\*\*|$)',
        r'Thinking:(.*?)(?=Question|<question>|$)'
    ]
    
    thinking_sections = []
    clean_content = content
    
    # Try each pattern
    for pattern in thinking_patterns:
        sections = re.findall(pattern, content, re.DOTALL)
        if sections:
            thinking_sections.extend(sections)
            # Remove these sections from the content
            clean_content = re.sub(pattern, '', clean_content, flags=re.DOTALL)
    
    # Debug output to see what's happening
    print(f"Found {len(thinking_sections)} thinking sections")
    if thinking_sections:
        print(f"First thinking section: {thinking_sections[0][:100]}...")
    
    return clean_content, thinking_sections

def clean_output_text(text):
    """
    Clean the output text by removing unwanted '#' characters while preserving markdown headers
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Process the text line by line to handle markdown headers properly
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Check if line starts with markdown header pattern (# followed by space)
        if re.match(r'^#+\s', line):
            # This is a markdown header, preserve it
            cleaned_lines.append(line)
        else:
            # For non-header lines, remove standalone '#' characters
            cleaned_line = re.sub(r'#', '', line)
            cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)

# ===============================
# Main Application
# ===============================

# Apply theme based on dark mode setting
apply_theme()

# Add this line to apply custom CSS for dark mode
styles.apply_custom_css(st.session_state.dark_mode)

# Sidebar for app controls and dark mode toggle
with st.sidebar:
    st.markdown('App Settings', unsafe_allow_html=True)
    
    # Dark mode toggle - add a unique key
    dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode, key="sidebar_dark_mode")
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    # Token usage counter
    st.markdown('', unsafe_allow_html=True)
    st.markdown(f"**API Token Usage:** {st.session_state.token_count} tokens")
    st.markdown('', unsafe_allow_html=True)
    
    # User info if authenticated
    if st.session_state.authenticated:
        st.markdown("---")
        st.markdown(f"**Logged in as:** {st.session_state.users[st.session_state.username]['name']}")
        if st.button("Logout"):
            logout_user()
            st.rerun()

# Create main navigation
if st.session_state.authenticated:
    selected = option_menu(
        menu_title=None,
        options=["Generate Questions", "History", "Profile", "Settings"],
        icons=["file-earmark-plus", "clock-history", "person", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
else:
    selected = option_menu(
        menu_title=None,
        options=["Generate Questions", "Login", "Register"],
        icons=["file-earmark-plus", "box-arrow-in-right", "person-plus"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

# ===============================
# Generate Questions Page
# ===============================

if selected == "Generate Questions":
    # Two-column layout for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        # Topic selection with custom icons
        st.markdown("### 📚 Select Physics Topic")
        topic_options = [
            "Mechanics - Motion",
            "Mechanics - Forces",
            "Electricity - Circuits",
            "Electricity - Fields",
            "Waves - Properties",
            "Waves - Optics",
            "Nuclear Physics",
            "Thermodynamics",
            "Particle Physics",
            "Magnetic Fields"
        ]
        
        # Use a selectbox with formatted options that include icons
        topic = st.selectbox(
            "Select a physics topic",
            options=topic_options,
            format_func=lambda x: f"{styles.get_topic_icon(x)} {x}",
            help="Choose the physics topic for your question"
        )
        
        # Display the selected topic with icon and color
        color = styles.get_topic_color(topic)
        st.markdown(
            f"""
            <div style="padding:10px; border-radius:5px; margin:10px 0; 
                        background-color:rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1); 
                        border-left:3px solid {color};">
                <span style="font-size:18px;">{styles.get_topic_icon(topic)}</span>
                <span style="font-weight:500; margin-left:5px;">{topic} selected</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Use Perplexity checkbox
        use_perplexity = st.checkbox("Use Perplexity AI (Higher quality questions)", value=True)
        
        # Additional options
        include_diagram = st.checkbox("Include Diagram", value=True, key="include_diagram_checkbox")
        include_hints = st.checkbox("Include Hints", value=True, key="include_hints_checkbox")
        
        # If user is logged in, show personalized recommendations
        if st.session_state.authenticated:
            username = st.session_state.username
            if 'topic_counts' in st.session_state.users[username] and st.session_state.users[username]['topic_counts']:
                st.markdown("### Recommended Topics")
                
                # Get top 3 topics
                topic_counts = st.session_state.users[username]['topic_counts']
                top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for topic_name, count in top_topics:
                    # Use the render_topic_card function to display each recommendation
                    recommendation_content = f"You've completed {count} questions on this topic"
                    st.markdown(
                        styles.render_topic_card(topic_name, recommendation_content),
                        unsafe_allow_html=True
                    )
    
    with col2:
        # Always show difficulty selection
        difficulty = st.select_slider(
            "Select Difficulty",
            options=["Easy", "Medium", "Hard"],
            value="Medium"
        )
        
        # If using Perplexity, also show question type selection
        if use_perplexity:
            question_type = st.multiselect(
                "Select Question Types",
                options=[
                    "Short-answer",
                    "Calculation",
                    "Application",
                    "Extended-response"
                ],
                default=["Calculation"]
            )
            # Convert list to comma-separated string for the API
            question_type_str = ", ".join(question_type)
    
    # Generate button
    generate_button = st.button("Generate Question", use_container_width=True)
    
    # Check if we should reuse parameters from history
    reuse_params = st.checkbox("Modify parameters from previous question", value=False, key="reuse_params_checkbox")
    if reuse_params and st.session_state.question_history:
        # Get the most recent question
        prev_question = st.session_state.question_history[-1]
        prev_params = prev_question['params']
        
        st.markdown("**Previous parameters:**")
        st.json(prev_params)
        
        # Allow modifications
        st.markdown("**Modify parameters:**")
        topic = st.selectbox("New Topic", options=[
            "Mechanics - Motion",
            "Mechanics - Forces",
            "Electricity - Circuits",
            "Electricity - Fields",
            "Waves - Properties",
            "Waves - Optics",
            "Nuclear Physics",
            "Thermodynamics",
            "Particle Physics",
            "Magnetic Fields"
        ], index=0)
        
        difficulty = st.select_slider(
            "New Difficulty",
            options=["Easy", "Medium", "Hard"],
            value=prev_params['difficulty']
        )
        
        include_diagram = st.checkbox("Include Diagram", value=prev_params['include_diagram'], key="modified_diagram_checkbox")
        include_hints = st.checkbox("Include Hints", value=prev_params['include_hints'], key="modified_hints_checkbox")
        
        if st.button("Generate Modified Question"):
            generate_button = True
    
    st.markdown('', unsafe_allow_html=True)
    
    # Generate and display question
    if generate_button:
        with st.spinner("Generating your physics question..."):
            # Call the API
            params = {
                "topic": topic,
                "difficulty": difficulty,
                "include_diagram": include_diagram,
                "include_hints": include_hints
            }
            
            # Add question_type parameter if using Perplexity
            if use_perplexity:
                params["question_type"] = question_type_str
            
            if use_perplexity:
                response = generate_perplexity_question(params)
            else:
                response = call_n8n_api(params)
            
            # Display the question in a card
            st.markdown('', unsafe_allow_html=True)
            st.markdown("## Generated Question 🧙")
            st.markdown(f"**Topic:** {styles.get_topic_icon(topic)} {topic}")
            st.markdown(f"**Difficulty:** {difficulty}")
            
            if use_perplexity:
                # First extract thinking sections
                if "question" in response:
                    clean_content, thinking_sections = extract_thinking_sections(response["question"])
                else:
                    # Handle the case where "question" key doesn't exist
                    print(f"Available keys in response: {response.keys()}")
                    
                    # Use a default or the entire response if appropriate
                    if isinstance(response, str):
                        clean_content, thinking_sections = extract_thinking_sections(response)
                    else:
                        clean_content, thinking_sections = "", []
                
                # Clean the content to remove unwanted characters
                clean_content = clean_output_text(clean_content)
                formatted_content = format_latex_content(clean_content)
                
                # Display thinking sections in a dropdown at the top with more visibility
                if thinking_sections:
                    with st.expander("🧠 View AI Thinking Process", expanded=False):
                        for i, thinking in enumerate(thinking_sections):
                            # Clean thinking sections as well
                            clean_thinking = clean_output_text(thinking)
                            st.markdown(clean_thinking)
                else:
                    st.info("No AI thinking process was found in this response.")
                    # Add a manual check to see if there are any tags visible in the content
                    if "<think" in response["question"] or "<thinking" in response["question"]:
                        st.warning("Thinking tags were detected but couldn't be properly extracted. Please check the format.")
                
                # Continue with the rest of the question display...
                # First, try to split the content by questions
                question_headers = re.findall(r'(?:\*\*)?Question\s*\d*\s*:?(?:\*\*)?|<question>', formatted_content)
                
                if question_headers:
                    # Split content by question headers
                    question_sections = re.split(r'((?:\*\*)?Question\s*\d*\s*:?(?:\*\*)?|<question>)', formatted_content)
                    # Remove empty strings and process
                    question_sections = [s for s in question_sections if s and s.strip()]
                    
                    # Group headers with content
                    questions = []
                    for i in range(0, len(question_sections), 2):
                        if i+1 < len(question_sections):
                            header = question_sections[i]
                            content = question_sections[i+1]
                            # Clean up any </question> tags
                            content = re.sub(r'</question>', '', content)
                            questions.append((header, content))
                    
                    # Create tabs for each question
                    if questions:
                        question_tabs = st.tabs([f"Q{i+1}" for i in range(len(questions))])
                        
                        for i, tab in enumerate(question_tabs):
                            with tab:
                                header, content = questions[i]
                                # Remove bold markers from header if present
                                header = re.sub(r'\*\*', '', header)
                                st.markdown(f"### {header}")
                                
                                # Check if this content contains a mark scheme
                                mark_scheme_match = re.search(r'((?:\*\*)?Mark\s*Scheme\s*:?(?:\*\*)?)(.*?)(?=(?:\*\*)?Question|$)', content, re.DOTALL | re.IGNORECASE)
                                
                                if mark_scheme_match:
                                    # Split content at mark scheme
                                    question_part = content[:mark_scheme_match.start()]
                                    mark_scheme_part = mark_scheme_match.group(2)
                                    
                                    # Display question and mark scheme separately
                                    st.markdown(question_part)
                                    with st.expander("View Mark Scheme", expanded=False):
                                        # Clean mark scheme text
                                        clean_mark_scheme = clean_output_text(mark_scheme_part)
                                        st.markdown(clean_mark_scheme)
                                else:
                                    # No mark scheme found, display entire content
                                    st.markdown(content)
                    else:
                        # Fallback if grouping failed
                        st.markdown(formatted_content)
                else:
                    # Alternative approach if no question headers found
                    # Try to find mark scheme sections directly
                    parts = re.split(r'(Mark\s*Scheme\s*:?)', formatted_content, flags=re.IGNORECASE)
                    
                    if len(parts) > 1:
                        # Display the question part
                        st.markdown(parts[0])
                        
                        # Combine all mark scheme parts
                        mark_scheme_content = "".join(parts[1:])
                        
                        # Display mark scheme in an expander
                        with st.expander("View Mark Scheme", expanded=False):
                            # Clean mark scheme content
                            clean_mark_scheme = clean_output_text(mark_scheme_content)
                            st.markdown(clean_mark_scheme)
                    else:
                        # If all parsing failed, just display the whole content
                        st.markdown(formatted_content)
            else:
                # Original display code
                st.markdown(response["question"])
                
                # Display diagram if included
                if include_diagram and response.get("diagram_data"):
                    st.markdown("### Diagram")
                    fig = generate_interactive_diagram(response["diagram_data"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Display hints if included
                if include_hints and response.get("hints"):
                    with st.expander("View Hints"):
                        for i, hint in enumerate(response["hints"]):
                            st.markdown(f"**Hint {i+1}:** {hint}")
            
            # Feedback buttons
            st.markdown("### Was this question helpful?")
            col1, col2, col3, _ = st.columns(4)
            
            with col1:
                if st.button("👍 Good"):
                    record_feedback(response["id"], "positive")
            
            with col2:
                if st.button("😐 Neutral"):
                    record_feedback(response["id"], "neutral")
            
            with col3:
                if st.button("👎 Poor"):
                    record_feedback(response["id"], "negative")
            
            st.markdown('', unsafe_allow_html=True)

# ===============================
# Login Page
# ===============================

elif selected == "Login":
    st.markdown('Login', unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if authenticate_user(username, password):
                st.success(f"Welcome back, {st.session_state.users[username]['name']}!")
                st.rerun()
            else:
                st.error("Invalid username or password")

# ===============================
# Register Page
# ===============================

elif selected == "Register":
    st.markdown('Register', unsafe_allow_html=True)
    
    with st.form("register_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        name = st.text_input("Full Name")
        submit_button = st.form_submit_button("Register")
        
        if submit_button:
            if not username or not password or not name:
                st.error("Please fill in all fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = register_user(username, password, name)
                if success:
                    st.success(message)
                    st.info("You can now log in with your credentials")
                else:
                    st.error(message)

# ===============================
# History Page
# ===============================

elif selected == "History" and st.session_state.authenticated:
    st.markdown('Question History', unsafe_allow_html=True)
    
    # Get user's question history
    user_history = retrieve_user_history(st.session_state.username)
    
    if not user_history:
        st.info("You haven't generated any questions yet.")
    else:
        # Export button
        if st.button("Export Questions as JSON"):
            json_str = export_questions_as_json()
            if json_str:
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'Download JSON File'
                st.markdown(href, unsafe_allow_html=True)
        
        # Display history in a table
        st.markdown("### Your Question History")
        
        # Convert to DataFrame for display
        history_data = []
        for item in user_history:
            history_data.append({
                "ID": item["id"][:8],
                "Topic": item["params"]["topic"],
                "Difficulty": item["params"]["difficulty"],
                "Date": item["timestamp"].split("T"),
                "Question": item["question"][:50] + "..."
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
        
        # Allow selecting a question to view in detail
        selected_id = st.selectbox("Select a question to view details", 
                                  options=[item["id"] for item in user_history],
                                  format_func=lambda x: f"{x[:8]} - {next((q['params']['topic'] for q in user_history if q['id'] == x), '')}")
        
        if selected_id:
            selected_question = next((q for q in user_history if q["id"] == selected_id), None)
            if selected_question:
                st.markdown('', unsafe_allow_html=True)
                st.markdown("## Question Details")
                
                # Use the topic icon in the topic display
                topic_icon = styles.get_topic_icon(selected_question['params']['topic'])
                st.markdown(f"**Topic:** {topic_icon} {selected_question['params']['topic']}")
                
                st.markdown(f"**Difficulty:** {selected_question['params']['difficulty']}")
                st.markdown(f"**Date:** {selected_question['timestamp'].split('T')}")
                st.markdown(selected_question["question"])
                
                # Display diagram if included
                if selected_question.get("diagram_data"):
                    st.markdown("### Diagram")
                    fig = generate_interactive_diagram(selected_question["diagram_data"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Display hints if included
                if selected_question.get("hints"):
                    with st.expander("View Hints"):
                        for i, hint in enumerate(selected_question["hints"]):
                            st.markdown(f"**Hint {i+1}:** {hint}")
                
                st.markdown('', unsafe_allow_html=True)

# ===============================
# Profile Page
# ===============================

elif selected == "Profile" and st.session_state.authenticated:
    username = st.session_state.username
    user_data = st.session_state.users[username]
    
    st.markdown(f'Profile: {user_data["name"]}', unsafe_allow_html=True)
    
    # User statistics
    st.markdown('', unsafe_allow_html=True)
    st.markdown("## User Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Questions Generated", user_data.get('completed_questions', 0))
        st.metric("Token Usage", st.session_state.token_count)
    
    with col2:
        # Topic preferences
        topic_stats = get_topic_statistics(username)
        if topic_stats:
            st.markdown("### Topic Preferences")
            
            # Create a bar chart of topic preferences
            topics = list(topic_stats.keys())
            counts = list(topic_stats.values())
            
            fig = px.bar(
                x=topics, y=counts,
                labels={"x": "Topic", "y": "Questions Generated"},
                title="Your Topic Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic statistics available yet. Generate some questions!")
    
    st.markdown('', unsafe_allow_html=True)
    
    # Personalized recommendations
    st.markdown('', unsafe_allow_html=True)
    st.markdown("## Personalized Recommendations")
    
    if topic_stats:
        # Find least practiced topics
        all_topics = [
            "Mechanics - Motion",
            "Mechanics - Forces",
            "Electricity - Circuits",
            "Electricity - Fields",
            "Waves - Properties",
            "Waves - Optics",
            "Nuclear Physics",
            "Thermodynamics",
            "Particle Physics",
            "Magnetic Fields"
        ]
        
        missing_topics = [topic for topic in all_topics if topic not in topic_stats]
        
        if missing_topics:
            st.markdown("### Topics to Explore")
            for topic in missing_topics:
                st.markdown(f"- {topic}")
        
        # Find most practiced topics
        if topic_stats:
            top_topics = sorted(topic_stats.items(), key=lambda x: x, reverse=True)[:3]
            
            st.markdown("### Your Strongest Topics")
            for topic, count in top_topics:
                st.markdown(f"- {topic} ({count} questions)")
    else:
        st.info("Generate more questions to get personalized recommendations!")
    
    st.markdown('', unsafe_allow_html=True)

# ===============================
# Settings Page
# ===============================

elif selected == "Settings" and st.session_state.authenticated:
    st.markdown('Settings', unsafe_allow_html=True)
    
    st.markdown('', unsafe_allow_html=True)
    st.markdown("## Application Settings")
    
    # Dark mode toggle
    dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode, key="settings_dark_mode")
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    # Reset token counter
    if st.button("Reset Token Counter"):
        st.session_state.token_count = 0
        st.success("Token counter has been reset.")
    
    # Clear question history
    if st.button("Clear Question History"):
        # Filter out questions from this user
        if st.session_state.authenticated:
            username = st.session_state.username
            st.session_state.pinecone_data = [q for q in st.session_state.pinecone_data if q.get('user') != username]
            st.session_state.question_history = [q for q in st.session_state.question_history if q.get('user') != username]
            st.session_state.users[username]['completed_questions'] = 0
            st.session_state.users[username]['topic_counts'] = {}
        else:
            st.session_state.question_history = []
        
        st.success("Question history has been cleared.")
    
    st.markdown('', unsafe_allow_html=True)
    
    # Account settings
    st.markdown('', unsafe_allow_html=True)
    st.markdown("## Account Settings")
    
    # Change name
    with st.form("update_name_form"):
        st.markdown("### Update Your Name")
        username = st.session_state.username
        user_data = st.session_state.users[username]
        new_name = st.text_input("New Name", value=user_data['name'])
        submit_name = st.form_submit_button("Update Name")
        
        if submit_name and new_name != user_data['name']:
            st.session_state.users[username]['name'] = new_name
            st.success("Name updated successfully!")
    
    # Change password
    with st.form("update_password_form"):
        st.markdown("### Change Password")
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        submit_password = st.form_submit_button("Update Password")
        
        if submit_password:
            if hashlib.sha256(current_password.encode()).hexdigest() != user_data['password']:
                st.error("Current password is incorrect.")
            elif new_password != confirm_password:
                st.error("New passwords do not match.")
            else:
                st.session_state.users[username]['password'] = hashlib.sha256(new_password.encode()).hexdigest()
                st.success("Password updated successfully!")
    
    st.markdown('', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2025 A-Level Physics Question Generator | Marco")

def generate_question():
    """Generate a physics question based on user inputs"""
    with st.spinner("Generating question... This may take a moment."):
        try:
            # Get user inputs
            topic = st.session_state.topic
            difficulty = st.session_state.difficulty
            question_type = st.session_state.question_type
            
            # Load guide and examples
            guide_content = pplx.process_guide("guide.txt")
            examples_content = pplx.process_examples("examples.docx")
            
            # Try to get RAG examples
            rag_examples = None
            try:
                from src.RAG import PhysicsRAG
                rag = PhysicsRAG()
                search_query = f"{topic} {difficulty} {question_type} physics question"
                rag_examples = rag.search(search_query, top_k=5)
                logger.info(f"Retrieved {len(rag_examples)} similar questions using RAG")
            except Exception as e:
                logger.warning(f"Failed to retrieve RAG examples: {e}")
            
            # Generate the question with RAG examples
            result = pplx.generate_physics_question(
                topic=topic,
                difficulty=difficulty,
                question_type=question_type,
                guide_content=guide_content,
                examples_content=examples_content,
                rag_examples=rag_examples
            )
            
            # Process the result
            if result and not result.startswith("Error"):
                # Parse the generated content
                questions = []
                current_question = {"text": "", "mark_scheme": ""}
                
                # Split by question and mark scheme
                lines = result.split("\n")
                mode = "none"
                
                for line in lines:
                    if line.strip().startswith("<question>"):
                        if current_question["text"]:
                            questions.append(current_question)
                            current_question = {"text": "", "mark_scheme": ""}
                        mode = "question"
                        line = line.replace("<question>", "").strip()
                    elif line.strip().startswith("<mark scheme>"):
                        mode = "mark_scheme"
                        line = line.replace("<mark scheme>", "").strip()
                    
                    if mode == "question" and line:
                        current_question["text"] += line + "\n"
                    elif mode == "mark_scheme" and line:
                        current_question["mark_scheme"] += line + "\n"
                
                # Add the last question
                if current_question["text"]:
                    questions.append(current_question)
                
                # Log the processed questions
                from src.utils import ensure_log_directory
                import datetime
                import os
                import json
                
                log_dir = ensure_log_directory()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_questions_{topic}_{difficulty}_{timestamp}.json"
                filepath = os.path.join(log_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump({
                        "topic": topic,
                        "difficulty": difficulty,
                        "question_type": question_type,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "num_questions": len(questions),
                        "questions": questions
                    }, f, indent=2)
                
                logger.info(f"Processed questions saved to {filepath}")
                
                # Store the questions
                if questions:
                    for q in questions:
                        question_id = str(uuid.uuid4())
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        question_data = {
                            "id": question_id,
                            "topic": topic,
                            "difficulty": difficulty,
                            "question_type": question_type,
                            "question": q["text"].strip(),
                            "mark_scheme": q["mark_scheme"].strip(),
                            "timestamp": timestamp,
                            "user": st.session_state.username if st.session_state.authenticated else "guest",
                            "feedback": None
                        }
                        
                        st.session_state.question_history.append(question_data)
                        st.session_state.current_question_id = question_id
                        
                        # Update user stats
                        if st.session_state.authenticated:
                            username = st.session_state.username
                            st.session_state.users[username]['completed_questions'] += 1
                            
                            # Update topic counts
                            if topic not in st.session_state.users[username]['topic_counts']:
                                st.session_state.users[username]['topic_counts'][topic] = 1
                            else:
                                st.session_state.users[username]['topic_counts'][topic] += 1
                    
                    st.success(f"Generated {len(questions)} questions successfully!")
                    st.session_state.token_count += 1
                else:
                    st.error("Failed to parse generated questions.")
            else:
                st.error(f"Error generating question: {result}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
