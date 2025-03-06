import streamlit as st

def apply_custom_css(dark_mode):
    """Apply custom CSS for dark mode adjustments"""
    if dark_mode:
        st.markdown("""
        <style>
        /* Set ALL text to white in dark mode */
        body, p, h1, h2, h3, h4, h5, h6, span, div, label, .stMarkdown, 
        .stText, .stTextInput, .stTextArea, .stButton, .stSelectbox, 
        .stMultiSelect, .stSlider, .stCheckbox, .stRadio, .stExpander, 
        .stTabs, .stDataFrame, .stTable, .stJson, .stCode, .stMetric, 
        .stProgress, .stAlert, .stInfo, .stWarning, .stError, .stSuccess {
            color: #FFFFFF !important;
        }
        
        /* Force all text elements to be white */
        * {
            color: #FFFFFF !important;
        }
        
        /* Exceptions for specific elements that should remain colored */
        div[data-testid="stHorizontalBlock"] div[role="tablist"] {
            background-color: white !important;
            border-radius: 8px;
            padding: 5px;
        }

        div[data-testid="stHorizontalBlock"] button {
            color: black !important;
        }
        
        /* Make sure links are visible */
        a {
            color: #4ECDC4 !important;
        }
        
        /* Make sure code blocks are readable */
        code {
            color: #F0F0F0 !important;
            background-color: #333333 !important;
        }
        
        /* Ensure dataframe text is visible */
        .dataframe {
            color: white !important;
        }
        
        .dataframe th, .dataframe td {
            color: white !important;
        }
        
        /* Fix for markdown text */
        .element-container div.stMarkdown p {
            color: white !important;
        }
        
        /* Fix for sidebar text */
        [data-testid="stSidebar"] [data-testid="stMarkdown"] p {
            color: white !important;
        }
        
        /* Fix for expander text */
        .streamlit-expanderContent {
            color: white !important;
        }
        
        /* Fix for tabs content */
        [data-testid="stTabsContent"] {
            color: white !important;
        }
        
        /* Custom Topic Icons Styling */
        .topic-icon {
            display: inline-block;
            width: 24px;
            height: 24px;
            margin-right: 8px;
            vertical-align: middle;
        }
        
        .topic-option {
            display: flex;
            align-items: center;
            padding: 8px 12px;
            border-radius: 4px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            background-color: #1E1E1E;
        }
        
        .topic-option:hover {
            background-color: #2D2D2D;
            transform: translateX(5px);
        }
        
        .topic-option.selected {
            background-color: #3B82F6;
            color: white;
        }
        
        /* Topic card styling */
        .topic-card {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            background-color: #2D2D2D;
            border-left: 4px solid;
            transition: all 0.2s ease;
        }
        
        .topic-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* Topic-specific colors */
        .mechanics-motion { border-color: #3B82F6; }
        .mechanics-forces { border-color: #8B5CF6; }
        .electricity-circuits { border-color: #EF4444; }
        .electricity-fields { border-color: #F59E0B; }
        .waves-properties { border-color: #10B981; }
        .waves-optics { border-color: #14B8A6; }
        .nuclear-physics { border-color: #EC4899; }
        .thermodynamics { border-color: #F97316; }
        .particle-physics { border-color: #6366F1; }
        .magnetic-fields { border-color: #8B5CF6; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        /* Reset to default colors in light mode */
        body, p, h1, h2, h3, h4, h5, h6, span, div, label {
            color: inherit;
        }
        
        div[data-testid="stHorizontalBlock"] button {
            color: inherit !important;
        }
        
        /* Custom Topic Icons Styling */
        .topic-icon {
            display: inline-block;
            width: 24px;
            height: 24px;
            margin-right: 8px;
            vertical-align: middle;
        }
        
        .topic-option {
            display: flex;
            align-items: center;
            padding: 8px 12px;
            border-radius: 4px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            background-color: #F3F4F6;
        }
        
        .topic-option:hover {
            background-color: #E5E7EB;
            transform: translateX(5px);
        }
        
        .topic-option.selected {
            background-color: #3B82F6;
            color: white;
        }
        
        /* Topic card styling */
        .topic-card {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            background-color: #F9FAFB;
            border-left: 4px solid;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .topic-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* Topic-specific colors */
        .mechanics-motion { border-color: #3B82F6; }
        .mechanics-forces { border-color: #8B5CF6; }
        .electricity-circuits { border-color: #EF4444; }
        .electricity-fields { border-color: #F59E0B; }
        .waves-properties { border-color: #10B981; }
        .waves-optics { border-color: #14B8A6; }
        .nuclear-physics { border-color: #EC4899; }
        .thermodynamics { border-color: #F97316; }
        .particle-physics { border-color: #6366F1; }
        .magnetic-fields { border-color: #8B5CF6; }
        </style>
        """, unsafe_allow_html=True)

def get_topic_icon(topic):
    """Return emoji icon for topic-specific icon"""
    # Map topics to appropriate icons
    icon_map = {
        "Mechanics - Motion": "🚀",  # Rocket for motion
        "Mechanics - Forces": "🧲",  # Magnet for forces
        "Electricity - Circuits": "⚡",  # Lightning for circuits
        "Electricity - Fields": "🔌",  # Electric plug for fields
        "Waves - Properties": "〰️",  # Wave symbol for wave properties
        "Waves - Optics": "🔍",  # Magnifying glass for optics
        "Nuclear Physics": "☢️",  # Radiation symbol for nuclear physics
        "Thermodynamics": "🔥",  # Fire for thermodynamics
        "Particle Physics": "⚛️",  # Atom symbol for particle physics
        "Magnetic Fields": "🧭"   # Compass for magnetic fields
    }
    
    # Return the icon or a default if topic not found
    return icon_map.get(topic, "📚")  # Default to book emoji

def get_topic_class(topic):
    """Return CSS class for topic styling"""
    # Convert topic to CSS-friendly class name
    if not topic:
        return "unknown-topic"
    
    # Remove spaces and convert to lowercase
    return topic.lower().replace(" - ", "-").replace(" ", "-")

def get_topic_color(topic):
    """Return color for topic"""
    color_map = {
        "Mechanics - Motion": "#3B82F6",  # Blue
        "Mechanics - Forces": "#8B5CF6",  # Purple
        "Electricity - Circuits": "#EF4444",  # Red
        "Electricity - Fields": "#F59E0B",  # Amber
        "Waves - Properties": "#10B981",  # Green
        "Waves - Optics": "#14B8A6",  # Teal
        "Nuclear Physics": "#EC4899",  # Pink
        "Thermodynamics": "#F97316",  # Orange
        "Particle Physics": "#6366F1",  # Indigo
        "Magnetic Fields": "#8B5CF6"   # Purple
    }
    
    return color_map.get(topic, "#6B7280")  # Default to gray

def render_topic_card(topic, content):
    """Render a topic card with icon and styled border"""
    icon = get_topic_icon(topic)
    color = get_topic_color(topic)
    
    return f"""
    <div style="padding:15px; border-radius:8px; margin-bottom:10px; border-left:4px solid {color}; background-color:rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);">
        <div style="display:flex; align-items:center; margin-bottom:8px;">
            <span style="font-size:20px; margin-right:8px;">{icon}</span>
            <span style="font-weight:bold; font-size:16px;">{topic}</span>
        </div>
        <div>{content}</div>
    </div>
    """

def display_topic_selector(topic_options, key_prefix="topic"):
    """Display a Streamlit-native topic selector with icons"""
    # Create a container for the topic options
    topic_container = st.container()
    
    # Use radio buttons for selection, but hide the actual radio buttons
    # and create our own styled options
    selected_topic = None
    
    # Create columns for a grid layout (3 columns)
    cols = st.columns(3)
    
    for i, topic in enumerate(topic_options):
        col_idx = i % 3
        with cols[col_idx]:
            icon = get_topic_icon(topic)
            color = get_topic_color(topic)
            
            # Create a clickable button with icon and topic name
            if st.button(
                f"{icon} {topic}", 
                key=f"{key_prefix}_{i}",
                use_container_width=True,
                help=f"Select {topic} as your question topic"
            ):
                selected_topic = topic
    
    return selected_topic 