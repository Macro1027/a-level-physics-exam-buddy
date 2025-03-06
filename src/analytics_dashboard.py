# Edexcel A-Level Physics Question Generator
# Analytics Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import io
import base64
import re
from streamlit_option_menu import option_menu
import styles

# ===============================
# Authentication Check
# ===============================

# Check if user is authenticated
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("⚠️ Please log in to access the Analytics Dashboard.")
    st.info("Go to the Login page to authenticate.")
    st.stop()

# ===============================
# Helper Functions
# ===============================

def get_user_data():
    """
    Retrieve question history and user data for the current user
    In a real app, this would query a database
    """
    username = st.session_state.username
    
    # Get user's question history
    user_history = [q for q in st.session_state.pinecone_data if q.get('user') == username]
    
    # Get user profile data
    user_data = st.session_state.users.get(username, {})
    
    return user_history, user_data

def get_all_users_data():
    """
    Retrieve aggregated data for all users (admin view)
    In a real app, this would query a database
    """
    # Get all question history
    all_history = st.session_state.pinecone_data
    
    # Get all user profiles
    all_users = st.session_state.users
    
    return all_history, all_users

def prepare_question_data(history):
    """
    Process question history into a pandas DataFrame for analysis
    """
    if not history:
        return pd.DataFrame()
    
    # Extract relevant fields from question history
    data = []
    for item in history:
        # Extract basic metadata
        entry = {
            "id": item.get("id", ""),
            "timestamp": item.get("timestamp", ""),
            "user": item.get("user", "guest"),
            "tokens_used": item.get("tokens_used", 0),
            "source": item.get("source", "n8n")
        }
        
        # Extract parameters
        params = item.get("params", {})
        entry["topic"] = params.get("topic", "Unknown")
        entry["difficulty"] = params.get("difficulty", "Medium")
        entry["include_diagram"] = params.get("include_diagram", False)
        entry["include_hints"] = params.get("include_hints", False)
        entry["question_type"] = params.get("question_type", "calculation")
        
        # Extract feedback if available
        if "id" in item and item["id"] in st.session_state.get("feedback_data", {}):
            feedback = st.session_state.feedback_data[item["id"]]
            entry["feedback_rating"] = feedback.get("rating", "none")
        else:
            entry["feedback_rating"] = "none"
        
        data.append(entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df["month"] = df["timestamp"].dt.month
        df["year"] = df["timestamp"].dt.year
        df["day_of_week"] = df["timestamp"].dt.day_name()
    
    return df

def generate_download_link(df, filename, text):
    """
    Generate a download link for a DataFrame
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def plot_question_type_distribution(df):
    """
    Create a pie chart showing distribution of question types
    """
    if df.empty or "question_type" not in df.columns:
        return go.Figure()
    
    # Count question types
    type_counts = df["question_type"].value_counts().reset_index()
    type_counts.columns = ["Question Type", "Count"]
    
    # Create pie chart
    fig = px.pie(
        type_counts, 
        values="Count", 
        names="Question Type",
        title="Question Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend_title="Question Types",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def plot_topic_distribution(df):
    """
    Create a bar chart showing distribution of topics
    """
    if df.empty or "topic" not in df.columns:
        return go.Figure()
    
    # Count topics
    topic_counts = df["topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    
    # Sort by count
    topic_counts = topic_counts.sort_values("Count", ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        topic_counts, 
        y="Topic", 
        x="Count",
        title="Topic Distribution",
        color="Count",
        color_continuous_scale=px.colors.sequential.Viridis,
        orientation='h'
    )
    
    fig.update_layout(
        xaxis_title="Number of Questions",
        yaxis_title="",
        height=400 + (len(topic_counts) * 20)  # Adjust height based on number of topics
    )
    
    return fig

def plot_difficulty_distribution(df):
    """
    Create a bar chart showing distribution of difficulty levels
    """
    if df.empty or "difficulty" not in df.columns:
        return go.Figure()
    
    # Define difficulty order
    difficulty_order = ["Easy", "Medium", "Hard"]
    
    # Count difficulties
    difficulty_counts = df["difficulty"].value_counts().reindex(difficulty_order).reset_index()
    difficulty_counts.columns = ["Difficulty", "Count"]
    
    # Create bar chart
    fig = px.bar(
        difficulty_counts, 
        x="Difficulty", 
        y="Count",
        title="Difficulty Level Distribution",
        color="Difficulty",
        color_discrete_map={
            "Easy": "#72B7B2",
            "Medium": "#F2B880",
            "Hard": "#F98866"
        },
        category_orders={"Difficulty": difficulty_order}
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Number of Questions"
    )
    
    return fig

def plot_time_series(df):
    """
    Create a time series chart showing questions over time
    """
    if df.empty or "date" not in df.columns:
        return go.Figure()
    
    # Group by date
    time_series = df.groupby("date").size().reset_index(name="Count")
    
    # Create line chart
    fig = px.line(
        time_series, 
        x="date", 
        y="Count",
        title="Questions Generated Over Time",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Questions"
    )
    
    return fig

def plot_feedback_distribution(df):
    """
    Create a pie chart showing distribution of feedback ratings
    """
    if df.empty or "feedback_rating" not in df.columns:
        return go.Figure()
    
    # Count feedback ratings
    feedback_counts = df["feedback_rating"].value_counts().reset_index()
    feedback_counts.columns = ["Rating", "Count"]
    
    # Create pie chart
    fig = px.pie(
        feedback_counts, 
        values="Count", 
        names="Rating",
        title="Feedback Distribution",
        color="Rating",
        color_discrete_map={
            "positive": "#72B7B2",
            "neutral": "#F2B880",
            "negative": "#F98866",
            "none": "#CCCCCC"
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend_title="Feedback",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def plot_topic_difficulty_heatmap(df):
    """
    Create a heatmap showing topic vs difficulty
    """
    if df.empty or "topic" not in df.columns or "difficulty" not in df.columns:
        return go.Figure()
    
    # Create cross-tabulation of topic vs difficulty
    heatmap_data = pd.crosstab(df["topic"], df["difficulty"])
    
    # Ensure all difficulty levels are present
    for diff in ["Easy", "Medium", "Hard"]:
        if diff not in heatmap_data.columns:
            heatmap_data[diff] = 0
    
    # Sort columns in correct order
    heatmap_data = heatmap_data[["Easy", "Medium", "Hard"]]
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Difficulty", y="Topic", color="Count"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="Viridis",
        title="Topic vs Difficulty Heatmap"
    )
    
    fig.update_layout(
        height=400 + (len(heatmap_data) * 20)  # Adjust height based on number of topics
    )
    
    # Add text annotations
    for i, topic in enumerate(heatmap_data.index):
        for j, diff in enumerate(heatmap_data.columns):
            fig.add_annotation(
                x=diff,
                y=topic,
                text=str(heatmap_data.loc[topic, diff]),
                showarrow=False,
                font=dict(color="white" if heatmap_data.loc[topic, diff] > 2 else "black")
            )
    
    return fig

def identify_syllabus_gaps(df, all_topics):
    """
    Identify gaps in syllabus coverage
    """
    if df.empty or "topic" not in df.columns:
        return []
    
    # Count topics
    covered_topics = set(df["topic"].unique())
    
    # Find gaps
    gaps = [topic for topic in all_topics if topic not in covered_topics]
    
    # Find underrepresented topics (less than 2 questions)
    topic_counts = df["topic"].value_counts()
    underrepresented = [topic for topic in topic_counts.index if topic_counts[topic] < 2]
    
    return gaps, underrepresented

def generate_recommendations(df, gaps, underrepresented):
    """
    Generate recommendations based on analytics
    """
    recommendations = []
    
    # Add recommendations for syllabus gaps
    if gaps:
        recommendations.append(f"**Syllabus Gaps:** Consider generating questions on {', '.join(gaps)}")
    
    # Add recommendations for underrepresented topics
    if underrepresented:
        recommendations.append(f"**Underrepresented Topics:** Generate more questions on {', '.join(underrepresented)}")
    
    # Add recommendations based on difficulty distribution
    if "difficulty" in df.columns and not df.empty:
        difficulty_counts = df["difficulty"].value_counts()
        if "Hard" in difficulty_counts and difficulty_counts["Hard"] < 0.2 * len(df):
            recommendations.append("**Difficulty Balance:** Consider adding more hard questions for better challenge")
        if "Easy" in difficulty_counts and difficulty_counts["Easy"] < 0.2 * len(df):
            recommendations.append("**Difficulty Balance:** Consider adding more easy questions for better scaffolding")
    
    # Add recommendations based on feedback
    if "feedback_rating" in df.columns and not df.empty:
        feedback_counts = df["feedback_rating"].value_counts()
        if "negative" in feedback_counts and feedback_counts["negative"] > 0.2 * len(df):
            recommendations.append("**Feedback Concerns:** High proportion of negative feedback. Review question quality.")
    
    return recommendations

# ===============================
# Main Dashboard
# ===============================

def main():
    # Set page title
    st.title("📊 Analytics Dashboard")
    st.markdown("Comprehensive analytics and insights for your question generation activity")
    
    # Apply theme based on dark mode setting
    styles.apply_custom_css(st.session_state.dark_mode)
    
    # Get user data
    user_history, user_data = get_user_data()
    
    # Check if there's data to analyze
    if not user_history:
        st.warning("No question data available for analysis. Generate some questions first!")
        return
    
    # Prepare data for analysis
    df = prepare_question_data(user_history)
    
    # Dashboard layout
    st.markdown("### 📈 Your Question Generation Activity")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", len(df))
    with col2:
        st.metric("Topics Covered", df["topic"].nunique())
    with col3:
        st.metric("Avg. Tokens/Question", int(df["tokens_used"].mean()) if "tokens_used" in df.columns else 0)
    with col4:
        positive_feedback = len(df[df["feedback_rating"] == "positive"])
        st.metric("Positive Feedback", f"{positive_feedback} ({int(positive_feedback/len(df)*100)}%)" if len(df) > 0 else "0 (0%)")
    
    # Date filter
    st.markdown("### 📅 Filter by Date Range")
    if "timestamp" in df.columns:
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date)
        with col2:
            end_date = st.date_input("End Date", max_date)
        
        # Filter data by date
        filtered_df = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)]
    else:
        filtered_df = df
    
    # Topic filter
    if "topic" in filtered_df.columns:
        selected_topics = st.multiselect(
            "Filter by Topics",
            options=sorted(filtered_df["topic"].unique()),
            default=sorted(filtered_df["topic"].unique())
        )
        
        if selected_topics:
            filtered_df = filtered_df[filtered_df["topic"].isin(selected_topics)]
    
    # Difficulty filter
    if "difficulty" in filtered_df.columns:
        selected_difficulties = st.multiselect(
            "Filter by Difficulty",
            options=["Easy", "Medium", "Hard"],
            default=["Easy", "Medium", "Hard"]
        )
        
        if selected_difficulties:
            filtered_df = filtered_df[filtered_df["difficulty"].isin(selected_difficulties)]
    
    # Check if filtered data is empty
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Create tabs for different analytics sections
    tabs = st.tabs([
        "Question Distribution", 
        "Time Analysis", 
        "Topic Coverage", 
        "Recommendations"
    ])
    
    # Tab 1: Question Distribution
    with tabs[0]:
        st.markdown("### Question Distribution Analysis")
        
        # Question type distribution
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_question_type_distribution(filtered_df), use_container_width=True)
        
        # Difficulty distribution
        with col2:
            st.plotly_chart(plot_difficulty_distribution(filtered_df), use_container_width=True)
        
        # Feedback distribution
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_feedback_distribution(filtered_df), use_container_width=True)
        
        # Topic vs Difficulty heatmap
        st.plotly_chart(plot_topic_difficulty_heatmap(filtered_df), use_container_width=True)
    
    # Tab 2: Time Analysis
    with tabs[1]:
        st.markdown("### Time Analysis")
        
        # Questions over time
        st.plotly_chart(plot_time_series(filtered_df), use_container_width=True)
        
        # Activity by day of week
        if "day_of_week" in filtered_df.columns:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_counts = filtered_df["day_of_week"].value_counts().reindex(day_order).reset_index()
            day_counts.columns = ["Day", "Count"]
            
            fig = px.bar(
                day_counts,
                x="Day",
                y="Count",
                title="Questions by Day of Week",
                color="Day",
                category_orders={"Day": day_order}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Topic Coverage
    with tabs[2]:
        st.markdown("### Topic Coverage Analysis")
        
        # Topic distribution
        st.plotly_chart(plot_topic_distribution(filtered_df), use_container_width=True)
        
        # Topic coverage table
        topic_stats = filtered_df.groupby("topic").agg({
            "id": "count",
            "difficulty": lambda x: x.value_counts().index[0],
            "feedback_rating": lambda x: (x == "positive").mean() * 100
        }).reset_index()
        
        topic_stats.columns = ["Topic", "Question Count", "Most Common Difficulty", "Positive Feedback %"]
        topic_stats["Positive Feedback %"] = topic_stats["Positive Feedback %"].round(1)
        
        # Add topic icons to the display
        topic_stats["Topic"] = topic_stats["Topic"].apply(lambda x: f"{styles.get_topic_icon(x)} {x}")
        
        st.dataframe(topic_stats, use_container_width=True)
    
    # Tab 4: Recommendations
    with tabs[3]:
        st.markdown("### 💡 Recommendations & Insights")
        
        # Define all possible topics
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
        
        # Identify syllabus gaps
        gaps, underrepresented = identify_syllabus_gaps(filtered_df, all_topics)
        
        # Generate recommendations
        recommendations = generate_recommendations(filtered_df, gaps, underrepresented)
        
        # Display recommendations
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.success("Great job! Your question generation is well-balanced across topics and difficulties.")
        
        # Topic recommendations
        st.markdown("### 🎯 Suggested Focus Areas")
        
        # Calculate topic scores based on count, difficulty balance, and feedback
        topic_scores = {}
        for topic in filtered_df["topic"].unique():
            topic_df = filtered_df[filtered_df["topic"] == topic]
            
            # Count score (inverse - fewer questions = higher priority)
            count_score = 1 / (len(topic_df) + 1)
            
            # Difficulty balance score
            difficulty_counts = topic_df["difficulty"].value_counts()
            difficulty_balance = 1 - (len(difficulty_counts) / 3)  # 3 is max number of difficulties
            
            # Feedback score (more negative feedback = higher priority)
            negative_feedback = len(topic_df[topic_df["feedback_rating"] == "negative"])
            feedback_score = negative_feedback / (len(topic_df) + 1)
            
            # Combined score
            topic_scores[topic] = (count_score * 0.4) + (difficulty_balance * 0.3) + (feedback_score * 0.3)
        
        # Sort topics by score
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Display top 3 focus areas
        if sorted_topics:
            focus_areas = sorted_topics[:3]
            for topic, score in focus_areas:
                topic_df = filtered_df[filtered_df["topic"] == topic]
                difficulties = topic_df["difficulty"].value_counts()
                
                # Determine missing difficulties
                missing_diff = [d for d in ["Easy", "Medium", "Hard"] if d not in difficulties.index]
                
                st.markdown(f"**{topic}**")
                if missing_diff:
                    st.markdown(f"- Add {', '.join(missing_diff)} difficulty questions")
                else:
                    st.markdown("- Good difficulty balance, consider adding more questions")
    
    # Export options
    st.markdown("### 📥 Export Data")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Filtered Data as CSV"):
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="physics_questions_analytics.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("Export Analytics Report"):
            # Create a more detailed report
            buffer = io.StringIO()
            buffer.write("# Physics Question Generator Analytics Report\n\n")
            buffer.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            buffer.write(f"User: {st.session_state.username}\n\n")
            buffer.write(f"Total Questions: {len(filtered_df)}\n")
            buffer.write(f"Topics Covered: {filtered_df['topic'].nunique()}\n")
            buffer.write(f"Date Range: {start_date} to {end_date}\n\n")
            
            buffer.write("## Topic Distribution\n\n")
            topic_counts = filtered_df["topic"].value_counts()
            for topic, count in topic_counts.items():
                buffer.write(f"- {topic}: {count} questions\n")
            
            buffer.write("\n## Difficulty Distribution\n\n")
            difficulty_counts = filtered_df["difficulty"].value_counts()
            for diff, count in difficulty_counts.items():
                buffer.write(f"- {diff}: {count} questions\n")
            
            buffer.write("\n## Recommendations\n\n")
            for rec in recommendations:
                buffer.write(f"- {rec}\n")
            
            report_text = buffer.getvalue()
            b64 = base64.b64encode(report_text.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="physics_analytics_report.txt">Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Privacy notice
    st.markdown("---")
    with st.expander("Privacy Notice"):
        st.markdown("""
        This analytics dashboard displays aggregated data about your question generation activity. 
        All data is stored locally in your browser session and is not shared with third parties.
        Individual student responses are not tracked or analyzed.
        """)

if __name__ == "__main__":
    main() 