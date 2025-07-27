import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from database import DatabaseManager
from scoring import PHQ9Scorer, GAD7Scorer
import os
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def apply_custom_css():
    st.markdown("""
    <style>
    .main-header {
        color: #2E8B57;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #87CEEB;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4682B4;
        margin: 0.5rem 0;
    }
    .severity-low { color: #228B22; font-weight: bold; }
    .severity-mild { color: #DAA520; font-weight: bold; }
    .severity-moderate { color: #FF8C00; font-weight: bold; }
    .severity-severe { color: #DC143C; font-weight: bold; }
    .stSelectbox > div > div { background-color: #F0F8FF; }
    .stButton > button {
        background-color: #4682B4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #2E8B57;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def get_data_manager():
    """Initialize and return data manager"""
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DatabaseManager()
    return st.session_state.data_manager

def login_page():
    """Display login and registration page"""
    st.markdown('<h1 class="main-header">🧠 Mental Health Tracker</h1>', unsafe_allow_html=True)
    st.markdown("### Track your mental health symptoms with clinical assessment tools")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.markdown("#### Sign In")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_button"):
                if username and password:
                    data_manager = get_data_manager()
                    user_id = data_manager.authenticate_user(username, password)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        # Set the user_id in the data manager
                        data_manager.set_user_id(user_id)
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
        
        with tab2:
            st.markdown("#### Create Account")
            new_username = st.text_input("Choose Username", key="reg_username")
            new_password = st.text_input("Choose Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            if st.button("Register", key="register_button"):
                if new_username and new_password and confirm_password:
                    if new_password == confirm_password:
                        data_manager = get_data_manager()
                        if data_manager.create_user(new_username, new_password):
                            st.success("Account created successfully! You can now login.")
                        else:
                            st.error("Username already exists. Please choose another.")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all fields")

def daily_tracking_page(data_manager):
    st.markdown('<h1 class="main-header">📊 Daily Symptom Tracking</h1>', unsafe_allow_html=True)
    
    # Date selection
    selected_date = st.date_input("Select Date", datetime.now().date())
    
    # Load existing entry for this date if available
    existing_entry = data_manager.get_entry_by_date(selected_date)
    
    if existing_entry:
        st.info(f"📝 You already have an entry for {selected_date}. You can update it below.")
    
    # Create tabs for PHQ-9 and GAD-7
    tab1, tab2 = st.tabs(["PHQ-9 (Depression)", "GAD-7 (Anxiety)"])
    
    with tab1:
        st.markdown("#### PHQ-9 Depression Assessment")
        st.caption("Are you experiencing any of these symptoms today?")
        
        # Quick option for no symptoms
        no_phq9_symptoms = st.checkbox(
            "✅ **No depression symptoms today**", 
            key="no_phq9_symptoms",
            help="Check this if you're not experiencing any depression symptoms today"
        )
        
        if not no_phq9_symptoms:
            phq9_questions = [
                "Little interest or pleasure in doing things",
                "Feeling down, depressed, or hopeless",
                "Trouble falling or staying asleep, or sleeping too much",
                "Feeling tired or having little energy",
                "Poor appetite or overeating",
                "Feeling bad about yourself - or that you are a failure or have let yourself or your family down",
                "Trouble concentrating on things, such as reading the newspaper or watching television",
                "Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual",
                "Thoughts that you would be better off dead, or of hurting yourself in some way"
            ]
            
            phq9_responses = []
            
            for i, question in enumerate(phq9_questions):
                # Pre-fill with existing data if available
                default_value = False
                if existing_entry and hasattr(existing_entry, f'phq9_{i}'):
                    default_value = getattr(existing_entry, f'phq9_{i}', False)
                
                response = st.checkbox(
                    f"{i+1}. {question}",
                    value=default_value,
                    key=f"phq9_{i}"
                )
                phq9_responses.append(response)
        else:
            # All responses are False when "no symptoms" is selected
            phq9_responses = [False] * 9
    
    with tab2:
        st.markdown("#### GAD-7 Anxiety Assessment")
        st.caption("Are you experiencing any of these symptoms today?")
        
        # Quick option for no symptoms
        no_gad7_symptoms = st.checkbox(
            "✅ **No anxiety symptoms today**", 
            key="no_gad7_symptoms",
            help="Check this if you're not experiencing any anxiety symptoms today"
        )
        
        if not no_gad7_symptoms:
            gad7_questions = [
                "Feeling nervous, anxious or on edge",
                "Not being able to stop or control worrying",
                "Worrying too much about different things",
                "Trouble relaxing",
                "Being so restless that it is hard to sit still",
                "Becoming easily annoyed or irritable",
                "Feeling afraid as if something awful might happen"
            ]
            
            gad7_responses = []
            
            for i, question in enumerate(gad7_questions):
                # Pre-fill with existing data if available
                default_value = False
                if existing_entry and hasattr(existing_entry, f'gad7_{i}'):
                    default_value = getattr(existing_entry, f'gad7_{i}', False)
                
                response = st.checkbox(
                    f"{i+1}. {question}",
                    value=default_value,
                    key=f"gad7_{i}"
                )
                gad7_responses.append(response)
        else:
            # All responses are False when "no symptoms" is selected
            gad7_responses = [False] * 7
    
    # Save button
    if st.button("💾 Save Daily Entry", type="primary"):
        # Save the entry
        success = data_manager.save_daily_entry(
            selected_date,
            phq9_responses,
            gad7_responses
        )
        
        if success:
            st.success("✅ Daily entry saved successfully!")
            
            # Calculate today's symptom counts
            phq9_count = sum(phq9_responses)
            gad7_count = sum(gad7_responses)
            
            # Get rolling scores for the past 14 days (including today)
            rolling_data = data_manager.get_rolling_data(selected_date, days=14)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Today's PHQ-9 Symptoms", f"{phq9_count}/9")
                
                # Show 14-day rolling average if we have data
                if not rolling_data.empty and len(rolling_data) > 0:
                    # Calculate rolling average
                    phq9_cols = [f'phq9_{i}' for i in range(9)]
                    rolling_avg = rolling_data[phq9_cols].sum(axis=1).mean()
                    st.caption(f"14-day average: {rolling_avg:.1f} symptoms/day")
                    
                    # Severity based on rolling average
                    if rolling_avg <= 1:
                        severity = "Low"
                        severity_class = "severity-low"
                    elif rolling_avg <= 3:
                        severity = "Moderate"
                        severity_class = "severity-mild"
                    elif rolling_avg <= 5:
                        severity = "High"
                        severity_class = "severity-moderate"
                    else:
                        severity = "Very High"
                        severity_class = "severity-severe"
                    
                    st.markdown(f'<p class="{severity_class}">Trend: {severity}</p>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Today's GAD-7 Symptoms", f"{gad7_count}/7")
                
                # Show 14-day rolling average if we have data
                if not rolling_data.empty and len(rolling_data) > 0:
                    # Calculate rolling average
                    gad7_cols = [f'gad7_{i}' for i in range(7)]
                    rolling_avg = rolling_data[gad7_cols].sum(axis=1).mean()
                    st.caption(f"14-day average: {rolling_avg:.1f} symptoms/day")
                    
                    # Severity based on rolling average
                    if rolling_avg <= 1:
                        severity = "Low"
                        severity_class = "severity-low"
                    elif rolling_avg <= 2:
                        severity = "Moderate"
                        severity_class = "severity-mild"
                    elif rolling_avg <= 4:
                        severity = "High"
                        severity_class = "severity-moderate"
                    else:
                        severity = "Very High"
                        severity_class = "severity-severe"
                    
                    st.markdown(f'<p class="{severity_class}">Trend: {severity}</p>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.rerun()
        else:
            st.error("❌ Error saving entry. Please try again.")

def historical_analysis_page(data_manager):
    st.markdown('<h1 class="main-header">📈 Historical Analysis</h1>', unsafe_allow_html=True)
    
    # Get historical data
    historical_data = data_manager.get_historical_data()
    
    if historical_data.empty:
        st.info("📊 No historical data available. Start by recording your daily symptoms!")
        return
    
    # Date range selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=historical_data['date'].min().date(),
            min_value=historical_data['date'].min().date(),
            max_value=historical_data['date'].max().date()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=historical_data['date'].max().date(),
            min_value=historical_data['date'].min().date(),
            max_value=historical_data['date'].max().date()
        )
    
    # Filter data by date range
    mask = (historical_data['date'].dt.date >= start_date) & (historical_data['date'].dt.date <= end_date)
    filtered_data = historical_data.loc[mask]
    
    if filtered_data.empty:
        st.warning("No data available for the selected date range.")
        return
    
    # Calculate scores
    phq9_scorer = PHQ9Scorer()
    gad7_scorer = GAD7Scorer()
    
    # Add daily symptom counts to the data
    filtered_data = filtered_data.copy()
    phq9_counts = []
    gad7_counts = []
    
    for _, row in filtered_data.iterrows():
        # Calculate PHQ-9 daily symptom count (boolean values)
        phq9_responses = [row[f'phq9_{i}'] for i in range(9) if f'phq9_{i}' in row]
        phq9_count = sum([r for r in phq9_responses if pd.notna(r) and r])
        phq9_counts.append(phq9_count)
        
        # Calculate GAD-7 daily symptom count (boolean values)
        gad7_responses = [row[f'gad7_{i}'] for i in range(7) if f'gad7_{i}' in row]
        gad7_count = sum([r for r in gad7_responses if pd.notna(r) and r])
        gad7_counts.append(gad7_count)
    
    filtered_data['phq9_count'] = phq9_counts
    filtered_data['gad7_count'] = gad7_counts
    
    # Display summary statistics
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Entries", len(filtered_data))
    
    with col2:
        avg_phq9 = filtered_data['phq9_count'].mean()
        st.metric("Avg PHQ-9 Symptoms/Day", f"{avg_phq9:.1f}")
    
    with col3:
        avg_gad7 = filtered_data['gad7_count'].mean()
        st.metric("Avg GAD-7 Symptoms/Day", f"{avg_gad7:.1f}")
    
    with col4:
        days_tracked = (end_date - start_date).days + 1
        completion_rate = (len(filtered_data) / days_tracked) * 100
        st.metric("Completion Rate", f"{completion_rate:.1f}%")
    
    # Create tabs for different views
    chart_tab1, chart_tab2 = st.tabs(["📊 Daily Symptoms", "📈 14-Day Rolling Averages"])
    
    with chart_tab1:
        st.subheader("Daily Symptom Counts")
        
        if len(filtered_data) > 1:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=filtered_data['date'],
                y=filtered_data['phq9_count'],
                mode='lines+markers',
                name='PHQ-9 Symptoms (Depression)',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=filtered_data['date'],
                y=filtered_data['gad7_count'],
                mode='lines+markers',
                name='GAD-7 Symptoms (Anxiety)',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Daily Symptom Counts Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Symptoms",
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_tab2:
        st.subheader("14-Day Rolling Averages")
        
        if len(filtered_data) >= 14:
            # Calculate 14-day rolling averages
            rolling_data = []
            dates_sorted = filtered_data.sort_values('date')
            
            for i in range(13, len(dates_sorted)):  # Start from day 14
                # Get the 14-day window ending on this date
                window_data = dates_sorted.iloc[i-13:i+1]
                date = window_data.iloc[-1]['date']
                
                # Calculate rolling averages
                phq9_rolling = window_data['phq9_count'].mean()
                gad7_rolling = window_data['gad7_count'].mean()
                
                rolling_data.append({
                    'date': date,
                    'phq9_rolling': phq9_rolling,
                    'gad7_rolling': gad7_rolling
                })
            
            if rolling_data:
                rolling_df = pd.DataFrame(rolling_data)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=rolling_df['date'],
                    y=rolling_df['phq9_rolling'],
                    mode='lines+markers',
                    name='PHQ-9 14-Day Average',
                    line=dict(color='#1f77b4', width=4),
                    marker=dict(size=10)
                ))
                
                fig.add_trace(go.Scatter(
                    x=rolling_df['date'],
                    y=rolling_df['gad7_rolling'],
                    mode='lines+markers',
                    name='GAD-7 14-Day Average',
                    line=dict(color='#ff7f0e', width=4),
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    title="14-Day Rolling Average Symptom Trends",
                    xaxis_title="Date",
                    yaxis_title="Average Symptoms per Day",
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show current rolling averages
                if len(rolling_df) > 0:
                    latest = rolling_df.iloc[-1]
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Current PHQ-9 Rolling Average", 
                            f"{latest['phq9_rolling']:.1f} symptoms/day",
                            delta=f"{latest['phq9_rolling'] - rolling_df.iloc[-2]['phq9_rolling']:.1f}" if len(rolling_df) > 1 else None
                        )
                    
                    with col2:
                        st.metric(
                            "Current GAD-7 Rolling Average", 
                            f"{latest['gad7_rolling']:.1f} symptoms/day",
                            delta=f"{latest['gad7_rolling'] - rolling_df.iloc[-2]['gad7_rolling']:.1f}" if len(rolling_df) > 1 else None
                        )
            else:
                st.info("Need at least 14 days of data to show rolling averages.")
        else:
            st.info(f"Need at least 14 days of data to show rolling averages. You have {len(filtered_data)} days of data.")
    
    # Symptom frequency analysis
    st.subheader("🎯 Symptom Frequency Analysis")
    
    # Calculate symptom frequencies
    phq9_freq = {}
    for i in range(9):
        col = f'phq9_{i}'
        if col in filtered_data.columns:
            phq9_freq[f'PHQ-9 Symptom {i+1}'] = filtered_data[col].sum()
    
    gad7_freq = {}
    for i in range(7):
        col = f'gad7_{i}'
        if col in filtered_data.columns:
            gad7_freq[f'GAD-7 Symptom {i+1}'] = filtered_data[col].sum()
    
    # Create frequency chart
    all_freq = {**phq9_freq, **gad7_freq}
    if all_freq:
        freq_df = pd.DataFrame(list(all_freq.items()), columns=['Symptom', 'Frequency'])
        
        fig = px.bar(
            freq_df,
            x='Symptom',
            y='Frequency',
            title="Symptom Frequency in Selected Period",
            color='Frequency',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def export_data_page(data_manager):
    st.markdown('<h1 class="main-header">📤 Export Your Data</h1>', unsafe_allow_html=True)
    
    # Get all user data
    historical_data = data_manager.get_historical_data()
    
    if historical_data.empty:
        st.info("📊 No data available to export. Start by recording your daily symptoms!")
        return
    
    # Date range selection for export
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Export Start Date",
            value=historical_data['date'].min().date(),
            min_value=historical_data['date'].min().date(),
            max_value=historical_data['date'].max().date(),
            key="export_start"
        )
    
    with col2:
        end_date = st.date_input(
            "Export End Date",
            value=historical_data['date'].max().date(),
            min_value=historical_data['date'].min().date(),
            max_value=historical_data['date'].max().date(),
            key="export_end"
        )
    
    # Filter data for export
    mask = (historical_data['date'].dt.date >= start_date) & (historical_data['date'].dt.date <= end_date)
    export_data = historical_data.loc[mask].copy()
    
    if export_data.empty:
        st.warning("No data available for the selected date range.")
        return
    
    # Calculate daily symptom counts for export
    phq9_counts = []
    gad7_counts = []
    
    for _, row in export_data.iterrows():
        # Calculate PHQ-9 daily symptom count
        phq9_responses = [row[f'phq9_{i}'] for i in range(9) if f'phq9_{i}' in row]
        phq9_count = sum([r for r in phq9_responses if pd.notna(r) and r])
        phq9_counts.append(phq9_count)
        
        # Calculate GAD-7 daily symptom count
        gad7_responses = [row[f'gad7_{i}'] for i in range(7) if f'gad7_{i}' in row]
        gad7_count = sum([r for r in gad7_responses if pd.notna(r) and r])
        gad7_counts.append(gad7_count)
    
    export_data['phq9_daily_count'] = phq9_counts
    export_data['gad7_daily_count'] = gad7_counts
    
    # Reorder columns for better readability
    columns_order = ['date', 'phq9_daily_count', 'gad7_daily_count']
    
    # Add individual symptom columns
    for i in range(9):
        col = f'phq9_{i}'
        if col in export_data.columns:
            columns_order.append(col)
    
    for i in range(7):
        col = f'gad7_{i}'
        if col in export_data.columns:
            columns_order.append(col)
    
    # Reorder the dataframe
    export_data = export_data[columns_order]
    
    # Preview the data
    st.subheader("📋 Data Preview")
    st.dataframe(export_data.head(10), use_container_width=True)
    
    # CSV export
    csv_data = export_data.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name=f"mental_health_data_{start_date}_to_{end_date}.csv",
        mime="text/csv"
    )
    
    # Summary statistics
    st.subheader("Data Summary")
    st.write(f"Total entries: {len(export_data)}")
    st.write(f"Date range: {start_date} to {end_date}")
    
    # Calculate average symptom frequencies
    phq9_cols = [col for col in export_data.columns if col.startswith('phq9_')]
    gad7_cols = [col for col in export_data.columns if col.startswith('gad7_')]
    
    if phq9_cols:
        avg_phq9_symptoms = export_data[phq9_cols].mean().mean()
        st.write(f"Average PHQ-9 symptoms per day: {avg_phq9_symptoms:.1f}")
    
    if gad7_cols:
        avg_gad7_symptoms = export_data[gad7_cols].mean().mean()
        st.write(f"Average GAD-7 symptoms per day: {avg_gad7_symptoms:.1f}")

def logout():
    """Handle user logout"""
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    if 'data_manager' in st.session_state:
        del st.session_state.data_manager
    st.rerun()

def main():
    st.set_page_config(
        page_title="Mental Health Tracker",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    apply_custom_css()
    
    # Check if user is logged in
    if not st.session_state.get('logged_in', False):
        login_page()
        return
    
    # Initialize data manager and set user_id if logged in
    data_manager = get_data_manager()
    if st.session_state.get('user_id'):
        data_manager.set_user_id(st.session_state.user_id)
    
    # Sidebar navigation with user info and logout
    st.sidebar.title("🧠 Mental Health Tracker")
    st.sidebar.write(f"👤 Welcome, {st.session_state.get('username', 'User')}!")
    
    if st.sidebar.button("🚪 Logout"):
        logout()
        return
    
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox("Navigate", ["Daily Tracking", "Historical Analysis", "Export Data"])
    
    if page == "Daily Tracking":
        daily_tracking_page(data_manager)
    elif page == "Historical Analysis":
        historical_analysis_page(data_manager)
    elif page == "Export Data":
        export_data_page(data_manager)

if __name__ == "__main__":
    main()
