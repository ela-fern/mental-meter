import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import pdfplumber
import PyPDF2
import re
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import io
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
from database import DatabaseManager
from scoring import PHQ9Scorer, GAD7Scorer

def apply_custom_css():
    """Apply custom CSS for the application"""
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        border: 1px solid #4CAF50;
        background-color: #4CAF50;
        color: white;
    }
    .stButton > button:hover {
        background-color: #45a049;
        border-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

def get_data_manager():
    """Get or create a DataManager instance"""
    if 'data_manager' not in st.session_state:
        user_id = st.session_state.get('user_id', None)
        st.session_state.data_manager = DatabaseManager(user_id=user_id)
    elif st.session_state.get('user_id') and not st.session_state.data_manager.user_id:
        # Update existing data manager with user ID
        st.session_state.data_manager.set_user_id(st.session_state.user_id)
    return st.session_state.data_manager

def login_page():
    """Display login/registration page"""
    st.title("🧠 Mental Health Tracker")
    st.caption("Your personal mental health tracking companion")
    
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login", type="primary")
            
            if login_submitted:
                if username and password:
                    # Create temporary database manager for authentication
                    temp_db = DatabaseManager()
                    success, user_id = temp_db.authenticate_user(username, password)
                    
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        # Update data manager with user ID
                        if 'data_manager' in st.session_state:
                            st.session_state.data_manager.set_user_id(user_id)
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
    
    with tab2:
        st.subheader("Create New Account")
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_submitted = st.form_submit_button("Register", type="primary")
            
            if register_submitted:
                if new_username and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        # Create temporary database manager for registration
                        temp_db = DatabaseManager()
                        success, message = temp_db.create_user(new_username, new_password)
                        
                        if success:
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error(message)
                else:
                    st.error("Please fill in all fields")

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
    
    # Initialize data manager
    data_manager = get_data_manager()
    
    # Sidebar navigation with user info and logout
    st.sidebar.title("🧠 Mental Health Tracker")
    st.sidebar.write(f"👤 Welcome, {st.session_state.get('username', 'User')}!")
    
    if st.sidebar.button("🚪 Logout"):
        logout()
        return
    
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox("Navigate", ["Daily Tracking", "Historical Analysis", "Import Data", "Export Data"])
    
    if page == "Daily Tracking":
        daily_tracking_page(data_manager)
    elif page == "Historical Analysis":
        historical_analysis_page(data_manager)
    elif page == "Import Data":
        import_pdf_page(data_manager)
    elif page == "Export Data":
        export_data_page(data_manager)

def daily_tracking_page(data_manager):
    st.title("Daily Symptom Tracking")
    
    # Date selection
    selected_date = st.date_input("Select Date", datetime.now().date())
    
    # Check if data already exists for this date
    existing_data = data_manager.get_data_for_date(selected_date)
    
    if existing_data is not None:
        st.info(f"Data already exists for {selected_date}. You can update it below.")
    
    # PHQ-9 Depression Symptoms
    st.subheader("PHQ-9 Depression Symptoms")
    st.caption("Check the symptoms you experienced today:")
    
    phq9_symptoms = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself or that you are a failure",
        "Trouble concentrating on things",
        "Moving or speaking slowly, or being fidgety/restless",
        "Thoughts that you would be better off dead or hurting yourself"
    ]
    
    phq9_responses = {}
    for i, symptom in enumerate(phq9_symptoms):
        key = f"phq9_{i}"
        default_value = False
        if existing_data is not None and key in existing_data:
            default_value = existing_data[key]
        phq9_responses[key] = st.checkbox(symptom, value=default_value, key=key)
    
    # GAD-7 Anxiety Symptoms
    st.subheader("GAD-7 Anxiety Symptoms")
    st.caption("Check the symptoms you experienced today:")
    
    gad7_symptoms = [
        "Feeling nervous, anxious, or on edge",
        "Not being able to stop or control worrying",
        "Worrying too much about different things",
        "Trouble relaxing",
        "Being so restless that it's hard to sit still",
        "Becoming easily annoyed or irritable",
        "Feeling afraid as if something awful might happen"
    ]
    
    gad7_responses = {}
    for i, symptom in enumerate(gad7_symptoms):
        key = f"gad7_{i}"
        default_value = False
        if existing_data is not None and key in existing_data:
            default_value = existing_data[key]
        gad7_responses[key] = st.checkbox(symptom, value=default_value, key=key)
    
    # Save button
    if st.button("Save Daily Entry", type="primary"):
        # Combine all responses
        all_responses = {**phq9_responses, **gad7_responses}
        all_responses['date'] = selected_date.isoformat()
        
        # Save data
        data_manager.save_daily_entry(selected_date, all_responses)
        st.success(f"Data saved for {selected_date}")
        st.rerun()
    
    # Display current rolling scores
    st.subheader("Current Rolling Scores (Past 14 Days)")
    
    # Get rolling scores
    phq9_scorer = PHQ9Scorer()
    gad7_scorer = GAD7Scorer()
    
    phq9_score = phq9_scorer.calculate_rolling_score(data_manager, selected_date)
    gad7_score = gad7_scorer.calculate_rolling_score(data_manager, selected_date)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("PHQ-9 Score", f"{phq9_score}/27")
        severity = phq9_scorer.get_severity_level(phq9_score)
        st.caption(f"Severity: {severity}")
    
    with col2:
        st.metric("GAD-7 Score", f"{gad7_score}/21")
        severity = gad7_scorer.get_severity_level(gad7_score)
        st.caption(f"Severity: {severity}")

def historical_analysis_page(data_manager):
    st.title("Historical Analysis")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now().date())
    
    if start_date > end_date:
        st.error("Start date must be before end date")
        return
    
    # Get historical data
    historical_data = data_manager.get_historical_data(start_date, end_date)
    
    if historical_data.empty:
        st.warning("No data available for the selected date range")
        return
    
    # Calculate rolling scores for each date
    phq9_scorer = PHQ9Scorer()
    gad7_scorer = GAD7Scorer()
    
    scores_data = []
    current_date = start_date
    while current_date <= end_date:
        phq9_score = phq9_scorer.calculate_rolling_score(data_manager, current_date)
        gad7_score = gad7_scorer.calculate_rolling_score(data_manager, current_date)
        
        scores_data.append({
            'date': current_date,
            'PHQ-9': phq9_score,
            'GAD-7': gad7_score
        })
        current_date += timedelta(days=1)
    
    scores_df = pd.DataFrame(scores_data)
    
    # Create charts
    st.subheader("Score Trends Over Time")
    
    # Line chart for scores
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=scores_df['date'],
        y=scores_df['PHQ-9'],
        mode='lines+markers',
        name='PHQ-9',
        line=dict(color='#2E8B57')
    ))
    
    fig.add_trace(go.Scatter(
        x=scores_df['date'],
        y=scores_df['GAD-7'],
        mode='lines+markers',
        name='GAD-7',
        line=dict(color='#4682B4')
    ))
    
    # Add severity level reference lines
    fig.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Mild (PHQ-9)")
    fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Moderate (PHQ-9)")
    fig.add_hline(y=15, line_dash="dash", line_color="darkred", annotation_text="Mod. Severe (PHQ-9)")
    fig.add_hline(y=20, line_dash="dash", line_color="purple", annotation_text="Severe (PHQ-9)")
    
    fig.update_layout(
        title="Mental Health Scores Over Time",
        xaxis_title="Date",
        yaxis_title="Score",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**PHQ-9 Statistics**")
        st.write(f"Average: {scores_df['PHQ-9'].mean():.1f}")
        st.write(f"Minimum: {scores_df['PHQ-9'].min()}")
        st.write(f"Maximum: {scores_df['PHQ-9'].max()}")
    
    with col2:
        st.write("**GAD-7 Statistics**")
        st.write(f"Average: {scores_df['GAD-7'].mean():.1f}")
        st.write(f"Minimum: {scores_df['GAD-7'].min()}")
        st.write(f"Maximum: {scores_df['GAD-7'].max()}")
    
    # Symptom frequency analysis
    st.subheader("Symptom Frequency Analysis")
    
    # Calculate symptom frequencies
    phq9_freq = {}
    gad7_freq = {}
    
    for i in range(9):
        col = f'phq9_{i}'
        if col in historical_data.columns:
            phq9_freq[f'PHQ-9 Symptom {i+1}'] = historical_data[col].sum()
    
    for i in range(7):
        col = f'gad7_{i}'
        if col in historical_data.columns:
            gad7_freq[f'GAD-7 Symptom {i+1}'] = historical_data[col].sum()
    
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

def import_pdf_page(data_manager):
    st.title("Import Assessment Data")
    st.caption("Upload PDF files or screenshots containing mental health assessment data to import into your tracker")
    
    # Tabs for different import types
    tab1, tab2 = st.tabs(["📄 PDF Import", "📷 Screenshot Import"])
    
    with tab1:
        st.subheader("PDF File Import")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_upload")
        
        if uploaded_file is not None:
            process_pdf_upload(uploaded_file, data_manager)
    
    with tab2:
        st.subheader("Screenshot Import")
        st.caption("Upload screenshots of mental health assessments (supports multiple files)")
        uploaded_images = st.file_uploader(
            "Choose screenshot files", 
            type=["png", "jpg", "jpeg", "bmp", "tiff"], 
            accept_multiple_files=True,
            key="image_upload"
        )
        
        if uploaded_images:
            process_image_uploads(uploaded_images, data_manager)

def process_pdf_upload(uploaded_file, data_manager):
    
    try:
        # Extract text from PDF
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        
        st.subheader("PDF Content Preview")
        st.text_area("Extracted Text", text[:1000] + "..." if len(text) > 1000 else text, height=200)
        
        # Parse and display extracted data
        parse_and_display_extracted_data(text, data_manager, "PDF")
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        st.caption("Please ensure the PDF is readable and try again, or use the manual entry form below.")

def preprocess_image(image):
    """Enhanced image preprocessing for better OCR results"""
    try:
        # Convert PIL Image to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Convert to grayscale if it's not already
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply different preprocessing techniques
        processed_images = []
        
        # Original grayscale
        processed_images.append(("Original", gray))
        
        # Gaussian blur + threshold
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        processed_images.append(("Threshold", thresh1))
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(("Adaptive", adaptive))
        
        # Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        processed_images.append(("Morphological", morph))
        
        # Histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray)
        processed_images.append(("Equalized", equalized))
        
        return processed_images
        
    except Exception as e:
        # Fallback to simple PIL processing if OpenCV fails
        enhanced_images = []
        
        # Convert to grayscale
        if image.mode != 'L':
            gray_img = image.convert('L')
        else:
            gray_img = image
        enhanced_images.append(("Original", gray_img))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray_img)
        contrast_img = enhancer.enhance(2.0)
        enhanced_images.append(("High Contrast", contrast_img))
        
        # Enhance sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(gray_img)
        sharp_img = sharpness_enhancer.enhance(2.0)
        enhanced_images.append(("Sharp", sharp_img))
        
        return enhanced_images

def extract_text_with_multiple_methods(image, progress_callback=None):
    """Extract text using optimized OCR methods with progress tracking"""
    all_texts = {}
    total_steps = 8  # Reduced from previous version for speed
    current_step = 0
    
    def update_progress():
        nonlocal current_step
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps)
    
    # Method 1: Fast primary configs (most effective ones only)
    try:
        # Optimized configs - removed redundant ones for speed
        primary_configs = [
            r'--oem 3 --psm 6 -l eng+spa',  # Best general purpose
            r'--oem 3 --psm 3 -l eng+spa',  # Auto page segmentation
            r'--oem 3 --psm 4 -l eng+spa',  # Single column of text
        ]
        
        for i, config in enumerate(primary_configs):
            try:
                text = pytesseract.image_to_string(image, config=config)
                if text.strip():
                    all_texts[f"Primary_{i+1}"] = text.strip()
                update_progress()
            except:
                update_progress()
                continue
    except Exception as e:
        for _ in range(3):  # Skip remaining primary configs
            update_progress()
    
    # Method 2: Quick preprocessing (only most effective ones)
    if OPENCV_AVAILABLE:
        try:
            # Simplified preprocessing for speed
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Only apply the most effective preprocessing
            processed_variants = [
                ("Original", gray),
                ("Threshold", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
            ]
            
            for name, processed_img in processed_variants:
                try:
                    pil_img = Image.fromarray(processed_img)
                    text = pytesseract.image_to_string(pil_img, config=r'--oem 3 --psm 6 -l eng+spa')
                    if text.strip():
                        all_texts[f"Processed_{name}"] = text.strip()
                    update_progress()
                except:
                    update_progress()
                    continue
        except Exception as e:
            for _ in range(2):  # Skip remaining preprocessing
                update_progress()
    else:
        for _ in range(2):  # Skip preprocessing steps if OpenCV not available
            update_progress()
    
    # Method 3: Enhanced PIL (quick contrast enhancement only)
    try:
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(2.0)  # Stronger enhancement
        text = pytesseract.image_to_string(enhanced, config=r'--oem 3 --psm 6 -l eng+spa')
        if text.strip():
            all_texts["Enhanced_PIL"] = text.strip()
        update_progress()
    except:
        update_progress()
    
    # Final fallback - basic OCR if nothing worked
    if not all_texts:
        try:
            text = pytesseract.image_to_string(image)
            if text.strip():
                all_texts["Fallback"] = text.strip()
        except:
            pass
        update_progress()
    else:
        update_progress()
    
    # Final progress update
    if progress_callback:
        progress_callback(1.0)
    
    # Return the longest/best text result
    if not all_texts:
        return "", {}
    
    # Find the text with most content (heuristic: longest meaningful text)
    best_text = max(all_texts.values(), key=lambda x: len([word for word in x.split() if len(word) > 2]))
    
    return best_text, all_texts

def process_image_uploads(uploaded_images, data_manager):
    try:
        all_extracted_text = ""
        
        st.subheader(f"Processing {len(uploaded_images)} image(s)")
        st.info("Using optimized OCR with multiple extraction methods")
        
        # Overall progress for all images
        if len(uploaded_images) > 1:
            overall_progress = st.progress(0)
            overall_status = st.empty()
        
        # Process each image
        for img_idx, uploaded_image in enumerate(uploaded_images):
            with st.expander(f"📷 Image {img_idx+1}: {uploaded_image.name}", expanded=(len(uploaded_images) == 1)):
                # Display the image
                image = Image.open(uploaded_image)
                st.image(image, caption=f"Uploaded Image {img_idx+1}", use_container_width=True)
                
                # Progress bar for current image processing
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                    if progress < 1.0:
                        status_text.text(f"Processing OCR methods... {int(progress * 100)}%")
                    else:
                        status_text.text("OCR processing complete!")
                
                # Extract text using enhanced methods with progress tracking
                try:
                    best_text, all_texts = extract_text_with_multiple_methods(image, update_progress)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    if best_text:
                        st.success(f"✅ Text extracted successfully from image {img_idx+1}")
                        st.text_area(f"Extracted Text from Image {img_idx+1}", 
                                   best_text[:500] + "..." if len(best_text) > 500 else best_text, 
                                   height=150, key=f"ocr_text_{img_idx}")
                        all_extracted_text += f"\n--- Image {img_idx+1} ---\n" + best_text
                        
                        # Compact debug info
                        with st.expander(f"🔍 Extraction Details for Image {img_idx+1}"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Text Length", f"{len(best_text)} chars")
                            with col2:
                                st.metric("Methods Used", len(all_texts))
                            with col3:
                                st.metric("Lines Found", len(best_text.splitlines()))
                            
                            # Show successful extraction methods
                            st.write("**Successful Methods:**")
                            for method, text in all_texts.items():
                                if text.strip() and len(text) > 10:
                                    st.write(f"• {method}: {len(text)} chars")
                            
                            # Show preview of first few meaningful lines
                            meaningful_lines = [line.strip() for line in best_text.splitlines() if line.strip() and len(line.strip()) > 3][:5]
                            if meaningful_lines:
                                st.write("**Content Preview:**")
                                for j, line in enumerate(meaningful_lines):
                                    st.write(f"{j+1}. {line}")
                    else:
                        st.warning(f"⚠️ No readable text detected in image {img_idx+1}")
                        
                        # Compact troubleshooting info
                        with st.expander(f"🔍 Troubleshooting for Image {img_idx+1}"):
                            if all_texts:
                                st.write("**All attempts returned minimal text:**")
                                for method, text in all_texts.items():
                                    st.write(f"• {method}: {len(text) if text else 0} characters")
                            else:
                                st.write("**Possible issues:**")
                                st.write("• Image resolution too low")
                                st.write("• Text too small or blurry")
                                st.write("• Handwritten text (OCR works best with printed text)")
                                st.write("• Poor contrast or lighting")
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error processing image {img_idx+1}: {str(e)}")
                
                # Update overall progress
                if len(uploaded_images) > 1:
                    overall_progress.progress((img_idx + 1) / len(uploaded_images))
                    overall_status.text(f"Completed {img_idx + 1} of {len(uploaded_images)} images")
        
        # Clear overall progress indicators
        if len(uploaded_images) > 1:
            overall_progress.empty()
            overall_status.empty()
        
        if all_extracted_text.strip():
            st.subheader("Combined Text Analysis")
            # Parse and display extracted data from all images
            parse_and_display_extracted_data(all_extracted_text, data_manager, "Screenshots")
        else:
            st.warning("No text was extracted from any of the uploaded images.")
            # Still show manual entry form
            show_manual_entry_form(data_manager, "image")
    
    except Exception as e:
        st.error(f"Error processing images: {str(e)}")

def parse_and_display_extracted_data(text, data_manager, source_type):
    # Try to parse PHQ-9 and GAD-7 scores
    st.subheader("Data Extraction")
    
    # Show debug information
    with st.expander("🔍 Debug: Full Extracted Text"):
        st.text_area("Complete extracted text", text, height=200)
        st.write(f"**Total characters:** {len(text)}")
        st.write(f"**Total lines:** {len(text.splitlines())}")
    
    # Comprehensive PHQ-9 patterns (very flexible, multiple languages)
    phq9_patterns = [
        # English patterns
        r'PHQ[-\s]*9[^\d]*(\d{1,2})',  # PHQ-9 followed by digits
        r'PHQ[-\s]*9.*?score[^\d]*(\d{1,2})',
        r'PHQ[-\s]*9.*?total[^\d]*(\d{1,2})', 
        r'depression.*?score[^\d]*(\d{1,2})',
        r'depression.*?total[^\d]*(\d{1,2})',
        r'(\d{1,2})[^\d]*PHQ[-\s]*9',  # Score before PHQ-9
        r'(\d{1,2})\s*/\s*27',  # X/27 format
        r'(\d{1,2})\s*out\s*of\s*27',
        r'(\d{1,2})\s*de\s*27',  # Spanish
        
        # Spanish patterns
        r'PHQ[-\s]*9.*?puntuación[^\d]*(\d{1,2})',
        r'PHQ[-\s]*9.*?puntaje[^\d]*(\d{1,2})',
        r'PHQ[-\s]*9.*?total[^\d]*(\d{1,2})',
        r'depresión.*?(\d{1,2})',
        r'depresión.*?puntuación[^\d]*(\d{1,2})',
        
        # French patterns
        r'PHQ[-\s]*9.*?score[^\d]*(\d{1,2})',
        r'dépression.*?(\d{1,2})',
        r'(\d{1,2})\s*/\s*27',
        
        # German patterns
        r'PHQ[-\s]*9.*?wert[^\d]*(\d{1,2})',
        r'depression.*?wert[^\d]*(\d{1,2})',
        
        # General number extraction near PHQ-9
        r'PHQ[-\s]*9\D*(\d{1,2})\D*',
        r'PHQ\D*9\D*(\d{1,2})',
        
        # Look for any two-digit number near depression-related terms
        r'(?:depression|depresión|dépression|depres)\D*(\d{1,2})',
        
        # Simple score formats
        r'score\s*:?\s*(\d{1,2})',
        r'total\s*:?\s*(\d{1,2})',
        r'puntuación\s*:?\s*(\d{1,2})',
        r'puntaje\s*:?\s*(\d{1,2})',
    ]
    
    phq9_matches = []
    for pattern in phq9_patterns:
        phq9_matches.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Comprehensive GAD-7 patterns (very flexible, multiple languages)
    gad7_patterns = [
        # English patterns
        r'GAD[-\s]*7[^\d]*(\d{1,2})',  # GAD-7 followed by digits
        r'GAD[-\s]*7.*?score[^\d]*(\d{1,2})',
        r'GAD[-\s]*7.*?total[^\d]*(\d{1,2})',
        r'anxiety.*?score[^\d]*(\d{1,2})',
        r'anxiety.*?total[^\d]*(\d{1,2})',
        r'(\d{1,2})[^\d]*GAD[-\s]*7',  # Score before GAD-7
        r'(\d{1,2})\s*/\s*21',  # X/21 format
        r'(\d{1,2})\s*out\s*of\s*21',
        r'(\d{1,2})\s*de\s*21',  # Spanish
        
        # Spanish patterns
        r'GAD[-\s]*7.*?puntuación[^\d]*(\d{1,2})',
        r'GAD[-\s]*7.*?puntaje[^\d]*(\d{1,2})',
        r'GAD[-\s]*7.*?total[^\d]*(\d{1,2})',
        r'ansiedad.*?(\d{1,2})',
        r'ansiedad.*?puntuación[^\d]*(\d{1,2})',
        
        # French patterns
        r'GAD[-\s]*7.*?score[^\d]*(\d{1,2})',
        r'anxiété.*?(\d{1,2})',
        r'(\d{1,2})\s*/\s*21',
        
        # German patterns
        r'GAD[-\s]*7.*?wert[^\d]*(\d{1,2})',
        r'angst.*?wert[^\d]*(\d{1,2})',
        
        # General number extraction near GAD-7
        r'GAD[-\s]*7\D*(\d{1,2})\D*',
        r'GAD\D*7\D*(\d{1,2})',
        
        # Look for any two-digit number near anxiety-related terms
        r'(?:anxiety|ansiedad|anxiété|angst)\D*(\d{1,2})',
        
        # Alternative GAD names
        r'generalized.{0,20}anxiety.{0,20}(\d{1,2})',
        r'trastorno.{0,20}ansiedad.{0,20}(\d{1,2})',
    ]
    
    gad7_matches = []
    for pattern in gad7_patterns:
        gad7_matches.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Comprehensive date patterns (very flexible, multiple formats)
    date_patterns = [
        # Numeric formats
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # 12/15/2024, 12-15-2024
        r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # 2024-12-15, 2024/12/15
        r'(\d{1,2}/\d{1,2}/\d{2,4})',  # US format
        r'(\d{1,2}-\d{1,2}-\d{2,4})',  # Dash format
        r'(\d{1,2}\.\d{1,2}\.\d{2,4})',  # European dot format
        r'(\d{1,2}\s+\d{1,2}\s+\d{2,4})',  # Space separated
        
        # Month name formats - English
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}',
        
        # Month name formats - Spanish
        r'(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+\d{1,2},?\s+\d{4}',
        
        # Month name formats - French
        r'(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{1,2},?\s+\d{4}',
        
        # Month name formats - German
        r'(januar|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember)\s+\d{1,2},?\s+\d{4}',
        
        # Date with context (multiple languages)
        r'(?:fecha|date|datum|data)[^\d]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(?:assessed|completed|evaluado|completado|evaluiert)[^\d]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(?:administered|aplicado|verabreicht)[^\d]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        
        # Today, yesterday patterns
        r'(today|aujourd\'hui|hoy|heute)',
        r'(yesterday|hier|ayer|gestern)',
        
        # Recent dates
        r'(\d{1,2}[/-]\d{1,2}[/-]202[0-9])',  # Any date in 2020s
    ]
    
    dates_found = []
    for pattern in date_patterns:
        dates_found.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Apply smart scoring - look for contextual clues
    def smart_score_detection(text_content, test_name, max_score):
        """Intelligently detect scores based on context"""
        potential_scores = []
        lines = text_content.lower().splitlines()
        
        for i, line in enumerate(lines):
            if test_name.lower() in line:
                # Look in this line and surrounding lines for numbers
                search_lines = [line]
                if i > 0:
                    search_lines.append(lines[i-1])
                if i < len(lines) - 1:
                    search_lines.append(lines[i+1])
                
                for search_line in search_lines:
                    numbers = re.findall(r'\b(\d{1,2})\b', search_line)
                    for num in numbers:
                        if num.isdigit() and 0 <= int(num) <= max_score:
                            potential_scores.append(int(num))
        
        # Also look for standalone valid numbers that could be scores
        all_numbers = re.findall(r'\b(\d{1,2})\b', text_content)
        valid_standalone = [int(n) for n in all_numbers if n.isdigit() and 0 <= int(n) <= max_score]
        
        return list(set(potential_scores + valid_standalone))
    
    # Smart detection for both tests
    smart_phq9 = smart_score_detection(text, "phq", 27)
    smart_gad7 = smart_score_detection(text, "gad", 21)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**PHQ-9 Scores Found:**")
        # Filter and combine pattern matches with smart detection
        valid_phq9_pattern = [score for score in phq9_matches if score.isdigit() and 0 <= int(score) <= 27]
        all_phq9_scores = list(set([int(s) for s in valid_phq9_pattern] + smart_phq9))
        
        if all_phq9_scores:
            for score in sorted(all_phq9_scores):
                st.write(f"• {score}")
        else:
            st.write("No PHQ-9 scores detected")
    
    with col2:
        st.write("**GAD-7 Scores Found:**")
        # Filter and combine pattern matches with smart detection
        valid_gad7_pattern = [score for score in gad7_matches if score.isdigit() and 0 <= int(score) <= 21]
        all_gad7_scores = list(set([int(s) for s in valid_gad7_pattern] + smart_gad7))
        
        if all_gad7_scores:
            for score in sorted(all_gad7_scores):
                st.write(f"• {score}")
        else:
            st.write("No GAD-7 scores detected")
    
    with col3:
        st.write("**Dates Found:**")
        if dates_found:
            unique_dates = list(set(dates_found))[:5]  # Remove duplicates, show first 5
            for date in unique_dates:
                st.write(f"• {date}")
        else:
            st.write("No dates detected")
    
    # Enhanced debugging with contextual analysis
    with st.expander("🔍 Enhanced Debug: Pattern Analysis"):
        st.write(f"**Pattern PHQ-9 matches:** {phq9_matches}")
        st.write(f"**Smart PHQ-9 detection:** {smart_phq9}")
        st.write(f"**Final PHQ-9 scores:** {all_phq9_scores}")
        st.write("---")
        st.write(f"**Pattern GAD-7 matches:** {gad7_matches}")
        st.write(f"**Smart GAD-7 detection:** {smart_gad7}")
        st.write(f"**Final GAD-7 scores:** {all_gad7_scores}")
        st.write("---")
        st.write(f"**Date matches:** {dates_found}")
        
        # Show all numbers found in context
        all_numbers = re.findall(r'\b\d{1,2}\b', text)
        st.write(f"**All numbers (1-2 digits):** {all_numbers[:30]}")
        
        # Show lines containing potential test names
        lines_with_tests = []
        for i, line in enumerate(text.splitlines()):
            if any(term in line.lower() for term in ['phq', 'gad', 'depression', 'anxiety', 'depres', 'ansied']):
                lines_with_tests.append(f"Line {i+1}: {line.strip()}")
        
        if lines_with_tests:
            st.write("**Lines with test-related terms:**")
            for line in lines_with_tests[:10]:  # Show first 10
                st.write(line)
    
    # Show manual entry form
    show_manual_entry_form(data_manager, source_type.lower())

def show_manual_entry_form(data_manager, source_type):
    # Manual data entry form
    st.subheader("Manual Data Entry / Entrada Manual de Datos")
    st.caption(f"If automatic extraction didn't work, you can manually enter the data from your {source_type}:")
    st.caption(f"Si la extracción automática no funcionó, puedes ingresar manualmente los datos de tu {source_type}:")
    
    with st.form(f"manual_{source_type}_entry"):
        entry_date = st.date_input("Date of Assessment", datetime.now().date())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**PHQ-9 Individual Scores (0-3 each):**")
            st.write("**PHQ-9 Puntuaciones Individuales (0-3 cada una):**")
            st.caption("0 = Not at all / Para nada")
            st.caption("1 = Several days / Varios días") 
            st.caption("2 = More than half the days / Más de la mitad de los días")
            st.caption("3 = Nearly every day / Casi todos los días")
            
            phq9_scores = []
            for i in range(9):
                score = st.selectbox(f"Symptom / Síntoma {i+1}", [0, 1, 2, 3], key=f"manual_{source_type}_phq9_{i}")
                phq9_scores.append(score)
            
            phq9_total = sum(phq9_scores)
            st.write(f"**Total PHQ-9 Score: {phq9_total}/27**")
        
        with col2:
            st.write("**GAD-7 Individual Scores (0-3 each):**")
            st.write("**GAD-7 Puntuaciones Individuales (0-3 cada una):**")
            st.caption("0 = Not at all / Para nada")
            st.caption("1 = Several days / Varios días")
            st.caption("2 = More than half the days / Más de la mitad de los días") 
            st.caption("3 = Nearly every day / Casi todos los días")
            
            gad7_scores = []
            for i in range(7):
                score = st.selectbox(f"Symptom / Síntoma {i+1}", [0, 1, 2, 3], key=f"manual_{source_type}_gad7_{i}")
                gad7_scores.append(score)
            
            gad7_total = sum(gad7_scores)
            st.write(f"**Total GAD-7 Score: {gad7_total}/21**")
        
        if st.form_submit_button("Import Data / Importar Datos", type="primary"):
            # Convert clinical scores back to daily entries for our system
            responses = {}
            
            # For PHQ-9: convert each symptom score back to daily tracking
            for i, score in enumerate(phq9_scores):
                if score == 0:
                    days_present = 0
                elif score == 1:
                    days_present = 4  # Several days (2-6 range, use middle)
                elif score == 2:
                    days_present = 8  # More than half (7-10 range, use middle)
                else:  # score == 3
                    days_present = 12  # Nearly every day (11-14 range, use middle)
                
                # Create entries for the calculated number of days
                for day_offset in range(14):
                    day_present = day_offset < days_present
                    responses[f'phq9_{i}'] = day_present
            
            # For GAD-7: same logic
            for i, score in enumerate(gad7_scores):
                if score == 0:
                    days_present = 0
                elif score == 1:
                    days_present = 4
                elif score == 2:
                    days_present = 8
                else:  # score == 3
                    days_present = 12
                
                for day_offset in range(14):
                    day_present = day_offset < days_present
                    responses[f'gad7_{i}'] = day_present
            
            # Save the data for the selected date
            data_manager.save_daily_entry(entry_date, responses)
            st.success(f"Data imported successfully for {entry_date} from {source_type}")
            st.rerun()

def export_data_page(data_manager):
    st.title("Export Data")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=30), key="export_start")
    with col2:
        end_date = st.date_input("End Date", datetime.now().date(), key="export_end")
    
    if start_date > end_date:
        st.error("Start date must be before end date")
        return
    
    # Get data for export
    export_data = data_manager.get_historical_data(start_date, end_date)
    
    if export_data.empty:
        st.warning("No data available for the selected date range")
        return
    
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(export_data)
    
    # Export options
    st.subheader("Export Options")
    
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

if __name__ == "__main__":
    main()
