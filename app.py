"""
Body Composition PDF Parser & Dashboard
A Streamlit web application for parsing body composition PDF reports
and visualizing health trends over time.
"""

import re
import io
from datetime import datetime

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import streamlit as st

# Import our database module
import database as db

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="Body Composition Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Authentication Functions
# =============================================================================

def show_login_form():
    """Display login form and handle authentication."""
    st.markdown("### 🔑 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if username and password:
                success, user_info = db.authenticate_user(username, password)
                if success:
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = user_info
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            else:
                st.warning("Please enter both username and password")


def show_register_form():
    """Display registration form and handle new user creation."""
    st.markdown("### 📝 Create Account")
    
    with st.form("register_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Create Account", use_container_width=True)
        
        if submit:
            # Validation
            if not all([name, email, username, password, password_confirm]):
                st.error("Please fill in all fields")
            elif password != password_confirm:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters")
            elif '@' not in email:
                st.error("Please enter a valid email address")
            else:
                success, message = db.create_user(username, email, password, name)
                if success:
                    st.success(f"✅ {message} Please login.")
                else:
                    st.error(f"❌ {message}")


def show_auth_page():
    """Display the authentication page with login/register tabs."""
    st.title("📊 Body Composition Dashboard")
    st.markdown("Track your health metrics and visualize your progress over time.")
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["� Login", "📝 Sign Up"])
        
        with tab1:
            show_login_form()
            st.markdown("---")
            st.info("**Demo Account:** Username: `demo` | Password: `demo123`")
        
        with tab2:
            show_register_form()
    
    return False


def is_authenticated():
    """Check if user is authenticated."""
    return st.session_state.get('authenticated', False)


def get_current_user():
    """Get the current logged-in user info."""
    return st.session_state.get('user', None)


def logout():
    """Clear session and logout user."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# =============================================================================
# Regex Patterns (All 28 Parameters)
# =============================================================================

PATTERNS = {
    "Name": r"Prevention Check\s+([A-Za-z ]+)",
    "Gender": r"\((male|female)\s+\d+\s+Years\)",
    "Age": r"\((?:male|female)\s+(\d+)\s+Years\)",
    "Date": r"Date:\s*([\d/]+)",
    "Time": r"Measures on .*? at ([\d:]+\s*(?:AM|PM))",
    "Height_cm": r"Height:\s*([\d.]+)\s*cm",
    "Weight_kg": r"Weight:\s*([\d.]+)\s*kg",
    "BMI": r"BMI:\s*([\d.]+)",
    "Body Fat %": r"Body Fat:\s*([\d.]+)\s*%",
    "Body Fat kg": r"Body Fat:[\s\S]*?=\s*([\d.]+)\s*kg",
    "Visceral Fat Level": r"Visceral fat:\s*([\d.]+)\s*Level",
    "Fat Free Mass_kg": r"Fat Free Mass:\s*([\d.]+)\s*kg",
    "Muscle Mass_kg": r"Muscle Mass:\s*([\d.]+)\s*kg",
    "Skeletal Muscle Mass %": r"Skeletal Muscle Mass:\s*([\d.]+)\s*%",
    "Skeletal Muscle Mass_kg": r"Skeletal Muscle Mass:[\s\S]*?\%\s*([\d.]+)\s*kg",
    "Bone Mass_kg": r"Bone Mass:\s*([\d.]+)\s*kg",
    "Sarcopenic Index": r"Sarcopenic Index:\s*([\d.]+)",
    "Body Water %": r"Body Water:\s*([\d.]+)\s*%",
    "Body Water_kg": r"Body Water:[\s\S]*?=\s*([\d.]+)\s*kg",
    "ECW_kg": r"ECW:\s*([\d.]+)\s*kg",
    "ECW/TBW %": r"ECW/TBW:\s*([\d.]+)\s*%",
    "ICW_kg": r"ICW:\s*([\d.]+)\s*kg",
    "ICW/TBW %": r"ICW/TBW:\s*([\d.]+)\s*%",
    "Phase Angle_deg": r"Phase angle:\s*([\d.]+)\s*°",
    "Impedance_Ohm": r"Impedance:\s*([\d.]+)\s*Ohm",
    "Metabolic Age": r"Metabolic Age:\s*([\d.]+)\s*Years",
    "BMR_kJ": r"Basal Metabolic Rate:\s*([\d.]+)\s*kJ",
    "BMR_kcal": r"Basal Metabolic Rate:[\s\S]*?=\s*([\d.]+)\s*kcal"
}

# =============================================================================
# Healthy Direction Mapping (True = Higher is Better, False = Lower is Better)
# =============================================================================

HEALTHY_DIRECTION = {
    "Weight_kg": False,
    "BMI": False,
    "Body Fat %": False,
    "Body Fat kg": False,
    "Visceral Fat Level": False,
    "Metabolic Age": False,
    "Impedance_Ohm": False,
    "Muscle Mass_kg": True,
    "Bone Mass_kg": True,
    "Body Water %": True,
    "Skeletal Muscle Mass %": True,
    "BMR_kcal": True,
    "Phase Angle_deg": True
}

# =============================================================================
# Data Extraction Functions
# =============================================================================

def extract_data(pdf_bytes: bytes) -> dict:
    """
    Extract body composition data from a PDF file.
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        Dictionary containing extracted values for all matched patterns
    """
    # Open PDF from bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Extract and concatenate text from all pages
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    
    # Normalize whitespace
    normalized_text = re.sub(r"\s+", " ", full_text)
    
    # Extract data using patterns
    extracted = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, normalized_text, re.IGNORECASE)
        if match:
            extracted[key] = match.group(1).strip()
        else:
            extracted[key] = None
    
    return extracted


def parse_datetime(date_str: str, time_str: str) -> datetime | None:
    """
    Parse date and time strings into a datetime object.
    
    Args:
        date_str: Date string in DD/MM/YYYY or MM/DD/YYYY format
        time_str: Time string with AM/PM
        
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str or not time_str:
        return None
    
    try:
        # Try DD/MM/YYYY format first
        combined = f"{date_str} {time_str}"
        for fmt in ["%d/%m/%Y %I:%M %p", "%m/%d/%Y %I:%M %p", 
                    "%d/%m/%Y %I:%M%p", "%m/%d/%Y %I:%M%p"]:
            try:
                return datetime.strptime(combined, fmt)
            except ValueError:
                continue
        return None
    except Exception:
        return None


# =============================================================================
# Trend Analysis Functions
# =============================================================================

def calculate_trend(values: list[float]) -> float | None:
    """
    Calculate the trend slope using Linear Regression.
    
    Args:
        values: List of numeric values over time
        
    Returns:
        Slope of the trend line, or None if insufficient data
    """
    if len(values) < 2:
        return None
    
    # Filter out NaN values
    valid_values = [(i, v) for i, v in enumerate(values) if pd.notna(v)]
    if len(valid_values) < 2:
        return None
    
    X = np.array([[idx] for idx, _ in valid_values])
    y = np.array([val for _, val in valid_values])
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model.coef_[0]


def is_healthy_trend(metric: str, slope: float) -> bool:
    """
    Determine if a trend is healthy based on the metric and slope direction.
    
    Args:
        metric: Name of the health metric
        slope: Trend slope (positive = increasing, negative = decreasing)
        
    Returns:
        True if the trend indicates improvement, False otherwise
    """
    if metric not in HEALTHY_DIRECTION:
        # Default: assume increasing is better
        return slope > 0
    
    higher_is_better = HEALTHY_DIRECTION[metric]
    
    if higher_is_better:
        # Increasing trend is healthy
        return slope > 0
    else:
        # Decreasing trend is healthy
        return slope < 0


def get_trend_color(metric: str, slope: float | None) -> str:
    """
    Get the color for a trend line based on health direction.
    
    Args:
        metric: Name of the health metric
        slope: Trend slope
        
    Returns:
        'green' for healthy trends, 'red' for unhealthy, 'gray' for neutral
    """
    if slope is None or slope == 0:
        return "#808080"  # Gray for no trend
    
    if is_healthy_trend(metric, slope):
        return "#2ECC71"  # Green for healthy
    else:
        return "#E74C3C"  # Red for unhealthy


# =============================================================================
# Visualization Functions
# =============================================================================

def create_dashboard(df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """
    Create a Plotly dashboard with trend charts for all metrics.
    
    Args:
        df: DataFrame with DateTime index and metric columns
        metrics: List of metric column names to visualize
        
    Returns:
        Plotly Figure with subplots
    """
    if not metrics:
        return go.Figure()
    
    # Calculate grid dimensions (3 columns for desktop, adapts for mobile)
    n_cols = 3
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    
    # Create subplot titles
    subplot_titles = [m.replace("_", " ") for m in metrics]
    
    # Fixed aspect ratio: 4:3 per chart
    # Base height per row for consistent proportions
    chart_height_per_row = 280
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,  # More spacing for readability
        horizontal_spacing=0.08
    )
    
    for idx, metric in enumerate(metrics):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        values = df[metric].tolist()
        dates = df["DateTime"].tolist()
        
        # Calculate trend
        slope = calculate_trend(values)
        color = get_trend_color(metric, slope)
        
        # Determine trend label
        if slope is not None:
            if is_healthy_trend(metric, slope):
                trend_label = "Improving ↑" if HEALTHY_DIRECTION.get(metric, True) else "Improving ↓"
            else:
                trend_label = "Declining ↓" if HEALTHY_DIRECTION.get(metric, True) else "Declining ↑"
        else:
            trend_label = "Stable"
        
        # Create trace with responsive marker sizes
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines+markers",
                name=metric,
                line=dict(color=color, width=2.5),
                marker=dict(
                    color=color, 
                    size=10,
                    line=dict(width=1, color='white')
                ),
                hovertemplate=(
                    f"<b>{metric.replace('_', ' ')}</b><br>"
                    "Value: %{y:.2f}<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    f"Trend: {trend_label}"
                    "<extra></extra>"
                ),
                showlegend=False
            ),
            row=row,
            col=col
        )
    
    # Responsive layout configuration
    fig.update_layout(
        # Fixed aspect ratio: height based on number of rows
        height=chart_height_per_row * n_rows,
        # Title configuration
        title=dict(
            text="📊 Body Composition Trends Dashboard",
            font=dict(size=22, weight="bold"),
            x=0.5,
            xanchor='center'
        ),
        # Clean template with grid
        template="plotly_white",
        # Responsive margins (percentage-based thinking)
        margin=dict(
            t=80,   # Top margin for title
            b=60,   # Bottom margin for x-axis labels
            l=50,   # Left margin for y-axis
            r=30,   # Right margin
            pad=4   # Padding between plot and axes
        ),
        # Enable auto-sizing for responsive behavior
        autosize=True,
        # Uniform font sizing
        font=dict(size=12),
        # Hover mode for better mobile experience
        hovermode='closest',
        # Responsive legend (if needed)
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update all x-axes for consistent, responsive formatting
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=10),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(128,128,128,0.4)',
        # Constrain to domain for fixed aspect ratio
        constrain='domain'
    )
    
    # Update all y-axes for consistent, responsive formatting
    fig.update_yaxes(
        tickfont=dict(size=10),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(128,128,128,0.4)',
        # Scale anchor for fixed aspect ratio
        scaleanchor=None,  # Allow independent scaling per chart
        automargin=True    # Auto-adjust margins for labels
    )
    
    # Make subplot titles more prominent
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13, weight='bold')
    
    return fig


# =============================================================================
# Main Application
# =============================================================================

def process_and_save_reports(uploaded_files, user_id):
    """Process uploaded PDF files and save to database."""
    new_reports_count = 0
    
    for uploaded_file in uploaded_files:
        try:
            pdf_bytes = uploaded_file.read()
            extracted = extract_data(pdf_bytes)
            
            # Parse DateTime for the report
            report_date = parse_datetime(extracted.get("Date"), extracted.get("Time"))
            
            # Save to database
            if db.save_report(user_id, uploaded_file.name, extracted, report_date):
                new_reports_count += 1
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    return new_reports_count


def show_dashboard(user):
    """Display the main dashboard for authenticated users."""
    
    # Header with user info and logout
    col_title, col_user = st.columns([4, 1])
    with col_title:
        st.title("📊 Body Composition Dashboard")
    with col_user:
        st.markdown(f"👤 **{user['name']}**")
        if st.button("Logout", use_container_width=True):
            logout()
    
    st.markdown("""
    Upload your body composition PDF reports to track health trends over time.
    Your data is **saved automatically** and available whenever you log in.
    """)
    
    st.divider()
    
    # Get user's saved report count
    report_count = db.get_report_count(user['id'])
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Reports")
        uploaded_files = st.file_uploader(
            "Add new PDF reports",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload body composition PDF reports to add to your history"
        )
        
        # Process uploaded files
        if uploaded_files:
            if st.button("💾 Save Reports", use_container_width=True):
                with st.spinner("Saving reports..."):
                    count = process_and_save_reports(uploaded_files, user['id'])
                    if count > 0:
                        st.success(f"✅ Saved {count} report(s)")
                        st.rerun()
        
        st.divider()
        
        # Report statistics
        st.header("📊 Your Data")
        st.metric("Saved Reports", report_count)
        
        st.divider()
        
        st.header("ℹ️ Legend")
        st.markdown("""
        - 🟢 **Green**: Healthy trend
        - 🔴 **Red**: Unhealthy trend  
        - ⚪ **Gray**: Stable
        """)
    
    # Main content - load user's historical data
    if report_count == 0:
        st.info("👈 Upload PDF reports using the sidebar to get started. Your data will be saved for future sessions!")
        
        with st.expander("📖 How to use this dashboard"):
            st.markdown("""
            ### Getting Started
            
            1. **Upload Reports**: Use the sidebar to upload your body composition PDF reports
            2. **Save Data**: Click "Save Reports" to store them in your account
            3. **Track Progress**: View your health trends over time
            4. **Add More**: Upload additional reports anytime to extend your history
            
            ### Your Data is Secure
            - Data is saved to your personal account
            - Only you can see your reports
            - Add more reports anytime to track your progress
            """)
        return
    
    # Load user's historical data
    with st.spinner("Loading your health data..."):
        df = db.get_user_reports_dataframe(user['id'])
        
        if df.empty:
            st.warning("No data found. Please upload some reports.")
            return
        
        # Process DateTime
        df["DateTime"] = df.apply(
            lambda row: parse_datetime(row.get("Date"), row.get("Time")),
            axis=1
        )
        
        # Identify numeric columns
        metadata_cols = ["Name", "Gender", "Date", "Time", "DateTime", "Source File", "Uploaded At"]
        numeric_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Convert numeric columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by DateTime
        if df["DateTime"].notna().any():
            df = df.sort_values("DateTime").reset_index(drop=True)
    
    # Summary metrics
    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if "Name" in df.columns and df["Name"].notna().any():
            st.metric("👤 Subject", df["Name"].dropna().iloc[0])
    
    with col2:
        if df["DateTime"].notna().any():
            earliest = df['DateTime'].min()
            latest = df['DateTime'].max()
            if earliest and latest:
                st.metric("📅 Date Range", f"{earliest:%Y-%m-%d} to {latest:%Y-%m-%d}")
    
    with col3:
        st.metric("📊 Total Reports", len(df))
    
    with col4:
        valid_metrics = sum(1 for col in numeric_cols if df[col].notna().any())
        st.metric("📈 Metrics Tracked", valid_metrics)
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "📋 Data Table", "🗑️ Manage Reports"])
    
    with tab1:
        # Dashboard charts
        valid_metrics = [col for col in numeric_cols if df[col].notna().any()]
        
        if len(df) < 2:
            st.warning("⚠️ Upload at least 2 reports to see trend analysis.")
        
        if valid_metrics:
            fig = create_dashboard(df, valid_metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trend summary
            if len(df) >= 2:
                st.divider()
                st.subheader("📊 Trend Summary")
                
                improving = []
                declining = []
                stable = []
                
                for metric in valid_metrics:
                    if metric in HEALTHY_DIRECTION:
                        values = df[metric].tolist()
                        slope = calculate_trend(values)
                        
                        if slope is None or abs(slope) < 0.001:
                            stable.append(metric)
                        elif is_healthy_trend(metric, slope):
                            improving.append(metric)
                        else:
                            declining.append(metric)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🟢 Improving")
                    if improving:
                        for m in improving:
                            st.markdown(f"- {m.replace('_', ' ')}")
                    else:
                        st.markdown("*None*")
                
                with col2:
                    st.markdown("### 🔴 Declining")
                    if declining:
                        for m in declining:
                            st.markdown(f"- {m.replace('_', ' ')}")
                    else:
                        st.markdown("*None*")
                
                with col3:
                    st.markdown("### ⚪ Stable")
                    if stable:
                        for m in stable:
                            st.markdown(f"- {m.replace('_', ' ')}")
                    else:
                        st.markdown("*None*")
    
    with tab2:
        # Data table view
        st.subheader("📋 All Your Reports")
        display_cols = ["DateTime", "Source File"] + [col for col in df.columns 
                                                       if col not in ["DateTime", "Source File", "Uploaded At"]]
        display_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)
    
    with tab3:
        # Manage reports - allow deletion
        st.subheader("🗑️ Manage Your Reports")
        st.warning("⚠️ Deleting reports cannot be undone!")
        
        reports = db.get_user_reports(user['id'])
        for report in reports:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.text(report['filename'])
            with col2:
                st.text(str(report['report_date'] or 'Unknown date'))
            with col3:
                if st.button("🗑️", key=f"del_{report['id']}"):
                    if db.delete_report(report['id'], user['id']):
                        st.success("Deleted!")
                        st.rerun()


def main():
    """Main application entry point."""
    
    # Check if user is authenticated
    if not is_authenticated():
        show_auth_page()
        return
    
    # Get current user
    user = get_current_user()
    if not user:
        logout()
        return
    
    # Show main dashboard
    show_dashboard(user)


if __name__ == "__main__":
    # Initialize database and create demo user on first run
    if not db.get_user_by_username('demo'):
        db.create_user('demo', 'demo@inbodyvis.com', 'demo123', 'Demo User')
    
    main()
