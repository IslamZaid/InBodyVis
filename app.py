"""
Body Composition PDF Parser & Dashboard
A Streamlit web application for parsing body composition PDF reports
and visualizing health trends over time.
"""

import re
import io
import yaml
from datetime import datetime

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import streamlit as st
import streamlit_authenticator as stauth

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
# Authentication Configuration
# =============================================================================

# Pre-hashed passwords (generated with bcrypt)
# demo123 and admin123 hashed with bcrypt
DEMO_PASSWORD_HASH = "$2b$12$kpZsxIljZph8RmyfvKUi..bWuq8uJ2XBpQTNyyvDdnS/6bExSfKrS"
ADMIN_PASSWORD_HASH = "$2b$12$lZJFDRHvyjgMYxfwLMcCre1z.qID7Avdp6iZGKq3Ps5rR50PGOF6m"

def get_authenticator():
    """Initialize and return the authenticator object."""
    # Default credentials with pre-hashed passwords
    credentials = {
        'usernames': {
            'demo': {
                'email': 'demo@inbodyvis.com',
                'name': 'Demo User',
                'password': DEMO_PASSWORD_HASH
            },
            'admin': {
                'email': 'admin@inbodyvis.com', 
                'name': 'Administrator',
                'password': ADMIN_PASSWORD_HASH
            }
        }
    }
    
    # Try to load from secrets if available
    try:
        if "credentials" in st.secrets:
            credentials = dict(st.secrets["credentials"])
    except:
        pass
    
    authenticator = stauth.Authenticate(
        credentials,
        'inbodyvis_cookie',           # Cookie name
        'inbodyvis_signature_key',    # Signature key
        cookie_expiry_days=30
    )
    
    return authenticator

def show_login_page(authenticator):
    """Display the login page."""
    st.title("📊 Body Composition Dashboard")
    st.markdown("### Welcome! Please sign in to continue.")
    
    # Login form
    name, authentication_status, username = authenticator.login('Login', 'main')
    
    if authentication_status == False:
        st.error('❌ Username or password is incorrect')
    elif authentication_status == None:
        st.info('👆 Enter your username and password above')
        
        # Show demo credentials
        with st.expander("🔑 Demo Credentials"):
            st.markdown("""
            **Demo Account:**
            - Username: `demo`
            - Password: `demo123`
            
            **Admin Account:**
            - Username: `admin`  
            - Password: `admin123`
            """)
    
    return authentication_status, name


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

def main():
    """Main Streamlit application."""
    
    # Initialize authenticator
    authenticator = get_authenticator()
    
    # streamlit-authenticator 0.4.x API - renders login widget
    authenticator.login(location='main')
    
    # Get authentication status from session state
    authentication_status = st.session_state.get('authentication_status')
    name = st.session_state.get('name')
    
    if authentication_status == False:
        st.error('❌ Username or password is incorrect')
        st.stop()
    elif authentication_status == None:
        st.title("📊 Body Composition Dashboard")
        st.markdown("### Welcome! Please sign in to continue.")
        st.info('👆 Enter your username and password above')
        
        # Show demo credentials
        with st.expander("🔑 Demo Credentials"):
            st.markdown("""
            **Demo Account:**
            - Username: `demo`
            - Password: `demo123`
            
            **Admin Account:**
            - Username: `admin`  
            - Password: `admin123`
            """)
        st.stop()
    
    # User is authenticated - show main app
    # Header with user greeting and logout
    col_title, col_user = st.columns([4, 1])
    with col_title:
        st.title("📊 Body Composition Dashboard")
    with col_user:
        st.markdown(f"👤 **{name}**")
        authenticator.logout(location='main')
    
    st.markdown("""
    Upload your body composition PDF reports to visualize health trends over time.
    The dashboard analyzes **28 health parameters** and shows whether each metric
    is **improving** (green) or **declining** (red).
    """)
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Reports")
        uploaded_files = st.file_uploader(
            "Select PDF reports",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more body composition PDF reports"
        )
        
        st.divider()
        
        st.header("ℹ️ Legend")
        st.markdown("""
        - 🟢 **Green**: Healthy trend (improving)
        - 🔴 **Red**: Unhealthy trend (declining)
        - ⚪ **Gray**: Stable / No significant trend
        """)
        
        st.divider()
        
        st.header("📋 Tracked Metrics")
        st.markdown("""
        **Composition:**
        - Weight, BMI, Body Fat %, Visceral Fat
        
        **Muscle & Bone:**
        - Muscle Mass, Skeletal Muscle, Bone Mass
        
        **Hydration:**
        - Body Water, ECW, ICW, Phase Angle
        
        **Metabolism:**
        - BMR, Metabolic Age, Impedance
        """)
    
    # Main content
    if not uploaded_files:
        st.info("👈 Upload PDF reports using the sidebar to get started.")
        
        # Show sample layout
        with st.expander("📖 How to use this dashboard"):
            st.markdown("""
            ### Getting Started
            
            1. **Upload Reports**: Use the file uploader in the sidebar to select one or more
               body composition PDF reports.
            
            2. **View Data**: Once uploaded, you'll see all extracted data in a table below.
            
            3. **Analyze Trends**: The dashboard will automatically create trend charts for
               all detected metrics, color-coded by health direction:
               - **Green lines**: The trend indicates improvement
               - **Red lines**: The trend indicates decline
            
            4. **Hover for Details**: Hover over any data point to see the exact value,
               date, and trend status.
            
            ### Supported Metrics
            
            This dashboard extracts and analyzes **28 different health parameters** including:
            - Body composition (weight, BMI, body fat percentage)
            - Muscle metrics (muscle mass, skeletal muscle mass, sarcopenic index)
            - Hydration (body water, ECW, ICW, phase angle)
            - Metabolism (BMR, metabolic age)
            - And more...
            """)
        return
    
    # Process uploaded files
    with st.spinner("📑 Processing PDF reports..."):
        all_data = []
        
        for uploaded_file in uploaded_files:
            try:
                pdf_bytes = uploaded_file.read()
                extracted = extract_data(pdf_bytes)
                extracted["Source File"] = uploaded_file.name
                all_data.append(extracted)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        if not all_data:
            st.error("No data could be extracted from the uploaded files.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Merge Date and Time into DateTime
        df["DateTime"] = df.apply(
            lambda row: parse_datetime(row.get("Date"), row.get("Time")),
            axis=1
        )
        
        # Identify numeric columns (exclude metadata)
        metadata_cols = ["Name", "Gender", "Date", "Time", "DateTime", "Source File"]
        numeric_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Convert numeric columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by DateTime
        if df["DateTime"].notna().any():
            df = df.sort_values("DateTime").reset_index(drop=True)
    
    st.success(f"✅ Successfully processed {len(all_data)} report(s)")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if "Name" in df.columns and df["Name"].notna().any():
            st.metric("👤 Subject", df["Name"].dropna().iloc[0])
    
    with col2:
        if "DateTime" in df.columns and df["DateTime"].notna().any():
            date_range = f"{df['DateTime'].min():%Y-%m-%d} to {df['DateTime'].max():%Y-%m-%d}"
            st.metric("📅 Date Range", date_range)
    
    with col3:
        st.metric("📊 Reports", len(all_data))
    
    with col4:
        valid_metrics = sum(1 for col in numeric_cols if df[col].notna().any())
        st.metric("📈 Metrics Found", valid_metrics)
    
    st.divider()
    
    # Data table
    with st.expander("📋 View Raw Data", expanded=False):
        # Reorder columns for better display
        display_cols = ["DateTime", "Source File"] + [col for col in df.columns 
                                                       if col not in ["DateTime", "Source File"]]
        display_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)
    
    st.divider()
    
    # Dashboard
    st.header("📈 Trend Analysis")
    
    # Get metrics that have at least some valid data
    valid_metrics = [col for col in numeric_cols if df[col].notna().any()]
    
    if len(df) < 2:
        st.warning("⚠️ Upload at least 2 reports to see trend analysis.")
        
        # Still show available data as single points
        if valid_metrics:
            fig = create_dashboard(df, valid_metrics)
            st.plotly_chart(fig, use_container_width=True)
    else:
        if valid_metrics:
            fig = create_dashboard(df, valid_metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trend summary
            st.divider()
            st.header("📊 Trend Summary")
            
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
        else:
            st.warning("⚠️ No numeric data found in the uploaded reports.")


if __name__ == "__main__":
    main()
