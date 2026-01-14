# 📊 Body Composition Dashboard

A Streamlit web application for parsing body composition PDF reports and visualizing health trends over time.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Multi-file PDF Upload** - Process multiple body composition reports at once
- **28 Health Parameters** - Extracts comprehensive health metrics including:
  - Body composition (Weight, BMI, Body Fat %)
  - Muscle metrics (Muscle Mass, Skeletal Muscle Mass)
  - Hydration (Body Water, ECW, ICW, Phase Angle)
  - Metabolism (BMR, Metabolic Age)
- **Trend Analysis** - LinearRegression-based trend detection
- **Interactive Dashboard** - Plotly charts with color-coded health indicators
  - 🟢 Green = Healthy/Improving trend
  - 🔴 Red = Unhealthy/Declining trend
- **Data Export** - View raw extracted data in expandable tables

## Installation

```bash
# Clone the repository
git clone https://github.com/IslamZaid/InBodyVis.git
cd InBodyVis

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Usage

1. Launch the application with `streamlit run app.py`
2. Upload your body composition PDF reports via the sidebar
3. View extracted data in the expandable table
4. Analyze trends in the interactive Plotly dashboard

## Requirements

- Python 3.9+
- PyMuPDF (fitz)
- Pandas
- Plotly
- Scikit-learn
- NumPy
- Streamlit

## License

MIT License - feel free to use and modify for your needs.
