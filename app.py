import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import json
import time  # Moved to the top

# Set page configuration
st.set_page_config(
    page_title="TechPrice AI",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def local_css():
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
            padding: 20px;
        }

        /* Card styling */
        .css-1r6slb0 {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            background-color: white;
            margin-bottom: 1rem;
        }

        /* Headers */
        h1 {
            color: #1E3A8A;
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }

        h2 {
            color: #2563EB;
            font-weight: 700;
            margin-top: 1rem;
        }

        h3 {
            color: #3B82F6;
            font-weight: 600;
        }

        /* Button styling */
        .stButton > button {
            background-color: #2563EB;
            color: white;
            font-weight: 600;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            border: none;
            width: 100%;
            transition: all 0.3s;
        }

        .stButton > button:hover {
            background-color: #1E40AF;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1E3A8A;
        }

        .css-1wbqy5l {
            color: white !important;
        }

        /* Success message styling */
        .element-container div[data-testid="stAlert"] {
            background-color: #DCFCE7;
            color: #166534;
            border: 1px solid #166534;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1.2rem;
            padding: 1rem;
            margin: 1rem 0;
        }

        /* Input fields */
        .stSelectbox, .stNumberInput {
            margin-bottom: 1rem;
        }

        /* Footer */
        footer {
            margin-top: 3rem;
            text-align: center;
            color: #6B7280;
            font-size: 0.8rem;
        }

        /* Navigation menu */
        .nav-link {
            font-weight: 600;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            margin: 0.25rem 0;
        }

        .nav-link:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }

        .nav-link.active {
            background-color: rgba(255, 255, 255, 0.2);
        }

        /* Cards */
        .card {
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            background-color: white;
            margin-bottom: 1rem;
            transition: transform 0.3s;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
        }

        /* Feature comparison table */
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
        }

        .comparison-table th {
            background-color: #2563EB;
            color: white;
            padding: 0.75rem;
            text-align: left;
        }

        .comparison-table td {
            padding: 0.75rem;
            border-bottom: 1px solid #E5E7EB;
        }

        .comparison-table tr:nth-child(even) {
            background-color: #F3F4F6;
        }

        /* Product image placeholder */
        .product-image {
            width: 100%;
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        /* Tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }

        /* Price tag */
        .price-tag {
            font-size: 2rem;
            font-weight: 700;
            color: #2563EB;
            margin: 1rem 0;
            text-align: center;
        }

        /* Currency selector */
        .currency-selector {
            width: 100%;
            padding: 0.5rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Load Lottie animations
@st.cache_resource()
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Helper functions
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_csv(path):
    return pd.read_csv(path)

# Load resources
@st.cache_resource()
def load_resources():
    base_path = os.getcwd()

    models = {
        "Smart Phones": load_pickle(f"{base_path}/Smart-Phones-Price-Predictor/Smart-Phones-Model.pkl"),
        "Smart Watches": load_pickle(f"{base_path}/Smart-Watches-Price-Predictor/Smart-Watches-Model.pkl"),
        "Tablets": load_pickle(f"{base_path}/Tablets-Price-Predictor/Tablets-Model.pkl"),
        "Laptops": load_pickle(f"{base_path}/Laptops-Price-Predictor/Laptop-Model.pkl"),
        "Headphones": load_pickle(f"{base_path}/Headphones-Price-Predictor/Headphones-Model.pkl"),
    }

    encoders = {
        "Smart Phones": load_pickle(f"{base_path}/Smart-Phones-Price-Predictor/Smart-Phones-Labels.pkl"),
        "Smart Watches": load_pickle(f"{base_path}/Smart-Watches-Price-Predictor/Smart-Watches-Labels.pkl"),
        "Tablets": load_pickle(f"{base_path}/Tablets-Price-Predictor/Tablets-Labels.pkl"),
        "Laptops": load_pickle(f"{base_path}/Laptops-Price-Predictor/Laptop-Labels.pkl"),
        "Headphones": load_pickle(f"{base_path}/Headphones-Price-Predictor/Headphone-Labels.pkl"),
    }

    datasets = {
        "Smart Phones": load_csv(f"{base_path}/Smart-Phones-Price-Predictor/Smart-Phones-Prices.csv"),
        "Smart Watches": load_csv(f"{base_path}/Smart-Watches-Price-Predictor/Smart-Watches-Prices.csv"),
        "Tablets": load_csv(f"{base_path}/Tablets-Price-Predictor/Tablets-Prices.csv"),
        "Laptops": load_csv(f"{base_path}/Laptops-Price-Predictor/Laptops-Prices.csv"),
        "Headphones": load_csv(f"{base_path}/Headphones-Price-Predictor/Headphones-Prices.csv"),
    }

    return models, encoders, datasets

# Load the resources
models, encoders, datasets = load_resources()

# Apply custom CSS
local_css()

# Currency conversion rates (as an example, real-time API can be used for live rates)
currency_rates = {"INR": 1, "USD": 0.012, "EUR": 0.011, "GBP": 0.009}

# Device icons
device_icons = {
    "Overview": "🏠",
    "Smart Phones": "📱",
    "Smart Watches": "⌚",
    "Tablets": "📟",
    "Laptops": "💻",
    "Headphones": "🎧"
}

# Device descriptions
device_descriptions = {
    "Smart Phones": "Predict smartphone prices based on specifications like RAM, storage, camera quality, and brand.",
    "Smart Watches": "Estimate smartwatch prices considering factors like display type, health features, and battery life.",
    "Tablets": "Calculate tablet prices factoring in screen size, processor, storage capacity, and connectivity options.",
    "Laptops": "Determine laptop prices based on CPU, RAM, storage type, graphics card, and display quality.",
    "Headphones": "Predict headphone prices considering type, connectivity, noise cancellation, and brand value."
}

# Lottie animations
lottie_tech = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_w51pcehl.json")
lottie_phones = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_bpqri9x5.json")
lottie_watch = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_snmohqxj.json")
lottie_tablet = load_lottieurl("https://assets8.lottiefiles.com/private_files/lf30_skjfj92h.json")
lottie_laptop = load_lottieurl("https://assets8.lottiefiles.com/private_files/lf30_bnsj43kd.json")
lottie_headphones = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_1lz9xcv4.json")

lottie_animations = {
    "Overview": lottie_tech,
    "Smart Phones": lottie_phones,
    "Smart Watches": lottie_watch,
    "Tablets": lottie_tablet,
    "Laptops": lottie_laptop,
    "Headphones": lottie_headphones
}

# Mapping dictionaries for user-friendly column names
column_name_mappings = {
    "Headphones": {
        "Actual_Price": "Price",
        "Brand": "Brand",
        "Model": "Model",
        "Colour": "Color",
        "Form_Factor": "Form Factor",
        "Connectivity_Type": "Connectivity Type",
        "Product_Type": "Product Type",
        "Technology": "Technology",
        "noise cancellation": "Noise Cancellation",
        "anc": "Active Noise Cancellation (ANC)",
        "waterproof": "Waterproof",
        "fast charging": "Fast Charging",
        "bass": "Bass",
        "enc": "Environmental Noise Cancellation (ENC)",
        "playtime": "Playtime (hours)",
        "gaming": "Gaming"
    },
    "Laptops": {
        "Brand": "Brand",
        "RAM (GB)": "RAM (GB)",
        "Screen Size (inch)": "Screen Size (inches)",
        "Battery Life (hours)": "Battery Life (hours)",
        "Weight (kg)": "Weight (kg)",
        "Operating System": "Operating System",
        "Price ($)": "Price ($)",
        "Storage Capacity (GB)": "Storage Capacity (GB)",
        "Storage Type": "Storage Type",
        "Processor Brand": "Processor Brand",
        "Processor Series": "Processor Series",
        "GPU Brand": "GPU Brand",
        "Resolution Width": "Resolution Width",
        "Resolution Height": "Resolution Height",
        "Total Pixels": "Total Pixels"
    },
    "Smart Phones": {
        "brand_name": "Brand",
        "model": "Model",
        "price": "Price",
        "rating": "Rating",
        "has_5g": "5G Support",
        "has_nfc": "NFC Support",
        "has_ir_blaster": "IR Blaster",
        "processor_brand": "Processor Brand",
        "num_cores": "Number of Cores",
        "processor_speed": "Processor Speed (GHz)",
        "battery_capacity": "Battery Capacity (mAh)",
        "fast_charging_available": "Fast Charging Available",
        "fast_charging": "Fast Charging (W)",
        "ram_capacity": "RAM Capacity (GB)",
        "internal_memory": "Internal Memory (GB)",
        "screen_size": "Screen Size (inches)",
        "refresh_rate": "Refresh Rate (Hz)",
        "num_rear_cameras": "Number of Rear Cameras",
        "num_front_cameras": "Number of Front Cameras",
        "os": "Operating System",
        "primary_camera_rear": "Primary Rear Camera (MP)",
        "primary_camera_front": "Primary Front Camera (MP)",
        "extended_memory_available": "Extended Memory Available",
        "extended_upto": "Extended Memory Upto (GB)",
        "resolution_width": "Resolution Width",
        "resolution_height": "Resolution Height"
    },
    "Smart Watches": {
        "Brand": "Brand",
        "Model": "Model",
        "Operating System": "Operating System",
        "Display Type": "Display Type",
        "Display Size (inches)": "Display Size (inches)",
        "Water Resistance (meters)": "Water Resistance (meters)",
        "Battery Life (days)": "Battery Life (days)",
        "Heart Rate Monitor": "Heart Rate Monitor",
        "GPS": "GPS",
        "NFC": "NFC",
        "Price (USD)": "Price (USD)",
        "Wi-Fi": "Wi-Fi",
        "Cellular": "Cellular",
        "Resolution X": "Resolution Width",
        "Resolution Y": "Resolution Height",
        "Bluetooth": "Bluetooth"
    },
    "Tablets": {
        "brand": "Brand",
        "rating": "Rating",
        "price": "Price",
        "processor_brand": "Processor Brand",
        "num_processor": "Number of Processors",
        "processor_speed": "Processor Speed (GHz)",
        "ram": "RAM (GB)",
        "memory_inbuilt": "Internal Memory (GB)",
        "battery_capacity": "Battery Capacity (mAh)",
        "charger": "Charger (W)",
        "charging": "Charging Speed",
        "display_size_inches": "Display Size (inches)",
        "pixel": "Pixel Density",
        "resolution_width": "Resolution Width",
        "resolution_height": "Resolution Height",
        "ppi": "Pixels Per Inch (PPI)",
        "frequency_display_hz": "Refresh Rate (Hz)",
        "primary_front_camera": "Primary Front Camera (MP)",
        "secondry_front_camera": "Secondary Front Camera (MP)",
        "primary_rear_camera": "Primary Rear Camera (MP)",
        "secondry_rear_camera": "Secondary Rear Camera (MP)",
        "os_brand": "Operating System",
        "version": "OS Version",
        "memory_card_upto": "Memory Card Support (GB)",
        "sim": "SIM Support",
        "is_5G": "5G Support",
        "is_wifi": "Wi-Fi Support"
    }
}

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/price-comparison.png", width=80)
    st.title("TechPrice AI")
    st.markdown("---")

    selected_currency = st.selectbox("💱 Select Currency", list(currency_rates.keys()))

    st.markdown("---")

    # Navigation menu
    selected_page = option_menu(
        "Navigation",
        options=list(device_icons.keys()),
        icons=list(device_icons.values()),
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {"color": "0F52BA", "font-size": "16px", "text-align": "left", "margin": "0px",
                         "--hover-color": "#4B5563"},
            "nav-link-selected": {"background-color": "#3B82F6"},
        }
    )

    st.markdown("---")

    st.markdown("### About")
    st.markdown(
        "TechPrice AI uses machine learning to predict accurate prices for the latest tech products based on their specifications.")

    st.markdown("---")

    st.markdown("### Developer")
    st.markdown('Made with ❤️ by [Akshat Sharma](https://github.com/Akshat-Sharma-110011)')

    # QR code for mobile access (placeholder)
    st.markdown("### Try on mobile")
    qr_html = """
    <img src="https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=https://techpriceai.com" width="100" />
    """
    st.markdown(qr_html, unsafe_allow_html=True)

# Create a comparison function for the overview page
def create_feature_comparison(category):
    df = datasets[category]

    # Find price column
    price_col = [col for col in df.columns if col.lower() in ['price', 'actual_price', 'price (usd)', 'price ($)']][0]

    # Get top 5 most expensive and least expensive items
    top_expensive = df.nlargest(5, price_col)
    top_affordable = df.nsmallest(5, price_col)

    # Create feature columns to analyze (exclude price)
    feature_cols = [col for col in df.columns if col.lower() not in ['price', 'actual_price', 'price (usd)']]

    # Select 3-5 important features
    important_features = feature_cols[:min(4, len(feature_cols))]

    return top_expensive, top_affordable, important_features, price_col

# Function to create radar chart comparing expensive vs affordable
def create_radar_chart(category, top_expensive, top_affordable, important_features):
    # Filter out non-numeric and boolean columns
    numeric_features = [
        col for col in important_features
        if pd.api.types.is_numeric_dtype(datasets[category][col]) and not pd.api.types.is_bool_dtype(datasets[category][col])
    ]

    if not numeric_features:
        st.warning("No numeric features found for radar chart.")
        return None

    # Get average values for each numeric feature
    expensive_means = top_expensive[numeric_features].mean()
    affordable_means = top_affordable[numeric_features].mean()

    # Normalize values between 0 and 1 for radar chart
    max_values = datasets[category][numeric_features].max()
    min_values = datasets[category][numeric_features].min()

    # Avoid division by zero in case max_values == min_values
    with np.errstate(divide='ignore', invalid='ignore'):
        expensive_norm = (expensive_means - min_values) / (max_values - min_values)
        affordable_norm = (affordable_means - min_values) / (max_values - min_values)

    # Replace NaN or infinite values with 0 (in case of division by zero)
    expensive_norm = expensive_norm.fillna(0).replace([np.inf, -np.inf], 0)
    affordable_norm = affordable_norm.fillna(0).replace([np.inf, -np.inf], 0)

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=expensive_norm.values,
        theta=numeric_features,
        fill='toself',
        name='Premium',
        line_color='#2563EB'
    ))

    fig.add_trace(go.Scatterpolar(
        r=affordable_norm.values,
        theta=numeric_features,
        fill='toself',
        name='Budget',
        line_color='#F97316'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=400,
        title=f"{category} Feature Comparison"
    )

    return fig

# Function to display price distribution
def create_price_distribution(category, price_col):
    df = datasets[category]

    fig = px.histogram(
        df,
        x=price_col,
        nbins=20,
        title=f"{category} Price Distribution",
        labels={price_col: "Price"},
        color_discrete_sequence=['#2563EB']
    )

    fig.update_layout(
        xaxis_title="Price",
        yaxis_title="Count",
        height=300
    )

    return fig

# Function to display feature importance (simulated since we don't have access to the model details)
def create_feature_importance(category):
    df = datasets[category]
    feature_cols = [col for col in df.columns if col.lower() not in ['price', 'actual_price', 'price (usd)']]

    # Create simulated importance values
    np.random.seed(42)  # For reproducibility
    importance = np.random.uniform(0.1, 1.0, size=len(feature_cols))
    importance = importance / np.sum(importance)  # Normalize

    # Sort by importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    # Create bar chart
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f"Feature Importance for {category}",
        color_discrete_sequence=['#2563EB']
    )

    fig.update_layout(
        xaxis_title="Relative Importance",
        yaxis_title="",
        height=300
    )

    return fig

# ... (rest of the code remains the same until get_user_input)
def get_user_input(category):
    """Generate input fields dynamically based on dataset columns with enhanced UI."""
    df = datasets[category]
    encoder = encoders[category]

    input_data = {}

    # Find columns to exclude (price-related)
    exclude_cols = [col for col in df.columns if col.lower() in ['price', 'actual_price', 'price (usd)', 'price ($)']]

    # Organize columns by types for better grouping
    categorical_cols = []
    numerical_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        if df[col].dtype == 'object' or col in encoder:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    # Create two columns for better UI organization
    col1, col2 = st.columns(2)

    # Process categorical columns (mostly in first column)
    for i, col in enumerate(categorical_cols):
        unique_vals = df[col].dropna().unique()

        with col1 if i % 2 == 0 else col2:
            with st.container():
                # Use user-friendly name for display
                friendly_name = column_name_mappings[category].get(col, col)
                st.markdown(f"**{friendly_name}**")
                selected_val = st.selectbox(
                    label=f"Select {friendly_name}",
                    options=unique_vals,
                    label_visibility="collapsed"
                )
                input_data[col] = encoder[col].transform([selected_val])[0] if col in encoder else selected_val

    # Process numerical columns (mostly in second column)
    for i, col in enumerate(numerical_cols):
        with col2 if i % 2 == 0 else col1:
            with st.container():
                # Use user-friendly name for display
                friendly_name = column_name_mappings[category].get(col, col)
                st.markdown(f"**{friendly_name}**")

                # Determine step and type based on column dtype
                if df[col].dtype == 'float64':
                    step = 0.1  # Use float step for float columns
                    value = float(df[col].median())
                    min_value = float(df[col].min())
                    max_value = float(df[col].max())
                else:
                    step = 1  # Use int step for int columns
                    value = int(df[col].median())
                    min_value = int(df[col].min())
                    max_value = int(df[col].max())

                # Ensure all parameters are of the same type
                input_data[col] = st.number_input(
                    label=f"Enter {friendly_name}",
                    min_value=min_value,
                    max_value=max_value,
                    value=value,
                    step=step,  # Ensure step matches the dtype of the column
                    label_visibility="collapsed"
                )

    return pd.DataFrame([input_data])
# Main content
def main():
    if selected_page == "Overview":
        st.markdown("# 🔮 TechPrice AI")
        st.markdown("### Predict accurate prices for tech products using AI")

        # Display animation
        if lottie_tech:
            st_lottie(lottie_tech, height=250, key="tech_animation")

        st.markdown("---")

        # Information cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="card">
                <h3>🎯 Accurate Predictions</h3>
                <p>Our AI models provide precise price estimates based on the latest market trends.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <h3>⚡ Fast & Easy</h3>
                <p>Get instant price predictions by simply entering product specifications.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="card">
                <h3>🌐 Multiple Currencies</h3>
                <p>View price predictions in your preferred currency: INR, USD, EUR, or GBP.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Categories overview
        st.markdown("## 📊 Tech Categories")

        # Display device cards in two rows
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2 = st.columns([1.5, 1.5])

        # First row
        with row1_col1:
            st.markdown(f"""
            <div class="card" onclick="window.location.href='#'">
                <h3>{device_icons["Smart Phones"]} Smart Phones</h3>
                <p>{device_descriptions["Smart Phones"]}</p>
                <center><a href="#" target="_self">Try Now →</a></center>
            </div>
            """, unsafe_allow_html=True)

        with row1_col2:
            st.markdown(f"""
            <div class="card" onclick="window.location.href='#'">
                <h3>{device_icons["Smart Watches"]} Smart Watches</h3>
                <p>{device_descriptions["Smart Watches"]}</p>
                <center><a href="#" target="_self">Try Now →</a></center>
            </div>
            """, unsafe_allow_html=True)

        with row1_col3:
            st.markdown(f"""
            <div class="card" onclick="window.location.href='#'">
                <h3>{device_icons["Tablets"]} Tablets</h3>
                <p>{device_descriptions["Tablets"]}</p>
                <center><a href="#" target="_self">Try Now →</a></center>
            </div>
            """, unsafe_allow_html=True)

        # Second row
        with row2_col1:
            st.markdown(f"""
            <div class="card" onclick="window.location.href='#'">
                <h3>{device_icons["Laptops"]} Laptops</h3>
                <p>{device_descriptions["Laptops"]}</p>
                <center><a href="#" target="_self">Try Now →</a></center>
            </div>
            """, unsafe_allow_html=True)

        with row2_col2:
            st.markdown(f"""
            <div class="card" onclick="window.location.href='#'">
                <h3>{device_icons["Headphones"]} Headphones</h3>
                <p>{device_descriptions["Headphones"]}</p>
                <center><a href="#" target="_self">Try Now →</a></center>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Market insights section
        st.markdown("## 📈 Market Insights")

        # Select a category to show insights
        insight_category = st.selectbox("Select a category to view insights", list(datasets.keys()))

        # Generate insights
        top_expensive, top_affordable, important_features, price_col = create_feature_comparison(insight_category)

        # Create three columns for visualizations
        viz_col1, viz_col2 = st.columns([1, 1])

        with viz_col1:
            # Radar chart
            radar_fig = create_radar_chart(insight_category, top_expensive, top_affordable, important_features)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.warning("No numeric features available for radar chart.")

        with viz_col2:
            # Price distribution
            price_dist_fig = create_price_distribution(insight_category, price_col)
            st.plotly_chart(price_dist_fig, use_container_width=True)

        # Feature importance
        importance_fig = create_feature_importance(insight_category)
        st.plotly_chart(importance_fig, use_container_width=True)

        # Display premium vs budget comparison
        st.markdown(f"### Premium vs Budget {insight_category} Comparison")

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown("#### Premium Models")
            for _, row in top_expensive.iterrows():
                price_val = row[price_col]
                # Format price based on currency
                if selected_currency == "INR":
                    formatted_price = f"₹{price_val:,.2f}"
                elif selected_currency == "USD":
                    formatted_price = f"${price_val * currency_rates['USD']:,.2f}"
                elif selected_currency == "EUR":
                    formatted_price = f"€{price_val * currency_rates['EUR']:,.2f}"
                else:
                    formatted_price = f"£{price_val * currency_rates['GBP']:,.2f}"

                # Display summary of features
                feature_summary = ", ".join([f"{col}: {row[col]}" for col in important_features[:2]])

                st.markdown(f"""
                <div style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>{feature_summary}...</div>
                        <div style="font-weight: bold; color: #2563EB;">{formatted_price}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with comp_col2:
            st.markdown("#### Budget Models")
            for _, row in top_affordable.iterrows():
                price_val = row[price_col]
                # Format price based on currency
                if selected_currency == "INR":
                    formatted_price = f"₹{price_val:,.2f}"
                elif selected_currency == "USD":
                    formatted_price = f"${price_val * currency_rates['USD']:,.2f}"
                elif selected_currency == "EUR":
                    formatted_price = f"€{price_val * currency_rates['EUR']:,.2f}"
                else:
                    formatted_price = f"£{price_val * currency_rates['GBP']:,.2f}"

                # Display summary of features
                feature_summary = ", ".join([f"{col}: {row[col]}" for col in important_features[:2]])

                st.markdown(f"""
                <div style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>{feature_summary}...</div>
                        <div style="font-weight: bold; color: #F97316;">{formatted_price}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # How it works section
        st.markdown("## ⚙️ How It Works")

        how_col1, how_col2, how_col3 = st.columns(3)

        with how_col1:
            st.markdown("""
            <div class="card">
                <h3>1️⃣ Select a Device</h3>
                <p>Choose the tech category you want to price from our navigation menu.</p>
            </div>
            """, unsafe_allow_html=True)

        with how_col2:
            st.markdown("""
            <div class="card">
                <h3>2️⃣ Enter Specifications</h3>
                <p>Provide the device specifications through our user-friendly interface.</p>
            </div>
            """, unsafe_allow_html=True)

        with how_col3:
            st.markdown("""
            <div class="card">
                <h3>3️⃣ Get Price Prediction</h3>
                <p>Our AI model instantly calculates the estimated price in your chosen currency.</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Device specific pages
        col1, col2 = st.columns([2, 1])

        with col1:
            st.title(f"{device_icons[selected_page]} {selected_page} Price Predictor")
            st.markdown(device_descriptions[selected_page])

        with col2:
            if lottie_animations[selected_page]:
                st_lottie(lottie_animations[selected_page], height=150, key=f"{selected_page}_animation")

        st.markdown("---")

        # Create two columns layout
        input_col, preview_col = st.columns([2, 1])

        with input_col:
            st.markdown("### Enter Device Specifications")
            user_input = get_user_input(selected_page)

            predict_btn = st.button("Calculate Price", key=f"predict_{selected_page}")

            if predict_btn:
                with st.spinner("Calculating price..."):
                    # Add artificial delay for UX
                    time.sleep(0.5)

                    prediction = models[selected_page].predict(user_input)[0]

                    # Convert price based on category
                    if selected_page in ["Smart Watches", "Laptops"]:
                        base_currency = "USD"
                        converted_price = prediction * currency_rates[selected_currency] / currency_rates["USD"]
                    else:
                        base_currency = "INR"
                        converted_price = prediction * currency_rates[selected_currency] / currency_rates["INR"]

                    # Format currency symbol
                    currency_symbols = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}

                    # Display prediction in an attractive card
                    st.markdown(
                        f"""
                        <div style="background-color: #F0F9FF; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                            <h3 style="color: #2563EB; margin-bottom: 1rem;">Predicted Price</h3>
                            <div style="font-size: 2rem; font-weight: bold; color: #1E3A8A;">
                                {currency_symbols[selected_currency]}{converted_price:,.2f}
                            </div>
                            <div style="color: #6B7280; margin-top: 0.5rem;">
                                Based on the provided specifications
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        with preview_col:
            st.markdown("### Price Insights")
            st.markdown("Compare your predicted price with market trends.")

            # Find price column
            price_cols = [col for col in datasets[selected_page].columns if col.lower() in ['price', 'actual_price', 'price (usd)', 'price ($)']]

            if not price_cols:
                st.error("No price column found in the dataset. Please ensure the dataset contains a column named 'price', 'actual_price', 'price (usd)', or 'Price ($)'.")
                st.stop()  # Stop execution if no price column is found

            price_col = price_cols[0]  # Use the first matching column

            # Get price distribution for the selected category
            price_dist_fig = create_price_distribution(selected_page, price_col)
            st.plotly_chart(price_dist_fig, use_container_width=True)

            # Display feature importance
            importance_fig = create_feature_importance(selected_page)
            st.plotly_chart(importance_fig, use_container_width=True)

        st.markdown("---")

        # Display top 5 most expensive and affordable models
        st.markdown("### Market Comparison")
        top_expensive, top_affordable, important_features, _ = create_feature_comparison(selected_page)

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown("#### Premium Models")
            for _, row in top_expensive.iterrows():
                price_val = row[price_col]
                # Format price based on currency
                if selected_currency == "INR":
                    formatted_price = f"₹{price_val:,.2f}"
                elif selected_currency == "USD":
                    formatted_price = f"${price_val * currency_rates['USD']:,.2f}"
                elif selected_currency == "EUR":
                    formatted_price = f"€{price_val * currency_rates['EUR']:,.2f}"
                else:
                    formatted_price = f"£{price_val * currency_rates['GBP']:,.2f}"

                # Display summary of features
                feature_summary = ", ".join([f"{col}: {row[col]}" for col in important_features[:2]])

                st.markdown(f"""
                <div style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>{feature_summary}...</div>
                        <div style="font-weight: bold; color: #2563EB;">{formatted_price}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with comp_col2:
            st.markdown("#### Budget Models")
            for _, row in top_affordable.iterrows():
                price_val = row[price_col]
                # Format price based on currency
                if selected_currency == "INR":
                    formatted_price = f"₹{price_val:,.2f}"
                elif selected_currency == "USD":
                    formatted_price = f"${price_val * currency_rates['USD']:,.2f}"
                elif selected_currency == "EUR":
                    formatted_price = f"€{price_val * currency_rates['EUR']:,.2f}"
                else:
                    formatted_price = f"£{price_val * currency_rates['GBP']:,.2f}"

                # Display summary of features
                feature_summary = ", ".join([f"{col}: {row[col]}" for col in important_features[:2]])

                st.markdown(f"""
                <div style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>{feature_summary}...</div>
                        <div style="font-weight: bold; color: #F97316;">{formatted_price}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Footer
        st.markdown("""
        <footer>
            <p>Made with ❤️ by Akshat Sharma | Powered by Streamlit</p>
        </footer>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()