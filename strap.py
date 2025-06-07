import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from math import floor, radians, tan, atan, degrees
from tensorflow.keras import layers
import keras.saving
from PIL import Image
import os
import math
import sys
import base64
from io import BytesIO
import joblib
from sklearn.preprocessing import PowerTransformer

# =================================================================
# Geogrid Model (Existing)
# =================================================================

# Model class
@keras.saving.register_keras_serializable(package='GeogridModels', name='GeogridPINN')
class GeogridPINN(tf.keras.Model):
    def __init__(self, hidden_layers=3, units_per_layer=1024, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer
        self.dense_layers = [layers.Dense(units_per_layer, activation='tanh', name=f'dense_{i}')
                             for i in range(hidden_layers)]
        self.u_output = layers.Dense(1, activation=None, name='output')

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.u_output(x)

# Load geogrid model
def load_geogrid_model():
    try:
        model_path = 'best_model9178.keras'
        custom_objects = {
            'GeogridModels>GeogridPINN': GeogridPINN,
            'GeogridPINN' : GeogridPINN
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Geogrid model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading geogrid model: {e}")
        raise

# =================================================================
# Geostrap Model (New)
# =================================================================

def load_geostrap_model():
    """Load the saved XGBoost model and preprocessing objects"""
    try:
        model = joblib.load('model_artifacts/best_xgboost_model.joblib')
        scaler = joblib.load('model_artifacts/power_transformer.joblib')
        feature_names = joblib.load('model_artifacts/feature_names.joblib')
        print("Geostrap model loaded successfully!")
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading geostrap model: {e}")
        raise

def predict_geostrap(model, scaler, feature_names, input_data):
    """
    Make predictions using the geostrap model
    Args:
        input_data: Pandas DataFrame or numpy array with the same features as training data
    Returns:
        predictions: Numpy array with model predictions
    """
    # Convert to numpy array if DataFrame
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
    
    # Ensure correct shape
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    
    # Preprocess the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    predictions = model.predict(input_scaled)
    
    return predictions[0]

# =================================================================
# Common Functions
# =================================================================

# Mappings
geogrid_classification_map = {
    'CH': 1, 'CL': 2, 'MH': 3, 'ML': 4, 'SW-SM': 5, 'SM': 6,
    'SP-SM': 7, 'SP': 8, 'SW': 9, 'GP': 10, 'GW': 11, 'GW-GM': 12
}

geostrap_classification_map = {
    'ML': 1, 'SM': 2, 'SP': 3, 'SP-SM': 4, 'SW': 5, 'GW': 6, 'GW-GM': 7
}

geogrid_type_map = {
    'NATURAL': 1, 'HDPE BIAXIAL': 2, 'PP BIAXIAL': 3, 'PP UNIAXIAL': 4,
    'HDPE UNIAXIAL': 5, 'PET BIAXIAL': 6, 'PET UNIAXIAL': 7
}

# Helper functions
def get_image_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    return None

def draw_geogrid(length, md, cmd, scale_factor=5):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    num_md = int(length / md) + 1
    num_cmd = 10

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, length * scale_factor)
    ax.set_ylim(0, num_cmd * cmd * scale_factor)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw CMD lines (horizontal)
    for i in range(num_cmd):
        y = i * cmd * scale_factor
        ax.plot([0, length * scale_factor], [y, y], color='blue', linewidth=2)

    # Draw MD lines (vertical)
    for j in range(num_md):
        x = j * md * scale_factor
        ax.plot([x, x], [0, (num_cmd - 1) * cmd * scale_factor], color='red', linewidth=1)

    # Add labels
    ax.text(length * scale_factor / 2, -10, f"Length: {length}mm", ha='center')
    ax.text(-20, num_cmd * cmd * scale_factor / 2, f"CMD: {cmd}mm", va='center', rotation=90)

    return fig

def calculate_geogrid_u(inputs):
    try:
        # Original prediction
        u_pred = geogrid_model.predict(inputs)[0][0]

        # Extract values from inputs
        phi = inputs[0][0]
        cohesion = inputs[0][1]
        normal_stress = inputs[0][2]
        length_mm = inputs[0][3]

        # Calculate P (kN/m)
        length_m = length_mm / 1000  # Convert to meters
        phi_rad = radians(phi)
        P = 2 * u_pred * length_m * (normal_stress * tan(phi_rad) + cohesion)

        return u_pred, P
    except Exception as e:
        st.error(f"Geogrid calculation error: {e}")
        return None, None

def calculate_geostrap_u(inputs):
    try:
        # Make prediction using XGBoost model
        u_pred = predict_geostrap(geostrap_model, geostrap_scaler, geostrap_feature_names, inputs)
        
        # Extract values from inputs
        phi = inputs[0][0]
        cohesion = inputs[0][1]
        normal_stress = inputs[0][2]
        length_mm = inputs[0][3]
        strap_width = inputs[0][13]  # New field
        num_straps = inputs[0][14]   # New field
        tensile_strength = inputs[0][15]  # New field

        # Calculate P (kN/m)
        length_m = length_mm / 1000  # Convert to meters
        phi_rad = radians(phi)
        P = 2 * u_pred * length_m * (normal_stress * tan(phi_rad) + cohesion)

        return u_pred, P
    except Exception as e:
        st.error(f"Geostrap calculation error: {e}")
        return None, None

# =================================================================
# Main App
# =================================================================

def main():
    st.set_page_config(layout="wide", page_title="μ* Prediction Tool")

    # Load models (only once)
    if 'geogrid_model' not in st.session_state:
        st.session_state.geogrid_model = load_geogrid_model()
    if 'geostrap_model' not in st.session_state:
        model, scaler, feature_names = load_geostrap_model()
        st.session_state.geostrap_model = model
        st.session_state.geostrap_scaler = scaler
        st.session_state.geostrap_feature_names = feature_names

    global geogrid_model, geostrap_model, geostrap_scaler, geostrap_feature_names
    geogrid_model = st.session_state.geogrid_model
    geostrap_model = st.session_state.geostrap_model
    geostrap_scaler = st.session_state.geostrap_scaler
    geostrap_feature_names = st.session_state.geostrap_feature_names

    # Initialize session state variables
    if 'scale_factor' not in st.session_state:
        st.session_state.scale_factor = 5
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Geogrid"

    # CSS for styling
    st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #2196F3;
        color: white;
    }
    .st-emotion-cache-1v0mbdj {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .header-text {
        text-align: center;
        font-weight: bold;
        font-style: italic;
        margin-bottom: 20px;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        border: 1px solid #dee2e6;
        color: #212529;
    }
    .number-input input {
        background-color: #ffffff !important;
    }
    .tab-button {
        font-size: 18px !important;
        padding: 10px 20px !important;
        margin: 5px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with logos
    col1, col2, col3 = st.columns([1, 2, 1])

    # Load and display logos if available
    csir_logo = get_image_base64("logo.png")
    vnit_logo = get_image_base64("VNIT.png")

    if csir_logo and col1:
        col1.markdown(
            f'<img src="data:image/png;base64,{csir_logo}" style="height: 100px;">',
            unsafe_allow_html=True
        )

    col2.markdown(
        '''
        <p style="font-size: 24px; font-weight: bold; text-align: center;">
            Pullout Interaction Coefficient Predictor using Machine Learning<br>
            Developed in collaboration with CSIR-CRRI and VNIT
        </p>
        ''',
        unsafe_allow_html=True
    )

    if vnit_logo and col3:
        col3.markdown(
            f'<img src="data:image/png;base64,{vnit_logo}" style="height: 150px;">',
            unsafe_allow_html=True
        )

    # Create main tabs with large buttons
    st.markdown("## Select Product Type")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Geogrid", key="geogrid_tab", help="Click to analyze geogrid parameters", 
                    use_container_width=True, type="primary" if st.session_state.current_tab == "Geogrid" else "secondary"):
            st.session_state.current_tab = "Geogrid"
    with col2:
        if st.button("Geostrap", key="geostrap_tab", help="Click to analyze geostrap parameters", 
                    use_container_width=True, type="primary" if st.session_state.current_tab == "Geostrap" else "secondary"):
            st.session_state.current_tab = "Geostrap"

    # Geogrid Tab
    if st.session_state.current_tab == "Geogrid":
        # Create tabs for Geogrid
        tab1, tab2 = st.tabs(["Soil Parameters", "Geogrid Parameters"])

        with tab1:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.header("Soil Parameters")

                # Soil input fields - all with empty default values
                normal_stress = st.number_input("Normal Stress (kPa):", value=None, placeholder="Enter value", step=0.1,
                                                format="%.2f", key="geo_normal_stress")
                phi = st.number_input("Φ' (degrees):", value=None, placeholder="Enter value", step=0.1, format="%.1f", key="geo_phi")
                cohesion = st.number_input("Cohesion c' (kPa):", value=None, placeholder="Enter value", step=0.1,
                                           format="%.2f", key="geo_cohesion")
                unit_weight = st.number_input("Unit Weight (kN/m³):", value=None, placeholder="Enter value", step=0.1,
                                              format="%.2f", key="geo_unit_weight")
                water_content = st.number_input("Water Content (%):", value=None, placeholder="Enter value", step=0.1,
                                                format="%.1f", key="geo_water_content")
                d50 = st.number_input("D50 (mm):", value=None, placeholder="Enter value", step=0.1, format="%.2f", key="geo_d50")
                soil_classification = st.selectbox("Soil Classification:", options=list(geogrid_classification_map.keys()), key="geo_soil_class")

            with col2:
                # Display soil image if available
                soil_img = get_image_base64("PulloutboxDiagram.jpg")
                if soil_img:
                    st.markdown(
                        f'''
                        <div style="display: flex; justify-content: center;">
                            <img src="data:image/jpg;base64,{soil_img}" style="max-width: 100%; width: 1200px; height: auto;">
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    "<p style='text-align: center; font-weight: bold; font-size: 20px;'>Soil-Geogrid Interaction</p>",
                    unsafe_allow_html=True
                )

        with tab2:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.header("Geogrid Parameters")

                # Geogrid input fields - all with empty default values
                length = st.number_input("Length (mm):", value=None, placeholder="Enter value", step=1.0, format="%.1f",
                                         key="geo_length", on_change=update_bearing_members)
                md_aperture = st.number_input("MD Aperture (mm):", value=None, placeholder="Enter value", step=0.1,
                                              format="%.1f",
                                              key="geo_md_aperture", on_change=update_bearing_members)
                cmd_aperture = st.number_input("CMD Aperture (mm):", value=None, placeholder="Enter value", step=0.1,
                                               format="%.1f",
                                               key="geo_cmd_aperture", on_change=update_bearing_members)
                bearing_members = st.number_input("Bearing Members:", value=None, placeholder="Enter value", step=1,
                                                  key="geo_bearing_members")
                tensile_strength = st.number_input("Tensile Strength (kN/m):", value=None, placeholder="Enter value",
                                                   step=0.1, format="%.1f", key="geo_tensile_strength")
                geogrid_type = st.selectbox("Geogrid Type:", options=list(geogrid_type_map.keys()), key="geo_geogrid_type")

            with col2:
                # Display geogrid image if available - moved to the right of input fields
                geogrid_img = get_image_base64("Geogrid.png")
                if geogrid_img:
                    st.markdown(
                        f'<img src="data:image/png;base64,{geogrid_img}" style="width: 100%;">',
                        unsafe_allow_html=True
                    )
                st.markdown("<p style='text-align: center; font-weight: bold;'>Geogrid Structure Reference</p>",
                            unsafe_allow_html=True)

                # Visualization controls
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Zoom In", key="geo_zoom_in"):
                        st.session_state.scale_factor *= 1.2
                with col2:
                    if st.button("Zoom Out", key="geo_zoom_out"):
                        st.session_state.scale_factor *= 0.8

                # Draw geogrid visualization
                if length and md_aperture and cmd_aperture:
                    fig = draw_geogrid(float(length), float(md_aperture), float(cmd_aperture),
                                       st.session_state.scale_factor)
                    st.pyplot(fig)

                # Run button
                if st.button("Run ▶", type="primary", key="geo_run"):
                    # Validate all required fields are filled
                    required_fields = [
                        normal_stress, phi, cohesion, length,
                        d50, unit_weight, water_content,
                        md_aperture, cmd_aperture, tensile_strength
                    ]

                    if None in required_fields:
                        st.error("Please fill in all required fields")
                    else:
                        inputs = np.array([
                            phi,
                            cohesion,
                            normal_stress,
                            length,
                            geogrid_classification_map[soil_classification],
                            d50,
                            unit_weight,
                            water_content,
                            geogrid_type_map[geogrid_type],
                            bearing_members if bearing_members else floor(float(length) / float(md_aperture)),
                            md_aperture,
                            cmd_aperture,
                            tensile_strength
                        ], dtype=np.float32).reshape(1, -1)

                        u_pred, P = calculate_geogrid_u(inputs)

                        if u_pred is not None:
                            st.session_state.result = f"μ* = {u_pred:.4f}   |   P = {P:.2f} kN/m "

    # Geostrap Tab
    elif st.session_state.current_tab == "Geostrap":
        # Create tabs for Geostrap
        tab1, tab2 = st.tabs(["Soil Parameters", "Geostrap Parameters"])

        with tab1:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.header("Soil Parameters")

                # Soil input fields - all with empty default values
                normal_stress = st.number_input("Normal Stress (kPa):", value=None, placeholder="Enter value", step=0.1,
                                                format="%.2f", key="strap_normal_stress")
                phi = st.number_input("Φ' (degrees):", value=None, placeholder="Enter value", step=0.1, format="%.1f", key="strap_phi")
                cohesion = st.number_input("Cohesion c' (kPa):", value=None, placeholder="Enter value", step=0.1,
                                           format="%.2f", key="strap_cohesion")
                unit_weight = st.number_input("Unit Weight (kN/m³):", value=None, placeholder="Enter value", step=0.1,
                                              format="%.2f", key="strap_unit_weight")
                water_content = st.number_input("Water Content (%):", value=None, placeholder="Enter value", step=0.1,
                                                format="%.1f", key="strap_water_content")
                d50 = st.number_input("D50 (mm):", value=None, placeholder="Enter value", step=0.1, format="%.2f", key="strap_d50")
                soil_classification = st.selectbox("Soil Classification:", 
                                                 options=list(geostrap_classification_map.keys()), key="strap_soil_class")

            with col2:
                # Display soil image if available
                soil_img = get_image_base64("PulloutboxDiagram.jpg")
                if soil_img:
                    st.markdown(
                        f'''
                        <div style="display: flex; justify-content: center;">
                            <img src="data:image/jpg;base64,{soil_img}" style="max-width: 100%; width: 1200px; height: auto;">
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    "<p style='text-align: center; font-weight: bold; font-size: 20px;'>Soil-Geostrap Interaction</p>",
                    unsafe_allow_html=True
                )

        with tab2:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.header("Geostrap Parameters")

                # Geostrap input fields - all with empty default values
                length = st.number_input("Length (mm):", value=None, placeholder="Enter value", step=1.0, format="%.1f",
                                         key="strap_length")
                
                # Additional Geostrap-specific fields
                strap_width = st.number_input("Width of Straps (mm):", value=None, placeholder="Enter value", 
                                             step=1.0, format="%.1f", key="strap_width")
                num_straps = st.number_input("Number of Straps:", value=None, placeholder="Enter value", 
                                            step=1, key="num_straps")
                strap_tensile_strength = st.number_input("Tensile Strength (kN):", value=None, placeholder="Enter value",
                                                         step=0.1, format="%.1f", key="strap_tensile_strength")

            with col2:
                # Display geostrap image if available
                geostrap_img = get_image_base64("Geostrap.png")
                if geostrap_img:
                    st.markdown(
                        f'<img src="data:image/png;base64,{geostrap_img}" style="width: 100%;">',
                        unsafe_allow_html=True
                    )
                st.markdown("<p style='text-align: center; font-weight: bold;'>Geostrap Structure Reference</p>",
                            unsafe_allow_html=True)

                # Run button
                if st.button("Run ▶", type="primary", key="strap_run"):
                    # Validate all required fields are filled
                    required_fields = [
                        normal_stress, phi, cohesion, length,
                        d50, unit_weight, water_content,
                        strap_width, num_straps, strap_tensile_strength
                    ]

                    if None in required_fields:
                        st.error("Please fill in all required fields")
                    else:
                        inputs = np.array([
                            phi,
                            cohesion,
                            normal_stress,
                            length,
                            geostrap_classification_map[soil_classification],
                            d50,
                            unit_weight,
                            water_content,
                            0,  # Placeholder (not used in geostrap model)
                            0,  # Placeholder
                            0,  # Placeholder
                            0,  # Placeholder
                            0,  # Placeholder
                            strap_width,
                            num_straps,
                            strap_tensile_strength
                        ], dtype=np.float32).reshape(1, -1)

                        u_pred, P = calculate_geostrap_u(inputs)

                        if u_pred is not None:
                            st.session_state.result = f"μ* = {u_pred:.4f}   |   P = {P:.2f} kN/m "

    # Control buttons and results (common for both tabs)
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button("Clear All", key="clear_all"):
            clear_all()

    with col2:
        uploaded_file = st.file_uploader("📂 Upload Excel for Prediction", type=["xlsx", "xls"], key="file_uploader")
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file

        if st.button("Import Excel", key="import_excel"):
            if 'uploaded_file' in st.session_state:
                import_excel(st.session_state.uploaded_file)
            else:
                st.warning("Please upload an Excel file first.")

        download_template()

    with col3:
        if st.button("Export to Excel", key="export_excel"):
            export_excel()

    with col4:
        if st.button("Data Format Info", key="data_info"):
            show_disclaimer()

    # Display results with improved styling
    if hasattr(st.session_state, 'result'):
        st.markdown(f"""
        <div class="result-box">
            <h3 style='color: #212529;'>Results:</h3>
            <p style='color: #212529; font-size: 18px;'>{st.session_state.result}</p>
        </div>
        """, unsafe_allow_html=True)

# =================================================================
# Helper Functions
# =================================================================

def update_bearing_members():
    try:
        length = st.session_state.geo_length
        md = st.session_state.geo_md_aperture
        if md and length:
            bearing_members = floor(float(length) / float(md))
            st.session_state.geo_bearing_members = bearing_members
    except:
        pass

def clear_all():
    st.session_state.clear()
    st.session_state.scale_factor = 5
    st.session_state.data = None
    st.session_state.current_tab = "Geogrid"
    st.rerun()

def show_disclaimer():
    disclaimer = """DATA FORMAT REQUIREMENTS:

For Soil Classification, use these exact values:
CH: 1, CL: 2, MH: 3, ML: 4, SW-SM: 5, SM: 6
SP-SM: 7, SP: 8, SW: 9, GP: 10, GW: 11, GW-GM: 12

For Geogrid Type, use these exact values:
NATURAL: 1, HDPE BIAXIAL: 2, PP BIAXIAL: 3
PP UNIAXIAL: 4, HDPE UNIAXIAL: 5
PET BIAXIAL: 6, PET UNIAXIAL: 7"""
    st.warning(disclaimer)

def import_excel(uploaded_file):
    if uploaded_file is None:
        st.warning("📂 Please upload an Excel file to begin predictions.")
        return

    try:
        df = pd.read_excel(uploaded_file)

        if st.session_state.current_tab == "Geogrid":
            # Validate columns for geogrid
            required_cols = ['phi', 'cohesion', 'normal_stress', 'length', 'soil_classification',
                             'd50', 'unit_weight', 'water_content', 'geogrid_type', 'bearing_members',
                             'md_aperture', 'cmd_aperture', 'tensile_strength']

            if not all(col in df.columns for col in required_cols):
                st.error("Missing required columns in the Excel file.")
                return

            # Map categorical values
            df['soil_classification'] = df['soil_classification'].map(geogrid_classification_map)
            df['geogrid_type'] = df['geogrid_type'].map(geogrid_type_map)

            # Predict with progress bar
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx in range(len(df)):
                row = df.iloc[idx]
                inputs = np.array([
                    row['phi'], row['cohesion'], row['normal_stress'], row['length'],
                    row['soil_classification'], row['d50'], row['unit_weight'],
                    row['water_content'], row['geogrid_type'], row['bearing_members'],
                    row['md_aperture'], row['cmd_aperture'], row['tensile_strength']
                ], dtype=np.float32).reshape(1, -1)

                u_pred, P = calculate_geogrid_u(inputs)
                results.append([u_pred, P])

                # Update progress
                progress = (idx + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Processing row {idx + 1} of {len(df)}")

            # Assign predictions
            df[['predicted_mu', 'P']] = results

            # Convert back to labels
            inv_classification_map = {v: k for k, v in geogrid_classification_map.items()}
            inv_geogrid_type_map = {v: k for k, v in geogrid_type_map.items()}
            df['soil_classification'] = df['soil_classification'].map(inv_classification_map)
            df['geogrid_type'] = df['geogrid_type'].map(inv_geogrid_type_map)

        else:  # Geostrap
            # Validate columns for geostrap
            required_cols = ['phi', 'cohesion', 'normal_stress', 'length', 'soil_classification',
                             'd50', 'unit_weight', 'water_content', 'strap_width', 
                             'num_straps', 'tensile_strength']

            if not all(col in df.columns for col in required_cols):
                st.error("Missing required columns in the Excel file.")
                return

            # Map categorical values
            df['soil_classification'] = df['soil_classification'].map(geostrap_classification_map)

            # Predict with progress bar
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx in range(len(df)):
                row = df.iloc[idx]
                inputs = np.array([
                    row['phi'], row['cohesion'], row['normal_stress'], row['length'],
                    row['soil_classification'], row['d50'], row['unit_weight'],
                    row['water_content'], 0, 0, 0, 0, 0,  # Placeholders
                    row['strap_width'], row['num_straps'], row['tensile_strength']
                ], dtype=np.float32).reshape(1, -1)

                u_pred, P = calculate_geostrap_u(inputs)
                results.append([u_pred, P])

                # Update progress
                progress = (idx + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Processing row {idx + 1} of {len(df)}")

            # Assign predictions
            df[['predicted_mu', 'P']] = results

            # Convert back to labels
            inv_classification_map = {v: k for k, v in geostrap_classification_map.items()}
            df['soil_classification'] = df['soil_classification'].map(inv_classification_map)

        st.session_state.data = df
        st.success(f"✅ Predictions completed for {len(df)} rows.")
        st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Failed to import and predict:\n{e}")

def export_excel():
    if st.session_state.data is None:
        st.error("No data to export. Please import and predict first.")
        return

    try:
        # Create a BytesIO buffer
        output = BytesIO()

        # Write dataframe to Excel
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            st.session_state.data.to_excel(writer, index=False, sheet_name='Results')

        # Create download button
        st.download_button(
            label="Download Excel File",
            data=output.getvalue(),
            file_name="geosynthetic_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Export failed: {e}")

def download_template():
    # Create sample data based on current tab
    if st.session_state.current_tab == "Geogrid":
        template = pd.DataFrame([
            {
                'phi': 30.0,
                'cohesion': 10.0,
                'normal_stress': 50.0,
                'length': 400.0,
                'soil_classification': 'CL',
                'd50': 0.5,
                'unit_weight': 18.5,
                'water_content': 12.0,
                'geogrid_type': 'HDPE BIAXIAL',
                'bearing_members': 8,
                'md_aperture': 50.0,
                'cmd_aperture': 40.0,
                'tensile_strength': 20.0
            }
        ])
        file_name = "geogrid_prediction_template.xlsx"
    else:
        template = pd.DataFrame([
            {
                'phi': 30.0,
                'cohesion': 10.0,
                'normal_stress': 50.0,
                'length': 400.0,
                'soil_classification': 'SM',
                'd50': 0.5,
                'unit_weight': 18.5,
                'water_content': 12.0,
                'strap_width': 50.0,
                'num_straps': 5,
                'tensile_strength': 15.0
            }
        ])
        file_name = "geostrap_prediction_template.xlsx"

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        template.to_excel(writer, index=False, sheet_name='Template')

    st.download_button(
        label="📥 Download Prediction Template (with samples)",
        data=output.getvalue(),
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
