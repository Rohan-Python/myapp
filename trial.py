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

# =================================================================
# Model Definitions (unchanged)
# =================================================================

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

def load_geostrap_model():
    try:
        import xgboost as xgb
        model = joblib.load('model_artifacts/best_xgboost_model.joblib')
        feature_names = joblib.load('model_artifacts/feature_names.joblib')
        if len(feature_names) != 11:
            st.error(f"Model expects 11 features, got {len(feature_names)}")
            return None, None
        return model, feature_names
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

# =================================================================
# Mappings and Helper Functions
# =================================================================

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
    for i in range(num_cmd):
        y = i * cmd * scale_factor
        ax.plot([0, length * scale_factor], [y, y], color='blue', linewidth=2)
    for j in range(num_md):
        x = j * md * scale_factor
        ax.plot([x, x], [0, (num_cmd - 1) * cmd * scale_factor], color='red', linewidth=1)
    ax.text(length * scale_factor / 2, -10, f"Length: {length}mm", ha='center')
    ax.text(-20, num_cmd * cmd * scale_factor / 2, f"CMD: {cmd}mm", va='center', rotation=90)
    return fig

def calculate_geogrid_u(inputs):
    try:
        u_pred = geogrid_model.predict(inputs)[0][0]
        phi = inputs[0][0]
        cohesion = inputs[0][1]
        normal_stress = inputs[0][2]
        length_mm = inputs[0][3]
        length_m = length_mm / 1000
        phi_rad = radians(phi)
        P = 2 * u_pred * length_m * (normal_stress * tan(phi_rad) + cohesion)
        return u_pred, P
    except Exception as e:
        st.error(f"Geogrid calculation error: {e}")
        return None, None

def calculate_geostrap_u(inputs):
    try:
        u_pred = predict_geostrap(geostrap_model, inputs)
        if u_pred is None:
            return None, None
        phi = inputs[0][0]
        cohesion = inputs[0][1]
        normal_stress = inputs[0][2]
        length_mm = inputs[0][3]
        length_m = length_mm / 1000
        phi_rad = radians(phi)
        interaction_term = normal_stress * tan(phi_rad) + cohesion
        P = 2 * u_pred * length_m * interaction_term
        return u_pred, P
    except Exception as e:
        st.error(f"Geostrap calculation failed: {str(e)}")
        return None, None

# =================================================================
# Main App (Fixed Version)
# =================================================================

def main():
    st.set_page_config(layout="wide", page_title="μ* Prediction Tool")

    # Initialize all session state variables
    if 'geogrid_model' not in st.session_state:
        st.session_state.geogrid_model = load_geogrid_model()
    if 'geostrap_model' not in st.session_state:
        model, feature_names = load_geostrap_model()
        st.session_state.geostrap_model = model
        st.session_state.geostrap_feature_names = feature_names
    if 'scale_factor' not in st.session_state:
        st.session_state.scale_factor = 5
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Geogrid"
    if 'active_subtab' not in st.session_state:
        st.session_state.active_subtab = "Soil Parameters"

    # Initialize all input variables
    geogrid_inputs = [
        'geo_normal_stress', 'geo_phi', 'geo_cohesion', 'geo_length',
        'geo_soil_class', 'geo_d50', 'geo_unit_weight', 'geo_water_content',
        'geo_md_aperture', 'geo_cmd_aperture', 'geo_tensile_strength',
        'geo_geogrid_type', 'geo_bearing_members'
    ]
    
    geostrap_inputs = [
        'strap_normal_stress', 'strap_phi', 'strap_cohesion', 'strap_length',
        'strap_soil_class', 'strap_d50', 'strap_unit_weight', 'strap_water_content',
        'strap_width', 'num_straps', 'strap_tensile_strength'
    ]
    
    for var in geogrid_inputs + geostrap_inputs:
        if var not in st.session_state:
            st.session_state[var] = None

    # CSS and Header (unchanged)
    st.markdown("""
    <style>
    .stButton>button { background-color: #4CAF50; color: white; font-weight: bold; }
    .stDownloadButton>button { background-color: #2196F3; color: white; }
    .result-box { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px; border: 1px solid #dee2e6; color: #212529; }
    </style>
    """, unsafe_allow_html=True)

    # Header with logos
    col1, col2, col3 = st.columns([1, 2, 1])
    csir_logo = get_image_base64("logo.png")
    vnit_logo = get_image_base64("VNIT.png")
    if csir_logo and col1:
        col1.markdown(f'<img src="data:image/png;base64,{csir_logo}" style="height: 100px;">', unsafe_allow_html=True)
    col2.markdown('''
        <p style="font-size: 24px; font-weight: bold; text-align: center;">
            Pullout Interaction Coefficient Predictor using Physics-Informed Neural Network (PINN)<br>
            Developed in collaboration with CSIR-CRRI and VNIT
        </p>''', unsafe_allow_html=True)
    if vnit_logo and col3:
        col3.markdown(f'<img src="data:image/png;base64,{vnit_logo}" style="height: 150px;">', unsafe_allow_html=True)

    # Main tabs
    st.markdown("## Select Product Type")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Geogrid", key="geogrid_tab", use_container_width=True, 
                    type="primary" if st.session_state.current_tab == "Geogrid" else "secondary"):
            st.session_state.current_tab = "Geogrid"
            st.session_state.active_subtab = "Soil Parameters"
            st.rerun()
    with col2:
        if st.button("Geostrap", key="geostrap_tab", use_container_width=True,
                    type="primary" if st.session_state.current_tab == "Geostrap" else "secondary"):
            st.session_state.current_tab = "Geostrap"
            st.session_state.active_subtab = "Soil Parameters"
            st.rerun()

    # Geogrid Tab
    if st.session_state.current_tab == "Geogrid":
        if st.session_state.active_subtab == "Soil Parameters":
            with st.container():
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.header("Soil Parameters")
                    st.session_state.geo_normal_stress = st.number_input("Normal Stress (kPa):", 
                        value=st.session_state.geo_normal_stress, step=0.1, format="%.2f", key="geo_normal_stress_input")
                    st.session_state.geo_phi = st.number_input("Φ' (degrees):", 
                        value=st.session_state.geo_phi, step=0.1, format="%.1f", key="geo_phi_input")
                    st.session_state.geo_cohesion = st.number_input("Cohesion c' (kPa):", 
                        value=st.session_state.geo_cohesion, step=0.1, format="%.2f", key="geo_cohesion_input")
                    st.session_state.geo_water_content = st.number_input("Water Content (%):", 
                        value=st.session_state.geo_water_content, step=0.1, format="%.1f", key="geo_water_content_input")
                    st.session_state.geo_soil_class = st.selectbox("Soil Classification:", 
                        options=list(geogrid_classification_map.keys()), key="geo_soil_class_input")
                    st.session_state.geo_d50 = st.number_input("D50 (mm):", 
                        value=st.session_state.geo_d50, step=0.1, format="%.2f", key="geo_d50_input")
                    st.session_state.geo_unit_weight = st.number_input("Unit Weight (kN/m³):", 
                        value=st.session_state.geo_unit_weight, step=0.1, format="%.2f", key="geo_unit_weight_input")
                
                if st.button("Next →", key="geo_next_soil_to_grid"):
                    st.session_state.active_subtab = "Geogrid Parameters"
                    st.rerun()

        elif st.session_state.active_subtab == "Geogrid Parameters":
            with st.container():
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.header("Geogrid Parameters")
                    st.session_state.geo_length = st.number_input("Length (mm):", 
                        value=st.session_state.geo_length, step=1.0, format="%.1f", key="geo_length_input",
                        on_change=update_bearing_members)
                    st.session_state.geo_md_aperture = st.number_input("MD Aperture (mm):", 
                        value=st.session_state.geo_md_aperture, step=0.1, format="%.1f", key="geo_md_aperture_input",
                        on_change=update_bearing_members)
                    st.session_state.geo_cmd_aperture = st.number_input("CMD Aperture (mm):", 
                        value=st.session_state.geo_cmd_aperture, step=0.1, format="%.1f", key="geo_cmd_aperture_input",
                        on_change=update_bearing_members)
                    st.session_state.geo_bearing_members = st.number_input("Bearing Members:", 
                        value=st.session_state.geo_bearing_members, step=1, key="geo_bearing_members_input")
                    st.session_state.geo_tensile_strength = st.number_input("Tensile Strength (kN/m):", 
                        value=st.session_state.geo_tensile_strength, step=0.1, format="%.1f", key="geo_tensile_strength_input")
                    st.session_state.geo_geogrid_type = st.selectbox("Geogrid Type:", 
                        options=list(geogrid_type_map.keys()), key="geo_geogrid_type_input")
                
                if st.button("← Previous", key="geo_prev_grid_to_soil"):
                    st.session_state.active_subtab = "Soil Parameters"
                    st.rerun()

                if st.button("Run ▶", type="primary", key="geo_run"):
                    required_fields = {
                        'normal_stress': st.session_state.geo_normal_stress,
                        'phi': st.session_state.geo_phi,
                        'cohesion': st.session_state.geo_cohesion,
                        'length': st.session_state.geo_length,
                        'd50': st.session_state.geo_d50,
                        'unit_weight': st.session_state.geo_unit_weight,
                        'water_content': st.session_state.geo_water_content,
                        'md_aperture': st.session_state.geo_md_aperture,
                        'cmd_aperture': st.session_state.geo_cmd_aperture,
                        'tensile_strength': st.session_state.geo_tensile_strength
                    }

                    if None in required_fields.values():
                        missing = [k for k, v in required_fields.items() if v is None]
                        st.error(f"Missing required fields: {', '.join(missing)}")
                    else:
                        inputs = np.array([
                            required_fields['phi'],
                            required_fields['cohesion'],
                            required_fields['normal_stress'],
                            required_fields['length'],
                            geogrid_classification_map[st.session_state.geo_soil_class],
                            required_fields['d50'],
                            required_fields['unit_weight'],
                            required_fields['water_content'],
                            geogrid_type_map[st.session_state.geo_geogrid_type],
                            st.session_state.geo_bearing_members if st.session_state.geo_bearing_members else floor(float(required_fields['length']) / float(required_fields['md_aperture'])),
                            required_fields['md_aperture'],
                            required_fields['cmd_aperture'],
                            required_fields['tensile_strength'],
                        ], dtype=np.float32).reshape(1, -1)
                    
                        u_pred, P = calculate_geogrid_u(inputs)
                        if u_pred is not None:
                            st.session_state.result = f"μ* = {u_pred:.4f}   |   P = {P:.2f} kN/m"
                            st.rerun()

    # Geostrap Tab
    elif st.session_state.current_tab == "Geostrap":
        if st.session_state.active_subtab == "Soil Parameters":
            with st.container():
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.header("Soil Parameters")
                    st.session_state.strap_normal_stress = st.number_input("Normal Stress (kPa):", 
                        value=st.session_state.strap_normal_stress, step=0.1, format="%.2f", key="strap_normal_stress_input")
                    st.session_state.strap_phi = st.number_input("Φ' (degrees):", 
                        value=st.session_state.strap_phi, step=0.1, format="%.1f", key="strap_phi_input")
                    st.session_state.strap_cohesion = st.number_input("Cohesion c' (kPa):", 
                        value=st.session_state.strap_cohesion, step=0.1, format="%.2f", key="strap_cohesion_input")
                    st.session_state.strap_water_content = st.number_input("Water Content (%):", 
                        value=st.session_state.strap_water_content, step=0.1, format="%.1f", key="strap_water_content_input")
                    st.session_state.strap_soil_class = st.selectbox("Soil Classification:", 
                        options=list(geostrap_classification_map.keys()), key="strap_soil_class_input")
                    st.session_state.strap_d50 = st.number_input("D50 (mm):", 
                        value=st.session_state.strap_d50, step=0.1, format="%.2f", key="strap_d50_input")
                    st.session_state.strap_unit_weight = st.number_input("Unit Weight (kN/m³):", 
                        value=st.session_state.strap_unit_weight, step=0.1, format="%.2f", key="strap_unit_weight_input")
                
                if st.button("Next →", key="strap_next_soil_to_strap"):
                    st.session_state.active_subtab = "Geostrap Parameters"
                    st.rerun()

        elif st.session_state.active_subtab == "Geostrap Parameters":
            with st.container():
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.header("Geostrap Parameters")
                    st.session_state.strap_length = st.number_input("Length (mm):", 
                        value=st.session_state.strap_length, step=1.0, format="%.1f", key="strap_length_input")
                    st.session_state.strap_width = st.number_input("Width of Straps (mm):", 
                        value=st.session_state.strap_width, step=1.0, format="%.1f", key="strap_width_input")
                    st.session_state.num_straps = st.number_input("Number of Straps:", 
                        value=st.session_state.num_straps, step=1, key="num_straps_input")
                    st.session_state.strap_tensile_strength = st.number_input("Tensile Strength (kN):", 
                        value=st.session_state.strap_tensile_strength, step=0.1, format="%.1f", key="strap_tensile_strength_input")
                
                if st.button("← Previous", key="strap_prev_strap_to_soil"):
                    st.session_state.active_subtab = "Soil Parameters"
                    st.rerun()

                if st.button("Run ▶", type="primary", key="strap_run"):
                    required_fields = {
                        'normal_stress': st.session_state.strap_normal_stress,
                        'phi': st.session_state.strap_phi,
                        'cohesion': st.session_state.strap_cohesion,
                        'length': st.session_state.strap_length,
                        'soil_class': st.session_state.strap_soil_class,
                        'd50': st.session_state.strap_d50,
                        'unit_weight': st.session_state.strap_unit_weight,
                        'water_content': st.session_state.strap_water_content,
                        'width': st.session_state.strap_width,
                        'num_straps': st.session_state.num_straps,
                        'tensile': st.session_state.strap_tensile_strength
                    }

                    if None in required_fields.values():
                        missing = [k for k, v in required_fields.items() if v is None]
                        st.error(f"Missing required fields: {', '.join(missing)}")
                    else:
                        inputs = np.array([
                            required_fields['phi'],
                            required_fields['cohesion'],
                            required_fields['normal_stress'],
                            required_fields['length'],
                            geostrap_classification_map[required_fields['soil_class']],
                            required_fields['d50'],
                            required_fields['unit_weight'],
                            required_fields['water_content'],
                            required_fields['width'],
                            required_fields['num_straps'],
                            required_fields['tensile']
                        ], dtype=np.float32).reshape(1, -1)
                        
                        u_pred, P = calculate_geostrap_u(inputs)
                        if u_pred is not None:
                            st.session_state.result = f"μ* = {u_pred:.4f}   |   P = {P:.2f} kN/m"
                            st.rerun()

    # Common controls
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("Clear All", key="clear_all"):
            st.session_state.clear()
            st.session_state.scale_factor = 5
            st.session_state.data = None
            st.session_state.current_tab = "Geogrid"
            st.session_state.active_subtab = "Soil Parameters"
            st.rerun()
    with col2:
        uploaded_file = st.file_uploader("📂 Upload Excel for Prediction", type=["xlsx", "xls"], key="file_uploader")
        if st.button("Import Excel", key="import_excel") and uploaded_file:
            import_excel(uploaded_file)
        download_template()
    with col3:
        if st.button("Export to Excel", key="export_excel"):
            export_excel()
    with col4:
        if st.button("Data Format Info", key="data_info"):
            show_disclaimer()

    if hasattr(st.session_state, 'result'):
        st.markdown(f"""
        <div class="result-box">
            <h3 style='color: #212529;'>Results:</h3>
            <p style='color: #212529; font-size: 18px;'>{st.session_state.result}</p>
        </div>
        """, unsafe_allow_html=True)

# Helper functions
def update_bearing_members():
    try:
        if 'geo_length' in st.session_state and 'geo_md_aperture' in st.session_state:
            if st.session_state.geo_length and st.session_state.geo_md_aperture:
                st.session_state.geo_bearing_members = floor(float(st.session_state.geo_length) / float(st.session_state.geo_md_aperture))
    except:
        pass

def clear_all():
    st.session_state.clear()
    st.session_state.scale_factor = 5
    st.session_state.data = None
    st.session_state.current_tab = "Geogrid"
    st.session_state.active_subtab = "Soil Parameters"
    st.rerun()

def show_disclaimer():
    disclaimer = """DATA FORMAT REQUIREMENTS:
For Geogrid Soil Classification, use these exact values:
CH: 1, CL: 2, MH: 3, ML: 4, SW-SM: 5, SM: 6
SP-SM: 7, SP: 8, SW: 9, GP: 10, GW: 11, GW-GM: 12

For Geostrap Soil Classification, use these exact values:
ML: 1, SM: 2, SP: 3, SP-SM: 4, SW: 5, GW: 6, GW-GM: 7

For Geogrid Type, use these exact values:
NATURAL: 1, HDPE BIAXIAL: 2, PP BIAXIAL: 3
PP UNIAXIAL: 4, HDPE UNIAXIAL: 5
PET BIAXIAL: 6, PET UNIAXIAL: 7"""
    st.warning(disclaimer)

def import_excel(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        if st.session_state.current_tab == "Geogrid":
            required_cols = ['phi', 'cohesion', 'normal_stress', 'length', 'soil_classification',
                           'd50', 'unit_weight', 'water_content', 'geogrid_type', 'bearing_members',
                           'md_aperture', 'cmd_aperture', 'tensile_strength']
            if not all(col in df.columns for col in required_cols):
                st.error("Missing required columns in the Excel file.")
                return
            df['soil_classification'] = df['soil_classification'].map(geogrid_classification_map)
            df['geogrid_type'] = df['geogrid_type'].map(geogrid_type_map)
            results = []
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
            df[['predicted_mu', 'P']] = results
            inv_classification_map = {v: k for k, v in geogrid_classification_map.items()}
            inv_geogrid_type_map = {v: k for k, v in geogrid_type_map.items()}
            df['soil_classification'] = df['soil_classification'].map(inv_classification_map)
            df['geogrid_type'] = df['geogrid_type'].map(inv_geogrid_type_map)
        else:
            required_cols = ['phi', 'cohesion', 'normal_stress', 'length', 'soil_classification',
                           'd50', 'unit_weight', 'water_content', 'strap_width', 
                           'num_straps', 'tensile_strength']
            if not all(col in df.columns for col in required_cols):
                st.error("Missing required columns in the Excel file.")
                return
            df['soil_classification'] = df['soil_classification'].map(geostrap_classification_map)
            results = []
            for idx in range(len(df)):
                row = df.iloc[idx]
                inputs = np.array([
                    row['phi'], row['cohesion'], row['normal_stress'], row['length'],
                    row['soil_classification'], row['d50'], row['unit_weight'],
                    row['water_content'], row['strap_width'], row['num_straps'], row['tensile_strength']
                ], dtype=np.float32).reshape(1, -1)
                u_pred, P = calculate_geostrap_u(inputs)
                results.append([u_pred, P])
            df[['predicted_mu', 'P']] = results
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
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            st.session_state.data.to_excel(writer, index=False, sheet_name='Results')
        st.download_button(
            label="Download Excel File",
            data=output.getvalue(),
            file_name="geosynthetic_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Export failed: {e}")

def download_template():
    if st.session_state.current_tab == "Geogrid":
        template = pd.DataFrame([{
            'phi': 30.0, 'cohesion': 10.0, 'normal_stress': 50.0, 'length': 400.0,
            'soil_classification': 'CL', 'd50': 0.5, 'unit_weight': 18.5,
            'water_content': 12.0, 'geogrid_type': 'HDPE BIAXIAL', 'bearing_members': 8,
            'md_aperture': 50.0, 'cmd_aperture': 40.0, 'tensile_strength': 20.0
        }])
        file_name = "geogrid_prediction_template.xlsx"
    else:
        template = pd.DataFrame([{
            'phi': 30.0, 'cohesion': 10.0, 'normal_stress': 50.0, 'length': 400.0,
            'soil_classification': 'SM', 'd50': 0.5, 'unit_weight': 18.5,
            'water_content': 12.0, 'strap_width': 50.0, 'num_straps': 5, 'tensile_strength': 15.0
        }])
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
