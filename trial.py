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
import xgboost as xgb

# =================================================================
# Geogrid Model (Existing - unchanged)
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

# =================================================================
# Geostrap Model Loader (Updated for new XGBoost)
# =================================================================

def load_geostrap_model():
    """Load the saved XGBoost model with version compatibility handling"""
    try:
        import xgboost as xgb
        print(f"Loaded XGBoost version: {xgb.__version__}")  # Debug
        
        # Load model with safe mode
        model = joblib.load('model_artifacts/best_xgboost_model.joblib')
        
        # Version compatibility fixes
        if hasattr(model, 'set_param'):
            model.set_param({'device': 'cpu', 'gpu_id': -1})
        elif hasattr(model, '_Booster'):
            model._Booster.set_param({'device': 'cpu', 'gpu_id': -1})
        
        # Load feature names
        feature_names = joblib.load('model_artifacts/feature_names.joblib')
        
        if len(feature_names) != 11:
            st.error(f"Model expects 11 features, got {len(feature_names)}")
            return None, None
            
        return model, feature_names
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

def predict_geostrap(model, input_data):
    """Robust prediction function with error handling"""
    try:
        # Input validation
        if input_data is None:
            st.error("Input data is None")
            return None
            
        # Conversion and shape handling
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values
            
        input_data = np.array(input_data, dtype=np.float32)
        
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
            
        if input_data.shape[1] != 11:
            st.error(f"Expected 11 features, got {input_data.shape[1]}")
            return None
            
        # Prediction with fallback
        try:
            prediction = model.predict(input_data)
        except AttributeError:
            # Fallback for older XGBoost versions
            if hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(input_data)
            elif hasattr(model, '_Booster'):
                dmatrix = xgb.DMatrix(input_data)
                prediction = model.predict(dmatrix)
            else:
                raise
                
        if prediction is None or len(prediction) == 0:
            st.error("Model returned no prediction")
            return None
            
        return float(prediction[0])
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# =================================================================
# Mappings and Helper Functions
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
        if 'geogrid_model' not in st.session_state or st.session_state.geogrid_model is None:
            st.error("Geogrid model not loaded")
            return None, None
            
        u_pred = st.session_state.geogrid_model.predict(inputs)[0][0]
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
    """Robust calculation with full error handling"""
    try:
        # Validate inputs first
        if inputs is None or len(inputs) == 0:
            st.error("No input data provided")
            return None, None
            
        if 'geostrap_model' not in st.session_state or st.session_state.geostrap_model is None:
            st.error("Geostrap model not loaded")
            return None, None
            
        u_pred = predict_geostrap(st.session_state.geostrap_model, inputs)
        if u_pred is None:
            return None, None
            
        # Extract and validate parameters
        try:
            phi = float(inputs[0][0])
            cohesion = float(inputs[0][1])
            normal_stress = float(inputs[0][2])
            length_mm = float(inputs[0][3])
        except (IndexError, TypeError, ValueError) as e:
            st.error(f"Invalid input values: {str(e)}")
            return None, None
            
        # Perform calculations
        try:
            length_m = length_mm / 1000
            phi_rad = radians(phi)
            interaction_term = normal_stress * tan(phi_rad) + cohesion
            P = 2 * u_pred * length_m * interaction_term
            return u_pred, P
        except Exception as calc_error:
            st.error(f"Calculation error: {str(calc_error)}")
            return None, None
    except Exception as e:
        st.error(f"Geostrap calculation failed: {str(e)}")
        return None, None

def update_bearing_members():
    try:
        if 'geo_length' in st.session_state and 'geo_md_aperture' in st.session_state:
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
    st.session_state.current_subtab = "Soil Parameters"
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
    if uploaded_file is None:
        st.warning("📂 Please upload an Excel file to begin predictions.")
        return

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

                progress = (idx + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Processing row {idx + 1} of {len(df)}")

            df[['predicted_mu', 'P']] = results

            inv_classification_map = {v: k for k, v in geogrid_classification_map.items()}
            inv_geogrid_type_map = {v: k for k, v in geogrid_type_map.items()}
            df['soil_classification'] = df['soil_classification'].map(inv_classification_map)
            df['geogrid_type'] = df['geogrid_type'].map(inv_geogrid_type_map)

        else:  # Geostrap
            required_cols = ['phi', 'cohesion', 'normal_stress', 'length', 'soil_classification',
                             'd50', 'unit_weight', 'water_content', 'strap_width', 
                             'num_straps', 'tensile_strength']

            if not all(col in df.columns for col in required_cols):
                st.error("Missing required columns in the Excel file.")
                return

            df['soil_classification'] = df['soil_classification'].map(geostrap_classification_map)

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx in range(len(df)):
                row = df.iloc[idx]
                inputs = np.array([
                    row['phi'], row['cohesion'], row['normal_stress'], row['length'],
                    row['soil_classification'], row['d50'], row['unit_weight'],
                    row['water_content'], row['strap_width'], row['num_straps'], row['tensile_strength']
                ], dtype=np.float32).reshape(1, -1)

                u_pred, P = calculate_geostrap_u(inputs)
                results.append([u_pred, P])

                progress = (idx + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Processing row {idx + 1} of {len(df)}")

            df[['predicted_mu', 'P']] = results

            inv_classification_map = {v: k for k, v in geostrap_classification_map.items()}
            df['soil_classification'] = df['soil_classification'].map(inv_classification_map)

        st.session_state.data = df
        st.success(f"✅ Predictions completed for {len(df)} rows.")
        st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Failed to import and predict:\n{e}")

def export_excel():
    if 'data' not in st.session_state or st.session_state.data is None:
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

# =================================================================
# Main App
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
    if 'current_subtab' not in st.session_state:
        st.session_state.current_subtab = "Soil Parameters"
    if 'result' not in st.session_state:
        st.session_state.result = None

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
    .result-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        border: 1px solid #dee2e6;
        color: #212529;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with logos
    col1, col2, col3 = st.columns([1, 2, 1])

    # Load and display logos
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
            Pullout Interaction Coefficient Predictor using Physics-Informed Neural Network (PINN)<br>
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

    # Create main tabs
    st.markdown("## Select Product Type")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Geogrid", key="geogrid_tab", 
                    use_container_width=True, type="primary" if st.session_state.current_tab == "Geogrid" else "secondary"):
            st.session_state.current_tab = "Geogrid"
    with col2:
        if st.button("Geostrap", key="geostrap_tab",
                    use_container_width=True, type="primary" if st.session_state.current_tab == "Geostrap" else "secondary"):
            st.session_state.current_tab = "Geostrap"

    # Geogrid Tab
    if st.session_state.current_tab == "Geogrid":
        if st.session_state.current_subtab == "Soil Parameters":
            col1, col2 = st.columns([1, 1])
    
            with col1:
                st.header("Soil Parameters")
                normal_stress = st.number_input("Normal Stress (kPa):", value=None, placeholder="Enter value", step=0.1,
                                              format="%.2f", key="geo_normal_stress")
                phi = st.number_input("Φ' (degrees):", value=None, placeholder="Enter value", step=0.1, 
                                    format="%.1f", key="geo_phi")
                cohesion = st.number_input("Cohesion c' (kPa):", value=None, placeholder="Enter value", step=0.1,
                                         format="%.2f", key="geo_cohesion")
                unit_weight = st.number_input("Unit Weight (kN/m³):", value=None, placeholder="Enter value", step=0.1,
                                            format="%.2f", key="geo_unit_weight")
                water_content = st.number_input("Water Content (%):", value=None, placeholder="Enter value", step=0.1,
                                              format="%.1f", key="geo_water_content")
                d50 = st.number_input("D50 (mm):", value=None, placeholder="Enter value", step=0.1, 
                                    format="%.2f", key="geo_d50")
                soil_classification = st.selectbox("Soil Classification:", 
                                                options=list(geogrid_classification_map.keys()), 
                                                key="geo_soil_class")
    
            with col2:
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
    
            # Navigation buttons at bottom
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Soil Parameters", disabled=True, help="You are currently on this tab"):
                    pass
            with col2:
                if st.button("Geogrid Parameters", key="geo_to_grid"):
                    st.session_state.current_subtab = "Geogrid Parameters"
                    st.rerun()
    
        elif st.session_state.current_subtab == "Geogrid Parameters":
            col1, col2 = st.columns([1, 1])
    
            with col1:
                st.header("Geogrid Parameters")
                length = st.number_input("Length (mm):", value=None, placeholder="Enter value", step=1.0, 
                                       format="%.1f", key="geo_length", on_change=update_bearing_members)
                md_aperture = st.number_input("MD Aperture (mm):", value=None, placeholder="Enter value", step=0.1,
                                            format="%.1f", key="geo_md_aperture", on_change=update_bearing_members)
                cmd_aperture = st.number_input("CMD Aperture (mm):", value=None, placeholder="Enter value", step=0.1,
                                             format="%.1f", key="geo_cmd_aperture", on_change=update_bearing_members)
                bearing_members = st.number_input("Bearing Members:", value=None, placeholder="Enter value", 
                                                step=1, key="geo_bearing_members")
                tensile_strength = st.number_input("Tensile Strength (kN/m):", value=None, placeholder="Enter value",
                                                step=0.1, format="%.1f", key="geo_tensile_strength")
                geogrid_type = st.selectbox("Geogrid Type:", options=list(geogrid_type_map.keys()), 
                                          key="geo_geogrid_type")
    
            with col2:
                geogrid_img = get_image_base64("Geogrid.png")
                if geogrid_img:
                    st.markdown(
                        f'<img src="data:image/png;base64,{geogrid_img}" style="width: 100%;">',
                        unsafe_allow_html=True
                    )
                st.markdown("<p style='text-align: center; font-weight: bold;'>Geogrid Structure Reference</p>",
                          unsafe_allow_html=True)
    
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Zoom In", key="geo_zoom_in"):
                        st.session_state.scale_factor *= 1.2
                        st.rerun()
                with col2:
                    if st.button("Zoom Out", key="geo_zoom_out"):
                        st.session_state.scale_factor *= 0.8
                        st.rerun()
    
                if 'geo_length' in st.session_state and 'geo_md_aperture' in st.session_state and 'geo_cmd_aperture' in st.session_state:
                    if st.session_state.geo_length and st.session_state.geo_md_aperture and st.session_state.geo_cmd_aperture:
                        fig = draw_geogrid(float(st.session_state.geo_length), 
                                          float(st.session_state.geo_md_aperture), 
                                          float(st.session_state.geo_cmd_aperture),
                                          st.session_state.scale_factor)
                        st.pyplot(fig)
    
            # Navigation buttons at bottom
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Soil Parameters", key="geo_to_soil"):
                    st.session_state.current_subtab = "Soil Parameters"
                    st.rerun()
            with col2:
                if st.button("Geogrid Parameters", disabled=True, help="You are currently on this tab"):
                    pass
    
            if st.button("Run ▶", type="primary", key="geo_run"):
                required_fields = [
                    st.session_state.get('geo_normal_stress'),
                    st.session_state.get('geo_phi'),
                    st.session_state.get('geo_cohesion'),
                    st.session_state.get('geo_length'),
                    st.session_state.get('geo_d50'),
                    st.session_state.get('geo_unit_weight'),
                    st.session_state.get('geo_water_content'),
                    st.session_state.get('geo_md_aperture'),
                    st.session_state.get('geo_cmd_aperture'),
                    st.session_state.get('geo_tensile_strength')
                ]
    
                if None in required_fields:
                    st.error("Please fill in all required fields")
                else:
                    inputs = np.array([
                        st.session_state.geo_phi,
                        st.session_state.geo_cohesion,
                        st.session_state.geo_normal_stress,
                        st.session_state.geo_length,
                        geogrid_classification_map[st.session_state.geo_soil_class],
                        st.session_state.geo_d50,
                        st.session_state.geo_unit_weight,
                        st.session_state.geo_water_content,
                        geogrid_type_map[st.session_state.geo_geogrid_type],
                        st.session_state.geo_bearing_members if st.session_state.geo_bearing_members else floor(float(st.session_state.geo_length) / float(st.session_state.geo_md_aperture)),
                        st.session_state.geo_md_aperture,
                        st.session_state.geo_cmd_aperture,
                        st.session_state.geo_tensile_strength
                    ], dtype=np.float32).reshape(1, -1)
    
                    u_pred, P = calculate_geogrid_u(inputs)
    
                    if u_pred is not None:
                        st.session_state.result = f"μ* = {u_pred:.4f}   |   P = {P:.2f} kN/m"

    # Geostrap Tab
    elif st.session_state.current_tab == "Geostrap":
        if st.session_state.current_subtab == "Soil Parameters":
            col1, col2 = st.columns([1, 1])
    
            with col1:
                st.header("Soil Parameters")
                normal_stress = st.number_input("Normal Stress (kPa):", value=None, placeholder="Enter value", step=0.1,
                                              format="%.2f", key="strap_normal_stress")
                phi = st.number_input("Φ' (degrees):", value=None, placeholder="Enter value", step=0.1, 
                                    format="%.1f", key="strap_phi")
                cohesion = st.number_input("Cohesion c' (kPa):", value=None, placeholder="Enter value", step=0.1,
                                         format="%.2f", key="strap_cohesion")
                length = st.number_input("Length (mm):", value=None, placeholder="Enter value", step=1.0,
                                       format="%.1f", key="strap_length")
                soil_classification = st.selectbox("Soil Classification:", 
                                                options=list(geostrap_classification_map.keys()), 
                                                key="strap_soil_class")
                d50 = st.number_input("D50 (mm):", value=None, placeholder="Enter value", step=0.1, 
                                    format="%.2f", key="strap_d50")
                unit_weight = st.number_input("Unit Weight (kN/m³):", value=None, placeholder="Enter value", step=0.1,
                                            format="%.2f", key="strap_unit_weight")
    
            with col2:
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
    
            # Navigation buttons at bottom
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Soil Parameters", disabled=True, help="You are currently on this tab"):
                    pass
            with col2:
                if st.button("Geostrap Parameters", key="strap_to_strap"):
                    st.session_state.current_subtab = "Geostrap Parameters"
                    st.rerun()
    
        elif st.session_state.current_subtab == "Geostrap Parameters":
            col1, col2 = st.columns([1, 1])
    
            with col1:
                st.header("Geostrap Parameters")
                water_content = st.number_input("Water Content (%):", value=None, placeholder="Enter value", step=0.1,
                                              format="%.1f", key="strap_water_content")
                strap_width = st.number_input("Width of Straps (mm):", value=None, placeholder="Enter value", 
                                            step=1.0, format="%.1f", key="strap_width")
                num_straps = st.number_input("Number of Straps:", value=None, placeholder="Enter value", 
                                           step=1, key="num_straps")
                tensile_strength = st.number_input("Tensile Strength (kN):", value=None, placeholder="Enter value",
                                                 step=0.1, format="%.1f", key="strap_tensile_strength")
    
            with col2:
                geostrap_img = get_image_base64("Geostrap.png")
                if geostrap_img:
                    st.markdown(
                        f'<img src="data:image/png;base64,{geostrap_img}" style="width: 100%;">',
                        unsafe_allow_html=True
                    )
                st.markdown("<p style='text-align: center; font-weight: bold;'>Geostrap Structure Reference</p>",
                          unsafe_allow_html=True)
    
            # Navigation buttons at bottom
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Soil Parameters", key="strap_to_soil"):
                    st.session_state.current_subtab = "Soil Parameters"
                    st.rerun()
            with col2:
                if st.button("Geostrap Parameters", disabled=True, help="You are currently on this tab"):
                    pass
    
            if st.button("Run ▶", type="primary", key="strap_run"):
                required_fields = [
                    st.session_state.get('strap_normal_stress'),
                    st.session_state.get('strap_phi'),
                    st.session_state.get('strap_cohesion'),
                    st.session_state.get('strap_length'),
                    st.session_state.get('strap_soil_class'),
                    st.session_state.get('strap_d50'),
                    st.session_state.get('strap_unit_weight'),
                    st.session_state.get('strap_water_content'),
                    st.session_state.get('strap_width'),
                    st.session_state.get('num_straps'),
                    st.session_state.get('strap_tensile_strength')
                ]
            
                if None in required_fields:
                    st.error("Please fill in all required fields")
                else:
                    inputs = np.array([
                        st.session_state.strap_phi,
                        st.session_state.strap_cohesion,
                        st.session_state.strap_normal_stress,
                        st.session_state.strap_length,
                        geostrap_classification_map[st.session_state.strap_soil_class],
                        st.session_state.strap_d50,
                        st.session_state.strap_unit_weight,
                        st.session_state.strap_water_content,
                        st.session_state.strap_width,
                        st.session_state.num_straps,
                        st.session_state.strap_tensile_strength
                    ], dtype=np.float32).reshape(1, -1)
                    
                    u_pred, P = calculate_geostrap_u(inputs)
            
                    if u_pred is not None:
                        st.session_state.result = f"μ* = {u_pred:.4f}   |   P = {P:.2f} kN/m"

    # Common controls
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("🔄 Clear All", 
                    key="clear_all",
                    help="Reset all inputs and results",
                    use_container_width=True):
            clear_all()
    
    with col2:
        uploaded_file = st.file_uploader(
            "📂 Upload Excel for Prediction", 
            type=["xlsx", "xls"], 
            key="file_uploader",
            help="Upload Excel file with input data"
        )
        
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
    
        if st.button("📤 Import Excel", 
                    key="import_excel",
                    help="Process uploaded Excel file",
                    use_container_width=True):
            if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
                import_excel(st.session_state.uploaded_file)
            else:
                st.warning("Please upload an Excel file first.")
    
        # Template download stays here to keep all file operations together
        if st.session_state.current_tab == "Geogrid":
            template_data = {
                'phi': [30.0],
                'cohesion': [10.0],
                'normal_stress': [50.0],
                'length': [400.0],
                'soil_classification': ['CL'],
                'd50': [0.5],
                'unit_weight': [18.5],
                'water_content': [12.0],
                'geogrid_type': ['HDPE BIAXIAL'],
                'bearing_members': [8],
                'md_aperture': [50.0],
                'cmd_aperture': [40.0],
                'tensile_strength': [20.0]
            }
        else:
            template_data = {
                'phi': [30.0],
                'cohesion': [10.0],
                'normal_stress': [50.0],
                'length': [400.0],
                'soil_classification': ['SM'],
                'd50': [0.5],
                'unit_weight': [18.5],
                'water_content': [12.0],
                'strap_width': [50.0],
                'num_straps': [5],
                'tensile_strength': [15.0]
            }
        
        template_df = pd.DataFrame(template_data)
        
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            template_df.to_excel(writer, index=False, sheet_name='Template')
        
        st.download_button(
            label="📥 Download Template",
            data=excel_buffer.getvalue(),
            file_name=f"{st.session_state.current_tab.lower()}_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download Excel template with sample data",
            use_container_width=True
        )
    
    with col3:
        if st.button("📥 Export to Excel", 
                    key="export_excel",
                    help="Export results to Excel",
                    disabled=('data' not in st.session_state or st.session_state.data is None),
                    use_container_width=True):
            export_excel()
    
    with col4:
        if st.button("ℹ️ Data Format Info", 
                    key="data_info",
                    help="Show required data formats",
                    use_container_width=True):
            show_disclaimer()
    
    # Results display
    if 'result' in st.session_state and st.session_state.result:
        st.markdown(f"""
        <div class="result-box">
            <h3 style='color: #212529;'>Results:</h3>
            <p style='color: #212529; font-size: 18px;'>{st.session_state.result}</p>
        </div>
        """, unsafe_allow_html=True)
    elif 'data' in st.session_state and st.session_state.data is not None:
        st.markdown(f"""
        <div class="result-box">
            <h3 style='color: #212529;'>Batch Results:</h3>
            <p style='color: #212529; font-size: 14px;'>
                Processed {len(st.session_state.data)} rows. See table below.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(st.session_state.data)
