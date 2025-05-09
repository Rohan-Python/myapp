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


# Model class
@keras.saving.register_keras_serializable()
class GeogridPINN(tf.keras.Model):
    def __init__(self, hidden_layers=6, units_per_layer=1024, **kwargs):
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


# Load model
def load_model():
    try:
        model_path = '84.keras'
        model = tf.keras.models.load_model(model_path, custom_objects={'GeogridPINN': GeogridPINN})
        print("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise


model = load_model()

# Mappings
classification_map = {
    'CH': 1, 'CL': 2, 'MH': 3, 'ML': 4, 'SW-SM': 5, 'SM': 6,
    'SP-SM': 7, 'SP': 8, 'SW': 9, 'GP': 10, 'GW': 11, 'GW-GM': 12
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


def calculate_u(inputs):
    try:
        # Original prediction
        u_pred = model.predict(inputs)[0][0]

        # Extract values from inputs
        phi = inputs[0][0]
        cohesion = inputs[0][1]
        normal_stress = inputs[0][2]
        length_mm = inputs[0][3]

        # Calculate P (kN/m)
        length_m = length_mm / 1000  # Convert to meters
        phi_rad = radians(phi)
        P = 2 * u_pred * length_m * (normal_stress * tan(phi_rad) + cohesion)

        # Calculate δ (degrees)
        delta_deg = 2*(phi)/3
        

        return u_pred, P, delta_deg
    except Exception as e:
        st.error(f"Calculation error: {e}")
        return None, None, None


# Main app
def main():
    st.set_page_config(layout="wide", page_title="μ* Prediction Tool")

    # Initialize session state variables
    if 'scale_factor' not in st.session_state:
        st.session_state.scale_factor = 5
    if 'data' not in st.session_state:
        st.session_state.data = None

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
    </style>
    """, unsafe_allow_html=True)

    # Header with logos
    col1, col2, col3 = st.columns([1, 2, 1])

    # Load and display logos if available
    csir_logo = get_image_base64("logo.png")
    vnit_logo = get_image_base64("VNIT.png")

    if csir_logo and col1:
        col1.markdown(
            f'<img src="data:image/png;base64,{csir_logo}" style="height: 80px;">',
            unsafe_allow_html=True
        )

    col2.markdown(
        '<p class="header-text">Pullout Interaction Coefficient Predictor using Physics-Informed Neural Network (PINN)<br>'
        'Developed in collaboration with CSIR-CRRI and VNIT</p>',
        unsafe_allow_html=True
    )

    if vnit_logo and col3:
        col3.markdown(
            f'<img src="data:image/png;base64,{vnit_logo}" style="height: 100px;">',
            unsafe_allow_html=True
        )

    # Create tabs
    tab1, tab2 = st.tabs(["Soil Parameters", "Geogrid Parameters"])

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("Soil Parameters")

            # Soil input fields - all with empty default values
            normal_stress = st.number_input("Normal Stress (kPa):", value=None, placeholder="Enter value", step=0.1,
                                            format="%.2f")
            phi = st.number_input("Φ' (degrees):", value=None, placeholder="Enter value", step=0.1, format="%.1f")
            cohesion = st.number_input("Cohesion c' (kPa):", value=None, placeholder="Enter value", step=0.1,
                                       format="%.2f")
            unit_weight = st.number_input("Unit Weight (kN/m³):", value=None, placeholder="Enter value", step=0.1,
                                          format="%.2f")
            water_content = st.number_input("Water Content (%):", value=None, placeholder="Enter value", step=0.1,
                                            format="%.1f")
            d50 = st.number_input("D50 (mm):", value=None, placeholder="Enter value", step=0.1, format="%.2f")
            soil_classification = st.selectbox("Soil Classification:", options=list(classification_map.keys()))

        with col2:
            # Display soil image if available
            soil_img = get_image_base64("Geogrid_1.png")
            if soil_img:
                st.markdown(
                    f'<img src="data:image/png;base64,{soil_img}" style="width: 100%;">',
                    unsafe_allow_html=True
                )
            st.markdown("<p style='text-align: center; font-weight: bold;'>Soil-Geogrid Interaction</p>",
                        unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("Geogrid Parameters")

            # Geogrid input fields - all with empty default values
            length = st.number_input("Length (mm):", value=None, placeholder="Enter value", step=1.0, format="%.1f",
                                     key="length", on_change=update_bearing_members)
            md_aperture = st.number_input("MD Aperture (mm):", value=None, placeholder="Enter value", step=0.1,
                                          format="%.1f",
                                          key="md_aperture", on_change=update_bearing_members)
            cmd_aperture = st.number_input("CMD Aperture (mm):", value=None, placeholder="Enter value", step=0.1,
                                           format="%.1f",
                                           key="cmd_aperture", on_change=update_bearing_members)
            bearing_members = st.number_input("Bearing Members:", value=None, placeholder="Enter value", step=1,
                                              key="bearing_members")
            tensile_strength = st.number_input("Tensile Strength (kN/m):", value=None, placeholder="Enter value",
                                               step=0.1, format="%.1f")
            geogrid_type = st.selectbox("Geogrid Type:", options=list(geogrid_type_map.keys()))

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
                if st.button("Zoom In"):
                    st.session_state.scale_factor *= 1.2
            with col2:
                if st.button("Zoom Out"):
                    st.session_state.scale_factor *= 0.8

            # Draw geogrid visualization
            if length and md_aperture and cmd_aperture:
                fig = draw_geogrid(float(length), float(md_aperture), float(cmd_aperture),
                                   st.session_state.scale_factor)
                st.pyplot(fig)

            # Run button
            if st.button("Run ▶", type="primary"):
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
                        classification_map[soil_classification],
                        d50,
                        unit_weight,
                        water_content,
                        geogrid_type_map[geogrid_type],
                        bearing_members if bearing_members else floor(float(length) / float(md_aperture)),
                        md_aperture,
                        cmd_aperture,
                        tensile_strength
                    ], dtype=np.float32).reshape(1, -1)

                    u_pred, P, delta_deg = calculate_u(inputs)

                    if u_pred is not None:
                        st.session_state.result = f"μ* = {u_pred:.4f}   |   P = {P:.2f} kN/m   |   δ = {delta_deg:.2f}°"

    # Control buttons and results
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button("Clear All"):
            clear_all()

    with col2:
        uploaded_file = st.file_uploader("📂 Upload Excel for Prediction", type=["xlsx", "xls"])
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file

        if st.button("Import Excel"):
            if 'uploaded_file' in st.session_state:
                import_excel(st.session_state.uploaded_file)
            else:
                st.warning("Please upload an Excel file first.")

        download_template()





    with col3:
        if st.button("Export to Excel"):
            export_excel()

    with col4:
        if st.button("Data Format Info"):
            show_disclaimer()

    # Display results with improved styling
    if hasattr(st.session_state, 'result'):
        st.markdown(f"""
        <div class="result-box">
            <h3 style='color: #212529;'>Results:</h3>
            <p style='color: #212529; font-size: 18px;'>{st.session_state.result}</p>
        </div>
        """, unsafe_allow_html=True)


def update_bearing_members():
    try:
        length = st.session_state.length
        md = st.session_state.md_aperture
        if md and length:
            bearing_members = floor(float(length) / float(md))
            st.session_state.bearing_members = bearing_members
    except:
        pass


def clear_all():
    st.session_state.clear()
    st.session_state.scale_factor = 5
    st.session_state.data = None
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
   

    uploaded_file = st.session_state.uploaded_file

    if uploaded_file is None:
        st.warning("📂 Please upload an Excel file to begin predictions.")
        return

    try:
        df = pd.read_excel(uploaded_file)

        # Validate columns
        required_cols = ['phi', 'cohesion', 'normal_stress', 'length', 'soil_classification',
                         'd50', 'unit_weight', 'water_content', 'geogrid_type', 'bearing_members',
                         'md_aperture', 'cmd_aperture', 'tensile_strength']

        if not all(col in df.columns for col in required_cols):
            st.error("Missing required columns in the Excel file.")
            return

        # Map categorical values
        df['soil_classification'] = df['soil_classification'].map(classification_map)
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

            u_pred, P, delta_deg = calculate_u(inputs)
            results.append([u_pred, P, delta_deg])

            # Update progress
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx + 1} of {len(df)}")

        # Assign predictions
        df[['predicted_mu', 'P', 'delta']] = results

        # Convert back to labels
        inv_classification_map = {v: k for k, v in classification_map.items()}
        inv_geogrid_type_map = {v: k for k, v in geogrid_type_map.items()}
        df['soil_classification'] = df['soil_classification'].map(inv_classification_map)
        df['geogrid_type'] = df['geogrid_type'].map(inv_geogrid_type_map)

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
            file_name="geogrid_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Export failed: {e}")

def download_template():
    # Create a sample dataframe with 5 rows
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
        },
        {
            'phi': 28.5,
            'cohesion': 8.0,
            'normal_stress': 60.0,
            'length': 500.0,
            'soil_classification': 'SP',
            'd50': 0.8,
            'unit_weight': 19.0,
            'water_content': 10.5,
            'geogrid_type': 'PP UNIAXIAL',
            'bearing_members': 10,
            'md_aperture': 45.0,
            'cmd_aperture': 35.0,
            'tensile_strength': 18.0
        },
        {
            'phi': 32.0,
            'cohesion': 12.0,
            'normal_stress': 55.0,
            'length': 450.0,
            'soil_classification': 'SW',
            'd50': 1.0,
            'unit_weight': 20.0,
            'water_content': 11.0,
            'geogrid_type': 'PET UNIAXIAL',
            'bearing_members': 9,
            'md_aperture': 50.0,
            'cmd_aperture': 40.0,
            'tensile_strength': 22.0
        },
        {
            'phi': 27.0,
            'cohesion': 6.0,
            'normal_stress': 45.0,
            'length': 380.0,
            'soil_classification': 'MH',
            'd50': 0.6,
            'unit_weight': 17.0,
            'water_content': 13.0,
            'geogrid_type': 'PP BIAXIAL',
            'bearing_members': 7,
            'md_aperture': 48.0,
            'cmd_aperture': 36.0,
            'tensile_strength': 19.0
        },
        {
            'phi': 29.0,
            'cohesion': 9.5,
            'normal_stress': 52.0,
            'length': 420.0,
            'soil_classification': 'GW',
            'd50': 0.9,
            'unit_weight': 18.8,
            'water_content': 11.5,
            'geogrid_type': 'HDPE UNIAXIAL',
            'bearing_members': 8,
            'md_aperture': 50.0,
            'cmd_aperture': 38.0,
            'tensile_strength': 21.0
        },
    ])

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        template.to_excel(writer, index=False, sheet_name='Template')

    st.download_button(
        label="📥 Download Prediction Template (with samples)",
        data=output.getvalue(),
        file_name="geogrid_prediction_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == "__main__":
    main()
