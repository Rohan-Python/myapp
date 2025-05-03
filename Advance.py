import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import keras.saving
from math import floor, radians, tan, atan, degrees
from PIL import Image
import os
import sys
import math


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


# Model class
@keras.saving.register_keras_serializable()
class GeogridPINN(tf.keras.Model):
    def __init__(self, hidden_layers=4, units_per_layer=64, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer
        self.dense_layers = [layers.Dense(units_per_layer, activation='tanh', name=f'dense_{i}') for i in
                             range(hidden_layers)]
        self.u_output = layers.Dense(1, activation=None, name='output')

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.u_output(x)


# Load model
try:
    model_path = resource_path('geogrid_pinn_model.keras')
    model = tf.keras.models.load_model(model_path, custom_objects={'GeogridPINN': GeogridPINN})

    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Mappings
classification_map = {
    'CH': 1, 'CL': 2, 'MH': 3, 'ML': 4, 'SW-SM': 5, 'SM': 6,
    'SP-SM': 7, 'SP': 8, 'SW': 9, 'GP': 10, 'GW': 11, 'GW-GM': 12
}
geogrid_type_map = {
    'NATURAL': 1, 'HDPE BIAXIAL': 2, 'PP BIAXIAL': 3, 'PP UNIAXIAL': 4,
    'HDPE UNIAXIAL': 5, 'PET BIAXIAL': 6, 'PET UNIAXIAL': 7
}


# Function to calculate μ*, P, and δ
def calculate_u(inputs, model):
    try:
        u_pred = model.predict(inputs)[0][0]

        # Parameters
        phi = inputs[0][0]
        cohesion = inputs[0][1]
        normal_stress = inputs[0][2]
        length_mm = inputs[0][3]
        length_m = length_mm / 1000  # Convert to meters

        # Calculate P (kN/m)
        phi_rad = radians(phi)
        P = 2 * u_pred * length_m * (normal_stress * tan(phi_rad) + cohesion)

        # Calculate δ (degrees)
        delta_rad = atan(P / (2 * length_m * normal_stress))
        delta_deg = degrees(delta_rad)

        return u_pred, P, delta_deg

    except Exception as e:
        print(f"Error in calculation: {e}")
        return None, None, None


# Streamlit UI Implementation
def main():
    st.title("μ* Prediction Tool")
    st.write(
        "This tool predicts the pullout interaction coefficient (μ*) for soil-geogrid interaction using a "
        "Physics-Informed Neural Network (PINN) model. Developed in collaboration with CSIR-CRRI and VNIT.")

    # Tabs for Soil and Geogrid Parameters
    tab_soil, tab_geo = st.tabs(["Soil Parameters", "Geogrid Parameters"])

    with tab_soil:
        st.header("Soil Parameters")
        
        # Input Fields
        normal_stress = st.number_input("Normal Stress (kPa):", value=100.0)
        phi = st.number_input("Φ' (degrees):", value=30.0)
        cohesion = st.number_input("Cohesion c' (kPa):", value=15.0)
        unit_weight = st.number_input("Unit Weight (kN/m³):", value=18.0)
        water_content = st.number_input("Water Content (%):", value=20.0)
        d50 = st.number_input("D50 (mm):", value=2.0)

        # Soil Classification Dropdown
        classification = st.selectbox("Soil Classification:", options=list(classification_map.keys()))

        # Next button
        if st.button("Next →"):
            st.session_state["soil_parameters"] = {
                "normal_stress": normal_stress,
                "phi": phi,
                "cohesion": cohesion,
                "unit_weight": unit_weight,
                "water_content": water_content,
                "d50": d50,
                "classification": classification
            }

    with tab_geo:
        st.header("Geogrid Parameters")

        # Input Fields
        length = st.number_input("Length (mm):", value=200.0)
        md_aperture = st.number_input("MD Aperture (mm):", value=10.0)
        cmd_aperture = st.number_input("CMD Aperture (mm):", value=5.0)
        bearing_members = st.number_input("Bearing Members:", value=20)
        tensile_strength = st.number_input("Tensile Strength (kN/m):", value=100.0)

        # Geogrid Type Dropdown
        geogrid_type = st.selectbox("Geogrid Type:", options=list(geogrid_type_map.keys()))

        # Calculate button
        if st.button("Run ▶"):
            # Check if all soil and geogrid parameters are filled
            if "soil_parameters" in st.session_state:
                soil_params = st.session_state["soil_parameters"]
                inputs = np.array([
                    soil_params["phi"], soil_params["cohesion"], soil_params["normal_stress"],
                    length, classification_map[soil_params["classification"]], soil_params["d50"],
                    soil_params["unit_weight"], soil_params["water_content"], geogrid_type_map[geogrid_type],
                    bearing_members, md_aperture, cmd_aperture, tensile_strength
                ], dtype=np.float32).reshape(1, -1)

                # Calculate μ*, P, and δ
                u_pred, P, delta_deg = calculate_u(inputs, model)

                if u_pred is not None:
                    st.success(f"Prediction Results: \nμ* = {u_pred:.4f} \nP = {P:.2f} kN/m \nδ = {delta_deg:.2f}°")
                else:
                    st.error("Error in prediction. Please check the inputs.")
            else:
                st.error("Please complete the soil parameters first.")

    # Additional functionality to import/export Excel
    st.sidebar.header("Data Management")
    action = st.sidebar.selectbox("Choose action", ["None", "Import Excel", "Export to Excel"])

    if action == "Import Excel":
        file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
        if file is not None:
            try:
                data = pd.read_excel(file)
                st.write(data.head())
                st.session_state["imported_data"] = data
                st.success(f"Data imported successfully! {len(data)} rows.")
            except Exception as e:
                st.error(f"Failed to import data: {e}")

    if action == "Export to Excel":
        if "imported_data" in st.session_state:
            data = st.session_state["imported_data"]
            output_file = st.sidebar.file_uploader("Save as Excel", type=["xlsx"])
            if output_file is not None:
                data.to_excel(output_file, index=False)
                st.success("Data exported successfully!")
        else:
            st.error("No data to export. Please import data first.")


if __name__ == "__main__":
    main()
