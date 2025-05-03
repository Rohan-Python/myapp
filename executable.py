import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tensorflow as tf
import numpy as np
import pandas as pd
from math import floor, radians, tan, atan, degrees
from tensorflow.keras import layers
import keras.saving
from PIL import Image, ImageTk
import os
import math
import sys

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


class GeogridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("μ* Prediction Tool")
        self.scale_factor = 5
        self.data = None
        self.setup_ui()

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.tab_soil = ttk.Frame(self.notebook)
        self.setup_soil_tab()
        self.tab_geo = ttk.Frame(self.notebook)
        self.setup_geogrid_tab()
        self.notebook.add(self.tab_soil, text="Soil Parameters")
        self.notebook.add(self.tab_geo, text="Geogrid Parameters")
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(control_frame, text="Clear All", command=self.clear_all).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Import Excel", command=self.import_excel).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Export to Excel", command=self.export_excel).pack(side="left", padx=5)

        # Modified output display to show all three parameters
        output_frame = ttk.Frame(control_frame)
        output_frame.pack(side="left", padx=5)

        self.output_value = tk.StringVar()
        ttk.Label(output_frame, text="Results:", font=('Helvetica', 10, 'bold')).pack(anchor='w')
        output_label = ttk.Label(output_frame, textvariable=self.output_value,
                                 font=('Helvetica', 10), justify='left')
        output_label.pack(anchor='w')

        # Add disclaimer button
        ttk.Button(control_frame, text="Data Format Info", command=self.show_disclaimer).pack(side="right", padx=5)

    def setup_soil_tab(self):
        # Create a frame for the left side (inputs)
        input_frame = ttk.Frame(self.tab_soil)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Create a frame for the right side (image)
        image_frame = ttk.Frame(self.tab_soil)
        image_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Configure grid weights
        self.tab_soil.grid_columnconfigure(0, weight=1)
        self.tab_soil.grid_columnconfigure(1, weight=1)
        self.tab_soil.grid_rowconfigure(0, weight=1)

        # Input fields in the left frame
        fields = [
            ("Normal Stress (kPa):", "normal_stress"),
            ("Φ' (degrees):", "phi"),
            ("Cohesion c' (kPa):", "cohesion"),
            ("Unit Weight (kN/m³):", "unit_weight"),
            ("Water Content (%):", "water_content"),
            ("D50 (mm):", "d50")
        ]
        for i, (label, name) in enumerate(fields):
            ttk.Label(input_frame, text=label).grid(row=i, column=0, padx=5, pady=5, sticky="e")
            entry = ttk.Entry(input_frame)
            entry.grid(row=i, column=1, padx=5, pady=5)
            setattr(self, f"entry_{name}", entry)

        ttk.Label(input_frame, text="Soil Classification:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
        self.classification_var = tk.StringVar(value=list(classification_map.keys())[0])
        ttk.Combobox(input_frame, textvariable=self.classification_var,
                     values=list(classification_map.keys())).grid(row=6, column=1, padx=5, pady=5, sticky="w")

        ttk.Button(input_frame, text="Next →", command=lambda: self.notebook.select(self.tab_geo)) \
            .grid(row=7, column=0, columnspan=2, pady=10)

        # Add logos
        logo_frame = ttk.Frame(image_frame)
        logo_frame.pack(fill="x", pady=(0, 5))

        try:
            # CSIR-CRRI logo
            csir_logo = Image.open(resource_path("logo.png")) if os.path.exists("logo.png") else Image.new('RGB', (120, 60), '#f0f0f0')
            csir_logo = csir_logo.resize((120, 60), Image.LANCZOS)
            self.csir_logo_img = ImageTk.PhotoImage(csir_logo)

            # VNIT logo
            vnit_logo = Image.open(resource_path("VNIT.png")) if os.path.exists("VNIT.png") else Image.new('RGB', (120, 90), '#f0f0f0')
            vnit_logo = vnit_logo.resize((120, 90), Image.LANCZOS)
            self.vnit_logo_img = ImageTk.PhotoImage(vnit_logo)

            ttk.Label(logo_frame, image=self.csir_logo_img).pack(side="left", padx=20, expand=True)
            ttk.Label(logo_frame, image=self.vnit_logo_img).pack(side="right", padx=20, expand=True)
        except Exception as e:
            print(f"Error loading logos: {e}")

        # Add heading
        heading_frame = ttk.Frame(image_frame)
        heading_frame.pack(fill="x", pady=(5, 10))

        heading_text = "Pullout Interaction Coefficient Predictor using Physics-Informed Neural Network (PINN)\nDeveloped in collaboration with CSIR-CRRI and VNIT"
        ttk.Label(heading_frame, text=heading_text, font=('Times New Roman', 10, 'bold italic'),
                  justify='center', wraplength=350).pack()

        # Add image
        try:
            if os.path.exists("Geogrid_1.png"):
                img = Image.open(resource_path("Geogrid_1.png"))
                img = img.resize((500, 350), Image.LANCZOS)
            else:
                img = Image.new('RGB', (300, 350), color='white')

            self.soil_img = ImageTk.PhotoImage(img)
            img_label = ttk.Label(image_frame, image=self.soil_img)
            img_label.pack(expand=True, fill="both")

            ttk.Label(image_frame, text="Soil-Geogrid Interaction", font=('Helvetica', 10, 'bold')).pack()
        except Exception as e:
            print(f"Error loading soil image: {e}")

    def setup_geogrid_tab(self):
        # Create a frame for the left side (inputs)
        input_frame = ttk.Frame(self.tab_geo)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Create a frame for the right side (image)
        image_frame = ttk.Frame(self.tab_geo)
        image_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Configure grid weights
        self.tab_geo.grid_columnconfigure(0, weight=1)
        self.tab_geo.grid_columnconfigure(1, weight=1)
        self.tab_geo.grid_rowconfigure(0, weight=1)

        # Input fields in the left frame
        params = [
            ("Length (mm):", "length"),
            ("MD Aperture (mm):", "md_aperture"),
            ("CMD Aperture (mm):", "cmd_aperture"),
            ("Bearing Members:", "bearing_members"),
            ("Tensile Strength (kN/m):", "tensile_strength")
        ]
        for i, (label, name) in enumerate(params):
            ttk.Label(input_frame, text=label).grid(row=i, column=0, padx=5, pady=5, sticky="e")
            entry = ttk.Entry(input_frame)
            entry.grid(row=i, column=1, padx=5, pady=5)
            setattr(self, f"entry_{name}", entry)
            if name in ["length", "md_aperture", "cmd_aperture"]:
                entry.bind("<KeyRelease>", lambda e: self.auto_update_bearing_and_draw())

        ttk.Label(input_frame, text="Geogrid Type:").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.geogrid_type_var = tk.StringVar(value=list(geogrid_type_map.keys())[0])
        ttk.Combobox(input_frame, textvariable=self.geogrid_type_var,
                     values=list(geogrid_type_map.keys())).grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # Visualization frame (below inputs)
        vis_frame = ttk.Frame(input_frame)
        vis_frame.grid(row=6, column=0, columnspan=2, pady=10)

        zoom_frame = ttk.Frame(vis_frame)
        zoom_frame.pack(side="top", fill="x")
        ttk.Button(zoom_frame, text="Zoom In", command=lambda: self.adjust_zoom(1.2)).pack(side="left", padx=5)
        ttk.Button(zoom_frame, text="Zoom Out", command=lambda: self.adjust_zoom(0.8)).pack(side="left", padx=5)

        canvas_frame = ttk.Frame(vis_frame)
        canvas_frame.pack(side="top", fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, width=400, height=200, bg="white", scrollregion=(0, 0, 800, 400))
        h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        style = ttk.Style()
        style.configure("GreenText.TButton", foreground="green", font=('Helvetica', 10, 'bold'))

        ttk.Button(input_frame, text="Run ▶", style="GreenText.TButton", command=self.calculate_u).grid(
            row=7, column=0, columnspan=2, pady=10
        )

        # Add image to the right frame
        try:
            # Load the actual image if available, otherwise create a placeholder
            if os.path.exists("Geogrid.png"):
                img = Image.open(resource_path("Geogrid.png"))
                img = img.resize((500, 400), Image.LANCZOS)
            else:
                img = Image.new('RGB', (300, 400), color='white')

            self.geogrid_img = ImageTk.PhotoImage(img)

            img_label = ttk.Label(image_frame, image=self.geogrid_img)
            img_label.pack(expand=True, fill="both")

            # Add caption
            ttk.Label(image_frame, text="Geogrid Structure Reference", font=('Helvetica', 10, 'bold')).pack()
        except Exception as e:
            print(f"Error loading geogrid image: {e}")

    def show_disclaimer(self):
        disclaimer = """DATA FORMAT REQUIREMENTS:

For Soil Classification, use these exact values:
CH: 1, CL: 2, MH: 3, ML: 4, SW-SM: 5, SM: 6
SP-SM: 7, SP: 8, SW: 9, GP: 10, GW: 11, GW-GM: 12

For Geogrid Type, use these exact values:
NATURAL: 1, HDPE BIAXIAL: 2, PP BIAXIAL: 3
PP UNIAXIAL: 4, HDPE UNIAXIAL: 5
PET BIAXIAL: 6, PET UNIAXIAL: 7"""
        messagebox.showinfo("Data Format Information", disclaimer)

    def clear_all(self):
        for name in ["normal_stress", "phi", "cohesion", "unit_weight", "water_content", "d50",
                     "length", "md_aperture", "cmd_aperture", "bearing_members", "tensile_strength"]:
            getattr(self, f"entry_{name}").delete(0, tk.END)
        self.classification_var.set(list(classification_map.keys())[0])
        self.geogrid_type_var.set(list(geogrid_type_map.keys())[0])
        self.output_value.set("")
        self.canvas.delete("all")
        self.scale_factor = 5

    def auto_update_bearing_and_draw(self):
        try:
            length = float(self.entry_length.get())
            md = float(self.entry_md_aperture.get())
            if md > 0:
                bearing_members = floor(length / md)
                self.entry_bearing_members.delete(0, tk.END)
                self.entry_bearing_members.insert(0, str(bearing_members))
                self.draw_geogrid()
        except Exception:
            pass

    def adjust_zoom(self, factor):
        self.scale_factor *= factor
        self.draw_geogrid()

    def draw_geogrid(self):
        self.canvas.delete("all")
        try:
            length = float(self.entry_length.get())
            md = float(self.entry_md_aperture.get())
            cmd = float(self.entry_cmd_aperture.get())

            num_md = int(length / md) + 1
            num_cmd = 10

            for i in range(num_cmd):
                y = 20 + i * cmd * self.scale_factor
                self.canvas.create_line(20, y, 20 + length * self.scale_factor, y, width=3, fill="blue")

            for j in range(num_md):
                x = 20 + j * md * self.scale_factor
                self.canvas.create_line(x, 20, x, 20 + (num_cmd - 1) * cmd * self.scale_factor, width=2, fill="red")

            self.canvas.create_text(20 + length * self.scale_factor / 2, 10,
                                    text=f"Length: {length}mm", anchor="n")
            self.canvas.create_text(10, 20 + num_cmd * cmd * self.scale_factor / 2,
                                    text=f"CMD: {cmd}mm", angle=90, anchor="e")

            total_width = 40 + length * self.scale_factor
            total_height = 40 + num_cmd * cmd * self.scale_factor
            self.canvas.configure(scrollregion=(0, 0, total_width, total_height))
        except Exception:
            pass

    def calculate_u(self):
        try:
            # Get input values needed for calculations
            phi = float(self.entry_phi.get())
            cohesion = float(self.entry_cohesion.get())
            length_mm = float(self.entry_length.get())
            length_m = length_mm / 1000  # Convert to meters
            bearing_members = float(self.entry_bearing_members.get())
            normal_stress = float(self.entry_normal_stress.get())  # <-- ADD THIS LINE

            # Original prediction
            inputs = np.array([
                phi,
                cohesion,
                normal_stress,
                length_mm,
                classification_map[self.classification_var.get()],
                float(self.entry_d50.get()),
                float(self.entry_unit_weight.get()),
                float(self.entry_water_content.get()),
                geogrid_type_map[self.geogrid_type_var.get()],
                bearing_members,
                float(self.entry_md_aperture.get()),
                float(self.entry_cmd_aperture.get()),
                float(self.entry_tensile_strength.get())
            ], dtype=np.float32).reshape(1, -1)

            u_pred = model.predict(inputs)[0][0]

            # Calculate P (kN/m)
            phi_rad = radians(phi)
            P = 2 * u_pred * length_m * (normal_stress * tan(phi_rad) + cohesion)

            # Calculate δ (degrees)
            delta_rad = atan(P / (2 * length_m * normal_stress))
            delta_deg = degrees(delta_rad)

            # Format the output
            result_text = f"μ* = {u_pred:.4f}   |   P = {P:.2f} kN/m   |   δ = {delta_deg:.2f}°"

            self.output_value.set(result_text)

        except Exception as e:
            self.output_value.set("Error")
            print(f"Calculation error: {e}")

    def import_excel(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if not file_path:
            return
        try:
            self.data = pd.read_excel(file_path)

            # Convert categorical data to numerical
            self.data['soil_classification'] = self.data['soil_classification'].map(classification_map)
            self.data['geogrid_type'] = self.data['geogrid_type'].map(geogrid_type_map)

            # Make predictions and calculate additional parameters
            def calculate_row(row):
                inputs = np.array([
                    row['phi'], row['cohesion'], row['normal_stress'], row['length'],
                    row['soil_classification'], row['d50'], row['unit_weight'],
                    row['water_content'], row['geogrid_type'], row['bearing_members'],
                    row['md_aperture'], row['cmd_aperture'], row['tensile_strength']
                ], dtype=np.float32).reshape(1, -1)

                u_pred = model.predict(inputs)[0][0]
                length_m = row['length'] / 1000
                phi_rad = radians(row['phi'])

                P = 2 * u_pred * length_m * (row['normal_stress'] * tan(phi_rad) + row['cohesion'])
                delta_rad = atan(P / (2 * length_m * row['normal_stress']))
                delta_deg = degrees(delta_rad)

                return pd.Series([u_pred, P, delta_deg])

            self.data[['predicted_mu', 'P', 'delta']] = self.data.apply(calculate_row, axis=1)

            messagebox.showinfo("Success", f"Predictions completed for {len(self.data)} rows.")
        except Exception as e:
            messagebox.showerror("Error",
                                 f"Failed to import and predict:\n{e}\n\nPlease ensure your Excel file matches the required format.")

    def export_excel(self):
        if self.data is None:
            messagebox.showerror("No Data", "No data to export. Please import and predict first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return
        try:
            self.data.to_excel(file_path, index=False)
            messagebox.showinfo("Export Successful", f"Results exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = GeogridApp(root)
    root.mainloop()