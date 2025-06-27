#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:59:28 2024

@author: jzhan124
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
# from io import StringIO
from opt_substrate_moss_design import (
    train_regression_models,
    calculate_saturation_history,
    map_moss_index_uniformly,
    round_to_nearest,
    evaporation_rate,
    objective_function,
    # bounds,
    match_packing,
    generate_positions,
    truncated_normal
)

# st.set_page_config(layout="wide")
# Example credentials
PASSWORD = "Utk37996"

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False  # Tracks if the "Run" button has been clicked

# Authentication logic
if not st.session_state.authenticated:
    st.title("Welcome! Please enter the passcode to use this tool.")

    password = st.text_input("Passcode", type="password")

    if st.button("Login"):
        if password == PASSWORD:
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid passcode.")
else:
    # Show description before the "Run" button is clicked
    # if not st.session_state.run_clicked:
    #     st.title("Design Tool for Bio-receptive Substrate with Moss")
    #     st.subheader("About the Tool")
    #     st.write("""
    #         This tool optimizes the design of moss substrates to achieve specific performance goals. 
    #         You can input parameters such as rainfall data, substrate properties and moss growth requirements. 
    #         Then the tool will determine optimal solutions for moss growth requirements.
    #     """)
    #     st.write("""
    #         Key Features:
    #         - Allows customization of substrate parameters, moss growth requirements.
    #         - Optimizes substrate design using advanced algorithms.
    #         - Visualizes results such as saturation levels and particle distribution.
    #     """)

    #     # Add a "Run" button to proceed
    #     if st.button("Let's Go!"):
    #         st.session_state.run_clicked = True  # Update session state to show the main content
    #         st.rerun()
    # else:
        # Streamlit app configuration
        st.title("Design Tool for Bio-receptive Substrate with Moss")
        st.subheader("About the Tool")
        st.write("""
            This tool optimizes the design of bio-receptive substrates to achieve specific performance goals. 
            You can input parameters such as substrate properties, rainfall data and moss growth requirements. 
            Then the tool will search optimal solutions for moss growth requirements.
        """)
        st.write("""
            Key Features:
            - Allows customization of substrate parameters, moss growth requirements.
            - Optimizes substrate design using differential evolution algorithms.
            - Visualizes results such as saturation levels and particle distribution.
        """)
        st.write("""
            Please customise any parameters in the sidebar (Left)
        """)
        st.sidebar.header("Input Options")
        
        # Features file input
        # st.sidebar.subheader("Feature Files")
        # feature_files = st.sidebar.file_uploader("Upload Feature Files (CSV)", type="csv", accept_multiple_files=True)
        # features_file_list = []
        # if feature_files: 
        #     for file in feature_files:
        #         features_file_list.append(file)
        features_file_list = [
            'solid_particles_features_1.csv',
            'solid_particles_features_2.csv',
            'solid_particles_features_3.csv'
        ]
        # Parameter bounds
        st.sidebar.subheader("Substrate Parameter Bounds")
        bounds = []
        # mean_radius_bounds = st.sidebar.slider("Mean Radius (μm)", 6, 15, (6, 15))
        # bounds.append((mean_radius_bounds[0] * 1e-6, mean_radius_bounds[1] * 1e-6))
        # Add a radio button to choose between fixed value or range
        # Mean Radius
        mean_radius_option = st.sidebar.radio(
            "Set Mean Radius:", ("Fix to a constant value", "Define a range")
        )
        if mean_radius_option == "Fix to a constant value":
            fixed_mean_radius = st.sidebar.number_input(
                "Mean Radius (Fixed, μm)", min_value=6.0, max_value=15.0, value=8.0, step=0.1
            )
            bounds.append((fixed_mean_radius * 1e-6, fixed_mean_radius * 1e-6))
        else:
            mean_radius_bounds = st.sidebar.slider(
                "Mean Radius Range (μm)", 6, 15, (6, 15)
            )
            bounds.append((mean_radius_bounds[0] * 1e-6, mean_radius_bounds[1] * 1e-6))
        
        # Std Radius
        std_radius_option = st.sidebar.radio(
            "Set Std Radius:", ("Fix to a constant value", "Define a range")
        )
        if std_radius_option == "Fix to a constant value":
            fixed_std_radius = st.sidebar.number_input(
                "Std Radius (Fixed, μm)", min_value=0.1, max_value=3.5, value=1.0, step=0.1
            )
            bounds.append((fixed_std_radius * 1e-6, fixed_std_radius * 1e-6))
        else:
            std_radius_bounds = st.sidebar.slider(
                "Std Radius Range (μm)", 0.1, 3.5, (0.1, 3.5)
            )
            bounds.append((std_radius_bounds[0] * 1e-6, std_radius_bounds[1] * 1e-6))
        
        # Min Radius
        min_radius_option = st.sidebar.radio(
            "Set Min Radius:", ("Fix to a constant value", "Define a range")
        )
        if min_radius_option == "Fix to a constant value":
            fixed_min_radius = st.sidebar.number_input(
                "Min Radius (Fixed, μm)", min_value=5.0, max_value=8.0, value=6.0, step=0.1
            )
            bounds.append((fixed_min_radius * 1e-6, fixed_min_radius * 1e-6))
        else:
            min_radius_bounds = st.sidebar.slider(
                "Min Radius Range (μm)", 5, 8, (5, 8)
            )
            bounds.append((min_radius_bounds[0] * 1e-6, min_radius_bounds[1] * 1e-6))
        
        # Max Radius
        max_radius_option = st.sidebar.radio(
            "Set Max Radius:", ("Fix to a constant value", "Define a range")
        )
        if max_radius_option == "Fix to a constant value":
            fixed_max_radius = st.sidebar.number_input(
                "Max Radius (Fixed, μm)", min_value=7.0, max_value=20.0, value=10.0, step=0.1
            )
            bounds.append((fixed_max_radius * 1e-6, fixed_max_radius * 1e-6))
        else:
            max_radius_bounds = st.sidebar.slider(
                "Max Radius Range (μm)", 7, 20, (7, 20)
            )
            bounds.append((max_radius_bounds[0] * 1e-6, max_radius_bounds[1] * 1e-6))
        
        # Porosity
        porosity_option = st.sidebar.radio(
            "Set Porosity:", ("Fix to a constant value", "Define a range")
        )
        if porosity_option == "Fix to a constant value":
            fixed_porosity = st.sidebar.number_input(
                "Porosity (Fixed)", min_value=0.6, max_value=0.7, value=0.65
            )
            bounds.append((fixed_porosity, fixed_porosity))
        else:
            porosity_bounds = st.sidebar.slider(
                "Porosity Range", 0.6, 0.7, (0.6, 0.7)
            )
            bounds.append((porosity_bounds[0], porosity_bounds[1]))
        
        # width_bounds = st.sidebar.slider("Width (m)", 0.1, 0.5, (0.1, 0.5))
        width_bounds = [1, 1]
        bounds.append((width_bounds[0], width_bounds[1]))
        
        # length_bounds = st.sidebar.slider("Length (m)", 0.1, 0.5, (0.1, 0.5))
        length_bounds = [1, 1]
        bounds.append((length_bounds[0], length_bounds[1]))
        
        # Thickness
        thickness_option = st.sidebar.radio(
            "Set Thickness:", ("Fix to a constant value", "Define a range")
        )
        if thickness_option == "Fix to a constant value":
            fixed_thickness = st.sidebar.number_input(
                "Thickness (Fixed, m)", min_value=0.01, max_value=0.06, value=0.02
            )
            bounds.append((fixed_thickness, fixed_thickness))
        else:
            thickness_bounds = st.sidebar.slider(
                "Thickness Range (m)", 0.01, 0.06, (0.01, 0.06)
            )
            bounds.append((thickness_bounds[0], thickness_bounds[1]))
        
        # moss_index_bounds = st.sidebar.slider("Moss Species Index", 0, 1, (0, 1))
        moss_index_bounds=[0, 1]
        bounds.append((moss_index_bounds[0], moss_index_bounds[1]))
        
        st.sidebar.subheader("Rainfall Data")
        # Rainfall data input
        uploaded_file = st.sidebar.file_uploader("Upload TMY Rainfall Data (m/hr)", type="csv")
        if uploaded_file:
            rainfall_data = pd.read_csv(uploaded_file, header=None, skiprows=1, names=["Rainfall_m_per_hr"])
        else:
            st.sidebar.info("Using example rainfall data (Knoxville, TN).")
            rainfall_data = pd.read_csv('USA_TN_Knoxville.Downtown.Island.AP.723264_TMYx.rain', header=None, skiprows=1, names=["Rainfall_m_per_hr"])
        
        # Moss requirements
        st.sidebar.subheader("Moss Growth Requirements")
        num_species = st.sidebar.number_input("Number of Moss Species", min_value=1, max_value=10, value=2)
        moss_species = []
        for i in range(num_species):
            st.sidebar.subheader(f"Species {i+1}")
            name = st.sidebar.text_input(f"Species {i+1} Name", value=f"Species {i+1}")
            min_saturation = st.sidebar.slider(f"Species {i+1} Min Saturation Level", 0.5, 1.0, 0.3)
            min_time_fraction = st.sidebar.slider(f"Species {i+1} Min Time Fraction", 0.5, 1.0, 0.95)
            moss_species.append({'name': name, 'min_saturation': min_saturation, 'min_time_fraction': min_time_fraction})
        
        # Optimization settings
        # st.sidebar.subheader("Optimization Algorithm Settings")
        iterations = 100
        # iterations = st.sidebar.number_input("Max Iterations", min_value=50, max_value=200, value=100)
        population_size = 15
        # population_size = st.sidebar.number_input("Population Size", min_value=10, max_value=50, value=15)
        
        # Button to run optimization
        if st.button("Start"):
            # Train regression models
            models, scaler = train_regression_models(features_file_list, target_columns=["E_avg", "k"])
            # Modify the objective function to include moss_species and rainfall_data
            # def updated_objective_function(params):
                # return objective_function(params, moss_species, rainfall_data)
    
            # Perform optimization
            result = differential_evolution(
                objective_function,
                bounds=bounds,
                args=(moss_species, rainfall_data, models, scaler),
                strategy='best1bin',
                maxiter=iterations,
                popsize=population_size,
                mutation=(0.5, 1.0),
                recombination=0.7,
                workers=8,  # Set to -1 for parallel processing
                polish=True
            )
    
            # Extract optimized parameters
            optimal_params = result.x
            optimal_fitness = result.fun
            st.write("Optimization Completed.")
            # Extract and format optimal results
            optimal_min_radius = round_to_nearest(optimal_params[2])
            optimal_max_radius = round_to_nearest(optimal_params[3])
            optimal_moss_index = map_moss_index_uniformly(optimal_params[8], len(moss_species))
            optimal_moss_species = moss_species[optimal_moss_index]
            
            # Create a dictionary for optimal results
            optimal_results = {
                "Parameter": [
                    "Mean Radius (μm)",
                    "Std Radius (μm)",
                    "Min Radius (μm)",
                    "Max Radius (μm)",
                    "Porosity",
                    # "Width (m)",
                    # "Length (m)",
                    "Thickness (m)",
                    "Optimal Moss Species"
                ],
                "Value": [
                    f"{optimal_params[0] * 1e6:.3f}",
                    f"{optimal_params[1] * 1e6:.3f}",
                    f"{optimal_min_radius * 1e6:.1f}",
                    f"{optimal_max_radius * 1e6:.1f}",
                    f"{optimal_params[4]:.3f}",
                    # f"{optimal_params[5]:.3f}",
                    # f"{optimal_params[6]:.3f}",
                    f"{optimal_params[7]:.3f}",
                    optimal_moss_species['name']
                ]
            }
            
            # Convert to a DataFrame for display
            optimal_results_df = pd.DataFrame(optimal_results)
            
            # Display the table in Streamlit
            st.subheader("Optimal Parameters")
            # st.table(optimal_results_df)
            st.dataframe(optimal_results_df, use_container_width=True)
    
            # Calculate saturation history
            optimal_substrate_params = {
                "mean_radius": optimal_params[0],
                "std_radius": optimal_params[1],
                "min_radius": optimal_min_radius,
                "max_radius": optimal_max_radius,
                "porosity": optimal_params[4],
                "width": optimal_params[5],
                "length": optimal_params[6],
                "thickness": optimal_params[7]
            }
            # Extract moss-specific requirements
            required_saturation = optimal_moss_species['min_saturation']
            required_time = optimal_moss_species['min_time_fraction']
            
            saturation_history = calculate_saturation_history(optimal_substrate_params, rainfall_data, models, scaler)
            satisfied = saturation_history >= required_saturation
            achieved_fraction = np.sum(satisfied) / len(saturation_history)
            
            # Input parameters
            mean_radius = optimal_params[0]
            std_radius = optimal_params[1]
            min_radius = optimal_min_radius
            max_radius = optimal_max_radius
            porosity = optimal_params[4]
            # Domain dimensions
            domain_width = 0.5e-3  # Width in meters
            domain_height = 0.5e-3  # Height in meters
            domain_area = domain_width * domain_height
            
            # Target solid fraction
            solid_fraction = 1 - porosity
            target_solid_area = solid_fraction * domain_area
            
            # Match the packing
            radii = match_packing(porosity, mean_radius, std_radius, min_radius, max_radius, domain_area)
            positions = generate_positions(radii, domain_width, domain_height)
            
            # Plot saturation history
            st.subheader("Time-history of Saturation Level for Substrate with Optimal Parameters")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(saturation_history, label="Substrate Saturation Level")
            ax.axhline(moss_species[map_moss_index_uniformly(optimal_params[8], num_species)]['min_saturation'], color='r', linestyle='--', label="Moss Requirement")
            ax.set_xlabel("Time (Hours)")
            ax.set_ylabel("Saturation Level")
            ax.legend()
            ax.text(
                0.05, 0.95,  # Position in axis coordinates (x, y)
                f"Time Fraction Above Requirement: {achieved_fraction:.1%}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.5)
            )
            st.pyplot(fig)
            
            # Plot particle arrangement
            st.subheader("Representative Micro Structure for Substrate")
            fig, ax = plt.subplots(figsize=(6, 6))
            for (x, y), r in zip(positions, radii):
                circle = plt.Circle((x*1e3, y*1e3), r*1e3, edgecolor='none', facecolor='black')
                ax.add_patch(circle)
            ax.set_xlim(0, domain_width*1e3)
            ax.set_ylim(0, domain_height*1e3)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_aspect('equal')
            st.pyplot(fig)
            
            st.subheader("Solid Particle Size Distribution in Representative Micro Structure for Substrate")
            # Display histogram of particle sizes
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(radii*1e6, bins=10, edgecolor='black')
            ax.set_xlabel("Radius (μm)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
