#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:31:15 2024

@author: jzhan124
"""

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution

def train_regression_models(features_file_list, target_columns):
    """
    Train Random Forest regression models to predict evaporation parameters.
    
    :param features_file_list: List of feature CSV file paths to be merged.
    :param target_columns: List of target columns (e.g., ["E_avg", "k"]).
    :return: Trained models and scaler.
    """
    # Merge feature data
    feature_dfs = [pd.read_csv(file) for file in features_file_list]
    features_data = pd.concat(feature_dfs, ignore_index=True)

    # Load evaporation model data
    evaporation_dfs = [pd.read_csv(f'evaporation_model_fit_{i+1}.csv') for i in range(len(features_file_list))]
    model_fit_data = pd.concat(evaporation_dfs, ignore_index=True)

    # Merge datasets
    merged_data = pd.concat([model_fit_data, features_data], axis=1)

    # Define features and targets
    X = merged_data[['mean_radius', 'std_radius', 'min_radius', 'max_radius', 'porosity', ]]
    y = merged_data[target_columns]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=8)

    # Train models for each target
    models = {}
    for target in y.columns:
        model = RandomForestRegressor(n_estimators=200, random_state=8)
        model.fit(X_train, y_train[target])
        models[target] = model

    # Print evaluation metrics
    for target, model in models.items():
        print(f"{target} R^2 on test data:", model.score(X_test, y_test[target]))

    return models, scaler

def predict_evaporation_params(substrate_params, models, scaler):
    """
    Predict evaporation parameters using trained Random Forest models.
    
    :param substrate_params: Dictionary of substrate properties.
    :param models: Trained Random Forest models.
    :param scaler: Scaler used during training.
    :return: Predicted E_avg and k, with S_critical set as a constant.
    """
    # Convert substrate_params to a DataFrame with appropriate feature names
    feature_vector = pd.DataFrame([substrate_params], columns=['mean_radius', 'std_radius', 'min_radius', 'max_radius', 'porosity'])

    # Scale features
    feature_vector_scaled = scaler.transform(feature_vector)

    # Predict evaporation parameters
    E_avg = models['E_avg'].predict(feature_vector_scaled)[0]
    k = models['k'].predict(feature_vector_scaled)[0]
    S_critical = 0.8  # Constant value

    return E_avg, k, S_critical

# Define evaporation rate function
def evaporation_rate(S, E_avg, S_critical, k):
    return E_avg / (0.5 + np.exp(-k * (S - S_critical)))

# Simulate time-history of saturation level
def calculate_saturation_history(substrate_params, rainfall, evap_models, evap_scaler):
    """
    Calculate the time-history of saturation level using Random Forest predictions.
    """
    L, W, H = substrate_params['length'], substrate_params['width'], substrate_params['thickness']
    phi = substrate_params['porosity']
    V_p = phi * L * W * H  # Total pore volume (m^3)
    A_p = W * L  # Surface area (m^2)
    S_max = 0.5  # Fully saturated
    A_max = 0.001  # Maximum absorption rate into substrate (kg/m^2s)
    rho_w = 1000 #water density
    
    # Predict E_avg and k using the regression models
    E_avg, k, S_critical = predict_evaporation_params(substrate_params, evap_models, evap_scaler)

    # Simulation parameters
    ts = 1.0  # Time step (hour)
    total_steps = len(rainfall)

    # Initialize saturation and mass history
    time_his = np.arange(0, total_steps, ts)
    saturation = np.zeros_like(time_his)
    saturation[0] = S_max
    mass_history = np.zeros_like(time_his)
    mass_history[0] = saturation[0] * V_p * rho_w  # Convert m^3 to kg

    # Simulation loop
    for i in range(1, len(time_his)):
        potential_absorption = rainfall["Rainfall_m_per_hr"].iloc[i] * A_p * rho_w
        rain = min(potential_absorption, A_max * (1 - saturation[i-1]) * A_p * ts * 3600)
        evaporation = evaporation_rate(saturation[i-1], E_avg, S_critical, k) * ts * A_p * 3600

        # Update mass and saturation
        mass_history[i] = mass_history[i-1] - evaporation + rain
        saturation[i] = mass_history[i] / (V_p * rho_w)
        saturation[i] = max(0, min(1, saturation[i]))  # Keep saturation within bounds

    return saturation

def map_moss_index_uniformly(value, num_species):
    """
    Uniformly map a continuous value to an integer moss index.
    
    :param value: Continuous value in [0, 1] range.
    :param num_species: Total number of moss species.
    :return: Integer moss index.
    """
    # Calculate the interval size
    interval_size = 1.0 / num_species

    # Find the corresponding index
    index = int(value / interval_size)

    # Handle edge case where value == 1.0
    index = min(index, num_species - 1)

    return index

def round_to_nearest(value, step=1e-7):
    """
    Round a value to the nearest step (e.g., 0.1 micron).
    
    :param value: The value to round.
    :param step: The step size for rounding (default: 0.1 micron = 1e-7 meters).
    :return: Rounded value.
    """
    return np.round(value / step) * step


# Define the objective function
def objective_function(params, moss_species, rainfall_data, models, scaler):
    """
    Objective function for optimizing porous parameters and moss species.
    """
    # mean_radius, std_radius, min_radius, max_radius, porosity, total_surface_area, width, length, thickness, moss_index_continuous = params
    mean_radius, std_radius, min_radius, max_radius, porosity,  width, length, thickness, moss_index_continuous = params
    
    # Round min and max radius to the nearest 0.1 micron
    min_radius = round_to_nearest(min_radius)
    max_radius = round_to_nearest(max_radius)

    # Add a penalty for invalid radius conditions
    if not (min_radius <= mean_radius <= max_radius):
        penalty = 1e6  # Large penalty for violating constraints
        return penalty
    
    substrate_params = {
        "mean_radius": mean_radius,
        "std_radius": std_radius,
        "min_radius": min_radius,
        "max_radius": max_radius,
        # "mean_nn_distance": mean_nn_distance,
        "porosity": porosity,
        # "total_surface_area": total_surface_area,
        "width": width,
        "length": length,
        "thickness": thickness
    }
    # Map the continuous moss index to an integer
    moss_index = map_moss_index_uniformly(moss_index_continuous, len(moss_species))
    moss_requirement = moss_species[moss_index]

    # Extract moss-specific requirements
    required_saturation = moss_requirement['min_saturation']
    required_time = moss_requirement['min_time_fraction']
    # Calculate saturation levels
    saturation = calculate_saturation_history(substrate_params, rainfall_data, models, scaler)

    satisfied = saturation >= required_saturation
    achieved_fraction = np.sum(satisfied) / len(saturation)

    # Calculate penalty
    penalty = max(0, required_time - achieved_fraction) * 1000
    return penalty

# Truncated normal distribution generator
def truncated_normal(mean, std, min_val, max_val, size):
    a, b = (min_val - mean) / std, (max_val - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

# Iterative adjustment of packing
def match_packing(target_porosity, mean_radius, std_radius, min_radius, max_radius, domain_area, max_iter=10000, tolerance=1e-4):
    num_particles = 1000  # Initial number of particles
    for _ in range(max_iter):
        # Generate radii
        radii = truncated_normal(mean_radius, std_radius, min_radius, max_radius, num_particles)
        radii = round_to_nearest(radii)
        particle_areas = np.pi * radii**2

        # Calculate porosity and surface area
        current_solid_area = np.sum(particle_areas)
        current_porosity = 1 - (current_solid_area / domain_area)

        # Check if both criteria are met
        if abs(current_porosity - target_porosity) < tolerance :
            return radii

        # Adjust the number of particles
        if current_porosity < target_porosity:
            num_particles -= int(0.1 * num_particles)  # Reduce particles
        else:
            num_particles += int(0.1 * num_particles)  # Increase particles

        # Adjust radii distribution if surface area is mismatchedd

    raise ValueError("Failed to match packing within the maximum number of iterations.")

# Generate particle positions
def generate_positions(radii, domain_width, domain_height):
    positions = []
    for r in radii:
        valid_position = False
        while not valid_position:
            # Generate random position within the domain
            x = np.random.uniform(r, domain_width - r)
            y = np.random.uniform(r, domain_height - r)

            # Check for non-overlap with existing particles
            if all(
                np.linalg.norm(np.array([x, y]) - np.array(pos)) >= r + radii[j] + 1e-6
                for j, pos in enumerate(positions)
            ):
                positions.append((x, y))
                valid_position = True
    return np.array(positions)

if __name__ == "__main__":
    
    # File paths for feature datasets
    features_file_list = [
        'solid_particles_features_1.csv',
        'solid_particles_features_2.csv',
        'solid_particles_features_3.csv'
    ]
    
    # Train regression models
    models, scaler = train_regression_models(features_file_list, target_columns=["E_avg", "k"])
    
    # Load rainfall data
    rainfall_data = pd.read_csv('USA_TN_Knoxville.Downtown.Island.AP.723264_TMYx.rain',header=None, skiprows=1, names=["Rainfall_m_per_hr"])  #
    
    # Define moss species requirements
    moss_species = [
        {'name': 'Species A', 'min_saturation': 0.3, 'min_time_fraction': 0.95},
        {'name': 'Species B', 'min_saturation': 0.4, 'min_time_fraction': 0.90},
        {'name': 'Species C', 'min_saturation': 0.5, 'min_time_fraction': 0.85},
        # Add more moss species here
    ]
    
    bounds = [
        (6e-6, 1.5e-5),  # Mean radius
        (0.1e-6, 3.5e-6),  # Std radius
        (5e-6, 8e-6),    # Min radius
        (7e-6, 2e-5),    # Max radius
        # (5e-6, 1e-5),    # Mean nearest neighbor distance
        (0.6, 0.7),      # Porosity
        # (0.005, 0.015),  # Total surface area
        (0.1, 0.5),      # Width (meters)
        (0.1, 0.5),      # Length (meters)
        (0.01, 0.03),    # Thickness (meters)
        (0, 1)  # Moss species index
    ]
    
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(moss_species, rainfall_data, models, scaler),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7,
        workers=8,
        polish=True  # Optional: Perform local optimization at the end
    )
    
    # Extract the optimal solution
    optimal_params = result.x
    optimal_fitness = result.fun
    
    # Map the moss species index to its corresponding species
    optimal_moss_index = map_moss_index_uniformly(optimal_params[-1], len(moss_species))
    optimal_moss_species = moss_species[optimal_moss_index]
    
    optimal_min_radius = round_to_nearest(optimal_params[2])
    optimal_max_radius = round_to_nearest(optimal_params[3])
    
    print("Optimal Parameters:")
    print(f"  Mean radius: {optimal_params[0] * 1e6:.3f} microns")
    print(f"  Std radius: {optimal_params[1] * 1e6:.3f} microns")
    print(f"  Min Radius: {optimal_min_radius * 1e6:.1f} microns")
    print(f"  Max Radius: {optimal_max_radius * 1e6:.1f} microns")
    # print(f"  Min radius: {optimal_params[2]:.4e}")
    # print(f"  Max radius: {optimal_params[3]:.4e}")
    # print(f"  Mean nearest neighbor distance: {optimal_params[3]:.4e}")
    print(f"  Porosity: {optimal_params[4]:.3f}")
    # print(f"  Total surface area: {optimal_params[5]:.4e}")
    print(f"  Width: {optimal_params[5]:.3f}")
    print(f"  Length: {optimal_params[6]:.3f}")
    print(f"  Thickness: {optimal_params[7]:.3f}")
    print(f"Optimal Moss Species: {optimal_moss_species['name']}")
    
    # Calculate saturation history
    optimized_saturation = calculate_saturation_history({
        "mean_radius": optimal_params[0],
        "std_radius": optimal_params[1],
        "min_radius": optimal_min_radius,
        "max_radius": optimal_max_radius,
        # "mean_nn_distance": optimal_params[3],
        "porosity": optimal_params[4],
        # "total_surface_area": optimal_params[5],
        "width": optimal_params[5],
        "length": optimal_params[6],
        "thickness": optimal_params[7]
    }, rainfall_data, models, scaler)
    
    # Get moss requirements
    moss_requirement = moss_species[optimal_moss_index]
    required_saturation = moss_requirement['min_saturation']
    
    time_steps = np.arange(len(optimized_saturation))
    
    plt.figure(figsize=(12*0.6, 6*0.6))
    plt.plot(time_steps, optimized_saturation, label='Saturation Level', color='b')
    plt.axhline(required_saturation, color='r', linestyle='--', label='Moss Minimum Saturation')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Saturation Level')
    plt.title('Saturation Time History for Optimized Parameters')
    plt.legend()
    plt.grid(True)
    plt.show()
    
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
    
    # Plot particle arrangement
    plt.figure(figsize=(5, 5))
    for (x, y), r in zip(positions, radii):
        circle =  plt.Circle((x, y), r, edgecolor='none', facecolor='black')
        plt.gca().add_patch(circle)
    plt.xlim(0, domain_width)
    plt.ylim(0, domain_height)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title("2D Particle Packing with Matched Porosity and Surface Area")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()
    
    # Validation
    actual_solid_area = np.sum(np.pi * radii**2)
    actual_porosity = 1 - (actual_solid_area / domain_area)
    actual_surface_area = np.sum(2 * np.pi * radii)
    print(f"Target Porosity: {porosity:.4f}, Actual Porosity: {actual_porosity:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.hist(radii, bins=20, edgecolor='black', color='blue')  # Convert radii to microns for readability
    plt.title("Particle Size Distribution")
    plt.xlabel("Radius (microns)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()