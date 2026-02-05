import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import csv

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import ticker 

# Parameters from Influenza SVIRS model (Pathogen 1)
beta1 = 2.668837
mu1 = 1.022717
xi1 = 0.010820

# Parameters from HMPV SIRS model (Pathogen 2)
beta2 = 6.281854
mu2 = 1.251611  
xi2 = 0.003081  

# Other fixed parameters in the model
theta1 = 0.360863 
upsilon = xi1
print(f"Using vaccine waning rate (upsilon) = {upsilon:.6f}")

try:
    df = pd.read_csv('data_Weekly_all.csv')
    print("Successfully loaded 'data_Weekly_all.csv'.")
except FileNotFoundError:
    print("Error: 'data_Weekly_all.csv' not found.")
    exit()

try:
    # Pathogen 1 (Influenza)
    inf_start_row = df[(df['Year'] == 2023) & (df['Week'] == 40)].iloc[0]
    I1_0_fitted = (inf_start_row['Weekly_InfA_Positive_Rate'] + inf_start_row['Weekly_InfB_Positive_Rate']) / 1000
    print(f"Influenza I0 (2023-W40) read from data: {I1_0_fitted:.6f}")

    # Pathogen 2 (HMPV) 
    hmpv_start_row = df[(df['Year'] == 2023) & (df['Week'] == 44)].iloc[0]
    I2_0_fitted = hmpv_start_row['Weekly_HMPV_Positive_Rate'] / 1000 # /100 /10 is /1000
    print(f"HMPV I0 (2023-W44) read from data: {I2_0_fitted:.6f}")
except (IndexError, KeyError):
    print("Error: Could not find required start weeks.")
    exit()

# --- Initial conditions for the model ---
# Pathogen 1 (Influenza)
V1_0_fitted = 0.2
R1_0_fitted = 0.404581
S1_0_fitted = 1.0 - V1_0_fitted - I1_0_fitted - R1_0_fitted

# Pathogen 2 (HMPV) 
R2_0_fitted = 0.767738  
S2_0_fitted = 1.0 - I2_0_fitted - R2_0_fitted

# Allocate initial I1 and R1 populations proportionally
total_susceptible_flu = S1_0_fitted + V1_0_fitted
if total_susceptible_flu > 0:
    prop_from_S = S1_0_fitted / total_susceptible_flu
    prop_from_V = V1_0_fitted / total_susceptible_flu
else:
    prop_from_S, prop_from_V = 1.0, 0.0

I1_u_0 = I1_0_fitted * prop_from_S
I1_v_0 = I1_0_fitted * prop_from_V
R1_u_0 = R1_0_fitted * prop_from_S
R1_v_0 = R1_0_fitted * prop_from_V

y0_unnormalized = np.array([
    S1_0_fitted * S2_0_fitted,     # 0: SS
    I1_u_0 * S2_0_fitted,          # 1: IS
    R1_u_0 * S2_0_fitted,          # 2: RS
    V1_0_fitted * S2_0_fitted,     # 3: VS
    I1_v_0 * S2_0_fitted,          # 4: IS_V
    R1_v_0 * S2_0_fitted,          # 5: RS_V
    R1_u_0 * I2_0_fitted,          # 6: RI
    R1_v_0 * I2_0_fitted,          # 7: RI_V
    R1_u_0 * R2_0_fitted,          # 8: RR
    R1_v_0 * R2_0_fitted,          # 9: RR_V
    S1_0_fitted * I2_0_fitted,     # 10: SI
    S1_0_fitted * R2_0_fitted,     # 11: SR
    V1_0_fitted * I2_0_fitted,     # 12: VI
    V1_0_fitted * R2_0_fitted,     # 13: VR
])
y0 = y0_unnormalized / np.sum(y0_unnormalized)
print("SS: {:.6f}, IS: {:.6f}, RS: {:.6f}, VS: {:.6f}, IS_V: {:.6f}, RS_V: {:.6f}, RI: {:.6f}, RI_V: {:.6f}, RR: {:.6f}, RR_V: {:.6f}, SI: {:.6f}, SR: {:.6f}, VI: {:.6f}, VR: {:.6f}".format(*y0))

def coinfection_model_14comp(y, t, beta1, mu1, xi1, beta2, mu2, xi2, upsilon, theta1, theta2, lambda12):
    SS, IS, RS, VS, IS_V, RS_V, RI, RI_V, RR, RR_V, SI, SR, VI, VR = y
    N = np.sum(y)
    if N <= 1e-9: return np.zeros(14)

    I_flu_total = IS + IS_V
    I_p2_total = VI + SI + RI + RI_V

    dSSdt = -beta1 * SS * I_flu_total / N - beta2 * SS * I_p2_total / N + xi1 * RS + xi2 * SR + upsilon * VS
    dISdt = beta1 * SS * I_flu_total / N - mu1 * IS
    dRSdt = mu1 * IS - lambda12 * beta2 * RS * I_p2_total / N + xi2 * RR - xi1 * RS
    dVSdt = -theta1 * beta1 * VS * I_flu_total / N - theta2 * beta2 * VS * I_p2_total / N - upsilon * VS + xi1 * RS_V + xi2 * VR
    dIS_Vdt = theta1 * beta1 * VS * I_flu_total / N - mu1 * IS_V
    dRS_Vdt = mu1 * IS_V - lambda12 * beta2 * RS_V * I_p2_total / N + xi2 * RR_V - xi1 * RS_V
    dRIdt = lambda12 * beta2 * RS * I_p2_total / N - mu2 * RI
    dRI_Vdt = lambda12 * beta2 * RS_V * I_p2_total / N - mu2 * RI_V
    dRRdt = mu2 * RI - xi2 * RR
    dRR_Vdt = mu2 * RI_V - xi2 * RR_V
    dSIdt = beta2 * SS * I_p2_total / N - mu2 * SI
    dSRdt = mu2 * SI - xi2 * SR
    dVIdt = theta2 * beta2 * VS * I_p2_total / N - mu2 * VI
    dVRdt = mu2 * VI - xi2 * VR

    return [dSSdt, dISdt, dRSdt, dVSdt, dIS_Vdt, dRS_Vdt, dRIdt, dRI_Vdt, dRRdt, dRR_Vdt, dSIdt, dSRdt, dVIdt, dVRdt]

print("\nSetting up 2D sensitivity analysis (14-Compartment Model)...")
start, stop, step = 0.5, 1.5, 0.01
lambda_grid = np.round(np.arange(start, stop + step/2, step), 2)
theta2_grid = np.round(np.arange(start, stop + step/2, step), 2)
t = np.arange(0, 52)

ratio_ri_vi_matrix = np.zeros((len(theta2_grid), len(lambda_grid)))

print("\nRunning 2D sensitivity simulations (this will take a while)...")
for i, theta2_val in enumerate(theta2_grid):
    if i % 10 == 0 or i == len(theta2_grid) - 1:
        print(f"   Processing row {i+1}/{len(theta2_grid)} (theta2 = {theta2_val:.2f})...")
        
    for j, lam_val in enumerate(lambda_grid):
        # --- Simulation 1: Main ---
        args_main = (beta1, mu1, xi1, beta2, mu2, xi2, upsilon, theta1, theta2_val, lam_val)
        sol_main = odeint(coinfection_model_14comp, y0, t, args=args_main)
        VS_m = sol_main[:, 3]; RS_V_m = sol_main[:, 5]
        N_m = np.sum(sol_main, axis=1); N_m[N_m == 0] = 1
        I_p2_m = sol_main[:, 12] + sol_main[:, 10] + sol_main[:, 6] + sol_main[:, 7]
        flux_VI_main = theta2_val * beta2 * VS_m * I_p2_m / N_m
        flux_RI_V_main = lam_val * beta2 * RS_V_m * I_p2_m / N_m
        C_VI_main = np.sum(flux_VI_main)
        C_RI_V_main = np.sum(flux_RI_V_main)

        # --- Simulation 2: RI Baseline (lambda = 1) ---
        args_ri_base = (beta1, mu1, xi1, beta2, mu2, xi2, upsilon, theta1, theta2_val, 1.0)
        sol_ri_base = odeint(coinfection_model_14comp, y0, t, args=args_ri_base)
        RS_V_ri_b = sol_ri_base[:, 5]
        N_ri_b = np.sum(sol_ri_base, axis=1); N_ri_b[N_ri_b == 0] = 1
        I_p2_ri_b = sol_ri_base[:, 12] + sol_ri_base[:, 10] + sol_ri_base[:, 6] + sol_ri_base[:, 7]
        flux_RI_V_ri_base = 1.0 * beta2 * RS_V_ri_b * I_p2_ri_b / N_ri_b
        C_RI_V_baseline = np.sum(flux_RI_V_ri_base)

        # --- Simulation 3: VI Baseline (theta2 = 1) ---
        args_vi_base = (beta1, mu1, xi1, beta2, mu2, xi2, upsilon, theta1, 1.0, lam_val)
        sol_vi_base = odeint(coinfection_model_14comp, y0, t, args=args_vi_base)
        VS_vi_b = sol_vi_base[:, 3]
        N_vi_b = np.sum(sol_vi_base, axis=1); N_vi_b[N_vi_b == 0] = 1
        I_p2_vi_b = sol_vi_base[:, 12] + sol_vi_base[:, 10] + sol_vi_base[:, 6] + sol_vi_base[:, 7]
        flux_VI_vi_base = 1.0 * beta2 * VS_vi_b * I_p2_vi_b / N_vi_b
        C_VI_baseline = np.sum(flux_VI_vi_base)

        # Calculate deltas and then the ratio of their absolute values
        numerator_delta = C_RI_V_main - C_RI_V_baseline
        denominator_delta = C_VI_main - C_VI_baseline
        
        abs_numerator = np.abs(numerator_delta)
        abs_denominator = np.abs(denominator_delta)
        
        ratio = abs_numerator / abs_denominator if abs_denominator > 1e-12 else np.nan
        ratio_ri_vi_matrix[i, j] = ratio

print("2D analysis complete.")

# Save the heatmap data to a CSV file 
output_filename = 'heatmap_14comp_abs_ratio_inf_hmpv_step001.csv'
print(f"\nSaving heatmap data to '{output_filename}'...")

with open(output_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    header = ['theta2/lambda'] + list(lambda_grid)
    writer.writerow(header)
    
    for i, theta2_val in enumerate(theta2_grid):
        row_to_write = [theta2_val] + list(ratio_ri_vi_matrix[i, :])
        writer.writerow(row_to_write)
print(f"Successfully saved data to '{output_filename}'.")


print("\nGenerating heatmap for the ratio of effect magnitudes...")
user_colors = ['#D93F49', '#E28187', '#EBBFC2', '#D5E1E3', '#AFC9CF', '#8FB4BE']
custom_cmap = LinearSegmentedColormap.from_list("custom_div_cmap", user_colors)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))


with np.errstate(divide='ignore'): 
    plot_data_log10 = np.log10(ratio_ri_vi_matrix)

lambda_extent = [lambda_grid[0] - step/2, lambda_grid[-1] + step/2]
theta2_extent = [theta2_grid[0] - step/2, theta2_grid[-1] + step/2]
extent = lambda_extent + theta2_extent

if np.any(np.isfinite(plot_data_log10)):
    v_min_log = np.nanmin(plot_data_log10[np.isfinite(plot_data_log10)])
    v_max_log = np.nanmax(plot_data_log10[np.isfinite(plot_data_log10)])


    plot_data_log10[np.isneginf(plot_data_log10)] = v_min_log
    norm = TwoSlopeNorm(vcenter=0, vmin=v_min_log, vmax=v_max_log)

    im = ax.imshow(plot_data_log10, aspect='auto', origin='lower', extent=extent, cmap=custom_cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    formatter = ticker.FuncFormatter(lambda x, pos: 
                                     "1" if x == 0 else (f"$10^{{{int(x)}}}$" if x == int(x) else f"$10^{{{x:.1f}}}$"))
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.tick_params(labelsize=28) 
    cbar.set_label('Ratio (Indirect/Direct Effect)', fontsize=28)

    contour_pos = ax.contour(lambda_grid, theta2_grid, ratio_ri_vi_matrix, levels=[1.0], colors='white', linestyles='--')
    ax.clabel(contour_pos, inline=True, fmt='Ratio = 1.0', fontsize=24)

    # --- Plot HMPV data point as "米" shape ---
    marker_lambda = 1.164  
    marker_theta2 = 1.146403  

    marker_size = 100    
    line_width = 2       
    
    # 1. Plot the (+) part
    ax.scatter(marker_lambda, marker_theta2, 
               marker='+', 
               color='yellow', 
               s=marker_size,
               linewidths=line_width, 
               zorder=5)
    
    # 2. Plot the (x) part
    ax.scatter(marker_lambda, marker_theta2, 
               marker='x', 
               color='yellow', 
               s=marker_size,
               linewidths=line_width, 
               zorder=5) 
    
    # Add the text label
    ax.text(marker_lambda + 0.015, marker_theta2, 'HMPV',  
            ha='left', 
            va='center', 
            color='black', 
            fontsize=24,
            fontweight='bold',
            zorder=6) 
    
else:
    print("Warning: Cannot generate a valid color scale. Plotting without color normalization.")
    im = ax.imshow(ratio_ri_vi_matrix, aspect='auto', origin='lower', extent=extent, cmap=custom_cmap)

ax.set_title('HMPV', fontsize=28) 


ax.set_xlabel(r'$\lambda_{12}$', fontsize=28)
ax.set_ylabel(r'$\theta_2$', fontsize=28) 

tick_step = 0.2
ax.set_xticks(np.arange(start, stop + step/2, tick_step))
ax.set_yticks(np.arange(start, stop + step/2, tick_step))
ax.tick_params(axis='both', which='major', labelsize=28)

plt.tight_layout()
plt.savefig("heatmap_14comp_abs_ratio_inf_hmpv_step001.png") 
plt.show()

print("\nScript finished. Heatmap plot saved as 'heatmap_14comp_abs_ratio_inf_hmpv_step001.png'") 