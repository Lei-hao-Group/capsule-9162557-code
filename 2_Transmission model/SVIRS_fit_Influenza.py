import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter 
from scipy.integrate import odeint
from scipy.optimize import minimize
import emcee

# --- Global Font Settings ---
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'lines.linewidth': 2.5
})

# --- Helper Functions ---
def calculate_diagnostics(chain, burn_in, param_names):
    samples = chain[burn_in:, :, :]
    n_steps, n_walkers, n_params = samples.shape
    results = {}
    for i in range(n_params):
        param_samples = samples[:, :, i]
        W = np.mean(np.var(param_samples, axis=0, ddof=1))
        walker_means = np.mean(param_samples, axis=0)
        grand_mean = np.mean(walker_means)
        B = (n_steps / (n_walkers - 1)) * np.sum((walker_means - grand_mean)**2)
        var_plus = (n_steps - 1) / n_steps * W + (1 / n_steps) * B
        r_hat = np.sqrt(var_plus / W) if W > 0 else 1.0
        
        flat_chain_for_ess = param_samples.flatten()
        total_var = np.var(flat_chain_for_ess, ddof=1)
        sum_rho = 0
        if total_var > 0:
            for t_lag in range(1, n_steps):
                demeaned_samples = param_samples - grand_mean
                autocov = np.sum(demeaned_samples[:-t_lag, :] * demeaned_samples[t_lag:, :])
                rho_t = autocov / ((n_steps - t_lag) * n_walkers * total_var)
                if rho_t > 0: sum_rho += rho_t
                else: break
        ess = (n_walkers * n_steps) / (1 + 2 * sum_rho) if (1 + 2 * sum_rho) != 0 else 0
        q2_5, q97_5 = np.percentile(flat_chain_for_ess, [2.5, 97.5])
        results[param_names[i]] = {'mean': grand_mean, 'sd': np.sqrt(var_plus), 
                                   '2.5%': q2_5, '97.5%': q97_5, 'rhat': r_hat, 'ess': ess}
    return results

def calculate_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)

# --- Data Loading (INF Data) ---
try:
    df = pd.read_csv('data_Weekly_all.csv')
    print("Successfully loaded 'data_Weekly_all.csv'.")
except FileNotFoundError:
    print("Error: 'data_Weekly_all.csv' not found.")
    exit()

try:
    # Data filtering logic: 2023-W40 to 2024-W39
    start_index = df[(df['Year'] == 2023) & (df['Week'] == 40)].index[0]
    end_rows = df[(df['Year'] == 2024) & (df['Week'] == 39)]
    
    if not end_rows.empty:
        end_index = end_rows.index[0]
        data_slice = df.loc[start_index:end_index]
    else:
        data_slice = df.loc[start_index:]
        print("Warning: 2024-W39 not found, using data until the end of the file.")
    
    # INF data merging logic
    observed_data = (data_slice['Weekly_InfA_Positive_Rate'] + data_slice['Weekly_InfB_Positive_Rate']).to_numpy() / 100 / 10
    print(f"Data filtered from week {data_slice['YearWeek'].iloc[0]} to {data_slice['YearWeek'].iloc[-1]}. Total weeks: {len(observed_data)}")
except (IndexError, KeyError):
    print("Error: Could not find the specified time range or columns for INF data."); exit()

time = np.arange(len(observed_data))

# --- Fixed Parameters ---
V0_fixed = 0.2
I0_fixed = max(observed_data[0], 1e-6)
e_fixed = 0.001
theta_fixed = 0.371396895216788

# --- Model Definition (SVIRS) ---
def svirs_model(y, t, beta, mu, xi, e, theta):
    S, V, I, R = y
    N = S + V + I + R
    if N <= 0: return [0, 0, 0, 0]
    dSdt = -e * S - (beta * I * S / N) + xi * R
    dVdt =  e * S - (beta * theta * I * V / N)
    dIdt = (beta * I * S / N) + (beta * theta * I * V / N) - mu * I
    dRdt =  mu * I - xi * R
    return [dSdt, dVdt, dIdt, dRdt]

# Define parameter bounds
PRIOR_BOUNDS = {
    'beta': (0.0, 10.0), 
    'mu':   (0.0, 5.0),
    'xi':   (0.0, 1.0)
    # R0 (Initial Recovered) upper bound dynamically depends on 1 - V0 - I0
}

def log_prior(theta_params, V0, I0):
    beta, mu, xi, R0_param = theta_params # R0_param refers to Recovered at t=0
    
    if (PRIOR_BOUNDS['beta'][0] < beta < PRIOR_BOUNDS['beta'][1] and 
        PRIOR_BOUNDS['mu'][0] < mu < PRIOR_BOUNDS['mu'][1] and 
        PRIOR_BOUNDS['xi'][0] < xi < PRIOR_BOUNDS['xi'][1] and 
        0.0 <= R0_param < (1.0 - V0 - I0)):
        return 0.0
    return -np.inf

def log_likelihood(theta_params, y_obs, t, V0, I0, e, theta_eff):
    beta, mu, xi, R0_param = theta_params
    
    # Calculate S0 dynamically
    S0 = 1.0 - V0 - I0 - R0_param
    if S0 < 0: return -np.inf
    y0 = [S0, V0, I0, R0_param]
    
    try:
        solution = odeint(svirs_model, y0, t, args=(beta, mu, xi, e, theta_eff))
        S_t, V_t, I_t, R_t = solution.T
        N_t = S_t + V_t + I_t + R_t
        N_t[N_t <= 0] = 1.0
        predicted_incidence = (beta * I_t * S_t / N_t) + (beta * theta_eff * I_t * V_t / N_t)
    except Exception:
        return -np.inf

    sigma_obs = 0.001
    log_lik_data = -0.5 * np.sum((y_obs - predicted_incidence)**2 / sigma_obs**2)
    
    # Periodicity penalty
    R_final = solution[-1, 3] 
    sigma_R = 0.005 
    log_penalty_periodicity = -0.5 * ((R_final - R0_param) / sigma_R)**2
    
    return log_lik_data + log_penalty_periodicity

def log_probability(theta_params, y_obs, t, V0, I0, e, theta_eff):
    lp = log_prior(theta_params, V0, I0)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta_params, y_obs, t, V0, I0, e, theta_eff)

# --- MCMC Settings ---
n_walkers = 50
n_dim = 4
n_steps = 10000
burn_in = 3000
param_labels = ["β", "μ", "ξ", "Initial_R"]

def neg_log_prob(theta, *args): return -log_probability(theta, *args)

# ==========================================================
# [Print] Initial Settings & Constraints
# ==========================================================
print("\n" + "="*60)
print("       MODEL SETTINGS & INITIAL CONDITIONS (t=0)")
print("="*60)
print(f"Fixed Vaccinated (V0)          : {V0_fixed:.6f}")
print(f"Fixed Initial Infection I(0)   : {I0_fixed:.6f}")
print(f"Constraint for S(0) + R(0)     : {1.0 - V0_fixed - I0_fixed:.6f}")
print("-" * 60)
print("Parameter Prior Bounds:")
print(f"  beta (β)       : ({PRIOR_BOUNDS['beta'][0]}, {PRIOR_BOUNDS['beta'][1]})")
print(f"  mu   (μ)       : ({PRIOR_BOUNDS['mu'][0]}, {PRIOR_BOUNDS['mu'][1]})")
print(f"  xi   (ξ)       : ({PRIOR_BOUNDS['xi'][0]}, {PRIOR_BOUNDS['xi'][1]})")
print(f"  Initial_R      : [0.0, {1.0 - V0_fixed - I0_fixed:.6f})")
print("="*60 + "\n")

initial_guess = np.array([2.0, 7/5, 0.01, 0.1])
sampler_args = (observed_data, time, V0_fixed, I0_fixed, e_fixed, theta_fixed)
soln = minimize(neg_log_prob, initial_guess, args=sampler_args)
print("Initial guess from optimization:", soln.x)
initial_pos = soln.x + 1e-4 * np.random.randn(n_walkers, n_dim)

sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, args=sampler_args)
print("\nStarting emcee sampling for INF SVIRS model...")
sampler.run_mcmc(initial_pos, n_steps, progress=True)
print("emcee sampling finished.")

flat_samples = sampler.get_chain(discard=burn_in, thin=15, flat=True)
raw_chain = sampler.get_chain()

# --- Results Summary ---
beta_m, mu_m, xi_m, R0_m = np.median(flat_samples, axis=0)

# [Print] Calculate and print final S(0), V(0), I(0), R(0)
S0_m = 1.0 - V0_fixed - I0_fixed - R0_m

print("\n" + "="*40)
print("   FINAL FITTED INITIAL STATE (t=0)")
print("="*40)
print(f"  S(0) : {S0_m:.4f}  (Calculated)")
print(f"  V(0) : {V0_fixed:.4f}  (Fixed)")
print(f"  I(0) : {I0_fixed:.4f}  (Fixed)")
print(f"  R(0) : {R0_m:.4f}  (Estimated Parameter)")
print(f"  SUM  : {S0_m + V0_fixed + I0_fixed + R0_m:.4f}")
print("="*40)

print(f"\n--- Median Parameters ---")
print(f"beta (β): {beta_m:.4f}")
print(f"mu (μ):   {mu_m:.4f}")
print(f"xi (ξ):   {xi_m:.4f}")
print(f"Initial_R:{R0_m:.4f}")

# Diagnostic Information
diagnostics = calculate_diagnostics(raw_chain, burn_in, param_labels)
diag_df = pd.DataFrame(diagnostics).T
print("\n[Diagnostic Table] MCMC Convergence:")
print(diag_df)

# --- Fitting and R2 ---
y0_fitted = [S0_m, V0_fixed, I0_fixed, R0_m]
best_fit_solution = odeint(svirs_model, y0_fitted, time, args=(beta_m, mu_m, xi_m, e_fixed, theta_fixed))
S_fit, V_fit, I_fit, R_fit = best_fit_solution.T
N_fit = S_fit + V_fit + I_fit + R_fit
final_predicted_infections = (beta_m * I_fit * S_fit / N_fit) + (beta_m * theta_fixed * I_fit * V_fit / N_fit)
r2 = calculate_r2(observed_data, final_predicted_infections)
print(f"\nR2 Score: {r2:.4f}")

# --- Plotting CI Calculation ---
print("\nCalculating 95% credible interval for INF fit...")
n_samples_for_ci = 500
random_indices = np.random.randint(0, flat_samples.shape[0], n_samples_for_ci)
param_samples_for_ci = flat_samples[random_indices]
predictions = []
for params in param_samples_for_ci:
    beta_s, mu_s, xi_s, R0_s = params
    S0_s = 1.0 - V0_fixed - I0_fixed - R0_s
    y0_s = [S0_s, V0_fixed, I0_fixed, R0_s]
    sol = odeint(svirs_model, y0_s, time, args=(beta_s, mu_s, xi_s, e_fixed, theta_fixed))
    S_s, V_s, I_s, R_s = sol.T
    N_s = S_s + V_s + I_s + R_s
    incidence_s = (beta_s * I_s * S_s / N_s) + (beta_s * theta_fixed * I_s * V_s / N_s)
    predictions.append(incidence_s)
predictions = np.array(predictions)
lower_bound, median_prediction, upper_bound = np.percentile(predictions, [2.5, 50, 97.5], axis=0)

# ============================================================
# Plotting Section (3-panel layout: 1 top, 2 bottom)
# ============================================================
print("\nGenerating Combined High-Res Plot (3 Panels)...")
fig = plt.figure(figsize=(20, 16)) 
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], width_ratios=[1, 1], wspace=0.25, hspace=0.25)

# --- Panel 1: Fit ---
ax1 = fig.add_subplot(gs[0, :])
ax1.fill_between(time, lower_bound, upper_bound, color='gray', alpha=0.4, label='95% Credible Interval')
ax1.plot(time, observed_data, 'o', color='black', alpha=0.7, label='Observed INF Data', markersize=6)
ax1.plot(time, median_prediction, color='red', lw=3, label='SVIRS Model Fit (Median)')

# [Modification]: Update X-axis label for the full date range
ax1.set_xlabel('Time (Weeks from 2023-W40 to 2024-W39)')
ax1.set_ylabel('Incidence') 
ax1.legend(loc='upper right', frameon=False)
ax1.grid(True, alpha=0.3)

# --- Panel 2: Trace ---
gs_trace = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1, 0], hspace=0.1)
for i in range(n_dim):
    ax_tr = fig.add_subplot(gs_trace[i, 0])
    ax_tr.plot(raw_chain[:, :, i], "k", alpha=0.3)
    ax_tr.set_xlim(0, n_steps)
    ax_tr.set_ylabel(param_labels[i])
    ax_tr.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax_tr.axvline(burn_in, color='red', linestyle='--', lw=2)
    if i < n_dim - 1: ax_tr.set_xticklabels([])
    else: ax_tr.set_xlabel("Step number")

# --- Panel 3: Hist ---
gs_hist = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 1], wspace=0.3, hspace=0.4)
for i in range(n_dim):
    row = i // 2; col = i % 2
    ax_hist = fig.add_subplot(gs_hist[row, col])
    ax_hist.hist(flat_samples[:, i], bins=30, color='gray', alpha=0.7, density=True)
    ax_hist.axvline(np.median(flat_samples[:, i]), color='red', linestyle='--', lw=2)
    ax_hist.set_xlabel(param_labels[i])
    ax_hist.set_yticklabels([]); ax_hist.set_ylabel("Density")

output_file = 'INF_svirs_3panel_results.png'
plt.savefig(output_file, dpi=1200, bbox_inches='tight')
print(f"Generated combined plot: '{output_file}'")

plt.show()