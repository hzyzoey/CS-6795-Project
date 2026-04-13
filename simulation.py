"""
CS 6795 Cognitive Science 

Updated: 2026-04-12

Title:  Modeling Mental Representations of Wildfire-Related Risk Under Uncertainty: A Computational Cognitive Approach
Author: Zeying Huang (zhuang647@gatech.edu)
Track:  Computational Model / Tool

Steps:
  Step 1: Vegetation recovery
  Step 2: Burn scar activation
  Step 3: Attribution weight
  Step 4: Raw cue integration
  Step 5: Learning decay
  Step 6: Shielding belief
  Step 7: Affect suppression
  Step 8: Belief updating
  Step 9: Residential utility

Simulation:
  T = 156 weeks (3 years)
  Fire events at weeks 10 and 90
  Five scenarios compared (S0–S4)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def expit(x):
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

T   = 156      # total weeks (3 years)
dt  = 1        # time step = 1 week

# Cue weights (salience-biased vs. ecologically valid)
# c0: smoke_vis 
# c1: pm25
# c2: wind_dir 
# c3: fire_history 
# c4: wui_dist
# c5: recent_fire_dist
# c6: composite_activity 

# Columns: [smoke_vis, pm25, wind_dir, fire_history, wui_dist, recent_fire_dist, composite_activity]
W_BIASED   = np.array([0.28, 0.22, 0.10, 0.04, 0.06, 0.18, 0.12])  # salience-biased
W_OPTIMAL  = np.array([0.10, 0.10, 0.18, 0.12, 0.20, 0.12, 0.18])  # ecologically valid

# Spatial and temporal decay constants for c5/c6
RHO     = 0.01   # spatial decay constant (km^{-1})
NU      = 0.02   # temporal decay constant (weeks^{-1})
D_AGENT = 20.0   # residential distance to typical fire ignition zone (km)
                 # distinct from active fire front distance in attribution_alpha

# Fire severity scores for composite fire activity index (c6)
FIRE_SEVERITY_MAP = {10: 1.0, 90: 0.6}  # s_k per event

# Belief updating
LAMBDA_UPD  = 0.35   # learning rate (how fast R updates toward R_perceived)
DELTA_DECAY = 0.04   # decay rate when no fire signal

# Vegetation / affect parameters
KAPPA = 0.015   # vegetation regrowth rate (exponential approx.)
TAU   = 0.02    # burn scar decay rate
SIGMA_0 = 0.25  # initial affect suppression strength

# Shielding / gambler's fallacy parameters
BETA_0 = 0.50   # initial shielding belief strength
ETA    = 0.40   # learning decay rate (beta and sigma shrink with experience)

# Amenity trade-off
GAMMA  = 0.18   # amenity sensitivity
A_BASE = 0.40   # baseline amenity value
DELTA_A = 0.30  # amenity gain from full vegetation recovery

# Attribution weight parameters (sigmoid coefficients)
PHI1 = 2.0   # wind alignment contribution
PHI2 = 1.5   # proximity contribution
PHI3 = 0.05  # time-since-fire penalty

# Fire event schedule (weeks)
FIRE_EVENTS = [10, 90]   # two fires over 3 years

# distance decay rate in attribution (km^{-1})
MU = 0.3     # half-decay distance ≈ ln2/0.3 ≈ 2.3 km

# ─────────────────────────────────────────────────────────────────────────────
# 1.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def make_objective_hazard(T, fire_events):
    """
    Objective wildfire hazard signal h(t) ∈ [0, 1].
    Spikes at fire events and decays with a long tail.
    """
    h = np.zeros(T)
    for t_fire in fire_events:
        for t in range(t_fire, T):
            h[t] += np.exp(-0.03 * (t - t_fire))
    return np.clip(h, 0, 1)



def make_cue_signals(T, fire_events, noise_level=0.05):
    """
    Generate 7 cue signals:
      c0: smoke_visibility        (high salience, correlated with fire)
      c1: pm25_level              (moderate salience, partially fire-driven)
      c2: wind_direction          (moderate salience, auxiliary)
      c3: fire_history            (low salience, distal)
      c4: wui_distance_inv        (low salience, distal, inverted so higher = closer)
      c5: recent_fire_distance    (high salience, distance-decay from most recent fire)
                                  c5(t) = exp(-rho * d_agent) * exp(-nu * (t - t_last))
      c6: composite_fire_activity (low salience, weighted sum over all past fires)
                                  c6(t) = sum_k s_k * exp(-rho * d_k) * exp(-nu * (t - t_k))

    Returns shape (T, 7).
    """
    rng = np.random.default_rng(42)
    obj = make_objective_hazard(T, fire_events)

    c0 = np.clip(obj * 0.9 + rng.normal(0, noise_level, T), 0, 1)            # smoke
    c1 = np.clip(obj * 0.7 + 0.1 + rng.normal(0, noise_level, T), 0, 1)      # PM2.5
    c2 = np.clip(obj * 0.5 + 0.2 + rng.normal(0, noise_level, T), 0, 1)      # wind
    c3 = np.clip(obj * 0.6 + 0.1 + rng.normal(0, noise_level, T), 0, 1)      # history
    c4 = np.clip(obj * 0.5 + 0.3 + rng.normal(0, noise_level, T), 0, 1)      # wui inv

    # c5: distance-decay proximity to most recent fire
    c5 = np.zeros(T)
    proximity = np.exp(-RHO * D_AGENT)   # spatial factor (fixed agent location)
    for t in range(T):
        past = [f for f in fire_events if f <= t]
        if past:
            t_last = max(past)
            c5[t] = proximity * np.exp(-NU * (t - t_last))
    c5 = np.clip(c5 + rng.normal(0, noise_level * 0.5, T), 0, 1)

    # c6: composite fire activity index (accumulates across all past fires)
    c6 = np.zeros(T)
    for t in range(T):
        val = 0.0
        for t_k in fire_events:
            if t_k <= t:
                s_k = FIRE_SEVERITY_MAP.get(t_k, 0.3)
                val += s_k * np.exp(-RHO * D_AGENT) * np.exp(-NU * (t - t_k))
        c6[t] = val
    c6 = np.clip(c6 + rng.normal(0, noise_level * 0.5, T), 0, 1)

    return np.stack([c0, c1, c2, c3, c4, c5, c6], axis=1)  # shape (T, 7)



def attribution_alpha(t, fire_events, wind_align=0.7, distance=2.0):
    """ 
    attribution_alpha: weight assigned to wildfire as PM2.5 source.
    Modulated by wind alignment, distance to fire, and time since last fire.
    Returns value in (0, 1).

    'distance' here is the distance to the nearest active fire front
    (assumed ~2 km when fire has spread to the WUI community boundary),
    distinct from D_AGENT (20 km), which is the residential distance to
    the typical fire ignition zone used in cue signal generation (c5/c6)
    """ 

    past = [f for f in fire_events if f <= t]
    if not past:
        return 0.1   # no fire has occurred yet → low attribution
    last_fire = max(past)
    delta_t = t - last_fire
    logit = PHI1 * wind_align + PHI2 * np.exp(-MU * distance) - PHI3 * delta_t

    return float(expit(logit))


def vegetation(t, fire_events):
    """V(t): fractional vegetation recovery since last fire, ∈ [0, 1]."""
    past = [f for f in fire_events if f <= t]
    if not past:
        return 1.0  # no fire → full pre-fire vegetation
    t_last = max(past)
    return float(1 - np.exp(-KAPPA * (t - t_last)))


def burn_scar(t, fire_events):
    """BurnScar(t): perceived residual firebreak effect, ∈ [0, 1]."""
    past = [f for f in fire_events if f <= t]
    if not past:
        return 0.0
    t_last = max(past)
    return float(np.exp(-TAU * (t - t_last)))


def fire_count_by(t, fire_events):
    """Number of fire events experienced up to and including week t."""
    return sum(1 for f in fire_events if f <= t)





# ─────────────────────────────────────────────────────────────────────────────
# 2.  CORE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate(T, fire_events, cues, weights, beta_0=0.0, sigma_0=0.0,
             label="Scenario"):
    """
    Parameters：
    beta_0  : float – initial shielding belief strength (0 = no shielding)
    sigma_0 : float – initial affect suppression strength (0 = no affect)
    weights : np.ndarray shape (7,) – cue weights for R_raw

    Returns：
    dict with arrays: R, R_raw_arr, V_arr, BurnScar_arr, alpha_arr, utility
    """
    R = np.zeros(T)
    R[0] = 0.1   # initial perceived risk (low prior)

    R_raw_arr   = np.zeros(T)
    V_arr       = np.zeros(T)
    BS_arr      = np.zeros(T)
    alpha_arr   = np.zeros(T)
    utility     = np.zeros(T)


    for t in range(T):
        # Steps 1: vegetation
        V_t  = vegetation(t, fire_events)
        V_arr[t]  = V_t

        # Steps 2: burn scar 
        BS_t = burn_scar(t, fire_events)
        BS_arr[t] = BS_t

        # Step 3: attribution weight 
        a_t = attribution_alpha(t, fire_events)
        alpha_arr[t] = a_t

        # Step 4: raw cue integration 
        c = cues[t]   # shape (7,)
        # Re-weight PM2.5 channel by attribution weight
        c_attributed = c.copy()
        c_attributed[1] *= a_t # PM2.5 only counts insofar as attributed to wildfire
        R_raw = float(np.dot(weights, c_attributed))
        R_raw_arr[t] = R_raw

        # Step 5: learning decay of beta and sigma 
        # The first fire activates the bias, subsequent fires provide corrective experience.
        corrective_experiences = max(fire_count_by(t, fire_events) - 1, 0)
        beta_t  = beta_0  * np.exp(-ETA * corrective_experiences)
        sigma_t = sigma_0 * np.exp(-ETA * corrective_experiences)

        # Step 6: shielding belief 
        R_shielded = R_raw * (1.0 - beta_t * BS_t)

        # Step 7: affect suppression (nutrient-regrowth paradox) 
        R_perceived = max(R_shielded - sigma_t * V_t, 0.0)

        # Step 8: bounded belief updating 
        fire_active = any(0 <= t - tf <= 3 for tf in fire_events)  # fire signal active (within 3 weeks following) a fire event
        if fire_active:
            R[t] = (1 - LAMBDA_UPD) * (R[t - 1] if t > 0 else 0.1) + LAMBDA_UPD * R_perceived
        else:
            prev = R[t - 1] if t > 0 else 0.1
            R[t] = max(prev * np.exp(-DELTA_DECAY * dt), R_perceived * LAMBDA_UPD * 0.5)

        R[t] = float(np.clip(R[t], 0, 1))

        # Step 9: residential utility 
        utility[t] = -R[t] + GAMMA * (A_BASE + DELTA_A * V_t)

    return {
        "label"     : label,
        "R"         : R,
        "R_raw"     : R_raw_arr,
        "V"         : V_arr,
        "BurnScar"  : BS_arr,
        "alpha"     : alpha_arr,
        "utility"   : utility,
    }





# ─────────────────────────────────────────────────────────────────────────────
# 3.  SCENARIO DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_all_scenarios():
    cues     = make_cue_signals(T, FIRE_EVENTS)
    obj_risk = make_objective_hazard(T, FIRE_EVENTS)

    scenarios = {}


    # S0: Salience-biased cue weighting, no shielding, no affect suppression
    scenarios["S0"] = simulate(T, FIRE_EVENTS, cues, W_BIASED,
                               beta_0=0.0, sigma_0=0.0,
                               label="S0: Salience-Biased Cue Integration")

    # S1: Baseline – ecologically-optimal weights, no shielding, no affect suppression
    scenarios["S1"] = simulate(T, FIRE_EVENTS, cues, W_OPTIMAL,
                               beta_0=0.0, sigma_0=0.0,
                               label="S1: Baseline (Near-Bayesian)")

    # S2: Shielding Belief only (Fuel-Depletion Myth) – salience weights, beta high
    scenarios["S2"] = simulate(T, FIRE_EVENTS, cues, W_BIASED,
                               beta_0=0.60, sigma_0=0.0,
                               label="S2: Shielding Belief (Gambler's Fallacy)")

    # S3: Affect Suppression only (Nutrient-Regrowth Paradox) – salience weights, sigma high
    scenarios["S3"] = simulate(T, FIRE_EVENTS, cues, W_BIASED,
                               beta_0=0.0, sigma_0=0.40,
                               label="S3: Affect Heuristic (Regrowth Paradox)")

    # S4: Full Cognitive-Biased Model – all mechanisms active
    scenarios["S4"] = simulate(T, FIRE_EVENTS, cues, W_BIASED,
                               beta_0=BETA_0, sigma_0=SIGMA_0,
                               label="S4: Full Biased Model")

    return scenarios, obj_risk, cues







# ─────────────────────────────────────────────────────────────────────────────
# 4.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "S0": "#607D8B",   # blue gray
    "S1": "#2196F3",   # blue
    "S2": "#FF9800",   # orange
    "S3": "#4CAF50",   # green
    "S4": "#F44336",   # red
}


def plot_main_comparison(scenarios, obj_risk, save_path=None):
    """Figure 1: Perceived risk trajectories for all scenarios vs. objective risk."""
    weeks = np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(weeks, obj_risk, alpha=0.12, color="gray")
    ax.plot(weeks, obj_risk, color="gray", lw=1.5, ls="--", label="Objective Hazard")

    for key, sc in scenarios.items():
        ax.plot(weeks, sc["R"], color=COLORS[key], lw=2, label=sc["label"])

    for tf in FIRE_EVENTS:
        ax.axvline(tf, color="black", ls=":", lw=1.2, alpha=0.7)
        ax.text(tf + 1, 0.95, f"Fire\n(wk {tf})", fontsize=8, va="top")

    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Perceived Risk R(t)", fontsize=12)
    ax.set_title("Perceived Wildfire Risk: Cognitive Scenarios (156 Weeks / 3 Years)", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, T)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_mechanisms_detail(scenarios, save_path=None):
    """Figure 2: Mechanism decomposition – α, V, BurnScar, and utility."""
    sc4 = scenarios["S4"]
    weeks = np.arange(T)

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # Panel A: Attribution weight α(t)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(weeks, sc4["alpha"], color="#9c27B0", lw=2)
    for tf in FIRE_EVENTS:
        ax1.axvline(tf, color="black", ls=":", lw=1)
    ax1.set_title("Attribution Weight α(t)\n(PM₂.₅ attributed to wildfire)", fontsize=10)
    ax1.set_ylabel("α(t)")
    ax1.set_ylim(0, 1)

    # Panel B: Vegetation recovery V(t)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(weeks, sc4["V"], color="#4CAF50", lw=2)
    for tf in FIRE_EVENTS:
        ax2.axvline(tf, color="black", ls=":", lw=1)
    ax2.set_title("Vegetation Recovery V(t)\n(Nutrient-Regrowth Paradox source)", fontsize=10)
    ax2.set_ylabel("V(t)")
    ax2.set_ylim(0, 1)

    # Panel C: Burn scar BurnScar(t)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(weeks, sc4["BurnScar"], color="#FF9800", lw=2)
    for tf in FIRE_EVENTS:
        ax3.axvline(tf, color="black", ls=":", lw=1)
    ax3.set_title("Burn Scar Activation BurnScar(t)\n(Fuel-Depletion Myth driver)", fontsize=10)
    ax3.set_ylabel("BurnScar(t)")
    ax3.set_ylim(0, 1)

    # Panel D: Residential utility U(t)
    ax4 = fig.add_subplot(gs[1, 1])
    for key, sc in scenarios.items():
        ax4.plot(weeks, sc["utility"], color=COLORS[key], lw=1.8, label=key)
    ax4.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
    for tf in FIRE_EVENTS:
        ax4.axvline(tf, color="black", ls=":", lw=1)
    ax4.set_title("Residential Utility U(t)\n(Amenity–Risk Trade-off)", fontsize=10)
    ax4.set_ylabel("U(t)")
    ax4.legend(fontsize=8)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Week")
        ax.set_xlim(0, T)

    fig.suptitle("Mechanism Decomposition – Full Biased Model (S4)", fontsize=12, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_pairwise_comparison(scenarios, obj_risk, save_path=None):
    """Figure 3: Side-by-side panels for each scenario vs. objective risk."""
    weeks = np.arange(T)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (key, sc) in enumerate(scenarios.items()):
        ax = axes[i]
        ax.fill_between(weeks, obj_risk, alpha=0.12, color="gray")
        ax.plot(weeks, obj_risk, color="gray", lw=1.2, ls="--", label="Objective")
        ax.plot(weeks, sc["R"],  color=COLORS[key], lw=2, label="Perceived")
        for tf in FIRE_EVENTS:
            ax.axvline(tf, color="black", ls=":", lw=1)
        ax.set_title(sc["label"], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("Week")
        ax.set_ylabel("Risk")

    for ax in axes[len(scenarios):]:
        ax.axis("off")

    fig.suptitle("Perceived vs. Objective Wildfire Risk – All Scenarios", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


def print_summary_statistics(scenarios, obj_risk):
    """Print key descriptive statistics for the Results section."""
    print("\n" + "="*65)
    print("SIMULATION SUMMARY STATISTICS (156 weeks / 3 years)")
    print("="*65)
    print(f"{'Scenario':<45} {'Mean R':<10} {'Max R':<10} {'Underest.':<10}")
    print("-"*65)
    obj_mean = obj_risk.mean()
    for key, sc in scenarios.items():
        mean_r = sc["R"].mean()
        max_r  = sc["R"].max()
        underest = obj_mean - mean_r
        print(f"{sc['label']:<45} {mean_r:<10.3f} {max_r:<10.3f} {underest:<10.3f}")
    print(f"\nObjective hazard mean: {obj_mean:.3f}")
    print("="*65 + "\n")


def save_summary_csv(scenarios, obj_risk, save_path):

    obj_mean = obj_risk.mean()
    rows = []
    for key, sc in scenarios.items():
        R = sc["R"]
        U = sc["utility"]
        mean_r     = R.mean()
        max_r      = R.max()
        under_abs  = obj_mean - mean_r
        under_pct  = under_abs / obj_mean * 100
        peak1      = R[10:20].max()
        peak1_wk   = 10 + int(R[10:20].argmax())
        peak2      = R[90:105].max()
        peak2_wk   = 90 + int(R[90:105].argmax())
        inter_mean = R[15:88].mean()
        u_pos_pct  = (U > 0).mean() * 100
        rows.append({
            "scenario"       : key,
            "label"          : sc["label"],
            "mean_R"         : round(mean_r,    4),
            "max_R"          : round(max_r,     4),
            "underest_abs"   : round(under_abs, 4),
            "underest_pct"   : round(under_pct, 2),
            "peak1"          : round(peak1,     4),
            "peak1_wk"       : peak1_wk,
            "peak2"          : round(peak2,     4),
            "peak2_wk"       : peak2_wk,
            "inter_fire_mean": round(inter_mean,4),
            "utility_pos_pct": round(u_pos_pct, 2),
        })
    # append objective hazard row
    rows.append({
        "scenario"       : "OBJ",
        "label"          : "Objective Hazard",
        "mean_R"         : round(obj_mean, 4),
        "max_R"          : 1.0,
        "underest_abs"   : 0.0,
        "underest_pct"   : 0.0,
        "peak1"          : "",
        "peak1_wk"       : "",
        "peak2"          : "",
        "peak2_wk"       : "",
        "inter_fire_mean": "",
        "utility_pos_pct": "",
    })
    fieldnames = ["scenario","label","mean_R","max_R","underest_abs","underest_pct",
                  "peak1","peak1_wk","peak2","peak2_wk","inter_fire_mean","utility_pos_pct"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {save_path}")


def save_trajectories_csv(scenarios, obj_risk, save_path):
    """
    Save full week-by-week R(t) and U(t) trajectories for all scenarios to CSV.
    """
    import csv
    weeks = list(range(T))
    fieldnames = ["week", "obj_hazard"] + \
                 [f"R_{k}" for k in scenarios] + \
                 [f"U_{k}" for k in scenarios]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in weeks:
            row = {"week": t, "obj_hazard": round(float(obj_risk[t]), 6)}
            for k, sc in scenarios.items():
                row[f"R_{k}"] = round(float(sc["R"][t]),       6)
                row[f"U_{k}"] = round(float(sc["utility"][t]), 6)
            writer.writerow(row)
    print(f"Saved: {save_path}")






# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    print("Running CS 6795 Wildfire Risk Perception Model...")
    scenarios, obj_risk, cues = run_all_scenarios()

    # Print summary stats to console
    print_summary_statistics(scenarios, obj_risk)

    # Output directories
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Save summary CSV 
    save_summary_csv(scenarios, obj_risk,
                     save_path=os.path.join(out_dir, "summary_statistics.csv"))

    # Save full trajectories CSV 
    save_trajectories_csv(scenarios, obj_risk,
                          save_path=os.path.join(out_dir, "trajectories.csv"))

    # Generate and save figures 
    plot_main_comparison(scenarios, obj_risk,
                         save_path=os.path.join(fig_dir, "fig1_scenario_comparison.png"))

    plot_mechanisms_detail(scenarios,
                           save_path=os.path.join(fig_dir, "fig2_mechanism_decomposition.png"))

    plot_pairwise_comparison(scenarios, obj_risk,
                             save_path=os.path.join(fig_dir, "fig3_pairwise.png"))

    print("\nAll outputs saved to:", out_dir)
    print(f"  summary_statistics.csv   — per-scenario summary stats")
    print(f"  trajectories.csv         — week-by-week R(t) and U(t)")
    print(f"  figures/fig1_scenario_comparison.png")
    print(f"  figures/fig2_mechanism_decomposition.png")
    print(f"  figures/fig3_pairwise.png")
