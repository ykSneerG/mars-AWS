import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def simulate_reflection(mu_a, mu_s, d=1.0):
    mu_eff = np.sqrt(mu_a * (mu_a + mu_s))
    R = np.exp(-2 * mu_eff * d)
    return np.clip(R, 0, 1)

def mix_optical(mu_a1, mu_s1, mu_a2, mu_s2, w1, w2):
    mu_a_mix = w1 * mu_a1 + w2 * mu_a2
    mu_s_mix = w1 * mu_s1 + w2 * mu_s2
    return simulate_reflection(mu_a_mix, mu_s_mix)

def multi_mix_loss(params, wavelengths, R_targets, weights_list):
    n = len(wavelengths)
    mu_a1 = np.abs(params[0:n])
    mu_s1 = np.abs(params[n:2*n])
    mu_a2 = np.abs(params[2*n:3*n])
    mu_s2 = np.abs(params[3*n:4*n])
    total_loss = 0
    for (w1, w2), R_target in zip(weights_list, R_targets):
        R_sim = mix_optical(mu_a1, mu_s1, mu_a2, mu_s2, w1, w2)
        total_loss += np.sum((R_sim - R_target)**2)
    return total_loss

def fit_turbid_media_model(wavelengths, R1, R2, mix_weights):
    R_mixes_measured = [w1 * R1 + w2 * R2 for w1, w2 in mix_weights]
    n = len(wavelengths)
    initial_guess = np.ones(4 * n) * 0.5

    result = minimize(
        multi_mix_loss,
        initial_guess,
        args=(wavelengths, R_mixes_measured, mix_weights),
        method='L-BFGS-B',
        options={'maxiter': 800}
    )

    mu_a1 = result.x[0:n]
    mu_s1 = result.x[n:2*n]
    mu_a2 = result.x[2*n:3*n]
    mu_s2 = result.x[3*n:4*n]

    return mu_a1, mu_s1, mu_a2, mu_s2, result

def plot_reflectance_predictions(wavelengths, R1, R2, mu_a1, mu_s1, mu_a2, mu_s2, mix_weights):
    R_mixes_measured = [w1 * R1 + w2 * R2 for w1, w2 in mix_weights]
    R_preds = [
        mix_optical(mu_a1, mu_s1, mu_a2, mu_s2, w1, w2)
        for w1, w2 in mix_weights
    ]

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red']
    for i, ((w1, w2), R_meas, R_pred) in enumerate(zip(mix_weights, R_mixes_measured, R_preds)):
        label = f"{int(w1*100)}/{int(w2*100)} Mischung"
        plt.plot(wavelengths, R_meas, label=f"Gemessen {label}", color=colors[i])
        plt.plot(wavelengths, R_pred, '--', label=f"Vorhersage {label}", color=colors[i])

    plt.xlabel("Wellenlänge (nm)")
    plt.ylabel("Reflexion")
    plt.title("Spektrale Vorhersage bei mehreren Mischverhältnissen")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
