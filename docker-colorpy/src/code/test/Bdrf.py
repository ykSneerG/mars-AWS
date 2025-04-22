import numpy as np
import matplotlib.pyplot as plt

# Wellenlängen
wavelengths = np.arange(380, 740, 10)

# Deine Originalspektren
spectra = np.array([
    [0.4689,0.5202,0.6240,0.7464,0.8459,0.8885,0.9010,0.9035,0.9050,0.9038,0.8992,0.8928,0.8840,0.8761,0.8710,0.8663,0.8627,0.8593,0.8525,0.8502,0.8457,0.8432,0.8426,0.8440,0.8479,0.8524,0.8590,0.8669,0.8752,0.8827,0.8892,0.8943,0.9002,0.9044,0.9061,0.9091],
    [0.4201,0.4900,0.6064,0.7309,0.8277,0.8730,0.8926,0.8990,0.9014,0.9002,0.8950,0.8871,0.8758,0.8627,0.8455,0.8138,0.7471,0.6017,0.3931,0.3233,0.3159,0.3230,0.3230,0.3300,0.3433,0.3435,0.3374,0.3310,0.3381,0.3383,0.3386,0.3453,0.3579,0.3807,0.4009,0.4273],
    [0.1674,0.2668,0.4334,0.5744,0.6664,0.7369,0.8053,0.8439,0.8551,0.8545,0.8424,0.8201,0.7869,0.7362,0.6554,0.5322,0.3573,0.1629,0.0550,0.0374,0.0359,0.0374,0.0374,0.0389,0.0419,0.0419,0.0405,0.0390,0.0405,0.0405,0.0405,0.0420,0.0449,0.0508,0.0565,0.0648],
    [0.0155,0.0334,0.0898,0.1687,0.2368,0.3236,0.4573,0.5647,0.6008,0.6010,0.5674,0.5074,0.4293,0.3293,0.2155,0.1144,0.0458,0.0130,0.0035,0.0023,0.0022,0.0023,0.0023,0.0024,0.0026,0.0026,0.0025,0.0024,0.0025,0.0025,0.0025,0.0026,0.0028,0.0032,0.0036,0.0042]
])

# --- Cook-Torrance Modell ---
def cook_torrance_simulation(spectra, f0=0.04, roughness=0.2, diffuse_factor=0.94):
    fresnel = f0 + (1 - f0) * (1 - spectra)**5
    geom = (1 - roughness) * 0.5 + 0.5
    specular = fresnel * geom
    reflected = specular + diffuse_factor * spectra
    return np.clip(reflected, 0, 1)

# --- Disney Modell ---
def disney_brdf_simulation(spectra, specular=0.5, roughness=0.3):
    fresnel = specular + (1 - specular) * (1 - spectra)**5
    diffuse = (1 - specular) * (1 - roughness * (1 - spectra)**2)
    reflected = fresnel + diffuse * spectra
    return np.clip(reflected, 0, 1)

# Simulationen durchführen
spectra_cook = cook_torrance_simulation(spectra, f0=0.04, roughness=0.2, diffuse_factor=0.25)
spectra_disney = disney_brdf_simulation(spectra, specular=0.05, roughness=0.2)

# --- Plotten ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i in range(4):
    axs[i].plot(wavelengths, spectra[i], 'k--', label='Original (SCE)')
    axs[i].plot(wavelengths, spectra_cook[i], 'b-', label='Cook-Torrance (SCI)')
    axs[i].plot(wavelengths, spectra_disney[i], 'r-', label='Disney (SCI)')
    axs[i].set_title(f'Sample {i+1}')
    axs[i].set_xlabel('Wellenlänge [nm]')
    axs[i].set_ylabel('Reflexion')
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_ylim(0, 1)

plt.tight_layout()
plt.show()