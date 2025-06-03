import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Deine Daten (je 40 Werte)
R_CM = np.array([0.01577, 0.01753, 0.01823, 0.02094, 0.02421, 0.02788, 0.03417, 0.045, 0.06382, 0.09689,
                0.157, 0.2582, 0.3798, 0.4463, 0.4177, 0.339, 0.2542, 0.1739, 0.1072, 0.06493,
                0.04275, 0.03224, 0.02557, 0.02172, 0.0207, 0.02107, 0.02252, 0.0262, 0.03139, 0.0336,
                0.03197, 0.02852, 0.02432, 0.02274, 0.02754, 0.04351])

R_C = np.array([0.02448, 0.06954, 0.153, 0.2819, 0.4415, 0.5606, 0.6673, 0.7357, 0.754, 0.7511,
                0.7255, 0.6875, 0.6344, 0.561, 0.4679, 0.3682, 0.2744, 0.1866, 0.1146, 0.07056,
                0.04836, 0.03824, 0.03197, 0.0282, 0.02748, 0.02809, 0.02989, 0.03396, 0.03921, 0.0418,
                0.04006, 0.03644, 0.03201, 0.0301, 0.03468, 0.05071])

R_M = np.array([0.02829, 0.02462, 0.02386, 0.02538, 0.02655, 0.02886, 0.03294, 0.04212, 0.05935, 0.09067,
                 0.1533, 0.273, 0.4589, 0.6535, 0.7756, 0.8232, 0.8378, 0.8416, 0.8397, 0.8421,
                 0.842, 0.8441, 0.8467, 0.8504, 0.8537, 0.8572, 0.8626, 0.8678, 0.8716, 0.873,
                 0.8733, 0.8741, 0.8762, 0.8789, 0.879, 0.8803])

# Fehlerfunktion
def yule_error(n):
    R1n = R_C ** (1/n)
    R2n = R_M ** (1/n)
    Rmixn = R_CM ** (1/n)
    Rpred = 0.5 * R1n + 0.5 * R2n
    return np.sum((Rmixn - Rpred) ** 2)

# Optimierung
res = minimize_scalar(yule_error, bounds=(0.4, 10.0), method='bounded')
n_opt = res.x
n_opt = 100

# Plot
R_pred = (0.5 * (R_C ** (1/n_opt)) + 0.5 * (R_M ** (1/n_opt))) ** n_opt

plt.plot(R_CM, label='Gemessen (C+M)')
plt.plot(R_pred, label=f'Predicted (n={n_opt:.2f})')
plt.legend()
plt.title("Yule-Nielsen-Mischung")
plt.xlabel("Wellenlänge (Index)")
plt.ylabel("Reflexion")
plt.grid(True)
plt.show()

print(f"Optimierter Yule-Nielsen Exponent: n = {n_opt:.3f}")