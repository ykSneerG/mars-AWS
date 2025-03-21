import numpy as np # type: ignore


class MetamerismIndex:
    
    # Function to convert XYZ to u'v' chromaticity coordinates
    def _xyz_to_upvp(self, X, Y, Z):
        denom = X + 15 * Y + 3 * Z
        u_prime = (4 * X) / denom
        v_prime = (9 * Y) / denom
        return u_prime, v_prime

    # Function to compute metamerism index (MI) given XYZ values
    def compute_metamerism_index(self, X1, Y1, Z1, X2, Y2, Z2):
        # Convert XYZ to u'v' chromaticity coordinates for both samples
        u1, v1 = self._xyz_to_upvp(X1, Y1, Z1)
        u2, v2 = self._xyz_to_upvp(X2, Y2, Z2)

        # Compute chromaticity difference (Metamerism Index)
        MI = np.sqrt((u1 - u2)**2 + (v1 - v2)**2)
        return MI
    
    def delta(sample_a_MetamerismIndex, sample_b_MetamerismIndex):
        return np.abs(sample_a_MetamerismIndex - sample_b_MetamerismIndex)
    
""" 

# Example XYZ values for two samples under two illuminants (D65 and A)
# Sample 1 under Illuminant D65
X1_D65, Y1_D65, Z1_D65 = 50.0, 60.0, 70.0  # Replace with your XYZ values
# Sample 2 under Illuminant D65
X2_D65, Y2_D65, Z2_D65 = 55.0, 65.0, 75.0  # Replace with your XYZ values

# Sample 1 under Illuminant A
X1_A, Y1_A, Z1_A = 45.0, 55.0, 65.0  # Replace with your XYZ values
# Sample 2 under Illuminant A
X2_A, Y2_A, Z2_A = 50.0, 60.0, 70.0  # Replace with your XYZ values

# Compute the Metamerism Index for the two illuminants (D65 and A)
MI_D65 = MetamerismIndex.compute_metamerism_index(X1_D65, Y1_D65, Z1_D65, X2_D65, Y2_D65, Z2_D65)
MI_A = MetamerismIndex.compute_metamerism_index(X1_A, Y1_A, Z1_A, X2_A, Y2_A, Z2_A)

# Print the results
print(f"Metamerism Index (D65): {MI_D65:.4f}")
print(f"Metamerism Index (A): {MI_A:.4f}")

# If you want to find the overall MI between the two illuminants
overall_MI = MetamerismIndex.delta(MI_D65,  MI_A)
print(f"Overall Metamerism Index (between D65 and A): {overall_MI:.4f}")
"""




""" 


# Standard observer 2° color matching functions (380-780 nm at 5 nm intervals)
# This data should be loaded from a proper dataset (CIE 1931 2° observer)
cie_xbar = np.array([...])  # Replace with actual values
cie_ybar = np.array([...])
cie_zbar = np.array([...])
wavelengths = np.arange(380, 785, 5)  # 5 nm intervals

# Example Illuminants (D65 and A) - Replace with actual spectral power distributions
illuminant_D65 = np.array([...])  
illuminant_A = np.array([...])

# Function to compute XYZ tristimulus values
def compute_xyz(reflectance, illuminant):
    k = 100 / np.sum(illuminant * cie_ybar)
    X = k * np.sum(reflectance * illuminant * cie_xbar)
    Y = k * np.sum(reflectance * illuminant * cie_ybar)
    Z = k * np.sum(reflectance * illuminant * cie_zbar)
    return X, Y, Z

# Function to convert XYZ to u'v' chromaticity coordinates
def xyz_to_upvp(X, Y, Z):
    denom = X + 15 * Y + 3 * Z
    u_prime = (4 * X) / denom
    v_prime = (9 * Y) / denom
    return u_prime, v_prime

# Function to compute metamerism index (MI)
def compute_metamerism_index(ref1, ref2, illum1, illum2):
    # Compute XYZ under first illuminant
    X1_ref1, Y1_ref1, Z1_ref1 = compute_xyz(ref1, illum1)
    X1_ref2, Y1_ref2, Z1_ref2 = compute_xyz(ref2, illum1)

    # Compute XYZ under second illuminant
    X2_ref1, Y2_ref1, Z2_ref1 = compute_xyz(ref1, illum2)
    X2_ref2, Y2_ref2, Z2_ref2 = compute_xyz(ref2, illum2)

    # Convert to u'v' chromaticity coordinates
    u1_ref1, v1_ref1 = xyz_to_upvp(X1_ref1, Y1_ref1, Z1_ref1)
    u1_ref2, v1_ref2 = xyz_to_upvp(X1_ref2, Y1_ref2, Z1_ref2)
    u2_ref1, v2_ref1 = xyz_to_upvp(X2_ref1, Y2_ref1, Z2_ref1)
    u2_ref2, v2_ref2 = xyz_to_upvp(X2_ref2, Y2_ref2, Z2_ref2)

    # Compute chromaticity differences (Metamerism Index)
    MI_illum1 = np.sqrt((u1_ref1 - u1_ref2)**2 + (v1_ref1 - v1_ref2)**2)
    MI_illum2 = np.sqrt((u2_ref1 - u2_ref2)**2 + (v2_ref1 - v2_ref2)**2)
    
    MI = np.abs(MI_illum1 - MI_illum2)  # Final metamerism index
    return MI

# Example reflectance data for two samples (Replace with actual data)
reflectance_1 = np.array([...])  
reflectance_2 = np.array([...])  

# Compute the metamerism index
MI = compute_metamerism_index(reflectance_1, reflectance_2, illuminant_D65, illuminant_A)
print(f"Metamerism Index (ISO 23603): {MI:.4f}")
 """