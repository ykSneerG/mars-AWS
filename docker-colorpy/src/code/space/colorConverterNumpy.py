from typing import Union
import numpy as np  # type: ignore

from src.code.space.colorSpace import CsXYZ, MAXVAL_CS_XYZ, MAXVAL_CS_RGB
from src.code.space.colorConstants.illuminant import OBSERVER, AdaptionBaseMatrix, Illuminant
from src.code.space.colorConstants.matrixRGB import MatrixRGB2XYZ
from src.code.space.colorConstants.weightningFunctions import Cmf2degNumpy, Cmf10degNumpy


CIE_E = 0.008856
"""
CIE constants / Actual CIE standard
216/24389
"""
CIE_K = 903.3
"""
CIE constants / Actual CIE standard
24389/27
"""


class ColorTrafoNumpy:
    """Optimized color space transformations with full NumPy vectorization"""
    
    def __init__(
        self, 
        observer: OBSERVER = OBSERVER.DEG2, 
        refWhite: CsXYZ = Illuminant.D50_DEG2
    ):
        self._set_observer(observer)
        self._set_illuminant(refWhite)
        self._precompute_constants()

    def _precompute_constants(self):
        """Precompute frequently used constants"""
        self.cie_e = np.float32(CIE_E)
        self.cie_k = np.float32(CIE_K)
        #self.lab_coeffs = np.array([116, 500, -200], dtype=np.float32)
        self.scale_xyz = np.float32(1/MAXVAL_CS_XYZ)
        self.scale_rgb = np.float32(MAXVAL_CS_RGB)

    def _set_observer(self, observer):
        """Optimized observer configuration"""
        cmf_map = {
            OBSERVER.DEG2: Cmf2degNumpy.weights,
            OBSERVER.DEG10: Cmf10degNumpy.weights
        }
        try:
            weights = cmf_map[observer].T.astype(np.float32, order='C')
        except KeyError:
            raise ValueError("Observer must be '2deg' or '10deg'")
            
        self.cmf_T = weights
        self.observer = observer

    def _set_illuminant(self, illuminant: CsXYZ = Illuminant.D50_DEG2):
        
        self.refWhite = illuminant.to_numpy().astype(np.float32)
        
        if illuminant == Illuminant.D50_DEG2:
            self.rgb_matrix = np.array(MatrixRGB2XYZ.SRGB_D50, dtype=np.float32)
        else:
            adaption_matrix = AdaptionBaseMatrix.get_matrix(
                AdaptionBaseMatrix.Bradford,
                AdaptionBaseMatrix.BradfordInvers,
                illuminant,
                Illuminant.D50_DEG2
            )
            
            self.rgb_matrix = np.dot(
                np.array(adaption_matrix, dtype=np.float32), 
                np.array(MatrixRGB2XYZ.SRGB_D50, dtype=np.float32)
            )


    def CS_SNM2XYZ(self, curveNM: np.ndarray) -> np.ndarray:
        """Vectorized spectral to XYZ conversion"""
        return np.dot(curveNM, self.cmf_T)

    def Cs_SNM2LAB(self, curveNM: np.ndarray) -> np.ndarray:
        xyz = self.CS_SNM2XYZ(curveNM)
        return self.Cs_XYZ2LAB(xyz)
    

    
    def Cs_SNM2OKLAB(self, curveNM: np.ndarray) -> np.ndarray:
        """Vectorized spectral to OKLAB conversion"""
        xyz = self.CS_SNM2XYZ(curveNM)
        return self.Cs_XYZ2OKLAB(xyz)
    
    def Cs_XYZ2OKLAB(self, xyz: np.ndarray) -> np.ndarray:
        """Optimized vectorized XYZ to OKLAB conversion"""
        
        # Precompute the transformation matrices
        xyz_to_lms = np.array([
            [ 0.8189330101,  0.3618667424, -0.1288597137],
            [ 0.0329845436,  0.9293118715,  0.0361456387],
            [ 0.0482003018,  0.2643662691,  0.6338517070]
        ])
        
        lms_to_oklab = np.array([
            [ 0.2104542553,  0.7936177850, -0.0040720468],
            [ 1.9779984951, -2.4285922050,  0.4505937099],
            [ 0.0259040371,  0.7827717662, -0.8086757660]
        ])
        
        # Step 1: Convert XYZ to LMS (matrix multiplication)
        LMS = np.dot(xyz, xyz_to_lms.T)  # shape (n, 3)
        
        # Step 2: Apply nonlinear transformation (cube root) for each component
        LMS = np.cbrt(LMS)  # Cube root of each element (vectorized operation)

        # Step 3: Convert LMS to OKLab (matrix multiplication)
        OKLab = np.dot(LMS, lms_to_oklab.T)  # shape (n, 3)

        return OKLab


    def Cs_SNM2CIECAM16(self, curveNM: np.ndarray) -> np.ndarray:
        """Convert XYZ to CIECAM16 color space"""
        xyz = self.CS_SNM2XYZ(curveNM)   
        return self.Cs_XYZ2CIECAM16(xyz)



    def Cs_XYZ2CIECAM16(self, xyz: np.ndarray, L_A=64, Y_b=20, F=1.0, c=0.69, N_c=1.0) -> np.ndarray:
        """
        XYZ to CIECAM16 conversion that maintains identical interface to OKLab version
        Returns Nx3 array of [J (lightness), a (red-green), b (yellow-blue)] in perceptual space
        """
        # Input validation and conversion
        if not isinstance(xyz, np.ndarray):
            xyz = np.array(xyz, dtype=np.float64)
        
        original_shape = xyz.shape
        if xyz.ndim == 1:
            xyz = xyz[np.newaxis, :]
        elif xyz.ndim > 2:
            raise ValueError("Input must be 1D or 2D array")
        
        if xyz.shape[1] != 3:
            raise ValueError(f"Input must have 3 columns. Got shape {original_shape}")

        # CIECAM16 transformation matrices
        xyz_to_lms = np.array([
            [0.401288,  0.650173, -0.051461],
            [-0.250268, 1.204414,  0.045854],
            [-0.002079, 0.048952,  0.953127]
        ], dtype=np.float64)

        # 1. Convert XYZ to LMS
        LMS = xyz @ xyz_to_lms.T  # Matrix multiplication

        # 2. Chromatic adaptation
        D = F * (1 - (1/3.6) * np.exp((-L_A - 42)/92))
        Y = xyz[:, 1].reshape(-1, 1)  # Ensure column vector
        LMS_c = LMS * (D * Y_b/Y + 1 - D)

        # 3. Nonlinear response
        k = 1 / (5*L_A + 1)
        F_L = 0.2 * k**4 * (5*L_A) + 0.1 * (1 - k**4)**2 * (5*L_A)**(1/3)
        LMS_prime = (F_L * LMS_c/100)**0.42
        LMS_a = 400 * np.sign(LMS_prime) * np.abs(LMS_prime) / (np.abs(LMS_prime) + 27.13)

        # 4. Calculate opponent dimensions
        a = LMS_a[:, 0] - (12/11)*LMS_a[:, 1] + (1/11)*LMS_a[:, 2]  # Red-Green
        b = (1/9)*(LMS_a[:, 0] + LMS_a[:, 1] - 2*LMS_a[:, 2])       # Yellow-Blue

        # 5. Calculate lightness (J)
        A = (2*LMS_a[:, 0] + LMS_a[:, 1] + 0.05*LMS_a[:, 2])/3.05
        LMS_w_prime = (F_L * Y_b/100)**0.42
        LMS_w_a = 400 * LMS_w_prime / (LMS_w_prime + 27.13)
        A_w = (2*LMS_w_a + LMS_w_a + 0.05*LMS_w_a)/3.05
        J = 100 * (A/A_w)**(c*F_L)

        # Return in same Cartesian format as OKLab: [Lightness, a, b]
        return np.array(np.column_stack((J, a, b)), dtype=np.float32).reshape(original_shape[0] if len(original_shape) == 1 else original_shape)


    def Cs_XYZ2RGB(self, xyz: np.ndarray) -> np.ndarray:
        """Batch XYZ to RGB conversion"""
        scaled_xyz = xyz * self.scale_xyz
        lin_rgb = np.dot(scaled_xyz, self.rgb_matrix.T)
        return self._sRGBcompanding(lin_rgb) * self.scale_rgb
    

    def Cs_XYZ2HEX(self, xyz: np.ndarray) -> Union[str, np.ndarray]:
        """Vectorized XYZ to HEX conversion"""
        rgb = self.Cs_XYZ2RGB(xyz)
        return self.Cs_RGB2HEX(rgb)
    
    def Cs_XYZ2LAB(self, xyz: np.ndarray) -> np.ndarray:
        """Optimized XYZ to LAB conversion"""
        # Normalize XYZ values
        rXYZ = (xyz * self.scale_xyz) / self.refWhite
        
        # Apply piecewise function
        mask = rXYZ > self.cie_e
        fXYZ = np.where(mask, np.cbrt(rXYZ), (self.cie_k * rXYZ + 16) / 116)
        
        # Calculate LAB components correctly
        L = 116 * fXYZ[..., 1] - 16      # L* from Y component
        a = 500 * (fXYZ[..., 0] - fXYZ[..., 1])  # a* = 500*(fX - fY)
        b = 200 * (fXYZ[..., 1] - fXYZ[..., 2])  # b* = 200*(fY - fZ)
        
        return np.stack([L, a, b], axis=-1)
    

    def Cs_RGB2HEX(self, rgb: np.ndarray) -> Union[str, np.ndarray]:
        """Batch RGB to HEX conversion"""
        if rgb.ndim == 1:
            return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        
        return np.apply_along_axis(
            lambda x: f"{int(x[0]):02X}{int(x[1]):02X}{int(x[2]):02X}", 
            axis=1, 
            arr=np.clip(rgb, 0, 255).astype(np.uint8)
        )
    

    def Cs_LAB2LCH(self, lab: np.ndarray) -> np.ndarray:
        """
        Vectorized LAB to LCH conversion for single or multiple colors
        Handles both 1D (single color) and 2D (multiple colors) inputs
        
        :param lab: LAB array(s) with shape (3,) or (N, 3)
        :return: LCH array(s) with matching shape
        """
        # Ensure array format for vectorized operations
        lab_arr = np.asarray(lab)
        
        # Calculate chroma using vectorized hypotenuse
        chroma = np.hypot(lab_arr[..., 1], lab_arr[..., 2])
        
        # Calculate hue in degrees with modulo 360
        hue = np.degrees(np.arctan2(lab_arr[..., 2], lab_arr[..., 1])) % 360
        
        # Stack components efficiently
        # Round and convert to float32 for memory efficiency
        return np.stack([
            np.round(lab[..., 0], 2).astype(np.float32),
            np.round(chroma, 2).astype(np.float32),
            np.round(hue, 2).astype(np.float32)
        ], axis=-1)

    


    def _sRGBcompanding(self, rgb: np.ndarray) -> np.ndarray:
        """Vectorized sRGB companding"""
        mask = rgb > 0.0031308
        return np.where(mask, 1.055 * np.cbrt(rgb) - 0.055, rgb * 12.92)


    # Convert multiple spectral data to multiple color spaces
    def Cs_SNM2MULTI(self, values: np.ndarray, incl_dst_values: dict = None) -> list:
        spectral = np.asarray(values, dtype=np.float32)
        if spectral.ndim == 1:
            spectral = spectral[np.newaxis, :]
         
        incl = incl_dst_values or {}
        
        # Precompute all conversions first
        xyz = lab = lch = hex = None
        rounded_lch = None
        
        if any([incl.get(k, False) for k in ["XYZ", "LAB", "LCH", "HEX"]]):
            xyz = self.CS_SNM2XYZ(spectral)
            
            if incl.get("HEX", False):
                hex = self.Cs_XYZ2RGB(xyz)
                hex = np.apply_along_axis(
                    lambda x: f"#{x[0]:02X}{x[1]:02X}{x[2]:02X}", 
                    1, 
                    np.clip(hex, 0, 255).astype(np.uint8)
                )
                
            if incl.get("LAB", False) or incl.get("LCH", False):
                lab = self.Cs_XYZ2LAB(xyz)
                
                if incl.get("LCH", False):
                    lch = self.Cs_LAB2LCH(lab)
                    rounded_lch = np.round(lch, 2)
        
        # Build results
        return [{
            "snm": [round(float(num), 4) for num in row],
            **({"xyz": np.round(xyz[i], 2).tolist()} if incl.get("XYZ") else {}),
            **({"lab": np.round(lab[i], 2).tolist()} if incl.get("LAB") else {}),
            **({"lch": {
                "L": round(float(rounded_lch[i, 0]), 2),
                "C": round(float(rounded_lch[i, 1]), 2),
                "H": round(float(rounded_lch[i, 2]), 2)
            }} if incl.get("LCH") else {}),
            **({"hex": hex[i]} if incl.get("HEX") else {})
        } for i, row in enumerate(spectral)]


    def Cs_SNM2MULTI_NP(self, values, incl_dst_values: dict = None) -> list:

        # return values.tolist()

        # Ensure input is a numpy array of float32
        # Filter out any non-numeric (e.g., dict) entries
        if isinstance(values, np.ndarray):
            spectral = values
        else:
            # Only keep rows that are list/tuple/np.ndarray of numbers
            filtered = []
            for v in values:
                if isinstance(v, (list, tuple, np.ndarray)) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in v):
                    filtered.append(v)
            spectral = np.asarray(filtered, dtype=np.float32)
        if spectral.ndim == 1:
            spectral = spectral[np.newaxis, :]
        
        incl = incl_dst_values or {}
        
        # Precompute all conversions first
        xyz = lab = lch = hex = None
        rounded_lch = None
        
        if any(incl.get(k, False) for k in ["XYZ", "LAB", "LCH", "HEX"]):
            xyz = self.CS_SNM2XYZ(spectral)
            
            if incl.get("HEX", False):
                rgb = self.Cs_XYZ2RGB(xyz)
                hex = np.apply_along_axis(
                    lambda x: f"#{int(x[0]):02X}{int(x[1]):02X}{int(x[2]):02X}", 
                    1, 
                    np.clip(rgb, 0, 255).astype(np.uint8)
                )
                
            if incl.get("LAB", False) or incl.get("LCH", False):
                lab = self.Cs_XYZ2LAB(xyz)
                
                if incl.get("LCH", False):
                    lch = self.Cs_LAB2LCH(lab)
                    rounded_lch = np.round(lch, 2)
        
        
        
        # Build results
        results = []
        for i, row in enumerate(spectral):
            entry = {
                "snm": [round(float(num), 4) for num in row]
            }
            if incl.get("XYZ"):
                entry["xyz"] = np.round(xyz[i], 2).tolist()
            if incl.get("LAB"):
                #entry["lab"] = np.round(lab[i], 2).tolist()
                entry["lab"] = [round(float(num), 2) for num in lab[i]]
            if incl.get("LCH"):
                entry["lch"] = {
                    "L": round(float(rounded_lch[i, 0]), 2),
                    "C": round(float(rounded_lch[i, 1]), 2),
                    "H": round(float(rounded_lch[i, 2]), 2)
                }
            if incl.get("HEX"):
                entry["hex"] = hex[i]
            results.append(entry)
        return results
