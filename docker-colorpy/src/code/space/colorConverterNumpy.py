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
