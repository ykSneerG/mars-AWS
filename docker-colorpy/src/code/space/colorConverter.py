import math
from typing import Union
import numpy as np  # type: ignore

from src.code.colorMath import CmMath
from src.code.space.colorSpace import CsLAB, CsLCH, CsSpectral, CsXYZ, CsRGB
from src.code.space.colorConstants.illuminant import OBSERVER, Illuminant
from src.code.space.colorConstants.matrixRGB import MatrixRGB2XYZ, MatrixXYZ2RGB
from src.code.space.colorConstants.weightningFunctions import (
    Cmf2deg,
    Cmf10deg,
    Cmf2degNumpy,
    Cmf10degNumpy,
)

from src.code.space.colorSpace import MAXVAL_CS_XYZ, MAXVAL_CS_RGB


CIE_E = 0.008856
"""
CIE constants / Actual CIE standard
"""
CIE_K = 903.3
"""
CIE constants / Actual CIE standard
"""


def Cs_Lab2LCH(lab: CsLAB) -> CsLCH:
    """
    Convert a CsLAB color to a CsLCH color.

    :param lab: The CsLAB color to convert.
    :return: The resulting CsLCH color.
    """

    """
    hue: float = math.degrees(math.atan2(lab.B, lab.A))
    hue: float = (hue + 360) % 360
    """

    hue: float = math.degrees(math.atan2(lab.B, lab.A)) % 360

    return CsLCH(lab.L, lab.to_chroma(), hue)


def Cs_LCH2Lab(lch: CsLCH) -> CsLAB:
    """
    Convert a CsLCH color to a CsLAB color.

    :param lch: The CsLCH color to convert.
    :return: The converted CsLAB color.
    """
    labA: float = lch.C * math.cos(math.radians(lch.H))
    labB: float = lch.C * math.sin(math.radians(lch.H))

    return CsLAB(lch.L, labA, labB)


def Cs_XYZ2LAB(xyz: CsXYZ, refWhite: CsXYZ = Illuminant.D50_DEG2) -> CsLAB:
    """
    Convert a CsXYZ color to a CsLAB color.

    :param xyz: The CsXYZ color to convert.
    :param refWhite: The reference white for the conversion as a normalized CsXYZ color.
    :return: The converted CsLAB color.
    """

    normalized_xyz: CsXYZ = xyz.normalize()

    rXYZ: list[float] = [
        normalized_xyz.X / refWhite.X,
        normalized_xyz.Y / refWhite.Y,
        normalized_xyz.Z / refWhite.Z,
    ]

    for i in range(len(rXYZ)):
        if rXYZ[i] > CIE_E:
            rXYZ[i] = pow(rXYZ[i], 1 / 3)
        else:
            rXYZ[i] = (CIE_K * rXYZ[i] + 16) / 116.0

    labL: float = 116.0 * rXYZ[1] - 16.0
    labA: float = 500.0 * (rXYZ[0] - rXYZ[1])
    labB: float = 200.0 * (rXYZ[1] - rXYZ[2])

    return CsLAB(labL, labA, labB)

def Cs_XYZ2LAB_Numpy(
    xyz: np.ndarray, refWhite: np.ndarray = Illuminant.D50_DEG2.to_numpy()
) -> np.ndarray:

    # Normalize the XYZ values and divide by the reference white
    rXYZ = (xyz / MAXVAL_CS_XYZ) / refWhite

    mask = rXYZ > CIE_E
    rXYZ = np.where(mask, np.power(rXYZ, 1 / 3), (CIE_K * rXYZ + 16) / 116.0)

    return np.array(
        [
            116.0 * rXYZ[1] - 16.0,
            500.0 * (rXYZ[0] - rXYZ[1]),
            200.0 * (rXYZ[1] - rXYZ[2]),
        ]
    )


def Cs_Lab2XYZ(lab: CsLAB, refWhite: CsXYZ) -> CsXYZ:
    """
    Convert a CsLAB color to a CsXYZ color.

    :param lab: The CsLAB color to convert.
    :param refWhite: The reference white for the conversion as a normalized CsXYZ color.
    :return: The converted CsXYZ color.
    """

    y: float = (lab.L + 16) / 116
    x: float = y + lab.A / 500
    z: float = y - lab.B / 200

    y = pow(y, 3) if (lab.L > CIE_E * CIE_K) else lab.L / CIE_K
    x = pow(x, 3) if (pow(x, 3) > CIE_E) else (116 * x - 16) / CIE_K
    z = pow(z, 3) if (pow(z, 3) > CIE_E) else (116 * z - 16) / CIE_K

    return CsXYZ(refWhite.X * x, refWhite.Y * y, refWhite.Z * z).denormalize()


def Cs_XYZ2Denisty(xyz: CsXYZ) -> float:
    """
    Convert a CsXYZ color to a density value.

    :param xyz: The CsXYZ color to convert.
    :return: The calculated denisty value.
    """

    return -math.log10(xyz.Y / 100)


def Cs_XYZ2RGB(xyz: CsXYZ) -> CsRGB:
    """
    Convert a CsXYZ color to a CsRGB color.

    :param xyz: The CsXYZ color to convert.
    :return: The converted CsRGB color.
    """

    # linear RGB
    varRGB: list[float] = CmMath.matrix3x3_1x3(
        MatrixRGB2XYZ.SRGB_D50, xyz.normalize().to_list()
    )

    # companded RGB
    normalized_rgb: CsRGB = __sRGBcompanding(CsRGB(varRGB[0], varRGB[1], varRGB[2]))

    return normalized_rgb.denormalize().clamp()

def Cs_XYZ2HEX_Numpy(xyz: CsXYZ) -> str:
    """
    Convert a CsXYZ color to a HEX color.

    :param xyz: The CsXYZ color to convert.
    :return: The converted HEX color.
    """
    
    # linear RGB
    varRGB: list[float] = CmMath.matrix3x3_1x3(
        MatrixRGB2XYZ.SRGB_D50, xyz.normalize().to_list()
    )

    # companded RGB
    normalized_rgb: CsRGB = __sRGBcompanding(CsRGB(varRGB[0], varRGB[1], varRGB[2]))

    rgb: CsRGB = normalized_rgb.denormalize().clamp()

    return rgb.to_hex()


def Cs_RGB2XYZ(rgb: CsRGB) -> CsXYZ:
    """
    Convert a CsRGB color to a XYZ color.

    :param rgb: The CsRGB color to convert.
    :return: The converted CsXYZ color.
    """

    # Convert sRGB to linear RGB
    linear_rgb: CsRGB = __SRGBcompandingInvers(rgb.normalize())

    # Apply the inverse of the SRGB_D50 matrix to get XYZ
    normalized_xyz: list[float] = CmMath.matrix3x3_1x3(
        MatrixXYZ2RGB.SRGB_D50, linear_rgb.to_list()
    )

    return CsXYZ(normalized_xyz[0], normalized_xyz[1], normalized_xyz[2]).denormalize()


def CS_Spectral2XYZ(curveNM: list[float], observer: OBSERVER = OBSERVER.DEG2) -> CsXYZ:
    """
    Spectral Data (380-730nm in 10nm steps) to XYZ Color (default: D50, 2 deg.)

    :param curveNM: array with 36 wavelengths (380-730nm in 10nm steps)
    :param observer: "Deg2" or "Deg10" specifies the color matching functions.
    """

    if observer == OBSERVER.DEG2:
        cmf = Cmf2deg
    elif observer == OBSERVER.DEG10:
        cmf = Cmf10deg
    else:
        raise ValueError("observer must be '2deg' or '10deg'")

    xyz = CsXYZ(0.0, 0.0, 0.0)

    for i, wavelength in enumerate(curveNM):
        xyz.X += cmf.WX[i] * wavelength
        xyz.Y += cmf.WY[i] * wavelength
        xyz.Z += cmf.WZ[i] * wavelength

    return xyz


def CS_Spectral2XYZ_Numpy(
    curveNM: np.ndarray, observer: OBSERVER = OBSERVER.DEG2
) -> np.ndarray:
    """
    Spectral Data (380-730nm in 10nm steps) to XYZ Color (default: D50, 2 deg.)

    :param curveNM: np.ndarray with 36 wavelengths (380-730nm in 10nm steps)
    :param observer: "Deg2" or "Deg10" specifies the color matching functions.
    """

    if observer == OBSERVER.DEG2:
        cmf_weights = Cmf2degNumpy.weights
    elif observer == OBSERVER.DEG10:
        cmf_weights = Cmf10degNumpy.weights
    else:
        raise ValueError("observer must be '2deg' or '10deg'")

    return cmf_weights @ curveNM


# -x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.


def CS_Spectral2LAB(curveNM: list[float], observer: OBSERVER = OBSERVER.DEG2) -> CsLAB:
    cs_xyz = CS_Spectral2XYZ(curveNM, observer)
    cs_lab = Cs_XYZ2LAB(cs_xyz)
    return cs_lab


def CS_Spectral2LCH(curveNM: list[float], observer: OBSERVER = OBSERVER.DEG2) -> CsLCH:
    cs_xyz = CS_Spectral2XYZ(curveNM, observer)
    cs_lab = Cs_XYZ2LAB(cs_xyz)
    cs_lch = Cs_Lab2LCH(cs_lab)
    return cs_lch


def Cs_XYZ2LCH(xyz: CsXYZ, refWhite: CsXYZ = Illuminant.D50_DEG2) -> CsLCH:
    cs_xyz = Cs_XYZ2LAB(xyz, refWhite)
    cs_lch = Cs_Lab2LCH(cs_xyz)
    return cs_lch


# -x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.


def __sRGBcompanding(linearRGB: CsRGB) -> list[float]:
    """
    sRGB companding to delinearize a given CsRGB value.

    :param rgb: CsRGB color
    :return: new CsRGB
    """
    tmp: list[float] = [
        (
            1.055 * math.pow(linrgb, 1.0 / 2.4) - 0.055
            if linrgb > 0.0031308
            else linrgb * 12.92
        )
        for linrgb in linearRGB.to_list()
    ]

    return CsRGB(tmp[0], tmp[1], tmp[2])

def __sRGBcompanding_Numpy(rgb: np.ndarray) -> np.ndarray:
    """
    sRGB companding to delinearize a given CsRGB value.

    :param rgb: numpy array with 3 values
    :return: numpy array with companded RGB values
    """
    mask = rgb > 0.0031308
    return np.where(mask, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055, rgb * 12.92)


def __SRGBcompandingInvers(rgb: CsRGB) -> CsRGB:
    """
    Inverse sRGB companding to linearize a given CsRGB value.

    :param rgb: CsRGB color
    :return: new CsRGB
    """

    sRGB: list[float] = [rgb.R, rgb.G, rgb.B]
    for i in range(3):
        if sRGB[i] > 0.04045:
            sRGB[i] = math.pow((sRGB[i] + 0.055) / 1.055, 2.4)
        else:
            sRGB[i] /= 12.92
    return CsRGB(sRGB[0], sRGB[1], sRGB[2])


# -x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.


def Cs_Spectral2Multi(values: list, incl_dst_values=None) -> list:
    incl_XYZ = incl_dst_values.get("XYZ", False) == True
    incl_LAB = incl_dst_values.get("LAB", False) == True
    incl_HEX = incl_dst_values.get("HEX", False) == True
    incl_LCH = incl_dst_values.get("LCH", False) == True
    # incl_SNM = incl_dst_values.get("SNM", "false") == 'true'

    result = []
    for item in values:

        tmp_SNM = item

        if incl_XYZ or incl_LAB or incl_LCH or incl_HEX:
            tmp_SNM = [round(element, 4) for element in item]
            tmp_XYZ = CS_Spectral2XYZ(tmp_SNM, OBSERVER.DEG2)
            tmp_HEX = Cs_XYZ2RGB(tmp_XYZ).to_hex()
            tmp_LAB = Cs_XYZ2LAB(tmp_XYZ)
            tmp_LCH = Cs_Lab2LCH(tmp_LAB)

        entry = {
            "snm": tmp_SNM,
        }
        if incl_LCH:
            entry["lch"] = tmp_LCH.to_json(2)
        if incl_HEX:
            entry["hex"] = tmp_HEX
        if incl_XYZ:
            entry["xyz"] = tmp_XYZ.to_json(2)
        if incl_LAB:
            entry["lab"] = tmp_LAB.to_json(2)

        result.append(entry)
    return result


# -x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.

class ColorTrafo:
    """Optimized color space transformations with full NumPy vectorization"""
    
    def __init__(self, observer: OBSERVER = OBSERVER.DEG2, 
                 refWhite: CsXYZ = Illuminant.D50_DEG2):
        self._set_observer(observer)
        self.refWhite = refWhite.to_numpy().astype(np.float32)
        self.rgb_matrix = np.array(MatrixRGB2XYZ.SRGB_D50, dtype=np.float32)
        self._precompute_constants()

    def _precompute_constants(self):
        """Precompute frequently used constants"""
        self.cie_e = np.float32(216/24389)
        self.cie_k = np.float32(24389/27)
        self.lab_coeffs = np.array([116, 500, -200], dtype=np.float32)
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

    def CS_SNM2XYZ(self, curveNM: np.ndarray) -> np.ndarray:
        """Vectorized spectral to XYZ conversion"""
        return np.dot(curveNM, self.cmf_T)

    def Cs_XYZ2RGB(self, xyz: np.ndarray) -> np.ndarray:
        """Batch XYZ to RGB conversion"""
        scaled_xyz = xyz * self.scale_xyz
        lin_rgb = np.dot(scaled_xyz, self.rgb_matrix.T)
        return self._sRGBcompanding(lin_rgb) * self.scale_rgb

    def Cs_XYZ2HEX(self, xyz: np.ndarray) -> Union[str, np.ndarray]:
        """Vectorized XYZ to HEX conversion"""
        rgb = self.Cs_XYZ2RGB(xyz)
        return self._vectorized_rgb_to_hex(rgb)

    def Cs_XYZ2LAB_OLD(self, xyz: np.ndarray) -> np.ndarray:
        """Optimized XYZ to LAB conversion"""
        rXYZ = (xyz * self.scale_xyz) / self.refWhite
        mask = rXYZ > self.cie_e
        
        rXYZ = np.where(mask, np.cbrt(rXYZ), (self.cie_k * rXYZ + 16) / 116)
        return np.dot(rXYZ, np.diag(self.lab_coeffs)) + [-16, 0, 0]
    
    def Cs_XYZ2LAB(self, xyz: np.ndarray) -> np.ndarray:
        """Corrected and optimized XYZ to LAB conversion"""
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
    
    def Cs_Lab2LCH(self, lab: np.ndarray) -> np.ndarray:
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

    # Optimized internal functions
    def _sRGBcompanding(self, rgb: np.ndarray) -> np.ndarray:
        """Vectorized sRGB companding"""
        mask = rgb > 0.0031308
        return np.where(mask, 1.055 * np.cbrt(rgb) - 0.055, rgb * 12.92)

    def _vectorized_rgb_to_hex(self, rgb: np.ndarray) -> Union[str, np.ndarray]:
        """Batch RGB to HEX conversion"""
        if rgb.ndim == 1:
            return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        
        return np.apply_along_axis(
            lambda x: f"{int(x[0]):02X}{int(x[1]):02X}{int(x[2]):02X}", 
            axis=1, 
            arr=np.clip(rgb, 0, 255).astype(np.uint8)
        )
        
    def Cs_Spectral2Multi(self, values: np.ndarray, incl_dst_values: dict = None) -> list:
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
                    lch = self.Cs_Lab2LCH(lab)
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
        
        
        
    def Cs_Spectral2Multi_Parallel(self, values: np.ndarray, incl_dst_values: dict = None) -> list:
        # Use ThreadPoolExecutor for parallel processing
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing
        
        # Get number of CPU cores, but limit to reasonable number for Lambda
        num_cores = min(multiprocessing.cpu_count(), 4)
        
        spectral = np.asarray(values, dtype=np.float32)
        if spectral.ndim == 1:
            spectral = spectral[np.newaxis, :]
            
        # Split data into chunks for parallel processing
        spectral_chunks = np.array_split(spectral, num_cores)
        
        incl = incl_dst_values or {}
        
        def process_chunk(chunk):
            # Process a single chunk of spectral data
            xyz = lab = lch = hex = None
            rounded_lch = None
            
            if any([incl.get(k, False) for k in ["XYZ", "LAB", "LCH", "HEX"]]):
                xyz = self.CS_SNM2XYZ(chunk)
                
                if incl.get("HEX", False):
                    hex = self.Cs_XYZ2RGB(xyz)
                    hex = np.apply_along_axis(
                        lambda x: f"#{int(x[0]):02X}{int(x[1]):02X}{int(x[2]):02X}", 
                        1, 
                        np.clip(hex, 0, 255).astype(np.uint8)
                    )
                    
                if incl.get("LAB", False) or incl.get("LCH", False):
                    lab = self.Cs_XYZ2LAB(xyz)
                    
                    if incl.get("LCH", False):
                        lch = self.Cs_Lab2LCH(lab)
                        rounded_lch = np.round(lch, 2)
                        
            # Build results for this chunk
            chunk_results = []
            for i, row in enumerate(chunk):
                entry = {"snm": [round(float(num), 4) for num in row]}
                
                if incl.get("XYZ") and xyz is not None:
                    entry["xyz"] = np.round(xyz[i], 2).tolist()
                if incl.get("LAB") and lab is not None:
                    entry["lab"] = np.round(lab[i], 2).tolist()
                if incl.get("LCH") and rounded_lch is not None:
                    entry["lch"] = {
                        "L": float(rounded_lch[i][0]),
                        "C": float(rounded_lch[i][1]),
                        "H": float(rounded_lch[i][2])
                    }
                if incl.get("HEX") and hex is not None:
                    entry["hex"] = hex[i]
                    
                chunk_results.append(entry)
                
            return chunk_results
            
        # Process chunks in parallel using ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            chunk_results = executor.map(process_chunk, spectral_chunks)
            for chunk_result in chunk_results:
                results.extend(chunk_result)
                
        return results       
        

        
    def Cs_Spectral2Multi_OLD(self, values: np.ndarray, incl_dst_values: dict = None) -> list:
        """Vectorized spectral to multiple color space converter"""
        # Convert input to numpy array and pre-process
        spectral = np.asarray(values, dtype=np.float32)
        if spectral.ndim == 1:
            spectral = spectral[np.newaxis, :]  # Handle single sample input
        
        # Round spectral values once for all elements
        rounded_spectral = np.round(spectral, 4)
        
        # Get conversion flags (default all False if not provided)
        incl_XYZ = incl_dst_values.get("XYZ", False)
        incl_LAB = incl_dst_values.get("LAB", False)
        incl_HEX = incl_dst_values.get("HEX", False)
        incl_LCH = incl_dst_values.get("LCH", False)
        
        # Initialize result containers
        results = []
        xyz = lab = lch = hex = None
        
        # Batch compute all required conversions
        if any([incl_XYZ, incl_LAB, incl_LCH, incl_HEX]):
            xyz = self.CS_SNM2XYZ(rounded_spectral)  # Vectorized
            
            if incl_HEX:
                hex = self.Cs_XYZ2RGB(xyz)  # Vectorized RGB conversion
                hex = np.apply_along_axis(
                    lambda x: f"#{int(x[0]):02X}{int(x[1]):02X}{int(x[2]):02X}", 
                    axis=1, 
                    arr=np.clip(hex, 0, 255).astype(np.uint8)
                )
            
            if incl_LAB or incl_LCH:
                lab = self.Cs_XYZ2LAB(xyz)  # Vectorized
                
                if incl_LCH:
                    lch = self.Cs_Lab2LCH(lab)  # Vectorized
        
        # Build results using vectorized operations
        for i in range(len(spectral)):
            entry = {"snm": rounded_spectral[i].tolist()}
            
            if incl_XYZ:
                entry["xyz"] = np.round(xyz[i], 2).tolist()
            if incl_LAB:
                entry["lab"] = np.round(lab[i], 2).tolist()
            if incl_LCH:
                # Extract and format LCH components
                l, c, h = np.round(lch[i], 2)
                entry["lch"] = {
                    "L": float(l),  # Convert numpy float to native float
                    "C": float(c),
                    "H": float(h)
                }
            if incl_HEX:
                entry["hex"] = hex[i]
                
            results.append(entry)
        
        return results


class ColorTrafo_OLD:
    '''    
    This class is responsible for color transformations between different color spaces.
    
    (INFO: NUMPY arrays are used for performance reasons.)
    '''

    def __init__(self, observer: OBSERVER = OBSERVER.DEG2, refWhite: CsXYZ = Illuminant.D50_DEG2):
        self.set_observer(observer)
        
        self.refWhite = refWhite.to_numpy()
        self.rgb_matrix = np.array(MatrixRGB2XYZ.SRGB_D50)
        
    def set_observer(self, observer):
        if observer == OBSERVER.DEG2:
            self.cmf_T = Cmf2degNumpy.weights.T.astype(np.float32, order='C')  # Even faster
        elif observer == OBSERVER.DEG10:
            self.cmf_T = Cmf10degNumpy.weights.T.astype(np.float32, order='C')  # Even faster
        else:
            raise ValueError("observer must be '2deg' or '10deg'")
        
        self.observer = observer


    def CS_SNM2XYZ(self, curveNM: np.ndarray) -> np.ndarray:
        """
        Spectral Data (380-730nm in 10nm steps) to XYZ Color (default: D50, 2 deg.)
        
        :param curveNM: np.ndarray with 36 wavelengths (380-730nm in 10nm steps) 
                       or array of multiple 36 wavelength arrays
        :return: XYZ values as np.ndarray. For multiple inputs returns array of XYZ values
        """
        """ if curveNM.ndim == 1:
            return self.cmf @ curveNM
        else:
            return np.array([self.cmf @ curve for curve in curveNM]) """
        return np.dot(curveNM, self.cmf_T)
        
    def Cs_XYZ2RGB(self, xyz: np.ndarray) -> str:
        """
        Convert a CsXYZ color to a RGB color.

        :param xyz: The CsXYZ color to convert.
        :return: The converted RGB color.
        """
        
        if xyz.ndim == 1:
            return self.xyz_to_rgb(xyz)
        else:
            return np.array([self.xyz_to_rgb(x) for x in xyz])
    
    def Cs_XYZ2HEX(self, xyz: np.ndarray) -> Union[str, np.ndarray]:        
        """
        Convert a CsXYZ color to a HEX color.

        :param xyz: The CsXYZ color to convert.
        :return: The converted HEX color.
        """
        
        if xyz.ndim == 1:
            return self._xyz_to_hex(xyz)
        else:
            return np.array([self._xyz_to_hex(x) for x in xyz])

    def Cs_XYZ2LAB(self, xyz: np.ndarray) -> np.ndarray:

        if xyz.ndim == 1:
            return self._xyz_to_lab(xyz)
        else:
            return np.array([self._xyz_to_lab(x) for x in xyz])

    # - X - X - X - INTERNAL FUNCTIONS - X - X - X -

    def _sRGBcompanding(self, rgb: np.ndarray) -> np.ndarray:
        """
        sRGB companding to delinearize a given CsRGB value.

        :param rgb: numpy array with 3 values
        :return: numpy array with companded RGB values
        """
        mask = rgb > 0.0031308
        return np.where(mask, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055, rgb * 12.92)

    def _xyz_to_lab(self, xyz: np.ndarray) -> np.ndarray:

        # Normalize the XYZ values and divide by the reference white
        rXYZ = (xyz / MAXVAL_CS_XYZ) / self.refWhite

        mask = rXYZ > CIE_E
        rXYZ = np.where(mask, np.power(rXYZ, 1 / 3), (CIE_K * rXYZ + 16) / 116.0)

        return np.array(
            [
                116.0 * rXYZ[1] - 16.0,
                500.0 * (rXYZ[0] - rXYZ[1]),
                200.0 * (rXYZ[1] - rXYZ[2]),
            ]
        )

    def _xyz_to_rgb(self, xyz: np.ndarray) -> str:
        """
        Convert a CsXYZ color to a RGB color.

        :param xyz: The CsXYZ color to convert.
        :return: The converted RGB color.
        """
        
        # Scales the CsXYZ into the range 0-1.
        scaled_xyz: np.ndarray = xyz / MAXVAL_CS_XYZ
        
        # Apply a matrix transformation to convert XYZ to linear RGB
        varRGB: np.ndarray = np.dot(self.rgb_matrix, scaled_xyz)

        # Apply sRGB companding to delinearize the RGB values
        companded_rgb: np.ndarray = self.sRGBcompanding(varRGB)
        
        # Apply scaling and than clipping to 0-255 range 
        return np.clip(companded_rgb * MAXVAL_CS_RGB, 0, MAXVAL_CS_RGB)

    def _xyz_to_hex(self, xyz: np.ndarray) -> str:
        """
        Convert a CsXYZ color to a HEX color.

        :param xyz: The CsXYZ color to convert.
        :return: The converted HEX color.
        """
        
        rgb = self._xyz_to_rgb(xyz)
    
        return ColorTrafo._rgb_to_hex(rgb)

    @staticmethod
    def _rgb_to_hex(rgb: np.ndarray, prefix: bool = True) -> str:
        """
        Convert RGB numpy array to hex string
        
        :param rgb: numpy array with RGB values (0-255)
        :param prefix: whether to include # prefix
        :return: hex color string
        """
        tmphex = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        hexstr = tmphex.upper()
        return hexstr if prefix else hexstr[1:]
