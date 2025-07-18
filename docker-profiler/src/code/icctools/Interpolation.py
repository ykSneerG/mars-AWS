import numpy as np    # type: ignore
from scipy.ndimage import uniform_filter1d

class Combinator:
    
    def __init__(self, num_channels:int=4, num_lattice_points:int=2):
        
        self.num_input = num_channels
        self.num_points = num_lattice_points
                
        self.log = []
                
        self.combos = self.generate_combinations_numpy(self.num_input, self.num_points, False).tolist()
        
    def generate_combinations_numpy(self, num_channels: int = 4, num_lattice_points: int = 2, reverse: bool = True) -> list:
        """
        Generate all possible combinations of output channels with values based on the number of lattice points.

        Parameters:
        output_channels (int): Number of output channels. Defaults to 4.
        num_lattice_points (int): Number of lattice points within the range 0 to 65535. Defaults to 2.
        reverse (bool): If True, the order of the combinations will be reversed. Defaults to True.

        Returns:
        List[Tuple[int]]: A list of tuples representing all combinations.
        """
        
        # Calculate lattice values
        lattice_values = np.round(np.linspace(0, 65535, num_lattice_points)).astype(int)
        
        # Create a grid of indices for combinations
        indices = np.indices((num_lattice_points,) * num_channels).reshape(num_channels, -1).T
        
        # Map indices to lattice values
        all_combos = lattice_values[indices]
        
        # Optionally reverse each combination
        if reverse:
            all_combos = all_combos[:, ::-1]

        return all_combos
    
    def logger(self, msg: str):
        self.log.append(msg)
        
    def print_log(self):
        return "\n".join(self.log)        
    
    @staticmethod
    def combinations_to_csv(combos: list[list[int]]) -> str:
        """
        Convert a list of combinations to a CSV string.

        Parameters:
        combos (List[Tuple[int]]): A list of tuples representing all combinations.

        Returns:
        str: A CSV string representing the combinations.
        """
        csv_str = ""
        for combo in combos:
            csv_str += ";".join(map(str, combo)) + "\n"
        return csv_str

    @staticmethod
    def redirect_channels(combos: list, order: list) -> list[list[int]]:
        """
        ### Redirect channels.
        """
                
        # check order fits to number of channels from combos !!!!!
        num_combos = len(combos[0])
        num_order = len(order)
        
        if num_combos != num_order:
            raise Exception(f"Order {order} does not fit to {combos[0]}")
        
        # order must have all number from 0 len(combos[0]), but not all numbers must be in order
        arr_nums = [i for i in range(len(combos[0]))]
        
        #check that order contains all numbers from arr_nums
        if not all(num in order for num in arr_nums):
            raise Exception(f"Order {order} does not contain all numbers from {arr_nums}")
        
        
        redirected_combos = []

        for combo in combos:
            new_combo = [0] * len(combo)
            for i, elem in enumerate(order):
                new_combo[elem] = combo[i]
                
            redirected_combos.append(new_combo)
        
        return redirected_combos


""" 
Table 23 — Rendering intents
Perceptual 0
Media-relative colorimetric 1
Saturation 2
ICC-absolute colorimetric 3 
"""

class LutMaker:
    
    def __init__(self):
        pass
    

    def perceptual_compression(
        self,
        lab_array: list,
        src_black=(0.0, 0.0, 0.0),
        dst_black=(0.0, 0.0, 0.0),
        src_white=(95.047, 0.0, 108.883),   # typical D65 white point in LAB
        dst_white=(100.0, 0.0, 0.0),        # PCS D50 white point in LAB (L=100)
        preserve_neutral=True,
        lightness_gamma=0.8,
        chroma_compression_power=0.7,
        chroma_threshold=5.0
    ) -> list:
        """
        Applies a perceptual compression transform on LAB colors with black point compensation.
        
        Parameters:
        - lab_array: Nx3 array of LAB values (L*, a*, b*).
        - src_black: LAB black point of source.
        - dst_black: LAB black point of destination.
        - src_white: LAB white point of source.
        - dst_white: LAB white point of destination.
        - preserve_neutral: if True, neutrals (a*=b*=0) won't get chroma compressed.
        - lightness_gamma: gamma curve exponent for lightness compression.
        - chroma_compression_power: exponent for chroma compression (>0 and <=1).
        - chroma_threshold: chroma threshold above which compression applies.
        
        Returns:
        - Nx3 array of transformed LAB values.
        """

        lab = np.array(lab_array, dtype=np.float64).copy()

        # Step 1: Normalize L* relative to black and white
        L = lab[:, 0]
        L_norm = (L - src_black[0]) / max(src_white[0] - src_black[0], 1e-8)
        L_norm = np.clip(L_norm, 0, 1)  # Clamp between 0 and 1

        # Apply gamma to compress lightness and scale to destination black-white range
        L_comp = dst_black[0] + (dst_white[0] - dst_black[0]) * np.power(L_norm, lightness_gamma)

        # Step 2: Calculate chroma relative to source black
        a = lab[:, 1] - src_black[1]
        b = lab[:, 2] - src_black[2]
        chroma = np.sqrt(a**2 + b**2)

        # Step 3: Compress chroma beyond threshold
        chroma_comp = np.where(
            chroma > chroma_threshold,
            chroma_threshold + np.power(chroma - chroma_threshold, chroma_compression_power),
            chroma
        )

        # Avoid division by zero in chroma ratio calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(chroma != 0, chroma_comp / chroma, 1.0)

        # Step 4: Preserve neutral colors if requested
        if preserve_neutral:
            neutral_mask = (a == 0) & (b == 0)
            ratio = np.where(neutral_mask, 1.0, ratio)

        # Step 5: Apply ratio and add back destination black offsets for a,b
        a_comp = dst_black[1] + a * ratio
        b_comp = dst_black[2] + b * ratio

        # Compose final LAB array
        lab_comp = np.stack((L_comp, a_comp, b_comp), axis=-1)

        return lab_comp.tolist()


class LutMakerHybrid:
    
    def __init__(self):
        pass

    def rot_mat(self, src_vec, dst_vec):
        """Rotation matrix from src_vec to dst_vec using Rodrigues' formula."""
        src = np.array(src_vec, dtype=float)
        dst = np.array(dst_vec, dtype=float)
        src /= np.linalg.norm(src)
        dst /= np.linalg.norm(dst)

        v = np.cross(src, dst)
        c = np.dot(src, dst)

        if np.linalg.norm(v) < 1e-8:
            return np.identity(3)

        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        r = np.identity(3) + vx + (vx @ vx) * ((1 - c) / (np.linalg.norm(v) ** 2))
        return r


    def vec_rot_mat(self, s1, s0, t1, t0):
        """Builds 3x4 matrix to map vector s0→s1 onto t0→t1"""
        ss = np.array(s1) - np.array(s0)
        tt = np.array(t1) - np.array(t0)
        r = self.rot_mat(ss, tt)
        rotated_s0 = r @ np.array(s0)
        m = np.zeros((3, 4))
        m[:, :3] = r
        m[:, 3] = np.array(t0) - rotated_s0
        return m


    def apply_3x4_matrix(self, matrix, lab_array):
        lab = np.array(lab_array)
        lab_transformed = (matrix[:, :3] @ lab.T).T + matrix[:, 3]
        return lab_transformed


    def hybrid_perceptual_lut(
        self,
        lab_array,
        src_black,
        src_white,
        dst_black=(0.0, 0.0, 0.0),
        dst_white=(100.0, 0.0, 0.0),
        lightness_gamma=0.8,
        chroma_compression_power=0.7,
        chroma_threshold=5.0,
        preserve_neutral=True
    ):
        lab_array = np.array(lab_array, dtype=np.float64)

        # 1. Build 3×4 transform matrix from src to dst space
        matrix = self.vec_rot_mat(src_white, src_black, dst_white, dst_black)

        # 2. Apply linear transform (scaling, rotation, shift)
        transformed = self.apply_3x4_matrix(matrix, lab_array)

        # 3. Lightness compression
        L = transformed[:, 0]
        L_norm = np.clip(L / 100.0, 0, 1)
        L_comp = 100.0 * np.power(L_norm, lightness_gamma)

        # 4. Chroma compression
        a, b = transformed[:, 1], transformed[:, 2]
        chroma = np.sqrt(a ** 2 + b ** 2)

        chroma_comp = np.where(
            chroma > chroma_threshold,
            chroma_threshold + np.power(chroma - chroma_threshold, chroma_compression_power),
            chroma
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(chroma != 0, chroma_comp / chroma, 1.0)

        # Optional neutral preservation
        if preserve_neutral:
            neutral_mask = (np.abs(a) < 1e-6) & (np.abs(b) < 1e-6)
            ratio = np.where(neutral_mask, 1.0, ratio)

        a_comp = a * ratio
        b_comp = b * ratio

        final = np.stack([L_comp, a_comp, b_comp], axis=-1)
        return final.tolist()


class VectorMapper:
    
    def __init__(self):
        self.m = None
        
    def apply_l_only(self, points, l_min, l_max):
        """
        Map L* from [l_min → l_max] to [0 → 100], preserving a*, b*.
        """
        points_np = np.array(points, dtype=np.float64)

        # Scale L* only
        L = points_np[:, 0]
        a = points_np[:, 1]
        b = points_np[:, 2]

        L_scaled = (L - l_min) / (l_max - l_min) * 100.0

        return np.stack([L_scaled, a, b], axis=1).tolist()
    
    def apply(self, points):

        # Convert to NumPy array (shape: N x 3)
        points_np = np.array(points)

        # Apply transformation matrix (3x4) to all points
        # Split the matrix into rotation and translation
        rotation = self.m[:, :3]     # 3x3
        translation = self.m[:, 3]   # 3x1

        # Apply: rotate and add translation
        transformed_points = (rotation @ points_np.T).T + translation  # shape: N x 3

        # Optionally convert back to list of lists
        transformed_list = transformed_points.tolist()

        # Print result
        """ for i, (src, dst) in enumerate(zip(points, transformed_list)):
            print(f"{i}: {src} --> {dst}") """
            
        return transformed_list


    def rigid_transform_3D(self, A, B):
        """
        Compute the rotation matrix R and translation vector t
        that align two sets of points A and B (Nx3).
        Returns 3x3 R and 3x1 t so that: R*A + t = B (best fit).
        """
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        assert A.shape == B.shape

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # Center the points
        AA = A - centroid_A
        BB = B - centroid_B

        H = AA.T @ BB

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T @ U.T

        t = centroid_B - R @ centroid_A
        return R, t


    def similarity_transform_3D(self, A, B):
        """
        Compute rotation R, translation t, and uniform scale s
        so that s * R * A + t ≈ B.
        """
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        assert A.shape == B.shape

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        AA = A - centroid_A
        BB = B - centroid_B

        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T @ U.T

        # Estimate scale
        var_A = np.sum(np.linalg.norm(AA, axis=1) ** 2)
        s = np.sum(S) / var_A

        t = centroid_B - s * R @ centroid_A

        return s, R, t


    def compute_affine_transform(self, src_points, dst_points):
        """
        Computes a full 3x4 affine transform matrix that maps src_points to dst_points exactly.
        src_points and dst_points must be Nx3 with N ≥ 4 for stable fit (or N=2 for exact match).
        """
        src = np.asarray(src_points)
        dst = np.asarray(dst_points)
        N = src.shape[0]

        # Append 1 for affine
        src_h = np.hstack([src, np.ones((N, 1))])  # N x 4

        # Solve: src_h @ M.T = dst → M.T = np.linalg.lstsq(src_h, dst)[0]
        M_T, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)
        M = M_T.T  # 3 x 4

        return M




    def rot_mat(self, src_vec, dst_vec):
        """Generate a rotation matrix that rotates src_vec to dst_vec."""
        src = np.array(src_vec, dtype=float)
        dst = np.array(dst_vec, dtype=float)
        
        src_norm = src / np.linalg.norm(src)
        dst_norm = dst / np.linalg.norm(dst)
        
        v = np.cross(src_norm, dst_norm)
        c = np.dot(src_norm, dst_norm)
        
        if np.linalg.norm(v) < 1e-8:
            return np.identity(3)

        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        r = np.identity(3) + vx + (vx @ vx) * ((1 - c) / (np.linalg.norm(v) ** 2))
        return r


    def mul_by_3x3(self, matrix, vec):
        """Multiply 3x3 matrix with 3D vector."""
        return np.dot(matrix, vec)


    def mul_3_by_3x4(self, matrix_3x4, vec):
        """Multiply a 3x4 matrix with a 3D vector, adding the translation part."""
        matrix = np.array(matrix_3x4)
        vec = np.array(vec)
        return np.dot(matrix[:, :3], vec) + matrix[:, 3]


    def lab_de_sq(self, lab1, lab2):
        """Squared delta E (Euclidean) between two LAB colors."""
        diff = np.array(lab1) - np.array(lab2)
        return np.dot(diff, diff)


    def vec_rot_mat_OLD3(self, s0, s1, t0, t1):
        M = self.compute_affine_transform([s0, s1], [t0, t1])
        self.m = M 
        return M

    def vec_rot_mat(self, s0, s1, t0, t1):
        A = np.array([s0, s1])
        B = np.array([t0, t1])
        
        R, t = self.rigid_transform_3D(A, B)
        m = np.zeros((3, 4))
        m[:, :3] = R
        m[:, 3] = t 
        
        """
        
        s, R, t = self.similarity_transform_3D(A, B)
        m = np.zeros((3, 4))
        m[:, :3] = s * R
        m[:, 3] = t
        """ 
        self.m = m
        return m

    def vec_rot_mat_OLD(self, s0, t0, s1,  t1):
        """
        Create a 3x4 matrix that transforms (rotation + translation) vector s0→s1 to t0→t1
        """
        s0 = np.array(s0, dtype=float)
        s1 = np.array(s1, dtype=float)
        
        t0 = np.array(t0, dtype=float)
        t1 = np.array(t1, dtype=float)
        
        ss = s1 - s0
        tt = t1 - t0

        # Rotation matrix
        rr = self.rot_mat(ss, tt)

        # Rotate s0
        rotated_s0 = self.mul_by_3x3(rr, s0)

        # Construct 3x4 matrix
        m = np.zeros((3, 4))
        m[:, :3] = rr
        m[:, 3] = t0 - rotated_s0

        # Debug / verification
        tt0 = self.mul_3_by_3x4(m, s0)
        if self.lab_de_sq(t0, tt0) > 1e-4:
            print(f"VecRotMat error t0: got {tt0}, expected {t0}")

        tt1 = self.mul_3_by_3x4(m, s1)
        if self.lab_de_sq(t1, tt1) > 1e-4:
            print(f"VecRotMat error t1: got {tt1}, expected {t1}")

        self.m = m

        return m




    def vec_rot_mat_X(self, s0, s1, t0, t1):
        """
        Compute an exact affine transform (3x4) from two point correspondences:
        s0 → t0 and s1 → t1.
        Supports translation, rotation, and uniform/non-uniform scaling.
        """

        # Prepare data
        src = np.array([s0, s1], dtype=np.float64)
        dst = np.array([t0, t1], dtype=np.float64)

        # Homogenize: add 1 as fourth element
        src_h = np.hstack([src, np.ones((2, 1))])  # Shape: (2, 4)

        # Solve for matrix using least squares
        M_T, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)  # M.T shape: (4, 3)
        M = M_T.T  # Transpose to shape (3, 4)

        self.m = M

        # Debug check
        tt0 = self.mul_3_by_3x4(M, s0)
        tt1 = self.mul_3_by_3x4(M, s1)
        if self.lab_de_sq(t0, tt0) > 1e-8:
            print(f"[Precision Error] t0: got {tt0}, expected {t0}")
        if self.lab_de_sq(t1, tt1) > 1e-8:
            print(f"[Precision Error] t1: got {tt1}, expected {t1}")

        return M



# --- FROM HERE IT WORKS

class VectorTranslate:
    
    def __init__(self, src_point, dst_point):
        self.src_point = np.array(src_point, dtype=float)
        self.dst_point = np.array(dst_point, dtype=float)
        self.translation = self.calculate_translation()
        
    def calculate_translation(self):
        """
        Calculate translation vector from src_points to dst_points.

        Returns:
        numpy.ndarray: Translation vector.
        """
        # Calculate translation vector
        translation = self.dst_point - self.src_point

        return translation
    
    def apply(self, points):
        """
        Apply translation to a list of 3D points.
        
        Parameters:
        points (list of list): List of 3D points to translate.
        
        Returns
        list: Translated 3D points.
        """
        translated_points = [(point + self.translation).tolist() for point in points]
        return translated_points
    
    
class VectorScale:

    def __init__(self):
        pass

    def apply_l_only(self, points, l_min, l_max):
        """
        Map L* from [l_min → l_max] to [0 → 100], preserving a*, b*.
        """
        points_np = np.array(points, dtype=np.float64)

        # Scale L* only
        L = points_np[:, 0]
        a = points_np[:, 1]
        b = points_np[:, 2]

        L_scaled = (L - l_min) / (l_max - l_min) * 100.0

        return np.stack([L_scaled, a, b], axis=1).tolist()

    def apply_l_only_NEW(self, points, l_min, l_max):
        """
        Map L* from [l_min → l_max] to [0 → 100], preserving a*, b*.
        
        Args:
            points: Input LAB colors as list or array (Nx3)
            l_min: Minimum L* value to map from
            l_max: Maximum L* value to map to
        
        Returns:
            Scaled LAB colors in same format as input
        """
        points_np = np.asarray(points, dtype=np.float64)
        
        # Handle single color case
        if points_np.ndim == 1:
            points_np = points_np[np.newaxis, :]
        
        # Scale L* channel only
        L_scaled = (points_np[:, 0] - l_min) * (100.0 / (l_max - l_min))
        
        # Clip to ensure values stay in [0, 100] range
        L_scaled = np.clip(L_scaled, 0, 100)
        
        # Combine with original a*, b* channels
        result = np.column_stack((L_scaled, points_np[:, 1], points_np[:, 2]))
        
        # Return in same format as input
        if isinstance(points, list):
            if len(points) > 0 and isinstance(points[0], (list, np.ndarray)):
                return result.tolist()
            return result[0].tolist()
        return result


    @staticmethod
    def apply_ab_factor(points, a_factor, b_factor):
        """
        Scale a* and b* channels by given factors.
        
        Parameters:
        points (list of list): List of LAB points.
        a_factor (float): Scaling factor for a* channel.
        b_factor (float): Scaling factor for b* channel.
        
        Returns:
        list: Scaled LAB points.
        """
        points_np = np.array(points, dtype=np.float64)
        
        # Scale a* and b* channels
        a_scaled = points_np[:, 1] * a_factor
        b_scaled = points_np[:, 2] * b_factor
        
        return np.stack([points_np[:, 0], a_scaled, b_scaled], axis=1).tolist()
    
    

    @staticmethod
    def scale_ab_saturation_photoshop_gamma(points, factor=1.2, hue_bins=360,
                                            smooth_hull=True, base_smooth_size=5,
                                            adaptive=True, gamma=0.5, alpha=0.2):
        """
        Photoshop-ähnliche Sättigungsskalierung mit Gamma-basiertem L*-Weighting.
        Optimiert für Cloud/Docker (schnell, speichersparend).

        Args:
            points (list): [[L*, a*, b*], ...] Punkte (z. B. aus CLUT).
            factor (float): Chroma-Skalierungsfaktor (0.4–1.0 erhöht Sättigung, >1.0 entsättigt).
            hue_bins (int): Anzahl Hue-Bins (180 für Speed, 360 für Präzision).
            smooth_hull (bool): Chroma-Hülle glätten.
            base_smooth_size (int): Basisfenster für Glättung.
            adaptive (bool): Passe Glättung an die Punktdichte pro Hue-Bin an.
            gamma (float): Steuert Mitteltöne (0.4–0.7 ist typisch).
            alpha (float): Minimalgewicht in Lichtern/Tiefen (0.1–0.3).

        Returns:
            list: LAB-Punkte mit angepasster Sättigung.
        """
        lab = np.asarray(points, dtype=np.float64)
        L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]

        # Chroma und Hue berechnen
        C = np.hypot(A, B)
        H = np.degrees(np.arctan2(B, A)) % 360
        hue_indices = np.floor(H * hue_bins / 360).astype(int)

        # Maximalchroma + Dichte pro Hue-Bin
        C_max = np.zeros(hue_bins, dtype=np.float64)
        counts = np.zeros(hue_bins, dtype=np.float64)
        np.maximum.at(C_max, hue_indices, C)
        np.add.at(counts, hue_indices, 1)
        C_max[C_max == 0] = 1e-6

        # Adaptive Glättung der Hülle
        if smooth_hull and base_smooth_size > 1:
            if adaptive:
                density_factor = 1.5 - (counts / (counts.max() + 1e-6))
                density_factor = np.clip(density_factor, 1.0, 3.0)
                smooth_sizes = np.clip((base_smooth_size * density_factor).astype(int), 1, hue_bins//4)
            else:
                smooth_sizes = np.full(hue_bins, base_smooth_size, dtype=int)

            max_window = smooth_sizes.max()
            smoothed = uniform_filter1d(C_max, size=max_window, mode="wrap")
            weight = (smooth_sizes - smooth_sizes.min()) / (smooth_sizes.max() - smooth_sizes.min() + 1e-6)
            C_max = (1 - weight) * C_max + weight * smoothed

        # Gamma-basierte L*-Gewichtung (Photoshop-Style)
        L_norm = np.clip(L / 100.0, 0, 1)
        l_weight = (L_norm ** gamma) * (1 - alpha) + alpha  # schützt Lichter/Tiefen

        # Ziel-Chroma
        C_target = C * (factor * l_weight + (1 - l_weight))

        # Soft-Clipping an geglättete Hülle
        C_hull = C_max[hue_indices]
        delta = C_hull - C
        t = np.clip((C_target - C) / (delta + 1e-6), 0, 1)
        C_final = C + t * delta

        # Neue a/b-Werte
        scale = np.divide(C_final, C, out=np.ones_like(C), where=C != 0)
        A *= scale
        B *= scale

        return np.column_stack((L, A, B)).tolist()

    

    @staticmethod
    def scale_ab_saturation_with_hull_photoshop2(points, factor=1.2, l_curve="cosine", hue_bins=360, smooth_hull=True, smooth_size=5):
        """
        Optimized saturation scaling in LAB space with chroma limits.
        Mimics Photoshop-like behavior with soft clipping and L* weighting.

        Sinnvolle Parameter:
        factor:
            greater 1.0 → will desaturate colors
            1.0 → no effect, original colors
            0.8–0.9 → slighter saturation enhancement
            0.4–0.6 → stronger saturation enhancement
            0.2–0.3 → extreme saturation enhancement, can lead to clipping
        hue_bins:
            180 ist schnell und ausreichend.
            360 ist feiner, besser bei großen LUTs (aber mehr Rechenaufwand).


        Args:
            points (list): List of [L*, a*, b*] values (e.g., CLUT points).
            factor (float): Chroma scaling factor (>1.0 boosts saturation).
            l_curve (str): L*-weighting ("cosine", "linear", "none").
            hue_bins (int): Number of hue bins for estimating chroma hull.
            smooth_hull (bool): Smooth the chroma hull for continuity.
            smooth_size (int): Kernel size for hull smoothing.

        Returns:
            list: LAB values with scaled saturation.
        """
        lab = np.asarray(points, dtype=np.float64)
        L, A, B = lab.T

        # Chroma & Hue
        C = np.hypot(A, B)
        H = np.degrees(np.arctan2(B, A)) % 360
        hue_indices = np.floor(H * hue_bins / 360).astype(int)

        # Chroma hull per hue bin
        C_max_per_hue = np.zeros(hue_bins)
        np.maximum.at(C_max_per_hue, hue_indices, C)
        C_max_per_hue[C_max_per_hue == 0] = 1e-6

        # Smooth hull (avoids jagged transitions between hues)
        if smooth_hull:
            C_max_per_hue = uniform_filter1d(C_max_per_hue, size=smooth_size, mode="wrap")

        # L* weighting
        if l_curve == "cosine":
            # Softere Variante: in Lichtern/Tiefen nicht komplett auf 0
            l_weight = (np.cos((L - 50) / 100 * np.pi) * 0.5 + 0.5) ** 2
        elif l_curve == "linear":
            l_weight = np.clip(L / 100.0, 0, 1)
        else:  # "none"
            l_weight = np.ones_like(L)

        # Target chroma Photohsop-like
        #C_target = C * (1 + (factor - 1) * l_weight)
        # Target chroma clamped to hull
        C_target = C * (factor * l_weight + (1 - l_weight))

        # Soft-Clipping: sanft an die Hülle annähern, nicht hart abschneiden
        C_hull = C_max_per_hue[hue_indices]
        delta = C_hull - C
        t = np.clip((C_target - C) / (delta + 1e-6), 0, 1)
        C_final = C + t * delta

        # Rescale A/B based on new chroma
        scale = np.divide(C_final, C, out=np.ones_like(C), where=C != 0)
        A_new = A * scale
        B_new = B * scale

        return np.column_stack((L, A_new, B_new)).tolist()

    
    
    @staticmethod
    def scale_ab_saturation_with_hull_photoshop(points, factor=1.2, l_curve="cosine", hue_bins=360):
        """
        Optimized saturation scaling in LAB space with chroma limits.
        
        Args:
            points (list): List of [L*, a*, b*] points (CLUT).
            factor (float): Chroma scaling factor.
            l_curve (str): L*-weighting ("cosine", "linear", "none").
            hue_bins (int): Number of hue bins for chroma limits.
            
        Returns:
            list: New LAB values with saturation scaling.
        """
        lab = np.asarray(points, dtype=np.float64)
        L, A, B = lab.T  # Transposed for direct unpacking

        # Chroma + hue
        C = np.hypot(A, B)
        H = np.degrees(np.arctan2(B, A)) % 360
        hue_indices = np.floor(H * hue_bins / 360).astype(int)

        # Max C per hue
        C_max_per_hue = np.zeros(hue_bins)
        np.maximum.at(C_max_per_hue, hue_indices, C)
        C_max_per_hue[C_max_per_hue == 0] = 1e-6
        
        # Optionaler Glättungsschritt 
        # C_max_per_hue = uniform_filter1d(C_max_per_hue, size=5, mode="wrap")

        # L* weighting
        if l_curve == "cosine":
            l_weight = np.cos(np.clip((L - 50) / 100 * np.pi, -np.pi / 2, np.pi / 2)) ** 2
        elif l_curve == "linear":
            l_weight = np.clip(L / 100.0, 0, 1)
        else:
            l_weight = np.ones_like(L)

        # Chroma scaling with clamping
        C_target = C * (1 + (factor - 1) * l_weight)
        C_final = np.minimum(C_target, C_max_per_hue[hue_indices])
        
        """ t = (C_target - C) / (C_max_per_hue[hue_indices] - C)
        t = np.clip(t, 0, 1)
        C_final = C + t * (C_max_per_hue[hue_indices] - C) """


        # AB rescaling
        scale = np.divide(C_final, C, out=np.ones_like(C), where=C != 0)
        a_new = A * scale
        b_new = B * scale

        return np.column_stack((L, a_new, b_new)).tolist()

    @staticmethod
    def scale_ab_saturation_with_hull(points, factor=1.2, l_curve="cosine", hue_bins=360):
        """
        Skaliert Chroma in der AB-Ebene mit Helligkeitsgewichtung,
        beschränkt durch Chroma-Hülle (pro Hue).

        Args:
            points (list): Liste von [L*, a*, b*] Punkten (CLUT).
            factor (float): Chroma-Skalierungsfaktor.
            l_curve (str): L*-Gewichtung ("cosine", "linear", "none").
            hue_bins (int): Anzahl der Hue-Bins zur Hüllenermittlung.

        Returns:
            list: Neue LAB-Werte mit saturationsgewichteter Skalierung.
        """
        lab = np.asarray(points, dtype=np.float64)
        L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

        # Chroma + Hue berechnen
        C = np.sqrt(a**2 + b**2)
        H = (np.degrees(np.arctan2(b, a)) + 360) % 360  # Hue in [0, 360)

        # Hüllenkurve: max C* pro Hue-Bin
        hue_indices = np.floor(H * hue_bins / 360).astype(int)
        C_max_per_hue = np.zeros(hue_bins)

        for i in range(hue_bins):
            C_max_per_hue[i] = np.max(C[hue_indices == i]) if np.any(hue_indices == i) else 1e-6

        # L*-gewichtung
        if l_curve == "cosine":
            l_weight = np.cos((L - 50) / 100 * np.pi) ** 2
        elif l_curve == "linear":
            l_weight = L / 100.0
        else:
            l_weight = np.ones_like(L)

        # Ziel-Chroma berechnen + clamping zur Hülle
        C_target = C * (1 + (factor - 1) * l_weight)
        C_max_clamp = C_max_per_hue[hue_indices]
        C_final = np.minimum(C_target, C_max_clamp)

        # zurück zu A/B
        H_rad = np.radians(H)
        a_new = C_final * np.cos(H_rad)
        b_new = C_final * np.sin(H_rad)

        result = np.stack([L, a_new, b_new], axis=1)
        return result.tolist()

    
    @staticmethod
    def scale_ab_saturation_l_weighted(points, factor, l_curve="cosine"):
        """
        Sättigung abhängig von L*: z. B. weniger in Schatten/Spitzlichtern.

        Args:
            points (list): [[L, a, b], ...]
            factor (float): Maximaler Chroma-Scaling-Faktor
            l_curve (str): Kurventyp ('cosine', 'linear', 'quad')

        Returns:
            list: LAB-Liste mit angepasster Sättigung
        """
        lab = np.asarray(points, dtype=np.float64)
        L = lab[:, 0]
        a = lab[:, 1]
        b = lab[:, 2]
        chroma = np.sqrt(a**2 + b**2)

        # Skalar (0–1) basierend auf L-Kurve
        if l_curve == "cosine":
            # Maximale Sättigung bei L=50, reduziert bei L=0/100
            l_weight = np.cos((L - 50) / 100 * np.pi) ** 2
        elif l_curve == "linear":
            l_weight = 1.0 - np.abs(L - 50) / 50
        elif l_curve == "quad":
            l_weight = 1.0 - ((L - 50) / 50) ** 2
        else:
            l_weight = np.ones_like(L)

        # Skaliere Chroma
        scaling = 1.0 + (factor - 1.0) * l_weight
        with np.errstate(divide='ignore', invalid='ignore'):
            a_scaled = a * scaling
            b_scaled = b * scaling

        return np.stack([L, a_scaled, b_scaled], axis=1).tolist()


    def apply_lch_factor(points, c_factor):
        """
        Scale chroma (C*) in LAB colors by a factor, preserving L and hue.

        Args:
            points (list of [L, a, b]): Input LAB colors.
            c_factor (float): Scaling factor for Chroma.

        Returns:
            list: LAB colors with scaled chroma.
        """
        points_np = np.asarray(points, dtype=np.float64)

        A = points_np[:, 1]
        B = points_np[:, 2]

        # Compute original chroma
        C = np.sqrt(A**2 + B**2)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(C > 0, (C * c_factor) / C, 1.0)

        A_scaled = A * scale
        B_scaled = B * scale

        result = np.stack([points_np[:, 0], A_scaled, B_scaled], axis=1)
        return result.tolist()


    @staticmethod
    def scale_l_to_0_100(lab_colors):
        """
        Forces L* channel to exactly 0-100 range (min → 0, max → 100)
        while keeping a* and b* unchanged.
        
        Args:
            lab_colors: Input LAB colors (list of lists or NumPy array)
            
        Returns:
            LAB colors with L* remapped to 0-100 (same format as input)
        """
        lab_np = np.asarray(lab_colors, dtype=np.float64)
        
        # Handle single color (shape = [3,]) vs multiple colors (shape = [N,3])
        if lab_np.ndim == 1:
            lab_np = lab_np[np.newaxis, :]
        
        L = lab_np[:, 0]  # Extract L* channel
        l_min, l_max = np.min(L), np.max(L)
        
        if l_max == l_min:
            # Edge case: All L* values are the same → set to 50 (mid-point)
            lab_np[:, 0] = 50.0
        else:
            # Linearly remap L* so min → 0, max → 100
            lab_np[:, 0] = (L - l_min) * (100.0 / (l_max - l_min))
        
        # Return in the same format as input
        if isinstance(lab_colors, list):
            return lab_np.tolist() if lab_np.shape[0] > 1 else lab_np[0].tolist()
        return lab_np if lab_np.shape[0] > 1 else lab_np[0]
    
    def scale_l_to_0_100_me(labs: list, l_min: float, l_max: float, bpc: bool = False) -> list:
        
        labs = np.asarray(labs, dtype=np.float64)
                
        labs[:, 0] = labs[:, 0] - l_min 
        labs[:, 0] = labs[:, 0] * 100.0 / (l_max - l_min)
        labs[:, 0] = np.clip(labs[:, 0], 0, 100)
        
        # apply a blackpoint compensation to L 
        # NOT WORKING
        if bpc:
            gamma: float = 2.2
            bpc_start: float = 75.0
            bpc_end: float = 0.0
            
            L = labs[:, 0]
            mask = L < bpc_start
            
            L_norm = L[mask] / bpc_start
            
            L_gamma = np.power(L_norm, 1.0 / gamma)
            print(f"scale_l_to_0_100_me: bpc_start={bpc_start}, gamma={gamma}, mask={mask}, L_norm={L_norm}, L_gamma={L_gamma}, min={np.min(L_gamma)}, max={np.max(L_gamma)}")
            
            labs[mask, 0] = L_gamma * bpc_start
            
            """ L = labs[:, 0]
            mask = L < bpc_start
            print(f"scale_l_to_0_100_me: bpc_start={bpc_start}, gamma={gamma}, mask={mask}")

            # Normieren in [0,1] innerhalb [0, bpc_start]
            L_norm = L[mask] / bpc_start
            print(f"scale_l_to_0_100_me: L_norm={L_norm} / Length={len(L_norm)}")

            # Gamma-Kurve → bleibt in [0,1]
            L_curve = np.power(L_norm, 1.0 / gamma)

            # Zurückskalieren auf [0, bpc_start]
            L_bpc = L_curve * bpc_start

            # Ersetzen nur der betroffenen Werte
            labs[mask, 0] = L_bpc """

        
        return labs.tolist()
    
    @staticmethod
    def scale_l_to_0_100_matrix(labs: list, l_min: float, l_max: float) -> list:
        """
        Linearly remap L* values to range 0–100 based on two reference indices.
        a* and b* remain unchanged.

        Args:
            lab_colors (list): List of [L, a, b] values.
            index_Min (int): Index of the LAB color with lowest L*.
            index_Max (int): Index of the LAB color with highest L*.

        Returns:
            list: Scaled LAB colors.
        """
        
        print(f"scale_l_to_0_100_matrix: l_min={l_min}, l_max={l_max}")
        
        # Convert input to NumPy array (handles lists and arrays)
        lab = np.asarray(labs, dtype=np.float64)
        
        # Avoid division by zero (already handled by l_max > l_min check)
        scale_L = 100.0 / (l_max - l_min)
        offset_L = -scale_L * l_min
        
        print(f"scale_L: {scale_L}, offset_L: {offset_L}")
        
        # Apply scaling only to L* channel (faster than matrix multiplication)
        lab_scaled = lab.copy()
        lab_scaled[:, 0] = lab[:, 0] * scale_L + offset_L
        
        #print(f"Scaled LAB: {lab_scaled[0]}")  # Print a few samples for debugging
        #darkest = np.min(lab_scaled[:, 0])
        #print(f"Darkest L* after scaling: {darkest}")

        # Return in same format as input
        return lab_scaled.tolist()

    @staticmethod
    def scale_l_to_0_100_indexed(lab_colors, index_Min, index_Max):
        """
        Linearly scale L* values to the 0–100 range based on two indices.
        a* and b* channels are left unchanged.

        Args:
            lab_colors (np.ndarray): Array of shape (N, 3) with LAB colors.
            index_Min (int): Index of color to map L* to 0.
            index_Max (int): Index of color to map L* to 100.

        Returns:
            np.ndarray: LAB array with scaled L* values.
        """
        lab_np = np.asarray(lab_colors, dtype=np.float64)
        if lab_np.ndim == 1:
            lab_np = lab_np[np.newaxis, :]

        N = lab_np.shape[0]
        if not (0 <= index_Min < N) or not (0 <= index_Max < N):
            raise IndexError("index_Min or index_Max out of bounds")

        l_min = lab_np[index_Min, 0]
        l_max = lab_np[index_Max, 0]

        if l_max == l_min:
            return lab_np.copy()  # avoid modifying in place

        # Scale L* channel
        lab_scaled = lab_np.copy()
        lab_scaled[:, 0] = (lab_scaled[:, 0] - l_min) * (100.0 / (l_max - l_min))
        lab_scaled[:, 0] = np.clip(lab_scaled[:, 0], 0, 100)

        return lab_scaled.tolist()

    @staticmethod
    def scale_l_to_0_100_indexed_OLD(lab_colors, index_Min, index_Max):
        """
        Forces L* channel to exactly 0-100 range (min → 0, max → 100)
        while keeping a* and b* unchanged.

        Args:
            lab_colors: Input LAB colors (list of lists or NumPy array)
            index_Min: Index for minimum L* (mapped to 0)
            index_Max: Index for maximum L* (mapped to 100)

        Returns:
            LAB colors with L* remapped to 0-100 (same format as input)
        """
        
        print(f"scale_l_to_0_100_indexed: index_Min={index_Min}, index_Max={index_Max}")
        
        
        lab_np = np.asarray(lab_colors, dtype=np.float64)

        # Handle single color (shape = [3,]) vs multiple colors (shape = [N,3])
        if lab_np.ndim == 1:
            lab_np = lab_np[np.newaxis, :]

        L = lab_np[:, 0]  # Extract L* channel

        # Use provided indices for min/max, fallback to np.min/np.max if out of bounds
        l_min = lab_np[index_Min, 0] if 0 <= index_Min < len(lab_np) else np.min(L)
        l_max = lab_np[index_Max, 0] if 0 <= index_Max < len(lab_np) else np.max(L)

        if l_max == l_min:
            # Avoid division by zero: return unchanged
            return lab_np.tolist() if lab_np.shape[0] > 1 else lab_np[0].tolist()
        else:
            # Linearly remap L* so l_min → 0, l_max → 100
            lab_np[:, 0] = (L - l_min) * (100.0 / (l_max - l_min))
            # Clip L* to [0, 100]
            lab_np[:, 0] = np.clip(lab_np[:, 0], 0, 100)

        scaled_clut = lab_np.tolist()
        print("LAB MAX:", scaled_clut[index_Max])
        print("LAB MIN:", scaled_clut[index_Min])

        # Return in the same format as input
        if isinstance(lab_colors, list):
            return lab_np.tolist() if lab_np.shape[0] > 1 else lab_np[0].tolist()
        return lab_np if lab_np.shape[0] > 1 else lab_np[0]

    @staticmethod
    def scale_l_to_0_100_indexed_OLD(lab_colors, index_Min, index_Max):
        """
        Forces L* channel to exactly 0-100 range (min → 0, max → 100)
        while keeping a* and b* unchanged.
        
        Args:
            lab_colors: Input LAB colors (list of lists or NumPy array)
            
        Returns:
            LAB colors with L* remapped to 0-100 (same format as input)
        """
        lab_np = np.asarray(lab_colors, dtype=np.float64)
        
        # Handle single color (shape = [3,]) vs multiple colors (shape = [N,3])
        if lab_np.ndim == 1:
            lab_np = lab_np[np.newaxis, :]
        
        L = lab_np[:, 0]  # Extract L* channel
        l_min = lab_np[index_Min, 0] if index_Min < len(lab_np) else np.min(L)
        l_max = lab_np[index_Max, 0] if index_Max < len(lab_np) else np.max(L)

        if l_max == l_min:
            return lab_np.tolist() if lab_np.shape[0] > 1 else lab_np[0].tolist()
        else:
            # Linearly remap L* so min → 0, max → 100
            lab_np[:, 0] = (L - l_min) * (100.0 / (l_max - l_min))
        
        # Return in the same format as input
        if isinstance(lab_colors, list):
            return lab_np.tolist() if lab_np.shape[0] > 1 else lab_np[0].tolist()
        return lab_np if lab_np.shape[0] > 1 else lab_np[0]
    
