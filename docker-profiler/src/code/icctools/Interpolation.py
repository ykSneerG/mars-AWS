import numpy as np    # type: ignore

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


