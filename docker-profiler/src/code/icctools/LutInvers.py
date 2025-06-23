import numpy as np
from tqdm import tqdm # type: ignore

from scipy.interpolate import interpn
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

class LUTInverter:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(11, 11, 11)):
        """
        Parameters:
            atob_lut: List of ((C,M,Y,K), (L,a,b)) tuples
            cmyk_grid_size: Size of the CMYK grid in AToB table (default 17)
            lab_grid_shape: Resolution of BToA output grid (default 33x33x33)
        """
        self.atob_lut = atob_lut
        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape
        self.interpolator = self._build_interpolator()
        
    def _prepare_lab_grid(self):
        """
        Reshape LAB values into a 4D grid for interpolation.
        """
        lab_vals = np.array([lab for _, lab in self.atob_lut])
        return lab_vals.reshape(
            (self.cmyk_grid_size,) * 4 + (3,)
        )

    def _build_interpolator(self):
        """
        Build 4D CMYK → LAB interpolator using linear interpolation.
        """
        lab_grid = self._prepare_lab_grid()
        points = [np.linspace(0, 1, self.cmyk_grid_size)] * 4
        return lambda cmyk: interpn(points, lab_grid, cmyk, bounds_error=False, method='linear')

    def _invert_lab_point(self, target_lab):
        """
        Invert a single LAB → CMYK using least-squares optimization.
        """
        def loss(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            return np.sum((lab - target_lab) ** 2)

        result = minimize(loss, x0=[0.5] * 4, bounds=[(0, 1)] * 4, method='L-BFGS-B')
        return result.x if result.success else np.array([0.5] * 4)

    def build_btoa_lut(self, verbose=True):
        """
        Generate LAB → CMYK LUT using inversion.
        Returns a 4D array of shape (L, a, b, 4)
        """
        L = np.linspace(0, 100, self.lab_grid_shape[0])
        a = np.linspace(-128, 127, self.lab_grid_shape[1])
        b = np.linspace(-128, 127, self.lab_grid_shape[2])

        Lg, ag, bg = np.meshgrid(L, a, b, indexing='ij')
        lab_points = np.stack([Lg, ag, bg], axis=-1).reshape(-1, 3)

        btoa = []
        for lab in tqdm(lab_points, desc="Inverting LAB→CMYK", disable=not verbose):
            cmyk = self._invert_lab_point(lab)
            btoa.append(cmyk)

        return np.array(btoa).reshape((*self.lab_grid_shape, 4))



    @staticmethod
    def smooth_lut(btoa_grid, sigma=1.0):
        """
        Apply Gaussian smoothing to CMYK channels in the BToA LUT.
        """
        smoothed = np.zeros_like(btoa_grid)
        for i in range(4):
            smoothed[..., i] = gaussian_filter(btoa_grid[..., i], sigma=sigma)
        return smoothed
    
    @staticmethod
    def smooth_lut_2(btoa_grid, sigma=1.0):
        """
        Apply Gaussian smoothing to the CMYK BToA grid,
        but preserve the boundary values (corners and edges).
        """
        smoothed = np.zeros_like(btoa_grid)

        # Smooth each CMYK channel independently
        for i in range(4):
            channel = btoa_grid[..., i]
            
            # Smooth full channel
            blurred = gaussian_filter(channel, sigma=sigma)
            
            # Copy only interior (excluding 1 voxel border)
            slices = tuple(slice(1, -1) for _ in range(3))
            smoothed[..., i] = channel  # Start with original
            smoothed[slices + (i,)] = blurred[slices]

        return smoothed




# # BACKUP VERSION - for reference, not used in production

from scipy.spatial import cKDTree #type: ignore
from itertools import product

# class LUTInverter_BACKUP:
#     def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(11, 11, 11)):
#         """
#         Parameters:
#             atob_lut: List of ((C,M,Y,K), (L,a,b)) tuples
#             cmyk_grid_size: Size of the CMYK grid in AToB table (default 17)
#             lab_grid_shape: Resolution of BToA output grid (default 33x33x33)
#         """
#         self.atob_lut = atob_lut
#         self.cmyk_grid_size = cmyk_grid_size
#         self.lab_grid_shape = lab_grid_shape
#         self.interpolator = self._build_interpolator()
        
        
#     # def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(33, 33, 33)):
#     #     self.atob_lut = atob_lut
#     #     self.cmyk_grid_size = cmyk_grid_size
#     #     self.lab_grid_shape = lab_grid_shape
#     #     self.cmyk_vals = np.array([cmyk for cmyk, _ in atob_lut])
#     #     self.lab_vals = np.array([lab for _, lab in atob_lut])
#     #     self.kdtree = cKDTree(self.lab_vals)

#     # def build_btoa_lut_fast(self, verbose=True):
#     #     """
#     #     Fast KD-tree based BToA LUT generation.
#     #     """
#     #     # Generate LAB target grid
#     #     L = np.linspace(0, 100, self.lab_grid_shape[0])
#     #     a = np.linspace(-128, 127, self.lab_grid_shape[1])
#     #     b = np.linspace(-128, 127, self.lab_grid_shape[2])

#     #     Lg, ag, bg = np.meshgrid(L, a, b, indexing='ij')
#     #     lab_grid = np.stack([Lg, ag, bg], axis=-1).reshape(-1, 3)

#     #     # Batch KD-tree lookup (vectorized)
#     #     _, indices = self.kdtree.query(lab_grid)
#     #     btoa_flat = self.cmyk_vals[indices]

#     #     if verbose:
#     #         print("BToA inversion completed using KD-tree (fast path).")

#     #     return btoa_flat.reshape((*self.lab_grid_shape, 4))

#     def _prepare_lab_grid(self):
#         """
#         Reshape LAB values into a 4D grid for interpolation.
#         """
#         lab_vals = np.array([lab for _, lab in self.atob_lut])
#         return lab_vals.reshape(
#             (self.cmyk_grid_size,) * 4 + (3,)
#         )

#     def _build_interpolator(self):
#         """
#         Build 4D CMYK → LAB interpolator using linear interpolation.
#         """
#         lab_grid = self._prepare_lab_grid()
#         points = [np.linspace(0, 1, self.cmyk_grid_size)] * 4
#         return lambda cmyk: interpn(points, lab_grid, cmyk, bounds_error=False, method='linear')

#     def _invert_lab_point(self, target_lab):
#         """
#         Invert a single LAB → CMYK using least-squares optimization.
#         """
#         def loss(cmyk):
#             if np.any((cmyk < 0) | (cmyk > 1)):
#                 return 1e6
#             lab = self.interpolator(cmyk)
#             return np.sum((lab - target_lab) ** 2)

#         result = minimize(loss, x0=[0.5] * 4, bounds=[(0, 1)] * 4, method='L-BFGS-B')
#         return result.x if result.success else np.array([0.5] * 4)

#     def build_btoa_lut(self, verbose=True):
#         """
#         Generate LAB → CMYK LUT using inversion.
#         Returns a 4D array of shape (L, a, b, 4)
#         """
#         L = np.linspace(0, 100, self.lab_grid_shape[0])
#         a = np.linspace(-128, 127, self.lab_grid_shape[1])
#         b = np.linspace(-128, 127, self.lab_grid_shape[2])

#         Lg, ag, bg = np.meshgrid(L, a, b, indexing='ij')
#         lab_points = np.stack([Lg, ag, bg], axis=-1).reshape(-1, 3)

#         btoa = []
#         for lab in tqdm(lab_points, desc="Inverting LAB→CMYK", disable=not verbose):
#             cmyk = self._invert_lab_point(lab)
#             btoa.append(cmyk)

#         return np.array(btoa).reshape((*self.lab_grid_shape, 4))

#     @staticmethod
#     def smooth_lut(btoa_grid, sigma=1.0):
#         """
#         Apply Gaussian smoothing to CMYK channels in the BToA LUT.
#         """
#         smoothed = np.zeros_like(btoa_grid)
#         for i in range(4):
#             smoothed[..., i] = gaussian_filter(btoa_grid[..., i], sigma=sigma)
#         return smoothed
    
#     @staticmethod
#     def smooth_lut_2(btoa_grid, sigma=1.0):
#         """
#         Apply Gaussian smoothing to the CMYK BToA grid,
#         but preserve the boundary values (corners and edges).
#         """
#         smoothed = np.zeros_like(btoa_grid)

#         # Smooth each CMYK channel independently
#         for i in range(4):
#             channel = btoa_grid[..., i]
            
#             # Smooth full channel
#             blurred = gaussian_filter(channel, sigma=sigma)
            
#             # Copy only interior (excluding 1 voxel border)
#             slices = tuple(slice(1, -1) for _ in range(3))
#             smoothed[..., i] = channel  # Start with original
#             smoothed[slices + (i,)] = blurred[slices]

#         return smoothed


# def build_btoa_lut(atob_lut, lab_grid_shape=(33, 33, 33)):
#     """
#     Inverts CMYK → LAB LUT to LAB → CMYK (BToA) using nearest neighbor.
    
#     Parameters:
#         atob_lut: list of tuples [((C,M,Y,K), (L,a,b))]
#         lab_grid_shape: shape of the BToA output grid (default 33³)

#     Returns:
#         btoa_grid: ndarray of shape (L,a,b,4), each entry is CMYK
#     """
#     # 1. Convert AToB LUT to NumPy arrays
#     cmyk_vals = np.array([cmyk for cmyk, _ in atob_lut])
#     lab_vals = np.array([lab for _, lab in atob_lut])

#     # 2. Build KD-tree on LAB
#     tree = cKDTree(lab_vals)

#     # 3. Generate uniform LAB grid
#     L_range = (0, 100)
#     a_range = (-128, 127)
#     b_range = (-128, 127)

#     L_lin = np.linspace(*L_range, lab_grid_shape[0])
#     a_lin = np.linspace(*a_range, lab_grid_shape[1])
#     b_lin = np.linspace(*b_range, lab_grid_shape[2])

#     Lg, ag, bg = np.meshgrid(L_lin, a_lin, b_lin, indexing='ij')
#     lab_grid = np.stack([Lg, ag, bg], axis=-1).reshape(-1, 3)

#     # 4. Query nearest LAB in AToB table
#     _, indices = tree.query(lab_grid)

#     # 5. Map back to CMYK
#     btoa_flat = cmyk_vals[indices]
#     btoa_grid = btoa_flat.reshape((*lab_grid_shape, 4))  # final shape: (L, a, b, 4)

#     return btoa_grid

