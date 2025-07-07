import numpy as np
from tqdm import tqdm # type: ignore

from scipy.interpolate import interpn
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from scipy.optimize import NonlinearConstraint

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.interpolate import interpn
from scipy.optimize import minimize
from tqdm import tqdm

from src.code.icctools.IccV4_Helper import ColorTrafo


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
            smoothed[..., i] = gaussian_filter(btoa_grid[..., i], sigma=sigma,)
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




import numpy as np
from scipy.interpolate import interpn
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize
#from colour.interpolation.tetrahedral import tetrahedral_interpolation  #type: ignore # ✅ Wichtig!

#from colour. .interpolation.tetrahedral import tetrahedral_interpolation

from scipy.interpolate import griddata

from concurrent.futures import ThreadPoolExecutor, as_completed

class LUTInverterTetrahedral:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(13, 13, 13), use_gcr=False,
                 black_start=0.2, black_width=0.5, black_strength=1.0):
        self.use_gcr = use_gcr
        self.black_start = black_start
        self.black_width = black_width
        self.black_strength = black_strength

        self.atob_lut = atob_lut
        self.cmyk_points = np.array([cmyk for cmyk, _ in atob_lut])
        self.lab_values = np.array([lab for _, lab in atob_lut])

        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape

    def _interpolator(self, query_cmyk):
        # query_cmyk: (N,4) array -> returns (N,3) array
        return griddata(self.cmyk_points, self.lab_values, query_cmyk, method='linear')

    def _gcr_curve(self, gray):
        if gray <= self.black_start:
            return 0.0
        elif gray >= self.black_start + self.black_width:
            return self.black_strength
        else:
            t = (gray - self.black_start) / self.black_width
            return np.clip(t ** 2.2 * self.black_strength, 0, 1)

    def _apply_gcr(self, cmyk):
        C, M, Y, K = cmyk
        gray = min(C, M, Y)
        replace = self._gcr_curve(gray) * gray
        return np.clip([C - replace, M - replace, Y - replace, K + replace], 0, 1)

    def _invert_lab_point(self, target_lab):
        def loss(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self._interpolator(cmyk[np.newaxis])[0]
            if lab is None or np.any(np.isnan(lab)):
                return 1e6
            return np.sum((lab - target_lab) ** 2)  # DE76

        result = minimize(loss, x0=[0.5]*4, bounds=[(0,1)]*4, method='L-BFGS-B')
        cmyk = result.x if result.success else np.array([0.5]*4)
        if self.use_gcr:
            cmyk = self._apply_gcr(cmyk)
        return cmyk

    def build_btoa_lut(self, verbose=True, max_workers=8):
        L = np.linspace(0, 100, self.lab_grid_shape[0])
        a = np.linspace(-128, 127, self.lab_grid_shape[1])
        b = np.linspace(-128, 127, self.lab_grid_shape[2])
        Lg, ag, bg = np.meshgrid(L, a, b, indexing='ij')
        lab_points = np.stack([Lg.ravel(), ag.ravel(), bg.ravel()], axis=-1)

        btoa = np.zeros((lab_points.shape[0], 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._invert_lab_point, lab): i for i, lab in enumerate(lab_points)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Inverting LAB→CMYK", disable=not verbose):
                i = futures[future]
                btoa[i] = future.result()

        return btoa.reshape((*self.lab_grid_shape, 4))

    
    
    
class LUTInverter_Multi:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(11, 11, 11),
                 interpolation_method='tetrahedral'):
        """
        Parameters:
            atob_lut: List of ((C,M,Y,K), (L,a,b)) tuples
            cmyk_grid_size: Grid resolution of the CMYK AToB LUT
            lab_grid_shape: Output LAB grid resolution (for BToA)
            interpolation_method: 'linear' or 'tetrahedral'
        """
        self.atob_lut = atob_lut
        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape
        self.interpolation_method = interpolation_method

        if interpolation_method == 'tetrahedral':
            self.interpolator = self._build_interpolator_tetrahedral()
        elif interpolation_method == 'linear':
            self.interpolator = self._build_interpolator_linear()
        else:
            raise ValueError("interpolation_method must be 'linear' or 'tetrahedral'")

    # --- LINEAR INTERPOLATION ---
    def _prepare_lab_grid(self):
        lab_vals = np.array([lab for _, lab in self.atob_lut])
        return lab_vals.reshape((self.cmyk_grid_size,) * 4 + (3,))

    def _build_interpolator_linear(self):
        lab_grid = self._prepare_lab_grid()
        points = [np.linspace(0, 1, self.cmyk_grid_size)] * 4
        return lambda cmyk: interpn(points, lab_grid, cmyk, bounds_error=False, method='linear')

    # --- TETRAHEDRAL INTERPOLATION ---
    def _prepare_tetrahedral_data(self):
        cmyk_vals = np.array([cmyk for cmyk, _ in self.atob_lut])
        lab_vals = np.array([lab for _, lab in self.atob_lut])
        tri = Delaunay(cmyk_vals)
        return tri, cmyk_vals, lab_vals

    def _build_interpolator_tetrahedral(self):
        tri, cmyk_vals, lab_vals = self._prepare_tetrahedral_data()

        def interpolate(cmyk):
            cmyk = np.asarray(cmyk)
            simplex = tri.find_simplex(cmyk)
            if simplex == -1:
                return np.array([0.0, 0.0, 0.0])  # Out of bounds
            X = tri.transform[simplex, :4]
            Y = cmyk - tri.transform[simplex, 4]
            bary = np.dot(X, Y)
            bary_coords = np.append(bary, 1 - bary.sum())
            indices = tri.simplices[simplex]
            return np.dot(bary_coords, lab_vals[indices])

        return interpolate

    # --- LOSS FUNCTION + INVERSION ---
    def _invert_lab_point(self, target_lab):
        def loss(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            return np.sum((lab - target_lab) ** 2)  # ΔE76

        result = minimize(loss, x0=[0.5] * 4, bounds=[(0, 1)] * 4, method='L-BFGS-B')
        return result.x if result.success else np.array([0.5] * 4)

    # --- BUILD LUT ---
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


import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from sklearn.neighbors import KDTree
from tqdm import tqdm


class LUTInverterKDTree:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(11, 11, 11), loss_function="de76", max_ink_limit=2.25):
        """
        Parameters:
            atob_lut: List of ((C,M,Y,K), (L,a,b)) tuples
            cmyk_grid_size: Resolution of AToB CMYK grid
            lab_grid_shape: Output resolution of the BToA LUT
            loss_function: One of 'de76', 'hybrid'
        """
        self.atob_lut = atob_lut
        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape
        self.loss_function = loss_function
        self.interpolator = self._build_interpolator()
        self.kdtree = None
        
        self.max_ink_limit = max_ink_limit  # z. B. 2.75 für 275%

    def _prepare_lab_grid(self):
        lab_vals = np.array([lab for _, lab in self.atob_lut])
        return lab_vals.reshape((self.cmyk_grid_size,) * 4 + (3,))

    def _build_interpolator(self):
        lab_grid = self._prepare_lab_grid()
        points = [np.linspace(0, 1, self.cmyk_grid_size)] * 4
        return RegularGridInterpolator(points, lab_grid, bounds_error=False, method='linear')

    def _build_kdtree(self, resolution=20):
        c = np.linspace(0, 1, resolution)
        grid = np.array(np.meshgrid(c, c, c, c, indexing='ij')).reshape(4, -1).T  # shape (N,4)
        lab_vals = self.interpolator(grid)  # shape (N,3)
        self.kdtree = KDTree(lab_vals)
        self.kdtree_cmyk = grid

    def _invert_lab_point_kdtree(self, target_lab):
        dist, idx = self.kdtree.query(np.array(target_lab).reshape(1, -1), k=1)
        return self.kdtree_cmyk[idx[0][0]]

    def _invert_lab_point_hybrid_OLD(self, target_lab):
        
        if np.allclose(target_lab, [100, 0, 0], atol=0.5):
            return np.array([0.0, 0.0, 0.0, 0.0])

        
        x0 = self._invert_lab_point_kdtree(target_lab)

        def loss_de76(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            return np.sum((lab - target_lab) ** 2)

        def loss_hybrid_nolimit(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            delta_e = np.sum((lab - target_lab) ** 2)
            penalty = 0.01 * np.sum(np.square(cmyk))
            return delta_e + penalty
        
        def loss_hybrid(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            delta_e = np.sum((lab - target_lab) ** 2)

            # TIL-Penalty (überschreitender Anteil wird bestraft)
            total_ink = np.sum(cmyk)
            ink_excess = max(0.0, total_ink - self.max_ink_limit)
            ink_penalty = 10.0 * ink_excess**2

            return delta_e + 0.01 * np.sum(cmyk**2) + ink_penalty

        def loss_smooth(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            delta_e = np.sum((lab - target_lab)**2)
            smoothness = np.sum(np.diff(cmyk)**2)
            ink_penalty = np.sum(cmyk)
            return delta_e + 0.05 * smoothness + 0.01 * ink_penalty

        def loss_linear_k_OLD(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            delta_e = np.sum((lab - target_lab) ** 2)
            k_target = 1 - target_lab[0] / 100  # L ∈ [0,100]
            k_penalty = (cmyk[3] - k_target) ** 2
            ink_penalty = 0.01 * np.sum(cmyk[:3] ** 2)
            return delta_e + 0.1 * k_penalty + ink_penalty #0.1 instead of 0.4
        
        def loss_linear_k_XXX(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            delta_e = np.sum((lab - target_lab) ** 2)

            # Zielwert für K aus L
            k_target = 1 - target_lab[0] / 100

            # Glättungsgewicht stärker im Dunkeln
            smooth_black_boost = 0.2 if target_lab[0] < 50 else 0.05
            k_penalty = smooth_black_boost * (cmyk[3] - k_target) ** 2

            # Optional: zusätzliche CMY-Tinte minimieren
            cmy_penalty = 0.01 * np.sum(cmyk[:3] ** 2)

            return delta_e + k_penalty + cmy_penalty

        def loss_linear_k(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6

            lab = np.asarray(self.interpolator(cmyk)).flatten()

            delta_e = np.sum((lab - target_lab) ** 2)

            # Zielwert für K basierend auf L
            k_target = 1 - target_lab[0] / 100
            smooth_black_boost = 0.2 if target_lab[0] < 50 else 0.05
            k_penalty = smooth_black_boost * (cmyk[3] - k_target) ** 2

            # Vermeide unnötige CMY
            cmy_penalty = 0.01 * np.sum(cmyk[:3] ** 2)

            # Neutralitätsstrafe (a, b nahe 0)
            neutrality_penalty = 0.05 * (lab[1] ** 2 + lab[2] ** 2)

            return delta_e + k_penalty + cmy_penalty + neutrality_penalty





        # Auswahl der Loss-Funktion:
        if self.loss_function == "hybrid":
            loss = loss_hybrid
        elif self.loss_function == "smooth":
            loss = loss_smooth
        elif self.loss_function == "linear_k":
            loss = loss_linear_k
            # Startwert: K anhand L abschätzen
            k_init = 1 - target_lab[0] / 100
            x0 = np.clip(np.array([0.5, 0.5, 0.5, k_init]), 0, 1)
        else:
            loss = loss_de76

        result = minimize(
            loss, x0=x0, bounds=[(0, 1)] * 4,
            method='L-BFGS-B',
            options={
                'maxiter': 50,
                'gtol': 1e-5,
                'ftol': 1e-5
            }
        )

        return result.x if result.success else x0

    def _invert_lab_point_hybrid(self, target_lab):
        if np.allclose(target_lab, [100, 0, 0], atol=0.5):
            return np.array([0.0, 0.0, 0.0, 0.0])

        x0 = self._invert_lab_point_kdtree(target_lab)

        def loss_de76(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            return np.sum((lab - target_lab) ** 2)

        def loss_hybrid(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            delta_e = np.sum((lab - target_lab) ** 2)
            penalty = 0.01 * np.sum(np.square(cmyk))
            return delta_e + penalty

        if self.loss_function == "hybrid":
            loss = loss_hybrid
        else:
            loss = loss_de76

        # 🧱 TIL: total CMYK must be ≤ self.max_ink_limit
        til_constraint = NonlinearConstraint(
            lambda cmyk: np.sum(cmyk),
            lb=0.0,
            ub=self.max_ink_limit
        )

        result = minimize(
            loss,
            x0=x0,
            bounds=[(0, 1)] * 4,
            constraints=[til_constraint],
            method='SLSQP',
            options={
                'maxiter': 100,
                'ftol': 1e-5,
                'disp': False
            }
        )

        return result.x if result.success else x0


    def build_btoa_lut(self, mode='hybrid', verbose=True, kdtree_resolution=20):
        """
        Build LAB → CMYK LUT.
        mode: 'kdtree' (fast), 'hybrid' (accurate)
        """
        if mode not in ['kdtree', 'hybrid']:
            raise ValueError("Mode must be 'kdtree' or 'hybrid'")

        self._build_kdtree(resolution=kdtree_resolution)

        L_vals = np.linspace(0, 100, self.lab_grid_shape[0])
        a_vals = np.linspace(-128, 127, self.lab_grid_shape[1])
        b_vals = np.linspace(-128, 127, self.lab_grid_shape[2])

        Lg, ag, bg = np.meshgrid(L_vals, a_vals, b_vals, indexing='ij')
        lab_points = np.stack([Lg, ag, bg], axis=-1).reshape(-1, 3)

        btoa = []
        invert_fn = self._invert_lab_point_kdtree if mode == 'kdtree' else self._invert_lab_point_hybrid_OLD

        for lab in tqdm(lab_points, desc=f"Building BToA ({mode})", disable=not verbose):
            cmyk = invert_fn(lab)
            btoa.append(cmyk)

        return np.array(btoa).reshape((*self.lab_grid_shape, 4))


class LUTInverterPur:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(11, 11, 11)):
        """
        Parameters:
            atob_lut: List of ((C,M,Y,K), (L,a,b)) tuples
            cmyk_grid_size: Size of the CMYK grid in AToB table (default 17)
            lab_grid_shape: Resolution of BToA output grid (default 11x11x11)
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
        return lab_vals.reshape((self.cmyk_grid_size,) * 4 + (3,))

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
        def loss_de76(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            return np.sum((lab - target_lab) ** 2)  # Euclidean (ΔE76)
        
        def loss_de00(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            return ColorTrafo.de00_lab(np.asarray(lab), np.asarray(target_lab))  # DeltaE 2000 (ΔE00)

        def loss_hybrid(cmyk):
            '''
            Hybrid Loss: ΔE76 + Smoothness Regularization
            '''
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            delta_e = np.sum((lab - target_lab) ** 2)  # DE76 squared
            # Optionally add a smoothness or ink saving penalty:
            penalty = 0.01 * np.sum(np.square(cmyk))  # encourages ink economy
            return delta_e + penalty

        def loss_smoothed(cmyk):
            '''
            Smoothness loss to encourage gradual transitions between CMY channels.
            '''
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            delta_e76 = np.sum((lab - target_lab) ** 2)
            reg = 0.01 * np.sum(np.diff(cmyk)**2)  # Glattheit zwischen Kanälen fördern
            return delta_e76 + reg

        result = minimize(loss_de76, x0=[0.5] * 4, bounds=[(0, 1)] * 4, method='L-BFGS-B')
        return result.x if result.success else np.array([0.5] * 4)

    def build_btoa_lut(self, verbose=True):
        """
        Generate LAB → CMYK LUT using inversion.
        Returns a 4D array of shape (L, a, b, 4)
        """
        L_vals = np.linspace(0, 100, self.lab_grid_shape[0])
        a_vals = np.linspace(-128, 127, self.lab_grid_shape[1])
        b_vals = np.linspace(-128, 127, self.lab_grid_shape[2])

        Lg, ag, bg = np.meshgrid(L_vals, a_vals, b_vals, indexing='ij')
        lab_points = np.stack([Lg, ag, bg], axis=-1).reshape(-1, 3)

        btoa = []
        for lab in tqdm(lab_points, desc="Inverting LAB→CMYK", disable=not verbose):
            cmyk = self._invert_lab_point(lab)
            btoa.append(cmyk)

        return np.array(btoa).reshape((*self.lab_grid_shape, 4))
    
    from scipy.ndimage import gaussian_filter

    @staticmethod
    def smooth_btoa(btoa, sigma=0.01):
        # sigma steuert die Stärke der Glättung
        smoothed = np.zeros_like(btoa)
        for i in range(4):  # Für jeden Kanal (C, M, Y, K)
            smoothed[..., i] = gaussian_filter(btoa[..., i], sigma=sigma)
        return smoothed
    
    @staticmethod
    def smooth_btoa_median(btoa, size=3):
        from scipy.ndimage import median_filter

        # Glätte über LAB-Achsen, nicht über den Kanal (4)
        # btoa shape is (L, a, b, 4), so size must be (size, size, size, 1)
        smoothed = median_filter(btoa, size=(size, size, size, 1))
        
        return smoothed # np.clip(smoothed, 0, 1)  # Sicherstellen, dass Werte im Bereich [0, 1] bleiben


#this works but creates zackige ud ungleichmäßige Übergänge, der sart und das ende sind nicht gleichmäßig
class LUTInverterGCR:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(11, 11, 11),
                 black_start=0.2, black_width=0.5, black_strength=1.0):
        """
        Parameters:
            atob_lut: List of ((C,M,Y,K), (L,a,b)) tuples
            cmyk_grid_size: Size of the CMYK grid in AToB table (default 17)
            lab_grid_shape: Resolution of BToA output grid (default 33x33x33)
            black_start: Gray level where K replacement begins (0–1)
            black_width: Width of the transition (0–1)
            black_strength: Maximum GCR strength (0–1)
        """
        self.atob_lut = atob_lut
        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape

        self.black_start = black_start
        self.black_width = black_width
        self.black_strength = black_strength

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

    def gcr_curve_sigmoid(self, gray):
        """
        GCR sigmoid curve: controls black generation from CMY gray.
        """
        x = (gray - self.black_start) / self.black_width
        ramp = 1 / (1 + np.exp(-10 * (x - 0.5)))
        return np.clip(ramp * self.black_strength, 0, 1)
    
    def gcr_curve(self, gray):
        """
        Benutzerdefinierte GCR-Kurve mit black_start, black_width und black_strength.
        Nutzt exponentiellen Verlauf innerhalb des Übergangsbereichs.
        """
        if gray <= self.black_start:
            return 0.0
        elif gray >= self.black_start + self.black_width:
            return self.black_strength
        else:
            # Normiere gray in [0, 1] innerhalb des Übergangs
            t = (gray - self.black_start) / self.black_width
            # Wende exponentielle Kurve auf t an (z. B. gamma)
            k = t ** 2.2  # oder andere Kurve
            return np.clip(k * self.black_strength, 0, 1)


    def apply_gcr(self, cmyk):
        """
        Apply black generation and reduce CMY accordingly.
        """
        C, M, Y, K = cmyk
        gray = min(C, M, Y)
        gcr_ratio = self.gcr_curve(gray)
        replace = gcr_ratio * gray

        C -= replace
        M -= replace
        Y -= replace
        K += replace

        return np.clip([C, M, Y, K], 0, 1)

    def _invert_lab_point(self, target_lab):
        """
        Invert a single LAB → CMYK using least-squares optimization + GCR.
        """
        def loss(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            return np.sum((lab - target_lab) ** 2)

        result = minimize(loss, x0=[0.5] * 4, bounds=[(0, 1)] * 4, method='L-BFGS-B')
        cmyk = result.x if result.success else np.array([0.5] * 4)
        return self.apply_gcr(cmyk)

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
        for i in range(4):
            channel = btoa_grid[..., i]
            blurred = gaussian_filter(channel, sigma=sigma)
            slices = tuple(slice(1, -1) for _ in range(3))
            smoothed[..., i] = channel
            smoothed[slices + (i,)] = blurred[slices]
        return smoothed
    
    
    def smooth_lut_3(btoa_grid, sigma=1.0):
        """
        Apply Gaussian smoothing to the CMYK BToA grid,
        but preserve the boundary values (corners and edges).
        """
        smoothed = np.zeros_like(btoa_grid)
        
    @staticmethod
    def smooth_lut_edge_aware(btoa_grid, sigma=1.0, edge_threshold=0.15):
        """
        Apply edge-aware Gaussian smoothing to CMYK channels while preserving:
        1) Boundary values (corners and edges of the LUT)
        2) Sharp transitions where LAB values change rapidly
        3) Color relationships between neighboring points
        
        Parameters:
            btoa_grid: 4D array of shape (L, a, b, 4) containing CMYK values
            sigma: Base smoothing strength
            edge_threshold: LAB difference that triggers edge preservation
        """
        # Convert LAB grid coordinates to actual LAB values
        L = np.linspace(0, 100, btoa_grid.shape[0])
        a = np.linspace(-128, 127, btoa_grid.shape[1])
        b = np.linspace(-128, 127, btoa_grid.shape[2])
        Lg, ag, bg = np.meshgrid(L, a, b, indexing='ij')
        lab_grid = np.stack([Lg, ag, bg], axis=-1)
        
        smoothed = np.zeros_like(btoa_grid)
        
        for i in range(4):  # For each CMYK channel
            channel = btoa_grid[..., i]
            
            # Calculate LAB differences between neighboring points
            lab_diff = np.zeros_like(channel)
            lab_diff[:-1, :, :] += np.sqrt(np.sum((lab_grid[1:, :, :] - lab_grid[:-1, :, :])**2, axis=-1))
            lab_diff[:, :-1, :] += np.sqrt(np.sum((lab_grid[:, 1:, :] - lab_grid[:, :-1, :])**2, axis=-1))
            lab_diff[:, :, :-1] += np.sqrt(np.sum((lab_grid[:, :, 1:] - lab_grid[:, :, :-1])**2, axis=-1))
            
            # Normalize and create edge mask
            lab_diff /= np.max(lab_diff)
            edge_mask = lab_diff > edge_threshold
            
            # Apply different smoothing strengths
            sigma_map = np.where(edge_mask, sigma*0.3, sigma)  # Reduce smoothing at edges
            
            # Process inner points (preserve boundaries)
            inner_slices = tuple(slice(1, -1) for _ in range(3))
            inner_channel = channel[inner_slices]
            inner_sigma = sigma_map[inner_slices]
            
            # Apply Gaussian filter with varying sigma
            blurred = np.zeros_like(inner_channel)
            unique_sigmas = np.unique(inner_sigma)
            
            for s in unique_sigmas:
                mask = (inner_sigma == s)
                if s == 0:  # No smoothing
                    blurred[mask] = inner_channel[mask]
                else:
                    temp = inner_channel.copy()
                    temp[~mask] = 0  # Only process current sigma region
                    temp_blurred = gaussian_filter(temp, sigma=s)
                    count = gaussian_filter(mask.astype(float), sigma=s)
                    temp_blurred[mask] = temp_blurred[mask] / count[mask]
                    blurred[mask] = temp_blurred[mask]
            
            # Combine results
            smoothed[..., i] = channel.copy()
            smoothed[inner_slices + (i,)] = blurred
            
            # Additional color relationship preservation
            if i < 3:  # For CMY channels only
                # Maintain relative proportions between CMY
                for j in range(i):
                    ratio = btoa_grid[..., j] / (btoa_grid[..., i] + 1e-6)
                    smoothed[..., i] = smoothed[..., j] / (ratio + 1e-6)
        
        return np.clip(smoothed, 0, 1)


import numpy as np
from scipy.interpolate import interpn
from scipy.optimize import minimize
from tqdm import tqdm


class LUTInverterGCR_BestButToLight:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(11, 11, 11),
                 black_levels=21, l_target=50, regularization=0.01):
        """
        Args:
            atob_lut: Liste von ((C, M, Y, K), (L, a, b)) Tupeln
            cmyk_grid_size: Auflösung des AToB-LUTs (Standard: 17)
            lab_grid_shape: Auflösung des BToA-Ausgabegitters (Standard: 11x11x11)
            black_levels: Anzahl diskreter K-Stufen für GCR-Lookup-Tabelle
            l_target: Ziel-Luminanzwert für neutrales Grau
            regularization: Gewichtung der CMY-Balance in Optimierung
        """
        self.atob_lut = atob_lut
        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape
        self.l_target = l_target
        self.regularization = regularization

        self.interpolator = self._build_interpolator()
        self.gcr_table = self._build_gcr_lookup_table(levels=black_levels)
        
        self.gcr_lambda = 0.01

    def _prepare_lab_grid(self):
        lab_vals = np.array([lab for _, lab in self.atob_lut])
        return lab_vals.reshape((self.cmyk_grid_size,) * 4 + (3,))

    def _build_interpolator(self):
        lab_grid = self._prepare_lab_grid()
        points = [np.linspace(0, 1, self.cmyk_grid_size)] * 4
        return lambda cmyk: interpn(points, lab_grid, cmyk, bounds_error=False, method='linear')[0]

    def _build_gcr_lookup_table(self, levels=21):
        """
        Erzeugt GCR-Lookup-Tabelle: K → [C, M, Y] (neutraler CMY-Ersatz für gegebenes K)
        """
        gcr_table = {}
        k_values = np.linspace(0, 1, levels)

        for k in k_values:
            target_lab = np.array([self.l_target, 0.0, 0.0])  # neutrales Grau

            def loss(cmy):
                cmy = np.clip(cmy, 0, 1)
                lab = self.interpolator([*cmy, k])
                delta_e = np.linalg.norm(lab - target_lab)
                balance = np.sum((cmy - np.mean(cmy)) ** 2)
                return delta_e + self.regularization * balance

            res = minimize(
                loss,
                x0=[1 - k] * 3,
                bounds=[(0, 1)] * 3,
                method='L-BFGS-B'
            )

            gcr_table[np.round(k, 4)] = res.x if res.success else [1 - k] * 3

        return gcr_table

    def apply_gcr(self, cmyk):
        """
        CMY durch neutrale GCR-Komponente ersetzen, basierend auf vordefinierter Tabelle
        """
        C, M, Y, K = cmyk
        k_rounded = np.round(K, 4)

        keys = sorted(self.gcr_table.keys())
        if k_rounded in self.gcr_table:
            neutral_cmy = self.gcr_table[k_rounded]
        else:
            k_low = max([k for k in keys if k <= K], default=0.0)
            k_high = min([k for k in keys if k >= K], default=1.0)
            cmy_low = np.array(self.gcr_table[k_low])
            cmy_high = np.array(self.gcr_table[k_high])
            t = (K - k_low) / (k_high - k_low + 1e-6)
            neutral_cmy = (1 - t) * cmy_low + t * cmy_high

        new_C = C - neutral_cmy[0]
        new_M = M - neutral_cmy[1]
        new_Y = Y - neutral_cmy[2]
        return np.clip([new_C, new_M, new_Y, K], 0, 1)

    def _invert_lab_point(self, target_lab):
        """
        Invertiert LAB → CMYK per Optimierung + GCR-Anwendung
        """
        def loss_OLD(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk)
            return np.linalg.norm(lab - target_lab)  # ΔE76
        
        def loss(cmyk):
            lab = self.interpolator(cmyk)
            delta_e = np.linalg.norm(lab - target_lab)

            # GCR penalty: encourage CMY ≈ gray if K is high
            C, M, Y, K = cmyk
            grayness = np.std([C, M, Y])  # less variance = more gray
            cmy_total = C + M + Y
            gcr_penalty = K * (grayness + 0.2 * cmy_total)  # penalize non-gray CMY when K is high

            return delta_e + self.gcr_lambda * gcr_penalty


        result = minimize(
            loss,
            x0=[0.5, 0.5, 0.5, 0.0],
            bounds=[(0, 1)] * 4,
            method='L-BFGS-B'
        )

        cmyk = result.x if result.success else np.array([0.5, 0.5, 0.5, 0.0])
        return self.apply_gcr(cmyk)

    def build_btoa_lut(self, verbose=True):
        """
        Baut LAB → CMYK LUT mittels Inversion (inkl. GCR)
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

class LUTInverterGCR_worksbutnotrealyabenefitslow:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(11, 11, 11),
                 black_curve=None, regularization=0.01):
        """
        Parameters:
            atob_lut: List of ((C,M,Y,K), (L,a,b)) tuples
            cmyk_grid_size: Size of the CMYK grid in AToB table (default 17)
            lab_grid_shape: Resolution of BToA output grid (default 11x11x11)
            black_curve: Optional callable L* → [k_levels]; defines GCR strategy
            regularization: Strength of CMY-smoothing regularizer
        """
        self.atob_lut = atob_lut
        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape

        self.black_curve = black_curve
        self.regularization = regularization

        self.interpolator = self._build_interpolator()

    def _prepare_lab_grid(self):
        """
        Reshape LAB values into a 5D array for interpn: (C, M, Y, K, 3)
        """
        lab_vals = np.array([lab for _, lab in self.atob_lut])
        return lab_vals.reshape((self.cmyk_grid_size,) * 4 + (3,))

    def _build_interpolator(self):
        """
        Build 4D CMYK → LAB interpolator
        """
        lab_grid = self._prepare_lab_grid()
        points = [np.linspace(0, 1, self.cmyk_grid_size)] * 4
        return lambda cmyk: interpn(points, lab_grid, np.atleast_2d(cmyk), bounds_error=False, method='linear')[0]

    def _invert_lab_point(self, target_lab):
        """
        Argyll-ähnliche GCR-Inversion: mehrere K-Stufen, ΔE76 + Regularisierung
        """
        if self.black_curve is not None:
            k_levels = self.black_curve(target_lab[0])  # L* → K-Stufen
        else:
            k_levels = np.linspace(0, 1, 7)

        best_loss = float('inf')
        best_cmyk = None

        for k in k_levels:
            def loss(cmy):
                cmy = np.clip(cmy, 0, 1)
                cmyk = np.array([*cmy, k])
                lab = self.interpolator(cmyk)

                # ΔE76 Farbabweichung
                delta_e76 = np.linalg.norm(lab - target_lab)

                # Regularisierung: CMY-Glätte
                gray = np.mean(cmy)
                reg = np.sum((cmy - gray) ** 2)

                return delta_e76 + self.regularization * reg

            # Startschätzung basierend auf Helligkeit
            gray_guess = 1.0 - target_lab[0] / 100.0
            x0 = np.clip([gray_guess] * 3, 0, 1)

            result = minimize(loss, x0=x0, bounds=[(0, 1)] * 3, method='L-BFGS-B')

            if result.success:
                cmy = result.x
                cmyk = [*cmy, k]
                current_loss = loss(cmy)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_cmyk = cmyk

        # Fallback: neutrales Grau
        if best_cmyk is None:
            gray = 1.0 - target_lab[0] / 100.0
            best_cmyk = [gray, gray, gray, 0]

        return np.clip(best_cmyk, 0, 1)

    def build_btoa_lut(self, verbose=True):
        """
        Invertiere LAB → CMYK LUT via optimierter GCR-Inversion.
        Rückgabe: LUT mit shape (L, a, b, 4)
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

class LUTInverterGCR_NOT:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(11, 11, 11),
                 black_strength=1.0, black_start=0.2, black_end=0.8):
        """
        Parameters:
            atob_lut: List of ((C,M,Y,K), (L,a,b)) tuples
            cmyk_grid_size: Size of the CMYK grid in AToB table (default 17)
            lab_grid_shape: Resolution of BToA output grid (default 11x11x11)
            black_strength: max GCR (0–1), how much gray is replaced
            black_start: Luminance (0–1) where K starts to build
            black_end: Luminance (0–1) where K reaches full strength
        """
        self.atob_lut = atob_lut
        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape

        self.black_strength = black_strength
        self.black_start = black_start
        self.black_end = black_end

        self.interpolator = self._build_interpolator()

    def _prepare_lab_grid(self):
        lab_vals = np.array([lab for _, lab in self.atob_lut])
        return lab_vals.reshape(
            (self.cmyk_grid_size,) * 4 + (3,)
        )

    def _build_interpolator(self):
        lab_grid = self._prepare_lab_grid()
        points = [np.linspace(0, 1, self.cmyk_grid_size)] * 4
        return lambda cmyk: interpn(points, lab_grid, cmyk, bounds_error=False, method='linear')

    def gcr_curve(self, lab):
        """
        GCR basierend auf Helligkeit (L) und Neutralität (a,b)
        """
        L, a, b = lab
        # Normalize L to 0–1
        L_norm = np.clip(L / 100.0, 0, 1)
        neutral = 1.0 - np.linalg.norm([a, b]) / 200  # 1.0 = perfekt neutral

        # GCR based on Luminance: sigmoid-style ramp
        t = (L_norm - self.black_start) / (self.black_end - self.black_start)
        ramp = 1 / (1 + np.exp(-10 * (t - 0.5)))

        return np.clip(ramp * self.black_strength * neutral, 0, 1)

    def _invert_lab_point_fixed_k_OLD(self, target_lab, fixed_k):
        """
        Invert LAB to CMYK with fixed K – optimize only CMY.
        """
        def loss(cmy):
            cmyk = np.clip([*cmy, fixed_k], 0, 1)
            lab = self.interpolator(cmyk)
            return np.sum((lab - target_lab) ** 2)

        result = minimize(
            loss,
            x0=[0.5, 0.5, 0.5],
            bounds=[(0, 1)] * 3,
            method='L-BFGS-B'
        )

        if result.success:
            c, m, y = result.x
        else:
            c, m, y = 0.5, 0.5, 0.5

        return np.clip([c, m, y, fixed_k], 0, 1)

    def _invert_lab_point_fixed_k(self, target_lab, fixed_k):
        """
        Invert LAB to CMYK with fixed K – optimize CMY only with regularization.
        """
        def loss(cmy):
            cmy = np.clip(cmy, 0, 1)
            cmyk = np.array([*cmy, fixed_k])
            lab = self.interpolator(cmyk)

            lab_loss = np.sum((lab - target_lab) ** 2)

            # Grau-Symmetrie erzwingen (C ≈ M ≈ Y)
            gray = np.mean(cmy)
            reg = np.sum((cmy - gray) ** 2)

            return lab_loss + 0.05 * reg  # Gewicht tunen

        # Startwert basierend auf Helligkeit
        gray_est = 1.0 - target_lab[0] / 100.0
        x0 = np.clip([gray_est] * 3, 0, 1)

        result = minimize(loss, x0=x0, bounds=[(0, 1)] * 3, method='L-BFGS-B')

        if result.success:
            c, m, y = result.x
        else:
            c, m, y = x0

        return np.clip([c, m, y, fixed_k], 0, 1)


    def build_btoa_lut(self, verbose=True):
        """
        Generate LAB → CMYK LUT using inversion.
        Returns a 4D array of shape (L, a, b, 4)
        """
        L_vals = np.linspace(0, 100, self.lab_grid_shape[0])
        a_vals = np.linspace(-128, 127, self.lab_grid_shape[1])
        b_vals = np.linspace(-128, 127, self.lab_grid_shape[2])

        Lg, ag, bg = np.meshgrid(L_vals, a_vals, b_vals, indexing='ij')
        lab_points = np.stack([Lg, ag, bg], axis=-1).reshape(-1, 3)

        btoa = []
        for lab in tqdm(lab_points, desc="Inverting LAB→CMYK", disable=not verbose):
            k_val = self.gcr_curve(lab)
            cmyk = self._invert_lab_point_fixed_k(lab, k_val)
            btoa.append(cmyk)

        return np.array(btoa).reshape((*self.lab_grid_shape, 4))

class LUTInverterGcrInksaver:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(11, 11, 11),
                 black_start=0.1, black_width=0.2, black_strength=1.0, max_delta_e=2.0):
        """
        Parameters:
            atob_lut: List of ((C,M,Y,K), (L,a,b)) tuples
            cmyk_grid_size: Size of CMYK grid in AToB table (default 17)
            lab_grid_shape: Resolution of BToA output grid
            black_start: CMY gray threshold to start GCR
            black_width: Range of gray where GCR transitions (soft zone)
            black_strength: Multiplier for how much CMY is replaced by K
            max_delta_e: Max allowed color error when applying GCR
        """
        self.atob_lut = atob_lut
        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape
        self.black_start = black_start
        self.black_width = black_width
        self.black_strength = black_strength
        self.max_delta_e = max_delta_e
        self.interpolator = self._build_interpolator()
        
    def _prepare_lab_grid(self):
        lab_vals = np.array([lab for _, lab in self.atob_lut])
        return lab_vals.reshape((self.cmyk_grid_size,) * 4 + (3,))

    def _build_interpolator(self):
        lab_grid = self._prepare_lab_grid()
        points = [np.linspace(0, 1, self.cmyk_grid_size)] * 4
        return lambda cmyk: interpn(points, lab_grid, cmyk, bounds_error=False, method='linear')

    def _apply_gcr(self, cmyk, target_lab):
        """
        Applies GCR to a CMYK value if it improves ink efficiency without color error.
        """
        c, m, y, k = cmyk
        gray = min(c, m, y)
        #print("C ", round(c, 3), " M ", round(m, 3), " Y ", round(y, 3), " K ", round(k, 3), " GRAY ", round(gray, 3))
        #print("_apply_gcr GRAY: ", gray, "BlackStart: ", self.black_start)

        # GCR weight from 0 to 1
        if gray < self.black_start:
            return cmyk  # No GCR
        elif gray > self.black_start + self.black_width:
            gcr_ratio = 1.0
            #print("gray > self.black_start + self.black_width", gcr_ratio)
        else:
            gcr_ratio = (gray - self.black_start) / self.black_width
            #print("(gray - self.black_start) / self.black_width", gcr_ratio)

        gcr_ratio *= self.black_strength
        k_new = k + gray * gcr_ratio
        c_new = c - gray * gcr_ratio
        m_new = m - gray * gcr_ratio
        y_new = y - gray * gcr_ratio

        new_cmyk = np.clip([c_new, m_new, y_new, k_new], 0, 1)

        # Only accept if Lab deviation is small
        predicted = self.interpolator([new_cmyk])[0]
        delta_e = np.linalg.norm(predicted - target_lab)
        if delta_e <= self.max_delta_e:
            return new_cmyk
        else:
            return cmyk  # Keep original

    def _invert_lab_point(self, target_lab):
        def loss(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator([cmyk])[0]
            return np.sum((lab - target_lab) ** 2)

        result = minimize(loss, x0=[0.5] * 4, bounds=[(0, 1)] * 4, method='L-BFGS-B')
        cmyk = result.x if result.success else np.array([0.5] * 4)
        return self._apply_gcr(cmyk, target_lab)

    def build_btoa_lut(self, verbose=True):
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
        smoothed = np.zeros_like(btoa_grid)
        for i in range(4):
            smoothed[..., i] = gaussian_filter(btoa_grid[..., i], sigma=sigma)
        return smoothed


# import numpy as np # type: ignore
from scipy.signal import savgol_filter # type: ignore  # for Savitzky-Golay smoothing

class SavitzkyGolaySmoothing:
    """Savitzky-Golay Smoothing for Spectral Data."""
    
    def __init__(self, pcs_data: np.ndarray, window_length: int = 11, polyorder: int = 3):
        """
        Initialize the smoothing class with spectral data.
        
        :param pcs_data: Spectral data (2D numpy array).
        :param window_length: Window size for SG filter (odd number).
        :param polyorder: Polynomial order for SG filter.
        """
        self.pcs_data = np.array(pcs_data, dtype=np.float32)
        self.window_length = window_length
        self.polyorder = polyorder

        # Ensure window_length is odd and valid
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be an odd integer.")
        
        if self.window_length <= self.polyorder:
            raise ValueError("window_length must be greater than polyorder.")

        self.smoothed_pcs = self.apply_smoothing()

    @property
    def smoothed_pcs_list(self) -> list:
        """Returns the smoothed spectral data as a list."""
        return self.smoothed_pcs.tolist()
    
    def apply_smoothing(self) -> np.ndarray:
        """Apply Savitzky-Golay smoothing using vectorized approach."""
        
        # Apply the Savitzky-Golay filter across all spectral bands (axis 0 - columns)
        smoothed_data = np.apply_along_axis(
            savgol_filter, 0, self.pcs_data, window_length=self.window_length, polyorder=self.polyorder
        )
        
        # Clip negative values to 0 (since spectral values can't be negative)
        smoothed_data = np.clip(smoothed_data, 0, None)  # Clips values below 0 to 0
        
        return smoothed_data


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

