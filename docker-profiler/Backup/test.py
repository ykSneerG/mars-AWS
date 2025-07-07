import numpy as np
from tqdm import tqdm
from scipy.interpolate import interpn
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

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

        # GCR weight from 0 to 1
        if gray < self.black_start:
            return cmyk  # No GCR
        elif gray > self.black_start + self.black_width:
            gcr_ratio = 1.0
        else:
            gcr_ratio = (gray - self.black_start) / self.black_width

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
