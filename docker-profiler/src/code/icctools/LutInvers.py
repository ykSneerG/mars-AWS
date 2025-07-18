import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize, NonlinearConstraint
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
    
    def invert_lab_batch_kdtree(self, target_labs):
        target_labs_np = np.array(target_labs, dtype=np.float32)
        _, idxs = self.kdtree.query(target_labs_np, k=1)
        return [self.kdtree_cmyk[int(i[0])] for i in idxs]


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

        Ls = np.linspace(0, 100, self.lab_grid_shape[0])
        As = np.linspace(-128, 127, self.lab_grid_shape[1])
        Bs = np.linspace(-128, 127, self.lab_grid_shape[2])

        Lg, Ag, Bg = np.meshgrid(Ls, As, Bs, indexing='ij')
        lab_points = np.stack([Lg, Ag, Bg], axis=-1).reshape(-1, 3)

        btoa = []
        invert_fn = self._invert_lab_point_kdtree if mode == 'kdtree' else self._invert_lab_point_hybrid_OLD

        for lab in tqdm(lab_points, desc=f"Building BToA ({mode})", disable=not verbose):
            cmyk = invert_fn(lab)
            btoa.append(cmyk)

        return np.array(btoa).reshape((*self.lab_grid_shape, 4))
    
    
    
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize, NonlinearConstraint
from sklearn.neighbors import KDTree
from tqdm import tqdm

from scipy.ndimage import gaussian_filter, sobel


class LUTInverterKDTree_V3:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(17, 17, 17), max_ink_limit=2.25):
        self.atob_lut = atob_lut
        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape
        self.max_ink_limit = max_ink_limit
        self.interpolator = self._build_interpolator()
        self.kdtree = None

    def _prepare_lab_grid(self):
        lab_vals = np.array([lab for _, lab in self.atob_lut])
        return lab_vals.reshape((self.cmyk_grid_size,) * 4 + (3,))

    def _build_interpolator(self):
        lab_grid = self._prepare_lab_grid()
        points = [np.linspace(0, 1, self.cmyk_grid_size)] * 4
        return RegularGridInterpolator(points, lab_grid, bounds_error=False, method='linear')

    def _build_kdtree(self, resolution=25):
        c = np.linspace(0, 1, resolution)
        grid = np.array(np.meshgrid(c, c, c, c, indexing='ij')).reshape(4, -1).T
        lab_vals = self.interpolator(grid)
        self.kdtree = KDTree(lab_vals)
        self.kdtree_cmyk = grid

    def invert_lab_batch_kdtree(self, lab_batch):
        labs = np.asarray(lab_batch)
        _, idxs = self.kdtree.query(labs, k=1)
        return self.kdtree_cmyk[idxs[:, 0]]

    def k_target_curve_OLD(self, L, k_start=0.01, k_max=1.0, width=100):
        """Sanfte, aber kontrollierte GCR-Kurve für Schwarzanteil."""
        x = (100 - L) / width
        x = np.clip(x, 0, 1)
        return k_start + (k_max - k_start) * (3 * x ** 2 - 2 * x ** 3)  # Smoothstep
    
    def k_target_curve(self, L, k_start=0.0, k_max=1.0, mid_point=50, softness=12):
        """
        GCR-Kurve mit kontrollierter S-Kurve (sigmoid).
        L: Lightness (0–100)
        k_start: Schwarzstart (0..1)
        k_max: Maximalwert für Schwarz (meist 1.0)
        mid_point: ab wo K zu steigen beginnt (je höher, desto später)
        softness: Übergangsbreite. Höher = weicher.
        """
        x = (mid_point - L) / softness
        sigmoid = 1 / (1 + np.exp(-x))  # Sigmoid (0..1)
        return k_start + (k_max - k_start) * sigmoid

    def _invert_lab_point_kdtree(self, target_lab):
        dist, idx = self.kdtree.query(np.array(target_lab).reshape(1, -1), k=1)
        return self.kdtree_cmyk[idx[0][0]]

    def _invert_lab_point_hybrid(self, target_lab):
        x0 = self._invert_lab_point_kdtree(target_lab)
        
        def compute_gcr_weights(L, steepness=0.15, midpoint=60.0, k_max=25.0, k_min=2.5, cmy_max=10.2, cmy_min=0.01):
            """
            Berechnet k_weight und cmy_weight abhängig von L (0–100)
            - steepness: Wie schnell der Übergang erfolgt
            - midpoint: Der L-Wert, bei dem der Übergang in der Mitte ist
            - k_max / k_min: Maximales/minimales Gewicht für K
            - cmy_max / cmy_min: Maximales/minimales Gewicht für CMY
            """
            
            # Normierter Übergangswert [0..1]
            x = 1 / (1 + np.exp(steepness * (L - midpoint)))  # sigmoid

            # Interpoliere die Gewichte
            k_weight = k_min + (k_max - k_min) * x
            cmy_weight = cmy_min + (cmy_max - cmy_min) * x

            return k_weight, cmy_weight

        def compute_gcr_weights_linear(L,
                                    L_min=0.0, L_mid=25.0, L_max=100.0,
                                    k_max=25.0, k_mid=10.0, k_min=3.0,
                                    cmy_max=0.2, cmy_mid=0.05, cmy_min=0.01):
            """
            Lineare Interpolation der GCR-Gewichte in drei Bereichen:
            - [L_min, L_mid]: Übergang von k_max → k_mid und cmy_max → cmy_mid
            - [L_mid, L_max]: Übergang von k_mid → k_min und cmy_mid → cmy_min
            """
            if L <= L_min:
                return k_max, cmy_max
            elif L <= L_mid:
                alpha = (L - L_min) / (L_mid - L_min)
                k = (1 - alpha) * k_max + alpha * k_mid
                cmy = (1 - alpha) * cmy_max + alpha * cmy_mid
                return k, cmy
            elif L <= L_max:
                alpha = (L - L_mid) / (L_max - L_mid)
                k = (1 - alpha) * k_mid + alpha * k_min
                cmy = (1 - alpha) * cmy_mid + alpha * cmy_min
                return k, cmy
            else:
                return k_min, cmy_min


        def loss(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6

            lab = np.asarray(self.interpolator(cmyk)).flatten()
            delta_e = np.sum((lab - target_lab) ** 2)

            k_target = self.k_target_curve(target_lab[0])

            # Verstärkte GCR-Steuerung je nach Helligkeit
            """ if target_lab[0] < 25:
                k_weight = 25.0
                cmy_weight = 0.2
            elif target_lab[0] < 50:
                k_weight = 10.0
                cmy_weight = 0.05
            else:
                k_weight = 3.0
                cmy_weight = 0.01 """
            k_weight, cmy_weight = compute_gcr_weights(target_lab[0])
            #k_weight, cmy_weight = compute_gcr_weights_linear(target_lab[0])

            k_penalty = k_weight * (cmyk[3] - k_target) ** 2
            cmy_penalty = cmy_weight * np.sum(cmyk[:3] ** 2)

            neutrality_penalty = 0.05 * (lab[1] ** 2 + lab[2] ** 2)

            return delta_e + k_penalty + cmy_penalty + neutrality_penalty

        def loss_1(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6

            lab = np.asarray(self.interpolator(cmyk)).flatten()
            delta_e = np.sum((lab - target_lab) ** 2)

            k_target = self.k_target_curve(target_lab[0])
            k_weight = 10.0 if target_lab[0] < 50 else 3.0
            k_penalty = k_weight * (cmyk[3] - k_target) ** 2

            cmy_weight = 0.05 if target_lab[0] > 70 else 0.01
            cmy_penalty = cmy_weight * np.sum(cmyk[:3] ** 2)

            neutrality_penalty = 0.05 * (lab[1] ** 2 + lab[2] ** 2)

            return delta_e + k_penalty + cmy_penalty + neutrality_penalty

        def loss_OLD(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = np.asarray(self.interpolator(cmyk)).flatten()
            delta_e = np.sum((lab - target_lab) ** 2)

            # Ziel-K berechnen
            k_target = self.k_target_curve(target_lab[0])
            # K-Penalty dynamisch anpassen je nach L-Bereich
            if target_lab[0] < 40:
                k_weight = 0.25
            elif target_lab[0] < 70:
                k_weight = 0.1
            else:
                k_weight = 0.03
            k_penalty = (cmyk[3] - k_target) ** 2 * k_weight

            # CMY minimieren, aber nicht zu stark bestrafen
            cmy_penalty = 0.005 * np.sum(cmyk[:3] ** 2)

            # Grauneutralität erhalten
            neutrality_penalty = 0.1 * (lab[1] ** 2 + lab[2] ** 2)

            return delta_e + k_penalty + cmy_penalty + neutrality_penalty

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
            options={'maxiter': 120, 'ftol': 1e-5}
        )

        return result.x if result.success else x0


    def build_btoa_lut(self, mode='hybrid', gray_eps=2.5, smoothing='tps', smoothing_kwargs=None):
        self._build_kdtree()

        # LAB-Gitter generieren
        Ls = np.linspace(0, 100, self.lab_grid_shape[0])
        As = np.linspace(-128, 127, self.lab_grid_shape[1])
        Bs = np.linspace(-128, 127, self.lab_grid_shape[2])
        Lg, Ag, Bg = np.meshgrid(Ls, As, Bs, indexing='ij')
        lab_points = np.stack([Lg, Ag, Bg], axis=-1).reshape(-1, 3)

        # Initiale grobe Invertierung mit KDTree
        cmyks = self.invert_lab_batch_kdtree(lab_points)

        # Optionale Hybrid-Korrektur entlang Grauachse
        if mode == 'hybrid':
            gray_mask = np.linalg.norm(lab_points[:, 1:3], axis=1) < gray_eps
            gray_idxs = np.where(gray_mask)[0]

            for idx in tqdm(gray_idxs, desc="Hybrid-Inversion Grauachse"):
                lab = lab_points[idx]
                cmyks[idx] = self._invert_lab_point_hybrid(lab)

        # Glättung anwenden
        smoothing_kwargs = smoothing_kwargs or {}

        if smoothing == 'none':
            smoothed_cmyks = cmyks

        elif smoothing == 'tps':
            # Volle CMYK-Glättung
            cmyk_channels = {
                'C': cmyks[:, 0],
                'M': cmyks[:, 1],
                'Y': cmyks[:, 2],
                'K': cmyks[:, 3],
            }
            smoother = TPSSmoother(lab_points, cmyk_channels, **smoothing_kwargs)
            smoothed_dict = smoother.predict(lab_points)
            smoothed_cmyks = np.stack([
                smoothed_dict['C'],
                smoothed_dict['M'],
                smoothed_dict['Y'],
                smoothed_dict['K'],
            ], axis=-1)

        elif smoothing == 'tps_cmy':
            # Nur C, M, Y glätten – K bleibt wie es ist
            cmy_channels = {
                'C': cmyks[:, 0],
                'M': cmyks[:, 1],
                'Y': cmyks[:, 2],
            }
            smoother = TPSSmoother(lab_points, cmy_channels, **smoothing_kwargs)
            smoothed_dict = smoother.predict(lab_points)
            smoothed_cmyks = np.stack([
                smoothed_dict['C'],
                smoothed_dict['M'],
                smoothed_dict['Y'],
                cmyks[:, 3],  # K bleibt unbearbeitet
            ], axis=-1)
                 
        elif smoothing == 'edge_cmy':
            smoothed_cmyks = LUTInverterKDTree_V3.smooth_cmy_channels_preserve_k(
                cmyks.reshape((*self.lab_grid_shape, 4)),
                sigma_edge=smoothing_kwargs.get('sigma_edge', 1.5),
                sigma_inner=smoothing_kwargs.get('sigma_inner', 0.5),
                edge_thresh=smoothing_kwargs.get('edge_thresh', 5.0)
            ).reshape(-1, 4)

        else:
            raise ValueError(f"Unbekannte Glättungsmethode: {smoothing}")

        # Rückgabe als 4D-LUT
        smoothed_grid = smoothed_cmyks.reshape((*self.lab_grid_shape, 4))
        return smoothed_grid



    @staticmethod
    def detect_edges(cmyk_grid, threshold=5.0):
        edges = np.zeros(cmyk_grid.shape[:-1], dtype=bool)
        for i in range(3):  # C, M, Y
            channel = cmyk_grid[..., i]
            grad_mag = np.sqrt(
                sobel(channel, axis=0, mode='nearest')**2 +
                sobel(channel, axis=1, mode='nearest')**2 +
                sobel(channel, axis=2, mode='nearest')**2
            )
            edges |= grad_mag > threshold
        return edges

    @staticmethod
    def smooth_cmy_channels_preserve_k(cmyk_grid, sigma_edge=1.5, sigma_inner=0.5, edge_thresh=5.0):
        grid = cmyk_grid.copy()
        edges = LUTInverterKDTree_V3.detect_edges(grid, threshold=edge_thresh)
        inner_mask = ~edges
        smoothed = np.empty_like(grid)

        for i in range(4):  # C, M, Y, K
            original = grid[..., i]
            if i < 3:  # Nur C, M, Y glätten
                edge_smooth = gaussian_filter(original, sigma=sigma_edge)
                inner_smooth = gaussian_filter(original, sigma=sigma_inner)
                combined = np.where(edges, edge_smooth, inner_smooth)
                smoothed[..., i] = combined
            else:
                smoothed[..., i] = original  # K bleibt erhalten
        return smoothed



    def build_btoa_lut_OLD2(self, mode='hybrid', gray_eps=2.5, smoothing='tps', smoothing_kwargs=None):
            self._build_kdtree()

            # Lab-Gitterpunkte generieren
            Ls = np.linspace(0, 100, self.lab_grid_shape[0])
            As = np.linspace(-128, 127, self.lab_grid_shape[1])
            Bs = np.linspace(-128, 127, self.lab_grid_shape[2])
            Lg, Ag, Bg = np.meshgrid(Ls, As, Bs, indexing='ij')
            lab_points = np.stack([Lg, Ag, Bg], axis=-1).reshape(-1, 3)

            # Grobe Invertierung mit KDTree
            cmyks = self.invert_lab_batch_kdtree(lab_points)

            # Optional: Hybrid-Grauachse verfeinern
            if mode == 'hybrid':
                gray_mask = np.linalg.norm(lab_points[:, 1:3], axis=1) < gray_eps
                gray_idxs = np.where(gray_mask)[0]

                for idx in tqdm(gray_idxs, desc="Hybrid-Inversion Grauachse"):
                    lab = lab_points[idx]
                    cmyks[idx] = self._invert_lab_point_hybrid(lab)

            smoothing_kwargs = smoothing_kwargs or {}

            # Optionale Glättung
            if smoothing == 'tps':
                smoother = TPSSmoother(lab_points, cmyks, **(smoothing_kwargs or {}))
                smoothed_cmyks = smoother.predict(lab_points)
            elif smoothing == 'none':
                smoothed_cmyks = cmyks
            elif smoothing == 'tps_cmy':
                smoothing_kwargs = smoothing_kwargs or {}

                cmy_channels = {
                    'C': cmyks[:, 0],
                    'M': cmyks[:, 1],
                    'Y': cmyks[:, 2],
                }

                # Nur CMY glätten
                smoother = TPSSmoother(lab_points, cmy_channels, **smoothing_kwargs)
                smoothed_dict = smoother.predict(lab_points)  # ← das ist ein dict

                # CMY aus dict, K unbearbeitet anhängen
                smoothed_cmyks = np.stack([
                    smoothed_dict['C'],
                    smoothed_dict['M'],
                    smoothed_dict['Y'],
                    cmyks[:, 3]  # ← K bleibt wie es ist
                ], axis=-1)

            else:
                raise ValueError(f"Unbekannte Glättungsmethode: {smoothing}")

            # Zurück ins LUT-Grid
            smoothed_grid = smoothed_cmyks.reshape((*self.lab_grid_shape, 4))
            return smoothed_grid


    def build_btoa_lut_OLD(self, mode='hybrid', gray_eps=2.5, verbose=True):
        self._build_kdtree()

        # LAB-Gitter
        Ls = np.linspace(0, 100, self.lab_grid_shape[0])
        As = np.linspace(-128, 127, self.lab_grid_shape[1])
        Bs = np.linspace(-128, 127, self.lab_grid_shape[2])
        Lg, Ag, Bg = np.meshgrid(Ls, As, Bs, indexing='ij')
        lab_points = np.stack([Lg, Ag, Bg], axis=-1).reshape(-1, 3)

        # Grob mit KDTree
        cmyks = self.invert_lab_batch_kdtree(lab_points)

        if mode == 'hybrid':
            gray_mask = np.linalg.norm(lab_points[:, 1:3], axis=1) < gray_eps
            gray_idxs = np.where(gray_mask)[0]

            for idx in tqdm(gray_idxs, desc="Hybrid-Inversion Grauachse", disable=not verbose):
                lab = lab_points[idx]
                cmyks[idx] = self._invert_lab_point_hybrid(lab)

        #result = cmyks.reshape((*self.lab_grid_shape, 4))

        # TPS-Glättung
        cmyk_flat = cmyks.reshape(-1, 4)
        cmyk_channels = {
            'C': cmyk_flat[:, 0],
            'M': cmyk_flat[:, 1],
            'Y': cmyk_flat[:, 2],
            'K': cmyk_flat[:, 3],
        }

        smoothing_kwargs = dict(function='thin_plate', smooth=0.25)
        tps_smoother = TPSSmoother(lab_points, cmyk_channels, **smoothing_kwargs)
        smoothed_cmyk = tps_smoother.predict(lab_points)

        smoothed_array = np.stack([
            smoothed_cmyk['C'],
            smoothed_cmyk['M'],
            smoothed_cmyk['Y'],
            smoothed_cmyk['K']
        ], axis=-1).reshape((*self.lab_grid_shape, 4))

        return smoothed_array



class LUTInverterKDTree_V2:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(17, 17, 17), max_ink_limit=2.25):
        self.atob_lut = atob_lut
        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape
        self.max_ink_limit = max_ink_limit
        self.interpolator = self._build_interpolator()
        self.kdtree = None

    def _prepare_lab_grid(self):
        lab_vals = np.array([lab for _, lab in self.atob_lut])
        return lab_vals.reshape((self.cmyk_grid_size,) * 4 + (3,))

    def _build_interpolator(self):
        lab_grid = self._prepare_lab_grid()
        points = [np.linspace(0, 1, self.cmyk_grid_size)] * 4
        return RegularGridInterpolator(points, lab_grid, bounds_error=False, method='linear')

    def _build_kdtree(self, resolution=25):
        c = np.linspace(0, 1, resolution)
        grid = np.array(np.meshgrid(c, c, c, c, indexing='ij')).reshape(4, -1).T
        lab_vals = self.interpolator(grid)
        self.kdtree = KDTree(lab_vals)
        self.kdtree_cmyk = grid

    def invert_lab_batch_kdtree(self, lab_batch):
        labs = np.asarray(lab_batch)
        _, idxs = self.kdtree.query(labs, k=1)
        return self.kdtree_cmyk[idxs[:, 0]]

    # def k_target_curve(self, L, k_start=0.0, k_max=1.0, width=60):
    #     """Definiert den Ziel-K für ein gegebenes L."""
    #     if L > (100 - width):
    #         return k_start + (100 - L) / width * (k_max - k_start)
    #     return k_max

    def k_target_curve(self, L, k_start=0.0, k_max=1.0, width=60):
        """Smoothe GCR: weicher Übergang, sigmoid-artig."""
        x = (100 - L) / width
        x = np.clip(x, 0, 1)
        return k_start + (k_max - k_start) * (3 * x ** 2 - 2 * x ** 3)  # smoothstep

    def _invert_lab_point_hybrid(self, target_lab):
        x0 = self._invert_lab_point_kdtree(target_lab)

        def loss(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = np.asarray(self.interpolator(cmyk)).flatten()
            delta_e = np.sum((lab - target_lab) ** 2)

            # GCR via Schwarzkurve
            k_target = self.k_target_curve(target_lab[0])
            k_penalty = (cmyk[3] - k_target) ** 2 * (0.3 if target_lab[0] < 50 else 0.05)

            # CMY sparen + neutral halten
            cmy_penalty = 0.01 * np.sum(cmyk[:3] ** 2)
            neutrality_penalty = 0.05 * (lab[1] ** 2 + lab[2] ** 2)

            return delta_e + k_penalty + cmy_penalty + neutrality_penalty

        til_constraint = NonlinearConstraint(
            lambda cmyk: np.sum(cmyk),
            lb=0.0,
            ub=self.max_ink_limit
        )

        result = minimize(
            loss, x0=x0, bounds=[(0, 1)] * 4,
            constraints=[til_constraint],
            method='SLSQP',
            options={'maxiter': 100, 'ftol': 1e-5}
        )

        return result.x if result.success else x0

    def _invert_lab_point_kdtree(self, target_lab):
        dist, idx = self.kdtree.query(np.array(target_lab).reshape(1, -1), k=1)
        return self.kdtree_cmyk[idx[0][0]]

    def build_btoa_lut(self, mode='hybrid', gray_eps=2.5, verbose=True):
        self._build_kdtree()

        # LAB-Gitter erzeugen
        Ls = np.linspace(0, 100, self.lab_grid_shape[0])
        As = np.linspace(-128, 127, self.lab_grid_shape[1])
        Bs = np.linspace(-128, 127, self.lab_grid_shape[2])
        Lg, Ag, Bg = np.meshgrid(Ls, As, Bs, indexing='ij')
        lab_points = np.stack([Lg, Ag, Bg], axis=-1).reshape(-1, 3)

        # KDTree-Inversion für alles
        cmyks = self.invert_lab_batch_kdtree(lab_points)

        if mode == 'hybrid':
            gray_mask = np.linalg.norm(lab_points[:, 1:3], axis=1) < gray_eps
            gray_idxs = np.where(gray_mask)[0]

            for idx in tqdm(gray_idxs, desc="Hybrid-Inversion Grauachse", disable=not verbose):
                lab = lab_points[idx]
                cmyks[idx] = self._invert_lab_point_hybrid(lab)

        return cmyks.reshape((*self.lab_grid_shape, 4))
    
    
    



# import numpy as np
# #import pandas as pd
# from scipy.signal import savgol_filter
# from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt

# class CMYKCurveSmoother:
#     def __init__(self, cmyk_dict, window_length=5, polyorder=2, interp_resolution=101):
#         self.L = np.array(cmyk_dict['L'])
#         self.C = np.array(cmyk_dict['C'])
#         self.M = np.array(cmyk_dict['M'])
#         self.Y = np.array(cmyk_dict['Y'])
#         self.K = np.array(cmyk_dict['K'])

#         self.window_length = window_length
#         self.polyorder = polyorder
#         self.interp_resolution = interp_resolution

#         self.smoothed = {}
#         self.interpolated = {}

#     def smooth_curves(self):
#         def smooth(channel):
#             return savgol_filter(channel, self.window_length, self.polyorder)

#         self.smoothed['C'] = np.clip(smooth(self.C), 0, 100)
#         self.smoothed['M'] = np.clip(smooth(self.M), 0, 100)
#         self.smoothed['Y'] = np.clip(smooth(self.Y), 0, 100)
#         self.smoothed['K'] = np.clip(smooth(self.K), 0, 100)
#         self.smoothed['L'] = self.L

#     def interpolate_curves(self):
#         L_dense = np.linspace(0, 100, self.interp_resolution)
#         self.interpolated['L'] = L_dense

#         for ch in ['C', 'M', 'Y', 'K']:
#             f_interp = interp1d(self.smoothed['L'], self.smoothed[ch], kind='cubic')
#             self.interpolated[ch] = np.clip(f_interp(L_dense), 0, 100)

#     def get_smoothed_dict(self):
#         return self.smoothed

#     def get_interpolated_dict(self):
#         return self.interpolated

#     """ def export_csv(self, filename='smoothed_cmyk.csv'):
#         df = pd.DataFrame(self.interpolated)
#         df.to_csv(filename, index=False)
#         print(f"Exported to {filename}") """

#     def plot_comparison(self):
#         for ch in ['C', 'M', 'Y', 'K']:
#             plt.figure(figsize=(6, 3))
#             plt.plot(self.L, getattr(self, ch), 'o-', label=f'{ch} original')
#             plt.plot(self.L, self.smoothed[ch], 'x--', label=f'{ch} smoothed')
#             plt.title(f'{ch} Curve Comparison')
#             plt.xlabel('L*')
#             plt.ylabel(f'{ch} Value')
#             plt.grid(True)
#             plt.legend()
#             plt.tight_layout()
#             plt.show()



from scipy.interpolate import Rbf
import numpy as np

from scipy.interpolate import Rbf
import numpy as np

class TPSSmoother:
    def __init__(self, lab_points, cmyk_channels, function='thin_plate', smooth=5.0):
        """
        lab_points: (N, 3) LAB-Werte
        cmyk_channels: dict mit keys ['C','M','Y','K'] oder ['C','M','Y'] – jeder ein Array (N,)
        function: RBF-Kern (z. B. 'thin_plate'), smooth: Glättungsstärke
        """
        self.models = {}
        self.lab_points = np.array(lab_points)

        for channel, values in cmyk_channels.items():
            values = np.array(values)
            assert values.shape[0] == self.lab_points.shape[0], f"Mismatch bei Channel {channel}"
            self.models[channel] = Rbf(*self.lab_points.T, values, function=function, smooth=smooth)

    def predict(self, lab_array):
        lab_array = np.array(lab_array)
        smoothed = {}
        for channel, model in self.models.items():
            smoothed[channel] = model(*lab_array.T)
        return smoothed