import numpy as np    # type: ignore
from src.code.icctools.LutInvers import LUTInverterKDTree_V3
from src.code.icctools.IccV4_Helper import Helper
from src.code.icctools.Interpolation import VectorScale
from scipy.interpolate import RegularGridInterpolator


class MakeXtoXBase:
    
    def __init__(self, lut):
        self._clut = lut
        self._index_lightest: int = 0
        self._index_darkest: int = len(self._clut) - 1
        
    @property
    def clut(self):
        return self._clut
    
    @clut.setter
    def clut(self, clut):
        """
        Set the CLUT (Color Look-Up Table).

        Args:
            clut (list): The CLUT (Color Look-Up Table) to set.
        """
        self._clut = clut

    @property
    def index_lightest(self):
        return self._index_lightest

    @index_lightest.setter
    def index_lightest(self, index):
        self._index_lightest = index
                
    @property
    def index_darkest(self):
        return self._index_darkest

    @index_darkest.setter
    def index_darkest(self, index):
        self._index_darkest = index
        
    @property
    def lightest(self):
        return self.clut[self.index_lightest]
    
    @property
    def lightest_rounded(self):
        return [round(x, 4) for x in self.clut[self.index_lightest]]

    @property
    def darkest(self):
        """
        Get the darkest LAB color.

        Returns:
            list: The darkest LAB color.
        """
        return self.clut[self.index_darkest]
    
    @property
    def darkest_rounded(self):
        return [round(x, 4) for x in self.clut[self.index_darkest]]
    

class MakeAtoB(MakeXtoXBase):

    def __init__(self, lut, dcs):
        super().__init__(lut)
        self.dcs = dcs

    @property
    def clut_as_uint16(self):
        labs = Helper.lab_to_uint16(self.clut)
        return Helper.flatten_list(labs)


    def generate_perceptual(self):
        """
        ### Generate the PERCEPTUAL LUT (AtoB0) 
        """
        self._map_darkest_L_NEW()

    def generate_saturation(self):
        """
        ### Generate the SATURATION LUT (AtoB2) 
        """
        self._map_darkest_L_NEW()
        self._scale_channel_AB()

    def generate_relative(self):
        """
        ### Generate the RELATIVE LUT (AtoB1) 
        """
        pass
    
    def report(self, freeInfo: str = ""):
        """
        Print the AtoB LUT information.
        """
        print(f"AtoB {freeInfo} LUT Information")
        print(f"{freeInfo}\tLightest - Index: ", self.index_lightest, "\t\tLAB: ", self.lightest_rounded)
        print(f"{freeInfo}\tDarkest. - Index: ", self.index_darkest, "\t\tLAB: ", self.darkest_rounded)
        print("")

    def _scale_channel_AB(self, scale_ab: float = 0.4):
        """ 
        Scale a* and b* channels from input LUT to output LUT.
        """
        # self.clut = VectorScale.apply_ab_factor(self.clut, scale_ab, scale_ab)
        # self.clut = VectorScale.apply_lch_factor(self.clut, scale_ab)
        # self.clut = VectorScale.scale_ab_saturation_l_weighted(self.clut, scale_ab, "cosine")
        # self.clut = VectorScale.scale_ab_saturation_with_hull(self.clut, factor=0.8, l_curve="cosine", hue_bins=360)
        # self.clut = VectorScale.scale_ab_saturation_with_hull_photoshop(self.clut, factor=scale_ab, l_curve="cosine", hue_bins=360)
        # self.clut = VectorScale.scale_ab_saturation_with_hull_photoshop2(self.clut, factor=scale_ab, l_curve="cosine", hue_bins=180, smooth_hull=True, smooth_size=30)
        self.clut = VectorScale.scale_ab_saturation_photoshop_gamma(self.clut, factor=scale_ab)


    def _map_darkest_L_NEW(self):
        """
        Map darkest L* to 0.
        """
        print("--Mapping L* to 0...100")
        print("Lightest", np.max(self.clut, axis=0).tolist())
        print("Darkest L* value before mapping: ", self.darkest_rounded[0])
        print("Lightest L* value before mapping: ", self.lightest_rounded[0])
        self.clut = VectorScale.scale_l_to_0_100_me(self.clut, self.darkest[0], self.lightest[0])
        # self.clut = VectorScale.scale_l_to_0_100_me(self.clut, 42, 100, bpc=True)
        print("Darkest L* value after mapping: ", self.darkest_rounded[0])
        print("Lightest L* value after mapping: ", self.lightest_rounded[0])
        print("--")


    def interpolate_clut(self, grid_size: int = 17):
        """
        Interpolate the LUT to a denser grid.
        
        Parameters:
        - grid_size: number of interpolation points per channel
        
        Returns:
        - dense_clut: ndarray of shape (grid_size**channels, 3)
        """
        
        dense_dcs, dense_lab = MakeAtoB.interpolate_grid(self.dcs, self.clut, grid_size)
        self.clut = dense_lab.tolist()
        self.dcs = dense_dcs.tolist()
        
    @staticmethod
    def interpolate_grid(dcs, lab, grid_size: int = 17):
        """
        Interpolate Lab values over a denser device coordinate grid.
        
        Parameters:
        - dcs: list of tuples or ndarray of shape (n, channels), values 0–100
        - lab: list of Lab values (L, a, b), shape (n, 3)
        - grid_size: number of interpolation points per channel
        
        Returns:
        - dense_dcs: ndarray of shape (grid_size**channels, channels), values 0–100
        - dense_lab: ndarray of shape (grid_size**channels, 3)
        """

        dcs = np.asarray(dcs, dtype=np.float32) / 100.0
        lab = np.asarray(lab, dtype=np.float32)

        num_points, num_channels = dcs.shape
        n = int(round(num_points ** (1 / num_channels)))
        assert n ** num_channels == num_points, f"Ungültige Anzahl von Punkten: {num_points} ist kein vollständiges {num_channels}-dimensionales Gitter."

        axis_vals = np.linspace(0, 1, n)
        lab_grid = lab.reshape([n] * num_channels + [3])

        # Interpolatoren für L, a, b
        interpolators = [
            RegularGridInterpolator([axis_vals] * num_channels, lab_grid[..., i], bounds_error=False, fill_value=None)
            for i in range(3)
        ]

        # Neues Gitter mit meshgrid erzeugen (statt itertools.product)
        dense_axis = np.linspace(0, 1, grid_size)
        mesh = np.meshgrid(*([dense_axis] * num_channels), indexing="ij")
        dense_dcs = np.stack([m.flatten() for m in mesh], axis=-1)  # Shape: (grid_size**num_channels, num_channels)

        # Interpoliere LAB
        dense_lab = np.stack([interp(dense_dcs) for interp in interpolators], axis=-1)

        # Zurückskalieren auf 0–100
        dense_dcs = dense_dcs * 100.0

        return dense_dcs, dense_lab
    

class MakeBtoA(MakeXtoXBase):
    
    def __init__(self, dcs, lut, grid_size=11):
        super().__init__(lut)
        self.dcs = dcs
        self.grid_size = grid_size

    @property
    def clut_as_uint16(self) -> list:
        """
        Convert the CLUT to a flat list of uint16 values.
        """
        arr = np.asarray(self.clut)
        flat = (arr * 65535.0).round().astype(np.uint16).ravel()
        return flat.tolist()

    def generate(self):
        """
        Generate the BtoA LUT..
        """
        
        inverter = LUTInverterKDTree_V3(
            atob_lut=list(zip(self.dcs, self.clut)),
            cmyk_grid_size=np.unique(np.array(self.dcs)).size,
            lab_grid_shape=(self.grid_size, self.grid_size, self.grid_size),
            max_ink_limit=3.0
        )
        # self.clut = inverter.build_btoa_lut(mode='hybrid', gray_eps=2.5)
        # self.clut = inverter.build_btoa_lut(mode='hybrid', gray_eps=22.5, smoothing='none')
        
        self.clut = inverter.build_btoa_lut(
            mode='hybrid',
            gray_eps=22.5,
            smoothing='edge_cmy',
            smoothing_kwargs={'sigma_edge': 0.5, 'sigma_inner': 0.2, 'edge_thresh': 3.5}
        )
        
        
        """ inverter = LUTInverterKDTree(
            atob_lut=list(zip(self.dcs, self.clut)),
            cmyk_grid_size=np.unique(np.array(self.dcs)).size,
            lab_grid_shape=(self.grid_size, self.grid_size, self.grid_size),
            loss_function='linear_k',
            max_ink_limit=4.00
        )
        self.clut = inverter.build_btoa_lut(mode='hybrid', kdtree_resolution=20) """

    def report(self, freeInfo: str = ""):
        """
        Print the BtoA LUT information.
        """
        #print("- + - + - + - + - + - + - + - + - + - + -")
        print("")
        print(f"BtoA {freeInfo} LUT Information")
        print("Shape: ", self.clut.shape)
        print("Length: ", int(self.clut.size / self.clut.shape[3]))
        #print("First entry: ", self.clut[0][0][0])
        #print("Last entry: ", self.clut[-1][-1][-1])
        print("Channel Count: ", self.clut.shape[3])
        print("Grid Size: ", self.grid_size)
        #print("")
        print("- + - + - + - + - + - + - + - + - + - + -")
        
    def get_interpolator(self):
        """
        Returns a BToAInterpolator instance to map LAB → CMYK.
        """
        L_vals = np.linspace(0, 100, self.grid_size)
        a_vals = np.linspace(-128, 127, self.grid_size)
        b_vals = np.linspace(-128, 127, self.grid_size)
        
        lab_grid = np.stack(np.meshgrid(L_vals, a_vals, b_vals, indexing='ij'), axis=-1)

        return BToAInterpolator(
            lab_grid_shape=self.clut.shape[:3],
            lab_vals=lab_grid.reshape(-1, 3),
            cmyk_vals=self.clut.reshape(-1, 4)
        )


class MakeCurve:
    
    @staticmethod
    def createZigZagCurve_LAB(low, high):
        """ 
        Create a zigzag curve between low and high values.

        Parameters:
        low (float): The low value in the range of 0 to 1
        high (float): The high value in the range of 0 to 1
        
        """
        TOP = 255  # 255 / 100 = 2.552

        low_scaled = int(low * TOP)
        high_scaled = int(TOP - high * TOP)

        mid = TOP - high_scaled - low_scaled
        step = TOP / mid
        
        curve = [0] * low_scaled
        
        for i in range(mid):
            curve.append(i * step)
            
        curve += [TOP] * high_scaled
        
        for i in range(len(curve)):
            curve[i] = int(curve[i] / 255 * 65534)

        if len(curve) > 256:
            curve = curve[:256]

        print("Curve: ", len(curve))

        print("low_scaled: ", low_scaled, "high_scaled: ", high_scaled)
        
        
        return curve

        """ [0, 0, 16384, 32768, 49152, 65534, 65534],
        [int(i / 6 * 65534) for i in range(7)], """
    
    @staticmethod
    def createZigZagCurve_LAB_OLD(low, high):
        
        # I need a make a curve that starts at 0 and goes to 65535. The x axis is the L of LAB range from 0 to 100, and the y axis is the L* value in the range of 0 to 65535.
        # The curve should be a zigzag pattern, with the first point at 0, and stays at 0 until the Low L* value is reached, then it goes up in a straint line to the High L* value.


        low_scaled = int(low * 2.55)
        high_scaled = int(high * 2.55)

        mid = 255 - high_scaled - (255 - low_scaled)
        step = 255 / mid
        
        curve = [0] * (255 - low_scaled)
        
        for i in range(mid):
            curve.append(i * step)
            
        curve += [255] * (high_scaled + 5)
        
        for i in range(len(curve)):
            curve[i] = int(curve[i] / 255 * 65534)

        if len(curve) > 256:
            curve = curve[:256]

        print("Curve: ", len(curve))

        print("low_scaled: ", low_scaled, "high_scaled: ", high_scaled)
        
        
        return curve

        """ [0, 0, 16384, 32768, 49152, 65534, 65534],
        [int(i / 6 * 65534) for i in range(7)], """


class BToAInterpolator:
    def __init__(self, lab_grid_shape, lab_vals, cmyk_vals):
        self.lab_grid_shape = lab_grid_shape
        self.lab_vals = lab_vals.reshape((*lab_grid_shape, 3))
        self.cmyk_vals = cmyk_vals.reshape((*lab_grid_shape, 4))
        self.L_axis = np.linspace(0, 100, lab_grid_shape[0])
        self.a_axis = np.linspace(-128, 127, lab_grid_shape[1])
        self.b_axis = np.linspace(-128, 127, lab_grid_shape[2])
        self.interpolator = RegularGridInterpolator(
            points=(self.L_axis, self.a_axis, self.b_axis),
            values=self.cmyk_vals,
            bounds_error=False,
            fill_value=np.nan
        )

    def lab_to_cmyk(self, lab_input):
        lab_input = np.asarray(lab_input)
        if lab_input.ndim == 1:
            lab_input = lab_input.reshape(1, 3)
            dcs = self.interpolator(lab_input)[0]
        elif lab_input.ndim == 2 and lab_input.shape[1] == 3:
            dcs = self.interpolator(lab_input)
        else:
            raise ValueError("Input must be shape (3,) or (N, 3)")

        cmyk_value_np = np.array(dcs, dtype=np.float32) * 100  # scale to 100%
        cmyk_value_np = np.clip(cmyk_value_np, 0, 100)  # clip to [0, 100]
        
        cmyk_C = cmyk_value_np[:, 0]
        cmyk_M = cmyk_value_np[:, 1]
        cmyk_Y = cmyk_value_np[:, 2]
        cmyk_K = cmyk_value_np[:, 3]
        
        lab_L = lab_input[:, 0]
        
        return {
            'L': np.round(lab_L, 2).tolist(),
            'C': np.round(cmyk_C, 2).tolist(),
            'M': np.round(cmyk_M, 2).tolist(),
            'Y': np.round(cmyk_Y, 2).tolist(),
            'K': np.round(cmyk_K, 2).tolist()
        }

    def verify_GCR(self, steps=10):
        """
        Verify if the GCR is applied correctly by checking the K channel.
        """
        
        # lab_value = [[l, 0, 0] for l in range(100, -1, -1)]  

        # L-Werte mit gegebener Schrittweite
        delta = 100 / steps
        L_values = np.arange(100, 0, -delta)
        #L_values = np.arange(100, -1, -steps)
        if L_values[-1] != 0:
            L_values = np.append(L_values, 0)
        lab_values = np.stack([L_values, np.zeros_like(L_values), np.zeros_like(L_values)], axis=1)
        
        

        # LAB → CMYK Konvertierung
        # cmyk = np.array(self.lab_to_cmyk(lab_values.tolist()))
        # print("LAB Values: ", lab_values.tolist())
        dcs = self.lab_to_cmyk(lab_values)
        # print("CMYK Values: ", dcs)

        
        # lab_value = [[l, 0, 0] for l in range(100, -1, -2)]  
        return dcs #self.lab_to_cmyk(lab_values)
