import numpy as np    # type: ignore
from src.code.icctools.LutInvers import LUTInverterKDTree
from src.code.icctools.IccV4_Helper import Helper
from src.code.icctools.Interpolation import VectorScale

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class MakeXtoXBase:
    
    def __init__(self, lut):
        self._clut = lut
        self._index_lightest: int = 0
        self._index_darkest: int = len(self._clut) - 1
        
    @property
    def clut(self):
        return self._clut

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
        return self.clut[self.index_darkest]
    
    @property
    def darkest_rounded(self):
        return [round(x, 4) for x in self._clut[self._index_darkest]]
    

class MakeAtoB(MakeXtoXBase):

    def __init__(self, lut, dcs):
        super().__init__(lut)
        self._dcs = dcs

    @property
    def clut_as_uint16(self):
        labs = Helper.lab_to_uint16(self._clut)
        flat_list = Helper.flatten_list(labs)
        return flat_list


    def generate_perceptual(self):
        """
        ### Generate the PERCEPTUAL LUT (AtoB0) 
        by applying the necessary transformations.
        """
        #self._media_relative()
        self._map_darkest_L_NEW()

    def generate_saturation(self):
        """
        ### Generate the SATURATION LUT (AtoB2) 
        by applying the necessary transformations.
        """
        #self._media_relative()
        self._map_darkest_L_NEW()
        self._scale_channel_AB()

    def generate_relative(self):
        """
        ### Generate the RELATIVE LUT (AtoB1) 
        by applying the necessary transformations.
        """
        #self._media_relative()
        pass
    
    def report(self, freeInfo: str = ""):
        """
        Print the AtoB LUT information.
        """
        #print("")
        print(f"AtoB {freeInfo} LUT Information")
        print(f"{freeInfo}\tLightest - Index: ", self.index_lightest, "\t\tLAB: ", self.lightest_rounded)
        print(f"{freeInfo}\tDarkest. - Index: ", self.index_darkest, "\t\tLAB: ", self.darkest_rounded)
        #print("- + - + - + - + - + - + - + - + - + - + -")
        print("")


    def _scale_channel_AB(self, scale_ab: float = 0.75):
        """ 
        Scale a* and b* channels from input LUT to output LUT.
        """
        self._clut = VectorScale.apply_ab_factor(self.clut, scale_ab, scale_ab)

    def _map_darkest_L_NEW(self):
        """
        Map darkest L* to 0.
        """
        self._clut = VectorScale.scale_l_to_0_100_indexed(self.clut, self.index_darkest, self.index_lightest)


    def interpolate_clut(self, grid_size: int = 17):
        """
        Interpolate the LUT to a denser grid.
        
        Parameters:
        - grid_size: number of interpolation points per channel
        
        Returns:
        - dense_clut: ndarray of shape (grid_size**channels, 3)
        """
        
        dense_dcs, dense_lab = MakeAtoB.interpolate_grid(self._dcs, self.clut, grid_size)
        self._clut = dense_lab.tolist()
        self._dcs = dense_dcs.tolist()
        
        

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
        self._dcs = dcs
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
        Generate the BtoA LUT by applying the necessary transformations.
        """
        
        #lut_atob = list(zip(self._dcs, self._clut))
        #print(lut_atob)

        # get cmyk_grid_size from the LUT
        #np_dcs_grid_size = np.unique(np.array(self._dcs)).size
        
        inverter = LUTInverterKDTree(
            atob_lut=list(zip(self._dcs, self._clut)),
            cmyk_grid_size=np.unique(np.array(self._dcs)).size,
            lab_grid_shape=(self.grid_size, self.grid_size, self.grid_size),
            loss_function='linear_k',
            max_ink_limit=4.00
        )
        
        self._clut = inverter.build_btoa_lut(mode='hybrid', kdtree_resolution=20)

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


import numpy as np
from scipy.interpolate import RegularGridInterpolator

class BToAInterpolator:
    def __init__(self, lab_grid_shape, lab_vals, cmyk_vals):
        self.lab_grid_shape = lab_grid_shape
        self.lab_vals = lab_vals.reshape((*lab_grid_shape, 3))
        self.cmyk_vals = cmyk_vals.reshape((*lab_grid_shape, 4))
        self.L_axis = np.linspace(0, 100, lab_grid_shape[0])
        self.a_axis = np.linspace(-128, 127, lab_grid_shape[1])
        self.b_axis = np.linspace(-128, 127, lab_grid_shape[2])
        self.interpolator = RegularGridInterpolator(
            (self.L_axis, self.a_axis, self.b_axis),
            self.cmyk_vals,
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
        print("LAB Values: ", lab_values.tolist())
        dcs = self.lab_to_cmyk(lab_values)
        print("CMYK Values: ", dcs)

        
        # lab_value = [[l, 0, 0] for l in range(100, -1, -2)]  
        return dcs #self.lab_to_cmyk(lab_values)

        


# --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED --- UNUSED ---


class MakeBtoA_UNUSED(MakeXtoXBase):
    
    def __init__(self, dcs, lut, grid_size=11):
        super().__init__(lut)
        self._dcs = dcs
        self.grid_size = grid_size
        
        #self.inverter = None

    """ @property
    def clut(self):
        return self._clut """

    @property
    def clut_as_uint16(self):
        """ btoa_uint16 = (self.clut * 65535.0).astype(np.uint16).tolist()
        flat_list = Helper.flatten_list(btoa_uint16)
        return flat_list """
        arr = np.asarray(self.clut)
        flat = (arr * 65535.0).round().astype(np.uint16).ravel()
        return flat.tolist()

    # @staticmethod
    # def argyll_style_gcr_curve(L):
    #     """
    #     Return a set of K levels based on lightness (L*),
    #     simulating Argyll -kr (robust GCR)
    #     """
    #     # Definiere maximale Schwarzmenge über L*
    #     if L > 95:
    #         k_max = 0.0
    #     elif L > 70:
    #         k_max = 0.3
    #     elif L > 50:
    #         k_max = 0.6
    #     else:
    #         k_max = 1.0

    #     return np.linspace(0, k_max, 5)  # 5 K-Kandidaten testen


    def generate(self):
        """
        Generate the BtoA LUT by applying the necessary transformations.
        """
        
        lut_atob = list(zip(self._dcs, self._clut))
        #print(lut_atob)

        # get cmyk_grid_size from the LUT
        np_dcs_grid_size = np.unique(np.array(self._dcs)).size
        
        # really the best and working with smallest de but curves are not so good
        """ inverter = LUTInverterPur(
            lut_atob, 
            cmyk_grid_size=np_dcs_grid_size, 
            lab_grid_shape=(self.grid_size, self.grid_size, self.grid_size)
        ) """
        inverter = LUTInverterKDTree(
            lut_atob,
            cmyk_grid_size=np_dcs_grid_size,
            lab_grid_shape=(self.grid_size, self.grid_size, self.grid_size),
            loss_function='linear_k',
            max_ink_limit=3.75
        )
        
        self._clut = inverter.build_btoa_lut(mode='hybrid', kdtree_resolution=20)
                
        """ inverter = LUTInverter_Multi(
            lut_atob, 
            cmyk_grid_size=7, 
            lab_grid_shape=(self.grid_size, self.grid_size, self.grid_size), 
            interpolation_method='tetrahedral',
        ) """
        
        """ inverter = LUTInverterTetrahedral(
            lut_atob, 
            cmyk_grid_size=7, 
            lab_grid_shape=(self.grid_size, self.grid_size, self.grid_size)
        ) """
        
        
        # This works
        """ inverter = LUTInverterGCR(
            lut_atob, 
            cmyk_grid_size=7, 
            lab_grid_shape=(self.grid_size, self.grid_size, self.grid_size), 
            black_start=0.05, 
            black_width=1.0, 
            black_strength=0.75
        ) """
        
        
        
        """ inverter = LUTInverterGCR(
            lut_atob, 
            cmyk_grid_size=7, 
            lab_grid_shape=(self.grid_size, self.grid_size, self.grid_size),
            black_strength=1.0,
            black_start=0.2, 
            black_end=0.8
        ) """
        """ inverter = LUTInverterGCR(
            atob_lut=lut_atob,
            cmyk_grid_size=7,
            lab_grid_shape=(self.grid_size, self.grid_size, self.grid_size),
            black_curve=MakeBtoA.argyll_style_gcr_curve,
            regularization=0.01
        ) """

        """ inverter = LUTInverterGCR(
            atob_lut=lut_atob, 
            cmyk_grid_size=7, 
            lab_grid_shape=(self.grid_size, self.grid_size, self.grid_size), 
            black_levels=21,
            l_target=50.0,
            regularization=0.1
        ) """
        # Build BToA LUT    
        
        # self._clut = inverter.build_btoa_lut(mode='hybrid')
        # self._clut = LUTInverterPur.gaussian_filter(self._clut, sigma=0.25)
        #self._clut = LUTInverterPur.smooth_btoa_median(self._clut, size=5)

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

    



    # def black_generation(self, dcs, pcs):
    #     '''
    #     1. Create a neutral LAB axis in 20 steps from 0,0,0 to 100,0,0.
    #     2. Interpolate self_clut to get the dcs combinations for the neutral LAB axis.
    #     '''
    #     # Create a neutral LAB axis
    #     neutral_lab = np.linspace(0, 100, 20).reshape(-1, 1)
    #     neutral_lab = np.concatenate([neutral_lab, np.zeros_like(neutral_lab), np.zeros_like(neutral_lab)], axis=1)
        
    #     list_dcs_pcs = list(zip(dcs, pcs))
        
    #     # Interpolate the LUT to get the dcs combinations for the neutral LAB axis
    #     interpolator = RegularGridInterpolator(            
    #         list_dcs_pcs,
    #         neutral_lab,
    #         bounds_error=False,
    #         fill_value=None
    #     )
    #     interpolated_dcs = interpolator(neutral_lab)
        
        

    # @staticmethod
    # def createBtoA(dcs, pcs, info, grid_size=7):
    #     """ 
    #     Create a BtoA LUT from the given device color space (dcs) and perceptual color space (pcs).
    #     The LUT is built using the LUTInverterGCR class, which handles the inversion of the LUT.
        
    #     Parameters:
    #     dcs (list): List of device color space values.
    #     pcs (list): List of perceptual color space values.
    #     info (str): Information string to be printed about the LUT.
    #     """
        
        
    #     print("- + - + - + - + - + - + - + - + - + - + -")
    #     print(info)
        
    #     lut_atob = list(zip(dcs, pcs))
    #     #print(lut_atob)

    #     """ inverter = LUTInverterGCR(
    #         lut_atob, 
    #         cmyk_grid_size=7, 
    #         lab_grid_shape=(grid_size, grid_size, grid_size), 
    #         black_start=0.0, 
    #         black_width=0.75, 
    #         black_strength=1.0
    #     ) """
        
    #     inverter = LUTInverter_ColorScience(
    #         lut_atob=lut_atob,
    #         cmyk_grid_size=7,
    #         lab_grid_shape=(grid_size, grid_size, grid_size)
    #     )
    #     # Build BToA LUT
    #     lut_btoa = inverter.build_btoa_lut()
        
    #     # --- Optionally smooth the result ---
        
    #     #smoothed_atob1 = LUTInverter.smooth_lut_2(inverter_atob1, sigma=1.0)
        
    #     #sgs = SavitzkyGolaySmoothing(inverter_atob1, 5, 2)
    #     #smoothed_atob1 = sgs.apply_smoothing()
        
    #     # smoothed_atob1 = inverter.smooth_lut_edge_aware(inverter_atob1, 1.0, 0.5)

    #     res_btoa = lut_btoa

    #     #print("Table: ", res_btoa)
    #     print("Shape: ", res_btoa.shape)
    #     print("Length: ", int(res_btoa.size / res_btoa.shape[3]))
    #     #print("First: ", res_btoa[0][0][0])

    #     btoa_uint16 = (res_btoa * 65535.0).astype(np.uint16).tolist()

    #     return btoa_uint16
