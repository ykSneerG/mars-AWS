from src.code.icctools.IccV4_Helper import Helper, ColorTrafo
from src.code.icctools.MakeXtoX import MakeAtoB, MakeBtoA, MakeCurve
from src.code.Devicelink import DevicelinkBase, PrinterlinkV2


class Profile_Printer(DevicelinkBase):
    """
    This class creates a printer profile.
    """

    def __init__(self, params):
        super().__init__(params)
        
        grid_size = params.get("grid_size")
        if grid_size is None:
            grid_size = {"atob": [9, 9, 9], "btoa": [11, 11, 11]}

        atob = grid_size.get("atob", [11, 11, 9])
        btoa = grid_size.get("btoa", [17, 21, 11])

        self.grid_size_atob0 = atob[0]
        self.grid_size_atob1 = atob[1]
        self.grid_size_atob2 = atob[2]

        self.grid_size_btoa0 = btoa[0]
        self.grid_size_btoa1 = btoa[1]
        self.grid_size_btoa2 = btoa[2]

        self.dcss = params.get("dcss_src", None)
        self.xyzs_abs = params.get("xyzs_src", None)
        self.labs_abs = ColorTrafo.xyz_to_lab(self.xyzs_abs)

        self.idx_light = 0
        self.idx_dark = 10

        # This is a workaround, needs imporvement
        self.detect_darkest_neutral_point()
        
        self.wtpt = self.find_wtpt()
        self.bktp = self.find_bktp()
        
        """ self.update_wtpt()
        self.update_bktp() """
        
        self.xyzs_rel = ColorTrafo.media_relative(self.xyzs_abs, self.wtpt)
        self.labs_rel = ColorTrafo.xyz_to_lab(self.xyzs_rel)
        
        self.gcr_result = {}
        
    def detect_darkest_neutral_point(self):
        """
        Detect the darkest neutral point in the absolute XYZ values.
        This is used to find the black point for the profile.
        """
        if self.xyzs_abs is not None and len(self.xyzs_abs) > 0:
            # Find the index of the darkest neutral point
            idx_dark_neutral = min(range(len(self.labs_abs)), 
                                key=lambda i: (self.labs_abs[i][0], abs(self.labs_abs[i][1]), abs(self.labs_abs[i][2])))
            print("Darkest neutral point index:", idx_dark_neutral)
            print("Darkest neutral point LAB:", self.labs_abs[idx_dark_neutral])
            #print("Darkest neutral point XYZ:", self.xyzs_abs[idx_dark_neutral])
            print("Darkest neutral point DCS:", self.dcss[idx_dark_neutral])

            idx_dark_noneutral = min(range(len(self.labs_abs)), 
                                key=lambda i: (self.labs_abs[i][0]))
            print("Darkest non-neutral point index:", idx_dark_noneutral)
            print("Darkest non-neutral point LAB:", self.labs_abs[idx_dark_noneutral])
            #print("Darkest non-neutral point XYZ:", self.xyzs_abs[idx_dark_noneutral])
            print("Darkest non-neutral point DCS:", self.dcss[idx_dark_noneutral])

            # get index where self.dcss is closest to [0, 0, 0, 100]
            idx_k100 = min(range(len(self.dcss)), 
                                key=lambda i: sum(abs(self.dcss[i][j] - (0 if j < 3 else 100)) for j in range(len(self.dcss[i]))))
            print("Closest to K100 index:", idx_k100)
            print("Closest to K100 LAB:", self.labs_abs[idx_k100])
            #print("Closest to K100 XYZ:", self.xyzs_abs[idx_k100])
            print("Closest to K100 DCS:", self.dcss[idx_k100])
            
            
            print("Last point index:", self.idx_dark)
            print("Last point LAB:", self.labs_abs[self.idx_dark])
            #print("Last point XYZ:", self.xyzs_abs[self.idx_dark])
            print("Last point DCS:", self.dcss[self.idx_dark])
            
            self.idx_dark = idx_dark_noneutral

    """ def update_wtpt(self):
        # find index where sum self.dcss[i] is smallest
        self.idx_light = max(range(len(self.dcss)), key=lambda i: sum(self.dcss[i]))
        self.wtpt = self.find_wtpt()

    def update_bktp(self):
        # find index where with max L value in labs_abs and A and B values close to 0
        self.idx_dark = min(range(len(self.labs_abs)), 
                       key=lambda i: (self.labs_abs[i][0], abs(self.labs_abs[i][1]), abs(self.labs_abs[i][2])))
        self.bktp = self.find_bktp()
        
        #self.idx_dark = min(range(len(self.labs_abs)), key=lambda i: self.labs_abs[i][0]) """

    def find_wtpt(self):
        if self.xyzs_abs is not None and len(self.xyzs_abs) > 0:            
            return self.xyzs_abs[self.idx_light]
        return [0.807, 0.829, 0.712]
    
    def find_bktp(self):
        if self.xyzs_abs is not None and len(self.xyzs_abs) > 0:
            return self.xyzs_abs[self.idx_dark]
        return [0.007, 0.008, 0.006]
    
    @staticmethod
    def build_btoa_table(dcs, lut, grid_size, info):
        make_btoa0 = MakeBtoA(dcs, lut, grid_size)
        make_btoa0.generate()
        make_btoa0.report(info)
        return make_btoa0
    
    def build_atob_table(self, id = 0) -> MakeAtoB:
        atob = MakeAtoB(self.labs_rel.copy(), self.dcss)
        atob.index_lightest = self.idx_light
        atob.index_darkest = self.idx_dark
        print("MakeAtoB Lightest Index: ", atob.index_lightest)
        print("MakeAtoB Darkest Index: ", atob.index_darkest)
        
        if id == 0:
            atob.generate_perceptual()
            atob.interpolate_clut(self.grid_size_atob0)
            atob.report("PERCEPTUAL")
        elif id == 1:
            atob.generate_relative()
            atob.interpolate_clut(self.grid_size_atob1)
            atob.report("RELATIVE")
        elif id == 2:
            atob.generate_saturation()
            atob.interpolate_clut(self.grid_size_atob2)
            atob.report("SATURATION")
            
        return atob
    
    def report_absolute(self):
        
        print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")
        print("Channel Count: ", len(self.dcss[0]))
        print("")
        print("ABSOLUTE\tLightest - Index: ", 0, "\t\tLAB: ", Helper.round_list(self.labs_abs[self.idx_light], 4))
        print("ABSOLUTE\tDarkest. - Index: ", len(self.labs_abs) - 1, "\tLAB: ", Helper.round_list(self.labs_abs[self.idx_dark], 4))
        print("")

    def create(self):
        
        self.report_absolute()
        
        make_atob0 = self.build_atob_table(0)
        make_atob1 = self.build_atob_table(1)
        make_atob2 = self.build_atob_table(2)
        
        make_btoa0 = self.build_btoa_table(make_atob0.dcs, make_atob0.clut, self.grid_size_btoa0, "PERCEPTUAL")
        make_btoa1 = self.build_btoa_table(make_atob1.dcs, make_atob1.clut, self.grid_size_btoa1, "RELATIVE")
        make_btoa2 = self.build_btoa_table(make_atob2.dcs, make_atob2.clut, self.grid_size_btoa2, "SATURATION")
        

        # Verify GCR BtoA1
        interp = make_btoa1.get_interpolator()
        self.gcr_result = interp.verify_GCR(100)


        self.params.update({
            'wtpt': self.wtpt,
            'bktp': self.bktp,
            'atob0_clut': make_atob0.clut_as_uint16,
            'atob1_clut': make_atob1.clut_as_uint16,
            'atob2_clut': make_atob2.clut_as_uint16,
            'btoa0_clut': make_btoa0.clut_as_uint16,
            'btoa1_clut': make_btoa1.clut_as_uint16,
            'btoa2_clut': make_btoa2.clut_as_uint16,
        })
        
        self.params["typenschild"] = DevicelinkBase.generate_typenschild(
            info="This is a printer profile.",
            name=self.name,
            text=f"Conversion: {self.params['input_type']['sig']} -> LAB -> {self.params['output_type']['sig']}",
            logs="Test1\nTest2",  # self.cblogs
            lic="<NO_LICENSE_INFO>",
        )

        self.dl = PrinterlinkV2(self.params)
        self.dl.check = self.logging
        self.icc = self.dl.create()
