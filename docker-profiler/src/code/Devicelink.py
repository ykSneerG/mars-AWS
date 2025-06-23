from src.code.icctools.IccV4_ValueConverter import ICCvalueconverter as vc
from src.code.icctools.IccV4_Elements import ICCheader_Link                                 
from src.code.icctools.IccV4_Elements import IccElements                              
from src.code.icctools.Interpolation import Combinator        


class DevicelinkV2:
    '''
    ### Devicelink - Icc Profile Version 2
    '''
    
    def __init__(self, params) -> None:
        
        self.name = params.get("name", "Default Name")
        
        input_space = params["input_type"]
        output_space = params["output_type"]
        
        self.input_channels: int = input_space["num"]
        self.input_type: str = input_space["sig"]
        self.output_channels: int = output_space["num"]
        self.output_type: str = output_space["sig"]
        
        self.grippoints = params.get("gridpoints", None)
        self.output_table = params.get("output_table", None)
        
        self.copyright = "Copyright by Heiko Pieper, Hannover/Germany 2024. All rights reserved."
        self.infotext: str = params.get("typenschild", "None")
        # self.description = params.get("description", f"Lin-DL: {self.name} ({self.input_type} -> {self.output_type})")
        self.description = params.get("description", f"Lin-DL: {self.name} ({self.input_type} -> {self.output_type})")
                
        self.icc: bytearray = bytearray()
        
        self.check: bool = True
    
    def create(self):
        
        profilesize: int = 128
        tagstart: int = 128
        tagentries: int = 0

        # Entry - desc
        # en_desc = IccElements.entry_desc(f"Lin-DL: {self.name} ({self.input_type} -> {self.output_type})")
        en_desc = IccElements.entry_mluc(self.description)
        en_desc_length = len(en_desc)
        profilesize += en_desc_length
        tagentries += 1       
        
        # Entry - cprt
        en_cprt = IccElements.entry_mluc(self.copyright)
        en_cprt_length = len(en_cprt)
        profilesize += en_cprt_length
        tagentries += 1
        
        # Entry - mft2
        en_a2b0 = IccElements.entry_mft2(
            num_input_channels=self.input_channels, 
            num_output_channels= self.output_channels, 
            lut_table=self.grippoints,
            output_table=self.output_table
            )
        en_a2b0_length = len(en_a2b0)
        #print(f"en_a2b0_length: {en_a2b0_length}")
        profilesize += en_a2b0_length
        tagentries += 1
        
        # Entry - pseq
        en_pseq = IccElements.entry_pseq(None)
        en_pseq_length = len(en_pseq)
        profilesize += en_pseq_length
        tagentries += 1
        
        # Entry - clrt / colorantTableOutTag
        en_clrt = IccElements.entry_clrt(self.output_channels)
        en_clrt_length = len(en_clrt)
        profilesize += en_clrt_length
        tagentries += 1
        
        # colorantTableTag (see 9.2.18) which is required only if the data colour space field is xCLR, where x is hexadecimal 2 to F (see 7.2.6);

        # Entry - info
        en_info = IccElements.entry_text(self.infotext)
        en_info_length = len(en_info)
        profilesize += en_info_length
        tagentries += 1

        # +++++++++++++++++++++++++++
        
        print(f"TagTable tagentries - count: {tagentries}")
        print(f"TagStart: {tagstart}")

        # Tag Table - count (Anzahl der Eintraege in die Tagtable)
        tt_count = vc.uint32_to_bytes(tagentries)
        tt_count_length = len(tt_count)
        
        profilesize += tt_count_length
        tagstart += tt_count_length     # tagcount (4 bytes)
        tagstart += (tagentries * 12)   # tagtables (every tagentry is 12 bytes)

        # +++++++++++++++++++++++++++

        # Tag Table - desc (Description)
        tt_desc, tt_desc_length = IccElements.tag_table_head_extend("desc", tagstart, en_desc_length)
        profilesize += tt_desc_length
        tagstart += en_desc_length

        # Tag Table - cprt (Copyright)
        tt_cprt, tt_cprt_length = IccElements.tag_table_head_extend("cprt", tagstart, en_cprt_length)
        profilesize += tt_cprt_length
        tagstart += en_cprt_length

        # Tag Table - a2b0 (A2B0 Tabelle,...)
        tt_a2b0, tt_a2b0_length = IccElements.tag_table_head_extend("A2B0", tagstart, en_a2b0_length)
        profilesize += tt_a2b0_length
        tagstart += en_a2b0_length

        # Tag Table - pseq (Profile Sequence Description)
        tt_pseq, tt_pseq_length = IccElements.tag_table_head_extend("pseq", tagstart, en_pseq_length)
        profilesize += tt_pseq_length
        tagstart += en_pseq_length

        # Tag Table - clrt (Colorant Table)
        tt_clrt, tt_clrt_length = IccElements.tag_table_head_extend("clot", tagstart, en_clrt_length)
        profilesize += tt_clrt_length
        tagstart += en_clrt_length
        
        # Tag Table - info (unknown tag - info)
        tt_info, tt_info_length = IccElements.tag_table_head_extend("info", tagstart, en_info_length)
        profilesize += tt_info_length
        tagstart += en_info_length

        # +++++ ICC Profile Header ++++++++++++++++++++++

        head = ICCheader_Link()
        head.dcs = self.input_type
        head.pcs = self.output_type
        header = head.header(profilesize)
        
        bytesIcc = bytearray(header)

        # +++++ Tag Table Count (Anzahl der Eintraege in die Tagtable) +++++++++++

        bytesIcc.extend(tt_count)

        # +++++ Tag Table +++++++++++

        bytesIcc.extend(tt_desc)
        self.check_size("tt_desc", tt_desc)
        bytesIcc.extend(tt_cprt)
        self.check_size("tt_cprt", tt_cprt)
        bytesIcc.extend(tt_a2b0)
        self.check_size("tt_a2b0", tt_a2b0)
        bytesIcc.extend(tt_pseq)
        self.check_size("tt_pseq", tt_pseq)
        bytesIcc.extend(tt_clrt)
        self.check_size("tt_clrt", tt_clrt)
        bytesIcc.extend(tt_info)
        self.check_size("tt_info", tt_info)

        # +++++ Entry +++++++++++++++

        bytesIcc.extend(en_desc)
        self.check_size("en_desc", en_desc)
        bytesIcc.extend(en_cprt)
        self.check_size("en_cprt", en_cprt)
        bytesIcc.extend(en_a2b0)
        self.check_size("en_a2b0", en_a2b0)
        bytesIcc.extend(en_pseq)
        self.check_size("en_pseq", en_pseq)
        bytesIcc.extend(en_clrt)
        self.check_size("en_clrt", en_clrt)
        bytesIcc.extend(en_info)
        self.check_size("en_info", en_info)

        # +++++++++++++++++++++++++++
        
        print(f"Length bytesIcc: {len(bytesIcc)}")
        bytesIccWithCustomProfileId = ICCheader_Link.update_profile_id(bytesIcc)
        self.icc = bytesIccWithCustomProfileId
        return bytesIccWithCustomProfileId
        '''
        self.icc = bytesIcc
        return bytesIcc
        '''

    def write(self, path) -> None:
        if not path:
            print("No path specified")
            return
        try:
            with open(path, "wb") as f:
                f.write(self.icc)
            print(f"File written successfully to {path}")
        except Exception as e:
            print(f"Failed to write file to {path}: {str(e)}")

    def check_size(self, name, element) -> None:
        if not self.check:
            return
        
        print(f"{name}: {len(element)}")
        div4 = (len(element) / 4) % 1 == 0
        print(f"element div4: {div4}")


class DevicelinkBase:
    '''
    This is the base class for creating different types of devicelinks.
    '''
    
    def __init__(self, params) -> None:
        self.params = params
        self.name: str = str(params.get("name", "NoName")) 
        self.output_channels: int = params["output_type"]["num"]
        self.icc = bytearray()
        self.logging = False
        self.dl  = None

    def create(self):
        raise NotImplementedError("Subclasses should implement this method")

    def generate_gridpoints(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def set_logging(self, check: bool) -> None:
        self.logging = check

    def write(self, path):
        self.dl.write(path)
        
    def result(self) -> bytes:
        return bytes(self.dl.icc)
    
    def filesize(self) -> int:
        return len(self.dl.icc)
    
    @staticmethod
    def generate_typenschild(info: str, name: str, text: str = "", logs: str = "") -> str:
        
        logs = "" if logs == "" else f"Report:\n{logs}"
        text = "" if text == "" else f"\n{text}\n"
        
        typenschild = \
            f"+++ {info} +++\n" + \
            f"{name}.icc\n" + \
            f"{text}" + \
            "\n" + \
            "The profile is created by Heiko Pieper\n" + \
            "Copyright by colorspace.cloud and Heiko Pieper\n" + \
            "Hannover / Germany 2024-2025\n" + \
            "\n" + \
            "LICENSE: <LICENSE_INFO>\n" + \
            "\n" + \
            logs
        
        return typenschild

    
class Devicelink_Redirect(DevicelinkBase):
    '''
    This class creates a linear devicelink to redirect all channels,
    where Input and Output color space is the same.
    '''
    
    def create(self):        
        self.name += "_redirect"
        self.order = self.params.get("channel_order", None)
        
        self.params["gridpoints"] = self.generate_gridpoints() 
        self.params["typenschild"] = DevicelinkBase.generate_typenschild(
            "This is a devicelink to redirect all channels. :)", 
            self.name, 
            f"New ouptut channel order: {self.order}", 
            self.cblogs
            )
                                
        self.dl = DevicelinkV2(self.params)
        self.dl.check = self.logging
        self.icc = self.dl.create()

    def generate_gridpoints(self):
        cb = Combinator(self.output_channels, 2)
        
        if self.order is None:
            return cb.combos
        
        res_redirection = cb.redirect_channels(cb.combos, self.order)
        self.cblogs = cb.print_log()
        return res_redirection


class Devicelink_OutputCurves(DevicelinkBase):
    '''
    This class creates a linear devicelink to redirect all channels,
    where Input and Output color space is the same.
    '''
                
    def create(self):        
        self.name += "_curvelink"
        
        self.params["gridpoints"] = self.generate_gridpoints()
        self.params["typenschild"] = DevicelinkBase.generate_typenschild(
            "This is a devicelink to adjust the output channel curves.",
            self.name
            )
                                
        self.dl = DevicelinkV2(self.params)
        self.dl.check = self.logging
        self.icc = self.dl.create()

    def generate_gridpoints(self):
        cb = Combinator(self.output_channels, 2)
        return cb.combos


class Devicelink_Dcsdata(DevicelinkBase):
    '''
    This class creates a devicelink via clut data.
    '''
    
    def __init__(self, params) -> None:
        super().__init__(params)
    
        self.name += "_dcsdata"
        num_input_channels: int = params["input_type"]["num"]
        num_output_channels: int = params["output_type"]["num"]
        self.params["description"] = f"DL: {self.name} ({num_input_channels} -> {num_output_channels})"
        
        self.cblogs = []

    def create(self):
        self.params["typenschild"] = DevicelinkBase.generate_typenschild(
            "This devicelink is based on LUT-data.",
            self.name
            )
                                        
        self.dl = DevicelinkV2(self.params)
        self.dl.check = self.logging
        self.icc = self.dl.create()
