import random
from src.code.icctools.IccV4_ValueConverter import ICCvalueconverter as vc
from src.code.icctools.IccV4_Elements import ICCheader_Prtr
from src.code.icctools.IccV4_Elements import IccElements
from src.code.icctools.Interpolation import Combinator
from src.code.Devicelink import DevicelinkBase


class PrinterlinkV2:
    """
    ### Printerlink - Icc Profile Version 2
    """

    def __init__(self, params) -> None:

        self.name = params.get("name", "Default Name")

        """ input_space = params["input_type"]
        output_space = params["output_type"] """

        """ self.input_channels: int = input_space["num"]
        self.input_type: str = input_space["sig"]
        self.output_channels: int = output_space["num"]
        self.output_type: str = output_space["sig"] """

        """
        icTagTypeSignature: mft2
        Tag_Bytes: 91482
        InputChan: 4
        OutputChan: 3
        Input_Entries: 256
        Output_Entries: 256
        Clut_Size: 11
        """
        self.wtpt = params.get("wtpt", [0.807, 0.829, 0.712])
        self.bktp = params.get("bktp", [0.007, 0.008, 0.006])

        clut_atob0 = params.get("atob0_clut", None)
        clut_atob1 = params.get("atob1_clut", None)
        clut_atob2 = params.get("atob2_clut", None)

        clut_btoa0 = params.get("btoa0_clut", None)
        clut_btoa1 = params.get("btoa1_clut", None)
        clut_btoa2 = params.get("btoa2_clut", None)

        self.atob0 = {
            "InputChan": 4,
            "OutputChan": 3,
            "CLUT": clut_atob0,
        }
        self.atob1 = {
            "InputChan": 4,
            "OutputChan": 3,
            "CLUT": clut_atob1,
        }
        self.atob2 = {
            "InputChan": 4,
            "OutputChan": 3,
            "CLUT": clut_atob2,
        }
        
        #print(f"atob: {self.atob}")
        self.btoa0 = {
            "InputChan": 3,
            "OutputChan": 4,
            "CLUT": clut_btoa0, #[(i + 1) * 255 for i in range(32)],
        }
        self.btoa1 = {
            "InputChan": 3,
            "OutputChan": 4,
            "CLUT": clut_btoa1, #[(i + 1) * 255 for i in range(32)],
        }
        self.btoa2 = {
            "InputChan": 3,
            "OutputChan": 4,
            "CLUT": clut_btoa2, #[(i + 1) * 255 for i in range(32)],
        }
        #print(f"btoa: {self.btoa}")

        # self.input_channels: int = 4
        self.input_type: str = "CMYK"
        # self.output_channels: int = 3
        self.output_type: str = "LAB "

        # self.grippoints = params.get("gridpoints", None)

        self.output_table = params.get("output_table", None)

        self.copyright = "Copyright by Heiko Pieper, Hannover/Germany 2024-2025. All rights reserved."
        self.infotext: str = params.get("typenschild", "None")
        self.description = params.get("description", f"ICC: {self.name}")

        self.icc: bytearray = bytearray()

        self.check: bool = True

    def create(self):

        profilesize: int = 128
        tagstart: int = 128
        tagentries: int = 0

        # Entry - desc
        en_desc = IccElements.entry_mluc(self.description)
        en_desc_length = len(en_desc)
        profilesize += en_desc_length
        tagentries += 1

        # Entry - cprt
        en_cprt = IccElements.entry_mluc(self.copyright)
        en_cprt_length = len(en_cprt)
        profilesize += en_cprt_length
        tagentries += 1

        # Entry -wptp
        en_wptp = IccElements.entry_wptp(self.wtpt)
        en_wptp_length = len(en_wptp)
        profilesize += en_wptp_length
        tagentries += 1

        # Entry - bkpt
        en_bkpt = IccElements.entry_wptp(self.bktp)
        en_bkpt_length = len(en_bkpt)
        profilesize += en_bkpt_length
        tagentries += 1

        # Entry - mft2 -- A2B0
        en_a2b0 = IccElements.entry_mft2(
            num_input_channels=self.atob0["InputChan"],
            num_output_channels=self.atob0["OutputChan"],
            lut_table=self.atob0["CLUT"],
            # output_table=self.output_table
        )
        en_a2b0_length = len(en_a2b0)
        #print(f"en_a2b0_length: {en_a2b0_length}")
        profilesize += en_a2b0_length
        tagentries += 1

        # Entry - mft2 -- B2A0
        en_b2a0 = IccElements.entry_mft2(
            num_input_channels=self.btoa0["InputChan"],
            num_output_channels=self.btoa0["OutputChan"],
            lut_table=self.btoa0["CLUT"],
            # output_table=self.output_table
        )
        en_b2a0_length = len(en_b2a0)
        #print(f"en_b2a0_length: {en_b2a0_length}")
        profilesize += en_b2a0_length
        tagentries += 1
        # +++++++++++++++++++++++++++
        # Entry - mft2 -- A2B0
        en_a2b1 = IccElements.entry_mft2(
            num_input_channels=self.atob1["InputChan"],
            num_output_channels=self.atob1["OutputChan"],
            lut_table=self.atob1["CLUT"],
            # output_table=self.output_table
        )
        en_a2b1_length = len(en_a2b1)
        #print(f"en_a2b1_length: {en_a2b1_length}")
        profilesize += en_a2b1_length
        tagentries += 1

        # Entry - mft2 -- B2A0
        en_b2a1 = IccElements.entry_mft2(
            num_input_channels=self.btoa1["InputChan"],
            num_output_channels=self.btoa1["OutputChan"],
            lut_table=self.btoa1["CLUT"],
            # output_table=self.output_table
        )
        en_b2a1_length = len(en_b2a1)
        #print(f"en_b2a1_length: {en_b2a1_length}")
        profilesize += en_b2a1_length
        tagentries += 1
        # +++++++++++++++++++++++++++
        # Entry - mft2 -- A2B0
        en_a2b2 = IccElements.entry_mft2(
            num_input_channels=self.atob2["InputChan"],
            num_output_channels=self.atob2["OutputChan"],
            lut_table=self.atob2["CLUT"],
            # output_table=self.output_table
        )
        en_a2b2_length = len(en_a2b2)
        #print(f"en_a2b2_length: {en_a2b2_length}")
        profilesize += en_a2b2_length
        tagentries += 1

        # Entry - mft2 -- B2A0
        en_b2a2 = IccElements.entry_mft2(
            num_input_channels=self.btoa2["InputChan"],
            num_output_channels=self.btoa2["OutputChan"],
            lut_table=self.btoa2["CLUT"],
            # output_table=self.output_table
        )
        en_b2a2_length = len(en_b2a2)
        #print(f"en_b2a2_length: {en_b2a2_length}")
        profilesize += en_b2a2_length
        tagentries += 1
        # +++++++++++++++++++++++++++

        # Entry - pseq
        """ en_pseq = IccElements.entry_pseq(None)
        en_pseq_length = len(en_pseq)
        profilesize += en_pseq_length
        tagentries += 1 """

        # Entry - clrt / colorantTableOutTag
        """ en_clrt = IccElements.entry_clrt(self.output_channels)
        en_clrt_length = len(en_clrt)
        profilesize += en_clrt_length
        tagentries += 1 """

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
        tagstart += tt_count_length  # tagcount (4 bytes)
        tagstart += tagentries * 12  # tagtables (every tagentry is 12 bytes)

        # +++++++++++++++++++++++++++

        # Tag Table - desc (Description)
        tt_desc, tt_desc_length = IccElements.tag_table_head_extend(
            "desc", tagstart, en_desc_length
        )
        profilesize += tt_desc_length
        tagstart += en_desc_length

        # Tag Table - cprt (Copyright)
        tt_cprt, tt_cprt_length = IccElements.tag_table_head_extend(
            "cprt", tagstart, en_cprt_length
        )
        profilesize += tt_cprt_length
        tagstart += en_cprt_length

        # +++++++++++++++++++++++++++

        tt_wptp, tt_wptp_length = IccElements.tag_table_head_extend(
            "wtpt", tagstart, en_wptp_length
        )
        profilesize += tt_wptp_length
        tagstart += en_wptp_length

        tt_bkpt, tt_bkpt_length = IccElements.tag_table_head_extend(
            "bkpt", tagstart, en_bkpt_length
        )
        profilesize += tt_bkpt_length
        tagstart += en_bkpt_length

        # Tag Table - a2b0 (A2B0 Tabelle,...)
        tt_a2b0, tt_a2b0_length = IccElements.tag_table_head_extend(
            "A2B0", tagstart, en_a2b0_length
        )
        profilesize += tt_a2b0_length
        tagstart += en_a2b0_length

        tt_b2a0, tt_b2a0_length = IccElements.tag_table_head_extend(
            "B2A0", tagstart, en_b2a0_length
        )
        profilesize += tt_b2a0_length
        tagstart += en_b2a0_length

        tt_a2b1, tt_a2b1_length = IccElements.tag_table_head_extend(
            "A2B1", tagstart, en_a2b1_length
        )
        profilesize += tt_a2b1_length
        tagstart += en_a2b1_length

        tt_b2a1, tt_b2a1_length = IccElements.tag_table_head_extend(
            "B2A1", tagstart, en_b2a1_length
        )
        profilesize += tt_b2a1_length
        tagstart += en_b2a1_length

        tt_a2b2, tt_a2b2_length = IccElements.tag_table_head_extend(
            "A2B2", tagstart, en_a2b2_length
        )
        profilesize += tt_a2b2_length
        tagstart += en_a2b2_length

        tt_b2a2, tt_b2a2_length = IccElements.tag_table_head_extend(
            "B2A2", tagstart, en_b2a2_length
        )
        profilesize += tt_b2a2_length
        tagstart += en_b2a2_length

        # Tag Table - pseq (Profile Sequence Description)
        """ tt_pseq, tt_pseq_length = IccElements.tag_table_head_extend("pseq", tagstart, en_pseq_length)
        profilesize += tt_pseq_length
        tagstart += en_pseq_length """

        # Tag Table - clrt (Colorant Table)
        """ tt_clrt, tt_clrt_length = IccElements.tag_table_head_extend("clot", tagstart, en_clrt_length)
        profilesize += tt_clrt_length
        tagstart += en_clrt_length """

        # Tag Table - info (unknown tag - info)
        tt_info, tt_info_length = IccElements.tag_table_head_extend(
            "info", tagstart, en_info_length
        )
        profilesize += tt_info_length
        tagstart += en_info_length

        # +++++ ICC Profile Header ++++++++++++++++++++++

        head = ICCheader_Prtr()
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

        bytesIcc.extend(tt_wptp)
        self.check_size("tt_wptp", tt_wptp)

        bytesIcc.extend(tt_bkpt)
        self.check_size("tt_bkpt", tt_bkpt)

        bytesIcc.extend(tt_a2b0)
        self.check_size("tt_a2b0", tt_a2b0)
        bytesIcc.extend(tt_b2a0)
        self.check_size("tt_b2a0", tt_b2a0)

        bytesIcc.extend(tt_a2b1)
        self.check_size("tt_a2b1", tt_a2b1)
        bytesIcc.extend(tt_b2a1)
        self.check_size("tt_b2a1", tt_b2a1)

        bytesIcc.extend(tt_a2b2)
        self.check_size("tt_a2b2", tt_a2b2)
        bytesIcc.extend(tt_b2a2)
        self.check_size("tt_b2a2", tt_b2a2)

        """ bytesIcc.extend(tt_pseq)
        self.check_size("tt_pseq", tt_pseq)
        bytesIcc.extend(tt_clrt)
        self.check_size("tt_clrt", tt_clrt) """
        bytesIcc.extend(tt_info)
        self.check_size("tt_info", tt_info)

        # +++++ Entry +++++++++++++++

        bytesIcc.extend(en_desc)
        self.check_size("en_desc", en_desc)
        bytesIcc.extend(en_cprt)
        self.check_size("en_cprt", en_cprt)

        bytesIcc.extend(en_wptp)
        self.check_size("en_wptp", en_wptp)

        bytesIcc.extend(en_bkpt)
        self.check_size("en_bkpt", en_bkpt)

        bytesIcc.extend(en_a2b0)
        self.check_size("en_a2b0", en_a2b0)
        bytesIcc.extend(en_b2a0)
        self.check_size("en_b2a0", en_b2a0)

        bytesIcc.extend(en_a2b1)
        self.check_size("en_a2b1", en_a2b1)
        bytesIcc.extend(en_b2a1)
        self.check_size("en_b2a1", en_b2a1)

        bytesIcc.extend(en_a2b2)
        self.check_size("en_a2b0", en_a2b2)
        bytesIcc.extend(en_b2a2)
        self.check_size("en_b2a0", en_b2a2)

        """ bytesIcc.extend(en_pseq)
        self.check_size("en_pseq", en_pseq)
        bytesIcc.extend(en_clrt)
        self.check_size("en_clrt", en_clrt) """
        bytesIcc.extend(en_info)
        self.check_size("en_info", en_info)

        # +++++++++++++++++++++++++++

        print(f"Length bytesIcc: {len(bytesIcc)}")
        bytesIccWithCustomProfileId = ICCheader_Prtr.update_profile_id(bytesIcc)
        self.icc = bytesIccWithCustomProfileId
        return bytesIccWithCustomProfileId
        """
        self.icc = bytesIcc
        return bytesIcc
        """

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


class Profile_Printer(DevicelinkBase):
    """
    This class creates a linear devicelink to redirect all channels,
    where Input and Output color space is the same.
    """

    def create(self):
        self.name += "_redirect"
        self.order = self.params.get("channel_order", None)

        #self.params["gridpoints"] = self.generate_gridpoints()
        self.params["typenschild"] = DevicelinkBase.generate_typenschild(
            info="This is a devicelink to redirect all channels. :)",
            name=self.name,
            text=f"New ouptut channel order: {self.order}",
            logs="Test1\nTest2",  # self.cblogs
        )

        self.dl = PrinterlinkV2(self.params)
        self.dl.check = self.logging
        self.icc = self.dl.create()

    """ def generate_gridpoints(self):
        cb = Combinator(self.output_channels, 2)

        if self.order is None:
            return cb.combos

        res_redirection = cb.redirect_channels(cb.combos, self.order)
        self.cblogs = cb.print_log()
        return res_redirection """
