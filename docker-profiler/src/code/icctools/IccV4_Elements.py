import datetime
from hashlib import md5
from enum import Enum
from src.code.icctools.IccV4_ValueConverter import ICCvalueconverter as vc
from src.code.icctools.Interpolation import Combinator as cb


class IccConstants:
    
    MIN16: int = 0
    '''
    Minimum value for 16-bit integer
    '''  
    
    MAX16: int = 65535
    '''
    Maximum value for 16-bit integer
    '''
    
    ZEROBYTE1: bytes = b'\x00'
    '''
    Zero byte (0x00) / 1 byte
    '''
    
    ZEROBYTE4: bytes = b'\x00\x00\x00\x00'
    '''
    Zero byte (0x00) / 4 byte
    '''

class ICCheader_Link:
    
    def __init__(self) -> None:
        self.dcs = "CMYK"
        self.pcs = "CMYK"
    
    # set the device color space
    def set_input_space(self, dcs: str) -> None:
        self.dcs = dcs
        
    # set the profile connection space
    def set_output_space(self, pcs: str) -> None:
        self.pcs = pcs
    
    def header(self, profile_size: int):
        print(f"Profile size: {profile_size}")
        
        header = bytearray()

        # Profile Size
        header.extend(vc.uint32_to_bytes(profile_size))

        # Preferred CMM type
        # If no preferred CMM is identified, this field shall be set to zero (00000000h).
        header.extend(vc.string_to_bytes("appl"))

        # Profile version number 4.4.0.0
        header.extend(vc.uint32_to_bytes(0x04400000))

        # Profile/Device class
        # abst = abstract profile
        # link = device link profile
        header.extend(vc.string_to_bytes("link"))

        # Colour space of data
        header.extend(vc.string_to_bytes(self.dcs))

        # PCS
        header.extend(vc.string_to_bytes(self.pcs))

        # Date and time this profile was first created
        header.extend(vc.datetime_to_bytes(datetime.datetime.now()))
        
        # acsp (61637370h) profile file signature
        header.extend(vc.string_to_bytes("acsp"))

        # Primary platform signature
        header.extend(vc.string_to_bytes("APPL"))

        # Profile flags
        # 0 1 Embedded profile (0 if not embedded, 1 if embedded in file)
        # 1 1 Profile cannot be used independently of the embedded colour data (set to 1 if true, 0 if false)
        profile_flags = [0, 1, 0, 0]
        header.extend(profile_flags)

        # Device manufacturer and model
        header.extend([0] * 8)  # 4 bytes each for manufacturer and model

        # Device attributes
        header.extend([0] * 8)  # Attributes

        # Rendering Intent
        rendering_intent = [0, 0, 0, 3] # absolute colorimetric
        header.extend(rendering_intent)

        # The PCS illuminant
        header.extend(vc.uint32_to_bytes(63190))
        header.extend(vc.uint32_to_bytes(65536))
        header.extend(vc.uint32_to_bytes(54061))

        # Profile creator signature
        header.extend(vc.string_to_bytes("HPDE"))

        # Profile ID (MD5 hash)
        header.extend([0] * 16)
        
        # Reserved bytes
        header.extend([0] * 28)
    
        return header

    def update_profile_id(profilebytes: bytearray) -> bytearray:
        
        data = bytearray(profilebytes)

        # Check if the profile is at least 128 bytes long (minimum size for ICC header)
        if len(data) < 128:
            raise ValueError("File is too small to be a valid ICC profile")

        # Zero out the Profile flags field (bytes 44 to 47)
        data[44:48] = b'\x00' * 4

        # Zero out the Rendering intent field (bytes 64 to 67)
        data[64:68] = b'\x00' * 4

        # Zero out the Profile ID field (bytes 84 to 99)
        data[84:100] = b'\x00' * 16

        # Calculate the MD5 hash of the modified data
        pid = md5(data).digest()
        
        profilebytes[84:100] = pid
        
        return profilebytes

class ICCheader_Prtr:
    
    def __init__(self) -> None:
        self.dcs = "CMYK"
        self.pcs = "LAB "
    
    # set the device color space
    def set_input_space(self, dcs: str) -> None:
        self.dcs = dcs
        
    # set the profile connection space
    def set_output_space(self, pcs: str) -> None:
        self.pcs = pcs
    
    def header(self, profile_size: int):
        print(f"Profile size: {profile_size}")
        
        header = bytearray()

        # Profile Size
        header.extend(vc.uint32_to_bytes(profile_size))

        # Preferred CMM type
        # If no preferred CMM is identified, this field shall be set to zero (00000000h).
        header.extend(vc.string_to_bytes("appl"))

        # Profile version number 4.4.0.0
        header.extend(vc.uint32_to_bytes(0x04400000))

        # Profile/Device class
        # abst = abstract profile
        # link = device link profile
        # spac = color space profile
        # prtr = printer profile
        header.extend(vc.string_to_bytes("prtr"))

        # Colour space of data
        header.extend(vc.string_to_bytes(self.dcs))

        # PCS
        #header.extend(vc.string_to_bytes("LAB "))
        header.extend("Lab ".encode('ascii'))

        # Date and time this profile was first created
        header.extend(vc.datetime_to_bytes(datetime.datetime.now()))
        
        # acsp (61637370h) profile file signature
        header.extend(vc.string_to_bytes("acsp"))

        # Primary platform signature
        header.extend(vc.string_to_bytes("APPL"))

        # Profile flags
        # 0 1 Embedded profile (0 if not embedded, 1 if embedded in file)
        # 1 1 Profile cannot be used independently of the embedded colour data (set to 1 if true, 0 if false)
        profile_flags = [0, 1, 0, 0]
        header.extend(profile_flags)

        # Device manufacturer and model
        header.extend([0] * 8)  # 4 bytes each for manufacturer and model

        # Device attributes
        header.extend([0] * 8)  # Attributes

        # Rendering Intent
        rendering_intent = [0, 0, 0, 3] # absolute colorimetric
        header.extend(rendering_intent)

        # The PCS illuminant
        header.extend(vc.uint32_to_bytes(63190))
        header.extend(vc.uint32_to_bytes(65536))
        header.extend(vc.uint32_to_bytes(54061))

        # Profile creator signature
        header.extend(vc.string_to_bytes("HPDE"))

        # Profile ID (MD5 hash)
        header.extend([0] * 16)
        
        # Reserved bytes
        header.extend([0] * 28)
    
        return header

    def update_profile_id(profilebytes: bytearray) -> bytearray:
        
        data = bytearray(profilebytes)

        # Check if the profile is at least 128 bytes long (minimum size for ICC header)
        if len(data) < 128:
            raise ValueError("File is too small to be a valid ICC profile")

        # Zero out the Profile flags field (bytes 44 to 47)
        data[44:48] = b'\x00' * 4

        # Zero out the Rendering intent field (bytes 64 to 67)
        data[64:68] = b'\x00' * 4

        # Zero out the Profile ID field (bytes 84 to 99)
        data[84:100] = b'\x00' * 16

        # Calculate the MD5 hash of the modified data
        pid = md5(data).digest()
        
        profilebytes[84:100] = pid
        
        return profilebytes


class IccElements:
        
    @staticmethod
    def entry_mluc(inhalt: str) -> bytes:
        '''
        multiLocalizedUnicodeType
        '''
        content = vc.string_to_bytes_utf16be(inhalt)        
        content = IccElements.fillup(content)
        
        output = bytearray()

        # mluc (0x6D6C7563) type signature
        output.extend(vc.string_to_bytes("mluc"))

        # reserved, must be set to 0
        output.extend(vc.uint32_to_bytes(0))

        # number of names (n): the number of name records that follow.
        output.extend(vc.uint32_to_bytes(1))

        # name record size
        output.extend(vc.uint32_to_bytes(12))

        # first name language code
        output.extend(vc.string_to_bytes("US"))

        # first name country code
        output.extend(vc.string_to_bytes("en"))

        # first name length
        output.extend(vc.uint32_to_bytes(len(content)))

        # first name offset
        first_name_offset = len(output) + 4
        output.extend(vc.uint32_to_bytes(first_name_offset))

        # String        
        output.extend(content)

        return output

    @staticmethod   
    def entry_mft2(
        num_input_channels: int,
        num_output_channels: int,
        matrix: list = None,
        input_table: list = None,
        lut_table: list = None,
        output_table: list = None
        ) -> bytes:        

        #region Default values

        if input_table == None:
            #input_table = [[IccConstants.MIN16, IccConstants.MAX16]] * num_input_channels
            input_table = IccElements.linear_table(num_input_channels)

        if output_table == None:
            #output_table = [[IccConstants.MIN16, IccConstants.MAX16]] * num_output_channels
            output_table = IccElements.linear_table(num_output_channels)

        if lut_table == None:
            lattice_points_default = 2
            lut_table = cb.generate_combinations_numpy(num_output_channels, lattice_points_default)
            #print(f"LUT table (default): {lut_table}")

        # calculate the number of lattice points from a list of gridpoints
        num_lattice_points = int(round((len(lut_table) / num_output_channels) ** (1 / num_input_channels)))

        # 3x3 matrix
        if matrix is None:
            eMax = 65536
            eMin = 0
            matrix = [
                eMax, eMin, eMin, 
                eMin, eMax, eMin, 
                eMin, eMin, eMax
            ]
            
        #endregion

        output = bytearray()

        # mft2 [6D667432h] [multi-function table with 2-byte precision] type signature
        output.extend(vc.string_to_bytes("mft2"))

        # reserved 
        output.extend(vc.uint32_to_bytes(0))

        # Number of Input Channels
        output.extend(vc.uint8_to_bytes(num_input_channels))

        # Number of Output Channels
        output.extend(vc.uint8_to_bytes(num_output_channels))

        # Number of CLUT grid points / identical for each side
        output.extend(vc.uint8_to_bytes(num_lattice_points))
        
        # Reserved
        output.extend(vc.uint8_to_bytes(0))

        # Matrix
        output.extend(vc.uint32_to_bytes(matrix))

        # Number of input table entries
        output.extend(vc.uint16_to_bytes(len(input_table[0])))

        # Number of output table entries
        output.extend(vc.uint16_to_bytes(len(output_table[0])))

        # Input tables
        output.extend(vc.uint16_to_bytes(input_table))

        # clut table
        output.extend(vc.uint16_to_bytes(lut_table))   

        # Output tables
        output.extend(vc.uint16_to_bytes(output_table))
        
        return IccElements.fillup(output)

    # UNTESTED !!! COULD WORK NEEDS FOR SURE A DEEPER LOOK INTO !!! NEEDS TO MFT2 OR NOT ????
    @staticmethod
    def entry_gamt(lut_table: list) -> bytes:
        '''
        ### gamutType encoding Version2
        lut_table is a list of 3-tuples (R, G, B) with values in range [0, 65535]
        '''
        output = bytearray()

        # gamt (0x67616d74) type signature
        output.extend(vc.string_to_bytes("gamt"))

        # reserved
        output.extend(vc.uint32_to_bytes(0))

        # Number of entries in the LUT
        num_entries = len(lut_table)
        output.extend(vc.uint32_to_bytes(num_entries))

        # LUT entries
        for entry in lut_table:
            output.extend(vc.uint16_to_bytes(entry))

        # add missing bytes to ensure divison by 4 is possible       
        return IccElements.fillup(output)

    # noch nicht fertig, works with default values !!! TO DO !!!
    @staticmethod
    def entry_clrt(num_channels, color_names = None, color_values = None) -> bytes:
        '''
        ### colorantTableType encoding / OutputColorantTableType encoding Version2
        color_values are in LAB format
        '''
        #region Default values
        
        if color_names == None:
            color_names = ["Cyan", "Magenta", "Yellow", "Black", "Spot_1", "Spot_2", "Spot_3", "Spot_4", "Spot_5", "Spot_6", "Spot_7", "Spot_8", "Spot_9", "Spot_10", "Spot_11", "Spot_12", "Spot_13", "Spot_14", "Spot_15"]
        
        if color_values == None:
            color_values = [
                [0, 0, 0],
                [0, 0, 0],
            ]
        
        '''
        if color_names != None and color_values != None:
            num_channels = len(color_names)
        '''
        
        #endregion
        
        output = bytearray()

        # clrt (0x636c7274) type signature
        output.extend(vc.string_to_bytes("clrt"))
        
        # reserved
        output.extend(vc.uint32_to_bytes(0))
        
        # Count of colorants (n)
        output.extend(vc.uint32_to_bytes(num_channels))
        
        for i in range(num_channels):
            # First colorant name (32-byte field, null terminated, unused bytes shall be set to zero)
            colorant_name = color_names[i].ljust(32, "\0")
            output.extend(vc.string_to_bytes(colorant_name))
            
            # PCS values of the first colorant
            output.extend(vc.uint16_to_bytes(45000 + i * 1000))
            output.extend(vc.uint16_to_bytes(12000 + i * 2000))
            output.extend(vc.uint16_to_bytes(32000 + i * 3000))
        
        # add missing bytes to ensure divison by 4 is possible       
        return IccElements.fillup(output)

    # !!! TO DO !!!
    def entry_colt():
        pass

    @staticmethod
    def entry_text(content: str) -> bytes:
        '''
        ### textType encoding Version2
        '''
        output = bytearray()

        # text (0x74657874) type signature
        output.extend(vc.string_to_bytes("text"))

        # reserved
        output.extend(vc.uint32_to_bytes(0))

        # Text description 
        # a string of (element size - 8) 7-bit ASCII characters      
        output.extend(vc.string_to_bytes(content))
        
        # terminat the string with 0x00
        output.extend(IccConstants.ZEROBYTE1)
        
        # add missing bytes to ensure divison by 4 is possible
        return IccElements.fillup(output)

    # noch nicht fertig
    @staticmethod
    def entry_desc(content: str) -> bytes:
        '''
        ### descType encoding / textDescriptionType encoding Version2
        
        6.4.32 profileDescriptionTag
        Tag Type: textDescriptionType
        Tag Signature: ‘desc’ (64657363h)
        Structure containing invariant and localizable versions of the profile description for display. 
        The content of this structure is described in 6.5.17. This invariant description has no fixed relationship 
        to the actual profile disk file name.
        '''
        '''
        Byte
        Offset Content Encoded as...
            0..3 ‘desc’ (64657363h) type signature
            4..7 reserved, must be set to 0
        8..11 ASCII invariant description count, including terminating null (description length) uInt32Number
        12..n-1 ASCII invariant description 7-bit ASCII
        n..n+3 Unicode language code uInt32Number
        n+4..n+7 Unicode localizable description count (description length) uInt32Number
        n+8..m-1 Unicode localizable description
        m..m+1 ScriptCode code uInt16Number
        m+2 Localizable Macintosh description count (description length) uInt8Number
        m+3..m+69 Localizable Macintosh description
        '''
        
        output = bytearray()
        
        # desc (0x64657363) type signature
        output.extend(vc.string_to_bytes("desc"))
        
        # reserved
        output.extend(vc.uint32_to_bytes(0))
        
        ascii_invariant_description_count = len(content) + 1
        # ASCII invariant description count
        output.extend(vc.uint32_to_bytes(ascii_invariant_description_count))
        # ASCII invariant description
        output.extend(vc.string_to_bytes(content))
        
        
        # Unicode language code
        output.extend(vc.uint32_to_bytes(0))
        # Unicode localizable description count
        output.extend(vc.uint32_to_bytes(0))
        # Unicode localizable description
        output.extend(vc.uint32_to_bytes(0))
        
        # ScriptCode code
        output.extend(vc.uint16_to_bytes(0))
        # Localizable Macintosh description count
        output.extend(vc.uint8_to_bytes(0))
        # Localizable Macintosh description
        output.extend([0] * 67)
        
        return output

    @staticmethod
    def entry_pseq(content: str) -> bytes:
        '''
        ### profileSequenceDescType encoding
        '''
        output = bytearray()

        # pseq (0x70736571) type signature
        output.extend(vc.string_to_bytes("pseq"))

        '''
        # reserved
        output.extend(vc.uint32_to_bytes(0))

        # ASCII invariant description count
        output.extend(vc.uint32_to_bytes(len(content)))

        # ASCII invariant description
        output.extend(vc.string_to_bytes(content))
        '''
        output.extend([0]* 8)
        
        return output
    
        '''
        9.2.44 profileSequenceDescTag
        Tag signature: ‘pseq’ (70736571h)
        Permitted tag type: profileSequenceDescType
        This tag describes the structure containing a description of the profile sequence from source to destination, typically used with the DeviceLink profile. The content of this structure is described in 10.17.
        '''


    def entry_wptp(illuminant: tuple[float, float, float], illuminant_name: str = "D50") -> bytes:
        '''
        Returns a whitePointTag (wtpt) as an XYZType
        Illuminant values must be float XYZ (e.g. D50 = (0.9642, 1.0000, 0.8249))
        '''

        def float_to_s15Fixed16Number(val: float) -> bytes:
            fixed_val = int(round(val * 65536))  # 2^16 = 65536
            return fixed_val.to_bytes(4, byteorder='big', signed=True)

        output = bytearray()

        # Tag type signature: 'XYZ '
        output.extend(b'XYZ ')

        # Reserved (4 bytes)
        output.extend((0).to_bytes(4, byteorder='big'))

        # XYZ values (each is 4-byte s15Fixed16Number)
        for component in illuminant:
            output.extend(float_to_s15Fixed16Number(component))

        return bytes(output)

    @staticmethod
    def tag_table_head(signature: str, offset: int, size: int) -> bytes:
        result = vc.string_to_bytes(signature)
        result.extend(vc.uint32_to_bytes(offset))
        result.extend(vc.uint32_to_bytes(size))
        return result
    
    @staticmethod
    def tag_table_head_extend(signature: str, offset: int, size: int) -> tuple[bytes, int]:
        result = vc.string_to_bytes(signature)
        result.extend(vc.uint32_to_bytes(offset))
        result.extend(vc.uint32_to_bytes(size))
                
        return (bytes(result), len(result))

    @staticmethod
    def fillup(content: bytearray) -> bytearray:
        '''
        Icc elements are always divisible by 4. The function calculates the number of bytes to be added to the content to ensure this.
        '''
        element_len = len(content)        
        if element_len % 4 != 0:
            add_num_bytes = 4 - (element_len % 4)
            content.extend(bytearray(add_num_bytes))
        return content                              

    @staticmethod
    def linear_table(num_channels: int) -> list:
        '''
        Generates a linear table with the entries for each channel [0, 65535].
        '''
        return [[IccConstants.MIN16, IccConstants.MAX16]] * num_channels


class DataColourSpaceSignatures(Enum):
    """
    ### Table 19 — Data colour space signatures
    {"sig": "GRAY", "num": 1}
    sig: Signature
    num: Number of channels
    """
    
    GRAY = {"sig": "GRAY", "num": 1}
    RGB  = {"sig": "RGB ", "num": 3}
    CMY  = {"sig": "CMY ", "num": 3}
    CMYK = {"sig": "CMYK", "num": 4}
    MC02 = {"sig": "2CLR", "num": 2}
    MC03 = {"sig": "3CLR", "num": 3}
    MC04 = {"sig": "4CLR", "num": 4}
    MC05 = {"sig": "5CLR", "num": 5}
    MC06 = {"sig": "6CLR", "num": 6}
    MC07 = {"sig": "7CLR", "num": 7}
    MC08 = {"sig": "8CLR", "num": 8}
    MC09 = {"sig": "9CLR", "num": 9}
    '''
    MC10 = {"sig": "ACLR", "num": 10}
    MC11 = {"sig": "BCLR", "num": 11}
    MC12 = {"sig": "CCLR", "num": 12}
    MC13 = {"sig": "DCLR", "num": 13}
    MC14 = {"sig": "ECLR", "num": 14}
    MC15 = {"sig": "FCLR", "num": 15}
    '''
    
    @staticmethod
    def get_list() -> list:
        # return a list of all enum values
        return list(DataColourSpaceSignatures)
    
    @staticmethod
    def get_dict_from_num(numChannels: int) -> dict:
        for i in DataColourSpaceSignatures:
            if i.value["num"] == numChannels:
                return i.value
        return None
    
    
    '''
    Signature NOT IMPLEMENTED !!!
    XYZ  
    Lab  
    Luv  
    YCbr 
    Yxy  
    HSV  
    HLS 
    '''
