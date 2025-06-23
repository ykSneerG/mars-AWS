import datetime


class ICCvalueconverter:

    @staticmethod
    def uint8_to_bytes(value: int) -> bytes:
        '''
        ### uInt8Number
        is an unsigned 1-byte (8-bit) quantity.
        '''
        return value.to_bytes(1, byteorder='big')
    
    @staticmethod
    def uint16_to_bytes(combos) -> bytearray:
        '''
        ### s15Fixed16Number 
        is a fixed signed 4-byte (32-bit) quantity which has 16 fractional bits.
        '''
        def single_uint16_to_bytes(value: int) -> bytes:
            # Converts a single integer (uint16) to bytes
            return value.to_bytes(2, byteorder='big')

        def process_entry(entry):
            if isinstance(entry, (list, tuple)):
                for item in entry:
                    process_entry(item)
            elif isinstance(entry, int):
                result.extend(single_uint16_to_bytes(entry))
            else:
                raise TypeError("Entries must be integers, lists, or tuples")

        result = bytearray()
        process_entry(combos)
        return result
    
    @staticmethod
    def uint32_to_bytes(combos) -> bytes:
        '''
        ### s15Fixed16Number 
        is a fixed signed 4-byte (32-bit) quantity which has 16 fractional bits.
        '''
        def single_uint16_to_bytes(value: int) -> bytes:
            # Converts a single integer (uint32) to bytes
            return value.to_bytes(4, byteorder='big')

        def process_entry(entry):
            if isinstance(entry, (list, tuple)):
                for item in entry:
                    process_entry(item)
            elif isinstance(entry, int):
                result.extend(single_uint16_to_bytes(entry))
            else:
                raise TypeError("Entries must be integers, lists, or tuples")

        result = bytearray()
        process_entry(combos)
        return result
        
    @staticmethod
    def string_to_bytes(value: str) -> bytearray:
        ba = bytearray(value.encode('ascii'))
        """ print(f"{repr(ba)}") 
        print(f"String to bytes (ASCII): {value} -> {ba}")
        print(ba) """
        return ba

    @staticmethod
    def string_to_bytes_utf16be(value: str) -> bytearray:
        return bytearray(value.encode('utf-16be'))
    
    @staticmethod
    def encode_to_7bit_ascii(input_string):
        encoded_chars = []
        for char in input_string:
            ascii_value = ord(char)
            if ascii_value > 127:
                raise ValueError(f"Character {char} is not a 7-bit ASCII character")
            encoded_chars.append(chr(ascii_value))
        return ''.join(encoded_chars)
    
    @staticmethod
    def datetime_to_bytes(value: datetime) -> bytes:
        timebytes = bytearray()
        timebytes.extend(ICCvalueconverter.uint16_to_bytes(value.year))
        timebytes.extend(ICCvalueconverter.uint16_to_bytes(value.month))
        timebytes.extend(ICCvalueconverter.uint16_to_bytes(value.day))
        timebytes.extend(ICCvalueconverter.uint16_to_bytes(value.hour))
        timebytes.extend(ICCvalueconverter.uint16_to_bytes(value.minute))
        timebytes.extend(ICCvalueconverter.uint16_to_bytes(value.second))
        return timebytes
