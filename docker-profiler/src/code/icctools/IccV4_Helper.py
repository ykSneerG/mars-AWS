import string
import random
import numpy as np
    

class Helper:
    
    @staticmethod
    def random_id_string(length):
        '''
        Generate a random string of fixed length
        '''

        return ''.join(random.choices(string.digits + string.ascii_uppercase, k=length))
    
    @staticmethod
    def random_id(blocks=4, block_length=5, separator='-'):
        '''
        Generate random String with the following pattern xxxxx-xxxxx-xxxxx-xxxxx, containing only numbers and uppercase letters
        '''
        
        return separator.join(Helper.random_id_string(block_length) for _ in range(blocks))
    
    
    @staticmethod
    def can_resort_to_equal(orderOld, orderNew):
        # Custom key function to convert all elements to strings for comparison
        def custom_key(x):
            return str(x)
        
        copied_old = orderOld.copy()
        copied_new = orderNew.copy()
        
        # Check if both inputs are lists
        if not isinstance(copied_old, list) or not isinstance(copied_new, list):
            return False
        
        # Check if all elements are unique in both lists
        if len(copied_old) != len(set(copied_old)) or len(copied_new) != len(set(copied_new)):
            return False
        
        # Sort both lists using the custom key
        sortedOld = sorted(copied_old, key=custom_key)
        sortedNew = sorted(copied_new, key=custom_key)
        
        # Compare the sorted lists
        return sortedOld == sortedNew

    @staticmethod
    def lab_to_uint16(lab):
        
        def convert_to_uint16(value):
            L, A, B = value
            L_enc = round((L / 100.39) * 65535.0) # round((L / 100.0) * 65535.0)
            a_enc = round(((A + 128.0) / 256.0) * 65535.0) #255.0
            b_enc = round(((B + 128.0) / 256.0) * 65535.0) #255.0
            
            # clamp values to ensure they are within the uint16 range
            L_enc = max(0, min(65535, L_enc))
            a_enc = max(0, min(65535, a_enc))
            b_enc = max(0, min(65535, b_enc))
            
            return [L_enc, a_enc, b_enc]
        
        if isinstance(lab, list):
            if all(isinstance(item, list) for item in lab):
                return [convert_to_uint16(value) for value in lab]
            else:
                return convert_to_uint16(lab)

    @staticmethod
    def uint16_to_lab(encoded):
        L_enc, a_enc, b_enc = encoded
        L = (L_enc / 65535.0) * 100.0
        a = ((a_enc / 65535.0) * 255.0) - 128.0
        b = ((b_enc / 65535.0) * 255.0) - 128.0
        return [L, a, b]
    
    @staticmethod
    def flatten_list(nested_list):
        flattened = []
        for sublist in nested_list:
            if isinstance(sublist, list):
                flattened.extend(Helper.flatten_list(sublist))
            else:
                flattened.append(sublist)
        return flattened
    
    @staticmethod
    def scale_to_range(value, old_min, old_max, new_min, new_max):
        
        value_np = np.array(value)
        
        result =   (value_np - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
        
        return result.tolist()

        """ def scale(value):
            # Calculate the scaling factor
            scale_factor = (new_max - new_min) / (old_max - old_min)
            # Apply the scaling factor and shift the value
            scaled_value = (value - old_min) * scale_factor + new_min
            return scaled_value

        if isinstance(value, list):
            return [scale(v) for v in value]
        else:
            return scale(value) """
    
    @staticmethod
    def divide_by_100(value):
        """
        Scale a value to the range of 0 to 1.
        """
        value_np = np.array(value)
        return (value_np / 100).tolist()
        
    @staticmethod
    def lab_to_lut(lab):
        """
        Convert LAB values to a LUT format.
        """
        uint16_lab = Helper.lab_to_uint16(lab)
        flattened = Helper.flatten_list(uint16_lab)
        return flattened
    
    @staticmethod
    def media_relative(cieXYZ, wtpt ):
        
        
        """ illuminant=[0.9642, 1.0000, 0.8249] """
        
        
        xyz_np = np.array(cieXYZ)
        ill_np = np.array(wtpt)
        return (xyz_np / ill_np).tolist()

        """ def media_relative_single(value, illuminant):

            result = []
            for i in range(len(cieXYZ)):
                result.append(cieXYZ[i] / illuminant[i])
            
            return result
        
        if  isinstance(cieXYZ, list):
            return [media_relative_single(value, illuminant) for value in cieXYZ]
        else:
            return media_relative_single(cieXYZ, illuminant) """
        
    
    @staticmethod
    def xyz_to_lab(xyz):

        def convert_to_lab(value):
            X, Y, Z = value
            """ X = X / 95.047
            Y = Y / 100.000
            Z = Z / 108.883 """

            # Calculate the values for the LAB color space
            L = 116 * (Y ** (1 / 3)) - 16
            a = 500 * ((X ** (1 / 3)) - (Y ** (1 / 3)))
            b = 200 * ((Y ** (1 / 3)) - (Z ** (1 / 3)))

            return [L, a, b]

        if isinstance(xyz, list):
            if all(isinstance(item, list) for item in xyz):
                return [convert_to_lab(value) for value in xyz]
            else:
                return convert_to_lab(xyz)
    
    @staticmethod
    def translate_vector(data, src, dst):
        src_np = np.array(src)
        dst_np = np.array(dst)
        translate = dst_np - src_np
        
        return (data + translate).tolist()
    
    
    @staticmethod
    def round_list(data, decimal_places=2):
        return [round(item, decimal_places) for item in data]
    
    
class ColorTrafo:
    
    ILLUMINANT_D50 = [0.9642, 1.0000, 0.8249]
    
    @staticmethod
    def media_relative(cieXYZ, wtpt):
        
        xyz_np = np.array(cieXYZ)
        ill_np = np.array(wtpt)
        return (xyz_np / ill_np).tolist()

    
    @staticmethod
    def xyz_to_lab_OLD(cieXYZ):

        def convert_to_lab(value):
            X, Y, Z = value
            """ X = X / ColorTrafo.ILLUMINANT_D50[0]
            Y = Y / ColorTrafo.ILLUMINANT_D50[1]
            Z = Z / ColorTrafo.ILLUMINANT_D50[2] """

            # Calculate the values for the LAB color space
            L = 116 * (Y ** (1 / 3)) - 16
            A = 500 * ((X ** (1 / 3)) - (Y ** (1 / 3)))
            B = 200 * ((Y ** (1 / 3)) - (Z ** (1 / 3)))

            return [L, A, B]

        if isinstance(cieXYZ, list):
            if all(isinstance(item, list) for item in cieXYZ):
                return [convert_to_lab(value) for value in cieXYZ]
            else:
                return convert_to_lab(cieXYZ)



    @staticmethod
    def xyz_to_lab(cieXYZ):
        # Convert input to numpy array for vectorized operations
        arr = np.asarray(cieXYZ, dtype=np.float64)
        
        # Handle both single color and multiple colors cases
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        
        # Normalize by D50 illuminant (uncomment if needed)
        # arr = arr / np.array(ColorTrafo.ILLUMINANT_D50)
        
        # Avoid division by zero and negative values
        arr = np.maximum(arr, 1e-16)
        
        # Calculate LAB components
        xyz_cbrt = np.cbrt(arr)  # Cube root is faster than power(1/3)
        
        L = 116 * xyz_cbrt[:, 1] - 16
        A = 500 * (xyz_cbrt[:, 0] - xyz_cbrt[:, 1])
        B = 200 * (xyz_cbrt[:, 1] - xyz_cbrt[:, 2])
        
        # Stack results
        lab = np.column_stack((L, A, B))
        
        # Return in same format as input
        if isinstance(cieXYZ, list):
            if all(isinstance(item, list) for item in cieXYZ):
                return lab.tolist()
            return lab[0].tolist()
        return lab
    
    @staticmethod
    def lab_to_xyz(cieLab):

        def convert_to_xyz(value):
            L, A, B = value
            Y = (L + 16) / 116
            X = A / 500 + Y
            Z = Y - B / 200

            # Convert to XYZ using the inverse of the D50 illuminant
            X = X ** 3 * ColorTrafo.ILLUMINANT_D50[0]
            Y = Y ** 3 * ColorTrafo.ILLUMINANT_D50[1]
            Z = Z ** 3 * ColorTrafo.ILLUMINANT_D50[2]

            return [X, Y, Z]

        if isinstance(cieLab, list):
            if all(isinstance(item, list) for item in cieLab):
                return [convert_to_xyz(value) for value in cieLab]
            else:
                return convert_to_xyz(cieLab)
            
    @staticmethod
    def de00_lab(lab1: np.ndarray, lab2: np.ndarray):
        """
        Compute the Delta E 2000 between two LAB colors (or arrays of colors).
        Inputs:
            lab1, lab2: arrays of shape (..., 3)
        Returns:
            delta_e: array of shape (...)
        """
        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

        avg_L = (L1 + L2) / 2
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        avg_C = (C1 + C2) / 2

        G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7)))
        a1p = (1 + G) * a1
        a2p = (1 + G) * a2
        C1p = np.sqrt(a1p**2 + b1**2)
        C2p = np.sqrt(a2p**2 + b2**2)
        avg_Cp = (C1p + C2p) / 2

        h1p = np.degrees(np.arctan2(b1, a1p)) % 360
        h2p = np.degrees(np.arctan2(b2, a2p)) % 360

        delta_Lp = L2 - L1
        delta_Cp = C2p - C1p

        dhp = h2p - h1p
        dhp = np.where(np.abs(dhp) > 180, dhp - np.sign(dhp) * 360, dhp)
        delta_hp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

        avg_Hp = np.where(
            np.abs(h1p - h2p) > 180,
            (h1p + h2p + 360) / 2,
            (h1p + h2p) / 2
        )
        avg_Hp = np.where((C1p * C2p) == 0, h1p + h2p, avg_Hp)

        T = (1
            - 0.17 * np.cos(np.radians(avg_Hp - 30))
            + 0.24 * np.cos(np.radians(2 * avg_Hp))
            + 0.32 * np.cos(np.radians(3 * avg_Hp + 6))
            - 0.20 * np.cos(np.radians(4 * avg_Hp - 63)))

        delta_ro = 30 * np.exp(-((avg_Hp - 275) / 25) ** 2)
        R_C = 2 * np.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7))
        S_L = 1 + ((0.015 * (avg_L - 50) ** 2) / np.sqrt(20 + (avg_L - 50) ** 2))
        S_C = 1 + 0.045 * avg_Cp
        S_H = 1 + 0.015 * avg_Cp * T
        R_T = -R_C * np.sin(np.radians(2 * delta_ro))

        delta_E = np.sqrt(
            (delta_Lp / S_L) ** 2 +
            (delta_Cp / S_C) ** 2 +
            (delta_hp / S_H) ** 2 +
            R_T * (delta_Cp / S_C) * (delta_hp / S_H)
        )

        return delta_E