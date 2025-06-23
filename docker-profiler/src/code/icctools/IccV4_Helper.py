import string
import random
    

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
            L, a, b = value
            L_enc = round((L / 100.0) * 65535.0)
            a_enc = round(((a + 128.0) / 255.0) * 65535.0)
            b_enc = round(((b + 128.0) / 255.0) * 65535.0)
            
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

        def scale(value):
            # Calculate the scaling factor
            scale_factor = (new_max - new_min) / (old_max - old_min)
            # Apply the scaling factor and shift the value
            scaled_value = (value - old_min) * scale_factor + new_min
            return scaled_value

        if isinstance(value, list):
            return [scale(v) for v in value]
        else:
            return scale(value)
    