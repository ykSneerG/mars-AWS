import string
import random


class RandomId:
    
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
        
        return separator.join(RandomId.random_id_string(block_length) for _ in range(blocks))
    