from io import BytesIO
# import numpy as np      # type: ignore
import tifffile as tf   # type: ignore


class SpyImage():
    
    def __init__(self, image_bytes: bytes) -> None:
        self.image_bytes = image_bytes
        
        if image_bytes:
            self.read()
    
    def read(self) -> list[list[int]]:
        """
        Read a spy image.

        Parameters:
        image_bytes (BytesIO): Image as BytesIO.
        Returns:
        List[List[int]]: The image as a list of lists of integers.
        """
        image_array = tf.imread(BytesIO(self.image_bytes))
        
        # Reshape the array to a 2D format
        height, width, channels = image_array.shape
        image_array_reshaped = image_array.reshape((height * width, channels))
        
        self.channels = channels
        self.width = width
        self.height = height
        self.gridpoints = image_array_reshaped.tolist()
        
        #return image_array_reshaped.tolist()
