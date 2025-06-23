class ColorAnalyser:
    def __init__(self):
        pass

    def lightest_color(self, lab, xyz, dcs):
        """
        Find the lightest point in the list of points.
        
        Parameters:
        points (list of list): List of points to search.
        
        Returns:
        list: The lightest point.
        """
        index = max(range(len(lab)), key=lambda i: lab[i][0])
        return (index, lab[index], xyz[index], dcs[index])