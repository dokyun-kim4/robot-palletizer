"""
Environment modeling and collision checking.
Maintains an internal representation of the workspace (robot, pallet, boxes, constraints)
to ensure safe operation and prevent both self-collisions and environment collisions.
"""
import numpy as np

class Pallet:
    def __init__(self, first_box_position, box_size = (0.1524, 0.1016, 0.0762), pallet_dim = (3,3,1)):
        self.first_box_position = first_box_position
        self.box_size = box_size
        self.pallet_dim = pallet_dim
        self.num_boxes = pallet_dim[0] * pallet_dim[1] * pallet_dim[2]
        self.box_positions = self._calculate_box_positions()
        self.pallet_occupancy = np.zeros(pallet_dim, dtype=bool)
    
    def _calculate_box_positions(self):
        positions = np.empty(self.pallet_dim, dtype=object)
        
        for i in range(self.pallet_dim[0]):
            for j in range(self.pallet_dim[1]):
                for k in range(self.pallet_dim[2]):
                    positions[i][j][k] = (
                                            self.first_box_position[0] + i * self.box_size[0], 
                                            self.first_box_position[1] + j * self.box_size[1], 
                                            self.first_box_position[2] + k * self.box_size[2],
                                        )       
        return positions
    
    def next_empty_position(self):
        """
        Returns next empty pallet position for a box and updates
        the internal occupancy grid of the pallet
        """
        pass

if __name__ == "__main__":
    # Example usage
    pallet = Pallet(first_box_position=(0.0, 0.0, 0.0))
    print("Box positions on the pallet:")
    for layer in pallet.box_positions:
        for pos in layer:
            print(pos)


