from typing import Protocol, Tuple
import numpy as np

class GuidanceProtocol(Protocol):

    def update(self, t: float, y: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
        Called every integrator step
        Returns
            vgo_mps:    float      - Remaining delta-v to target in m/s
            thurst_dir: Vector     - Unit vector of desired thrust direction
            throttle:   float      - 0 = coast, 1 = full throttle
    
        
        """
        ...
    
    def is_active(self, t:float, y: np.ndarray) -> bool:
        """Returns False -> coast. Used for clean burn/coast segmentation"""
        ...

