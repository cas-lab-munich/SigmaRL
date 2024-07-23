import xml.etree.ElementTree as ET
import torch
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

# Scientific plotting
# import scienceplots
# plt.rcParams.update({'figure.dpi': '100'}) # Avoid DPI problem (https://github.com/garrettj403/SciencePlots/issues/60)
# plt.style.use(['science','ieee']) # The science + ieee styles for IEEE papers (can also be one of 'ieee' and 'science' )

import os
import sys
script_dir = os.path.dirname(__file__) # Directory of the current script
project_root = os.path.dirname(script_dir) # Project root directory
if project_root not in sys.path:
    sys.path.append(project_root)
    
from utilities.colors import Color # Do not remove (https://github.com/garrettj403/SciencePlots)
# print(plt.style.available) # List all available style

class MapParseBase(ABC):
    """Base class for map parse.

    This is the class that map parse inherit from.

    The methods that are **compulsory to instantiate** are:

    - :class:`_parse_xml`
    - :class:`reset_world_at`
    - :class:`observation`
    - :class:`reward`

    The methods that are **optional to instantiate** are:

    - :class:`info`
    - :class:`extra_render`
    - :class:`process_action`

    """
    def __init__(self, map_path: str, device = "cpu"):
        self._map_path = map_path  # Path to the map data
        self._device = device  # Torch device
        self.bounds = {
            "min_x": float("inf"),
            "min_y": float("inf"),
            "max_x": float("-inf"),
            "max_y": float("-inf"),
        }  # Bounds of the map
        
        self.map_data = []  # A list of dict
        
        self.reference_paths = []
        self.reference_paths_intersection = []
        self.reference_paths_merge_in = []
        self.reference_paths_merge_out = []
                
        self._linewidth = 0.5
        self._fontsize = 12
        
        # Extract file name
        file_name_with_extension = os.path.basename(map_path)
        self._file_name, file_extension = os.path.splitext(file_name_with_extension)
    
    staticmethod
    def get_center_length_yaw_polyline(polyline: torch.Tensor):
        """
        This function calculates the center points, lengths, and yaws of all line segments of the given polyline.
        """
        
        center_points = polyline.unfold(0, 2, 1).mean(dim=2)
        
        polyline_vecs = polyline.diff(dim=0)
        lengths = polyline_vecs.norm(dim=1)
        yaws = torch.atan2(polyline_vecs[:, 1], polyline_vecs[:, 0])

        return center_points, lengths, yaws, polyline_vecs
    
    @abstractmethod
    def _parse_map_file():
        raise NotImplementedError()