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

class ParseMapBase(ABC):
    """
    Base class for map parse.
    """
    def __init__(self, scenario_type, device, **kwargs):
        self._scenario_type = scenario_type  # Path to the map data
        self._device = device  # Torch device
        
        self._get_map_path()
        
        self._is_visualize_map = kwargs.pop("is_visualize_map", False)
        self._is_save_fig = kwargs.pop("is_save_fig", False)
        self._is_plt_show = kwargs.pop("is_plt_show", False)
        self._is_visu_lane_ids = kwargs.pop("is_visu_lane_ids", False)
        
        self._width = kwargs.pop("width", [])  # Width of the lane
        self._scale = kwargs.pop("scale", [])  # Scale of the map
        self._is_share_lanelets = kwargs.pop("is_share_lanelets", False)  # Whether agents can move to nearing lanelets

        self.bounds = {
            "min_x": float("inf"),
            "min_y": float("inf"),
            "max_x": float("-inf"),
            "max_y": float("-inf"),
        }  # Bounds of the map
        
        self.lanelets_all = []  # A list of dict. Each dict stores relevant data of a lane such as its center line, left boundary, and right boundary
        self.neighboring_lanelets_idx = []  # Neighboring lanelets of each lanelet
        
        self.reference_paths = []
        self.reference_paths_intersection = []
        self.reference_paths_merge_in = []
        self.reference_paths_merge_out = []

        self._linewidth = 0.5
        self._fontsize = 12
        
        # Use the same name for data to be stored as the scenario type
        self._scenario_type = self._scenario_type
        
    def _get_map_path(self):
        # Get the path to the corresponding map for the given scenario type
        # The most easy way to get the path of the target scenario type is to name the map file with `scenario_type` and store it at assets/maps/
        map_path_tentative_osm = f"assets/maps/{self._scenario_type}.osm"
        
        if ("cpm" in self._scenario_type) or ("CPM" in self._scenario_type):
            self._map_path = "assets/maps/cpm.xml"
        elif os.path.exists(map_path_tentative_osm):
            self._map_path = map_path_tentative_osm
        else:
            raise ValueError(f"No map file can be found for {self._scenario_type}. The map file must have the same name as the scenario type and stored at 'assets/maps/'")
        
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