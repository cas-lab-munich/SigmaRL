import xml.etree.ElementTree as ET
import torch
import matplotlib.pyplot as plt
# Set Matplotlib to use Type 1 or TrueType fonts instead of Type 3
plt.rcParams['pdf.fonttype'] = 42  # Ensures TrueType font (TT) is used
# plt.rcParams['ps.fonttype'] = 42   # Same setting for PostScript (PS) output

import numpy as np

# Scientific plotting
# import scienceplots
# plt.rcParams.update({'figure.dpi': '100'}) # Avoid DPI problem (https://github.com/garrettj403/SciencePlots/issues/60)
# plt.style.use(['science','ieee']) # The science + ieee styles for IEEE papers (can also be one of 'ieee' and 'science' )

import os
import sys
script_dir = os.path.dirname(__file__) # Directory of the current scrit
project_root = os.path.dirname(script_dir) # Project root directory
if project_root not in sys.path:
    sys.path.append(project_root)

from utilities.helper_scenario import get_rectangle_vertices
    
from utilities.colors import Color # Do not remove (https://github.com/garrettj403/SciencePlots)
# print(plt.style.available) # List all available style

intersection_outer_boundary_ids = [
    18, 44, 96, 70
] # Take the right boundaries of these lanelets to form the outer boundaries of the intersection


intersection_lanelets_ids = [
    25, 26, 52, 72, 18, 17, 43, 73, 51, 50, 102, 20, 44, 45, 97, 21, 103, 104, 78, 46, 96, 95, 69, 47, 77, 76, 24, 98, 70, 71, 19, 99,
]

def parse_point(element, device):
    """ Parses a point element to extract x and y coordinates. """
    x = float(element.find("x").text) if element.find("x") is not None else None
    y = float(element.find("y").text) if element.find("y") is not None else None
    return torch.tensor([x, y], device=device)

def parse_bound(element, device):
    """ Parses a bound (left boundary or right boundary) element to extract points and line marking. """
    points = [parse_point(point, device) for point in element.findall("point")]
    points = torch.vstack(points)
    line_marking = element.find("lineMarking").text if element.find("lineMarking") is not None else None
    return points, line_marking

def get_center_length_yaw_polyline(polyline: torch.Tensor):
    """
    This function calculates the center points, lengths, and yaws of all line segments of the given polyline.
    """
    
    center_points = polyline.unfold(0,2,1).mean(dim=2)
    
    polyline_vecs = polyline.diff(dim=0)
    lengths = polyline_vecs.norm(dim=1)
    yaws = torch.atan2(polyline_vecs[:,1], polyline_vecs[:,0])

    return center_points, lengths, yaws, polyline_vecs

def parse_lanelet(element, device):
    """ Parses a lanelet element to extract detailed information. """
    lanelet_data = {
        "id": int(element.get("id")),
        
        "left_boundary": [],
        "left_boundary_center_points": [],
        "left_boundary_lengths": [],
        "left_boundary_yaws": [],
        "left_line_marking": None,
        
        "right_boundary": [],
        "right_boundary_center_points": [],
        "right_boundary_lengths": [],
        "right_boundary_yaws": [],
        "right_line_marking": None,
        
        "center_line": [],
        "center_line_center_points": [],
        "center_line_lengths": [],
        "center_line_yaws": [],
        "center_line_marking": "dashed",
        
        "predecessor": [],
        "successor": [],
        "adjacent_left": None,
        "adjacent_right": None,
        "lanelet_type": None
    }
    for child in element:
        if child.tag == "leftBound":
            lanelet_data["left_boundary"], lanelet_data["left_line_marking"] = parse_bound(child, device)
        elif child.tag == "rightBound":
            lanelet_data["right_boundary"], lanelet_data["right_line_marking"] = parse_bound(child, device)
        elif child.tag == "predecessor":
            lanelet_data["predecessor"].append(int(child.get("ref")))
        elif child.tag == "successor":
            lanelet_data["successor"].append(int(child.get("ref")))
        elif child.tag == "adjacentLeft":
            lanelet_data["adjacent_left"] = {
                "id": int(child.get("ref")),
                "drivingDirection": child.get("drivingDir")
            }
        elif child.tag == "adjacentRight":
            lanelet_data["adjacent_right"] = {
                "id": int(child.get("ref")),
                "drivingDirection": child.get("drivingDir")
            }
        elif child.tag == "lanelet_type":
            lanelet_data["lanelet_type"] = child.text


    lanelet_data["center_line"] = (lanelet_data["left_boundary"] + lanelet_data["right_boundary"]) / 2
    
    lanelet_data["center_line_center_points"], lanelet_data["center_line_lengths"], lanelet_data["center_line_yaws"], _ = get_center_length_yaw_polyline(polyline=lanelet_data["center_line"])
    lanelet_data["left_boundary_center_points"], lanelet_data["left_boundary_lengths"], lanelet_data["left_boundary_yaws"], _ = get_center_length_yaw_polyline(polyline=lanelet_data["left_boundary"])
    lanelet_data["right_boundary_center_points"], lanelet_data["right_boundary_lengths"], lanelet_data["right_boundary_yaws"], _ = get_center_length_yaw_polyline(polyline=lanelet_data["right_boundary"])
    
    return lanelet_data

# Function to parse intersections
def parse_intersections(element):
    intersection_info = []

    for incoming in element.findall("incoming"):
        incoming_info = {
            "incomingLanelet": int(incoming.find("incomingLanelet").get("ref")), # The starting lanelet of a part of the intersection
            "successorsRight": int(incoming.find("successorsRight").get("ref")), # The successor right lanelet of the incoming lanelet
            "successorsStraight": [int(s.get("ref")) for s in incoming.findall("successorsStraight")], # The successor lanelet(s) of the incoming lanelet
            "successorsLeft": int(incoming.find("successorsLeft").get("ref")), # The successor left lanelet of the incoming lanelet
        }
        intersection_info.append(incoming_info)

    return intersection_info


def visualize_and_save_map(lanelets, intersection_info, is_save_fig = False, is_visualize = False, is_show_vehicles = True,):
    x_lim = 4.5 # [m] Dimension in x-direction 
    y_lim = 4.0 # [m] Dimension in y-direction 

    # Set up the plot
    plt.figure(figsize=(x_lim*0.8, y_lim*0.8))  # Size in inches, adjusted for 4.0m x 4.5m dimensions
    plt.axis("equal")  # Ensure x and y dimensions are equally scaled

    line_width = 0.35
    fontsize = 11
    # fontname = "Time New Roman"

    for i in range(len(lanelets)):
        lanelet = lanelets[i]
        # Extract coordinates for left, right, and center lines
        left_bound = lanelet["left_boundary"]
        right_bound = lanelet["right_boundary"]
        center_line = lanelet["center_line"]

        # Extract line markings
        left_line_marking = lanelet["left_line_marking"]
        right_line_marking = lanelet["right_line_marking"]
        center_line_marking = lanelet["center_line_marking"]

        is_use_random_color = False
        
        # Choose color
        color = np.random.rand(3,) if is_use_random_color else "grey"

        # Plot left boundary
        plt.plot(left_bound[:, 0], left_bound[:, 1], linestyle="--" if left_line_marking == "dashed" else "-", color=color, linewidth=line_width)
        # Plot right boundary
        plt.plot(right_bound[:, 0], right_bound[:, 1], linestyle="--" if right_line_marking == "dashed" else "-", color=color, linewidth=line_width)
            
        # Plot center line
        # plt.plot(center_line[:, 0], center_line[:, 1], linestyle="--" if center_line_marking == "dashed" else "-", color=color, linewidth=line_width)
        # Adding lanelet ID as text
        
        is_show_id = False
        if is_show_id:
            plt.text(center_line[int(len(center_line)/2), 0], center_line[int(len(center_line)/2), 1], str(lanelet["id"]), color=color, fontsize=fontsize)
            
    is_highlight_intersection = True
    if is_highlight_intersection:
        intersection_boundary = []
        for j in intersection_outer_boundary_ids:
            lanelet = lanelets[j-1]
            right_bound = lanelet["right_boundary"]
            intersection_boundary.append(right_bound)

            # # Plot right boundary
            # plt.plot(right_bound[:, 0], right_bound[:, 1], linestyle="--" if right_line_marking == "dashed" else "-", color=color, linewidth=line_width)
            
        intersection_boundary = torch.vstack([b for b in intersection_boundary])

        plt.fill(intersection_boundary[:, 0], intersection_boundary[:, 1], color='grey', alpha=0.2)

    if is_show_vehicles:
        w = 0.08 # Width
        l = 0.16 # Length
        
        rec_x = [0.208, 0.208, 0.208 + w, 0.208 + w, 0.208]
        rec_y = [2.1, 2.1 - l, 2.1 - l, 2.1, 2.1]
        
        lanelet_ids = [71, 46, 20, 69]
        point_indices = [2, 3, 3, 3]
        
        for k in range(len(lanelet_ids)):
            
            lanelet_id = lanelet_ids[k]
            lanelet = lanelets[lanelet_id-1]
            
            center_line = lanelet["center_line"]
            center_line_jaws = lanelet["center_line_yaws"]
            
            point_index = point_indices[k]
            pos = center_line[point_index]
            yaw = center_line_jaws[point_index]
            
            vertices = get_rectangle_vertices(
                center=pos,
                yaw=yaw, 
                width=w, 
                length=l,
                is_close_shape=True
            ).squeeze(0)
            
            if k == 0:
                color_veh = Color.blue100
                # Get the observation area of the first agent
                # area_observation = get_rectangle_vertices(
                #     center=pos,
                #     yaw=yaw, 
                #     width=l*10,
                #     length=l*10,
                #     is_close_shape=True
                # ).squeeze(0)
                # plt.fill(area_observation[:, 0], area_observation[:, 1], color=color_veh, alpha=0.1)
                circle = plt.Circle(pos, l*5, color=color_veh, fill=False, linestyle="--", linewidth=0.5)
                plt.gca().add_patch(circle)

            else:
                color_veh = Color.black50
            
            plt.fill(vertices[:, 0], vertices[:, 1], color=color_veh)
            plt.text(pos[0] + 0.08, pos[1], k+1, color='black', fontsize=fontsize)

    # plt.xlabel(r"$x$ [m]", fontsize=18)
    # plt.ylabel(r"$y$ [m]", fontsize=18)
    plt.xlim((0, x_lim))
    plt.ylim((0, y_lim))
    # plt.xticks(np.arange(0, x_lim+0.05, 0.5), fontsize=12)
    # plt.yticks(np.arange(0, y_lim+0.05, 0.5), fontsize=12)
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel("Map visualization", fontsize=fontname)
    # Remove the outer box
    ax = plt.gca()  # Get current axes
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Save fig
    if is_save_fig:
        file_name = "map_visualization_paper.pdf"
        plt.tight_layout() # Set the layout to be tight to minimize white space
        plt.savefig(file_name, format="pdf", bbox_inches="tight")

        file_name = "map_visualization_paper.png"
        plt.savefig(file_name, format="png", bbox_inches="tight", transparent=True, dpi=600)
        
        print(f"A fig has been saved unter {file_name}.")
        
    if is_visualize:
        plt.show()


# Parse the XML file
def get_map_data(is_save_fig = False, is_visualize = False, is_show_vehicles = True, **kwargs):
    xml_file_path = kwargs.get("xml_file_path", "./assets/maps/cpm_lab_map.xml")
    device = kwargs.get("device", torch.device("cpu"))
    
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    lanelets = []
    intersection_info = []
    for child in root:
        if child.tag == "lanelet":
            lanelets.append(parse_lanelet(child, device))
        elif child.tag == "intersection":
            intersection_info = parse_intersections(child)
    
    # Calculate the mean lane width
    mean_lane_width = torch.mean(torch.norm(torch.vstack([lanelets[i]["left_boundary"] for i in range(len(lanelets))]) - torch.vstack([lanelets[i]["right_boundary"] for i in range(len(lanelets))]), dim=1))

    # Storing all the data in a single tree variable
    map_data = {
        "lanelets": lanelets,
        "intersection_info": intersection_info,
        "mean_lane_width": mean_lane_width,
    }
    
    
    
    # Visualization
    if is_visualize | is_save_fig:
        visualize_and_save_map(lanelets, intersection_info, is_save_fig, is_visualize, is_show_vehicles)
        
    return map_data


if __name__ == "__main__":
    map_data = get_map_data(
        is_visualize=False, # Rendering may be slow due to the usage of the package `scienceplots`. You may want to disable it by commenting the related codes out.
        is_save_fig=True, 
        is_show_vehicles=True,
    )
    print(map_data)

# Example of how to use 
# map_data["lanelets"][4] # Access all the information about the lanelet with ID 5
# map_data["lanelets"][4]["left_boundary"] # Access the coordinates of the left boundary of the lanelet with ID 5
# map_data["intersection_info"][1] # Access the information about the second part of the intersection