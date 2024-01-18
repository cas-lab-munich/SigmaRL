import xml.etree.ElementTree as ET
import torch
import matplotlib.pyplot as plt
import numpy as np

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

def get_center_length_yaw_polyline(polyline):
    """
    This function calculates the center points, lengths, and yaws of all line segments of the given polyline.
    """
    
    center_points = polyline.unfold(0,2,1).mean(dim=2)
    
    polyline_vecs = polyline.diff(dim=0)
    lengths = polyline_vecs.norm(dim=1)
    yaws = torch.atan2(polyline_vecs[:,1], polyline_vecs[:,0])

    return center_points, lengths, yaws

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
    
    lanelet_data["center_line_center_points"], lanelet_data["center_line_lengths"], lanelet_data["center_line_yaws"] = get_center_length_yaw_polyline(polyline=lanelet_data["center_line"])
    lanelet_data["left_boundary_center_points"], lanelet_data["left_boundary_lengths"], lanelet_data["left_boundary_yaws"] = get_center_length_yaw_polyline(polyline=lanelet_data["left_boundary"])
    lanelet_data["right_boundary_center_points"], lanelet_data["right_boundary_lengths"], lanelet_data["right_boundary_yaws"] = get_center_length_yaw_polyline(polyline=lanelet_data["right_boundary"])
    
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


def visualize_map(lanelets):
    x_lim = 4.5 # [m] Dimension in x-direction 
    y_lim = 4.0 # [m] Dimension in y-direction 

    # Set up the plot
    plt.figure(figsize=(x_lim*3, y_lim*3))  # Size in inches, adjusted for 4.0m x 4.5m dimensions
    plt.axis("equal")  # Ensure x and y dimensions are equally scaled

    is_use_random_color = True

    save_to_pdf = False
    file_name = "cpm_lab_map_visualization.pdf"

    line_width = 1.0
    font_size = 14

    for lanelet in lanelets:
        # Extract coordinates for left, right, and center lines
        left_bound = lanelet["left_boundary"]
        right_bound = lanelet["right_boundary"]
        center_line = lanelet["center_line"]

        # Extract line markings
        left_line_marking = lanelet["left_line_marking"]
        right_line_marking = lanelet["right_line_marking"]
        center_line_marking = lanelet["center_line_marking"]

        # Choose color
        color = np.random.rand(3,) if is_use_random_color else "black"

        # Plot left boundary
        plt.plot(left_bound[:, 0], left_bound[:, 1], linestyle="--" if left_line_marking == "dashed" else "-", color=color, linewidth=line_width)
        # Plot right boundary
        plt.plot(right_bound[:, 0], right_bound[:, 1], linestyle="--" if right_line_marking == "dashed" else "-", color=color, linewidth=line_width)
        # Plot center line
        plt.plot(center_line[:, 0], center_line[:, 1], linestyle="--" if center_line_marking == "dashed" else "-", color=color, linewidth=line_width)
        # Adding lanelet ID as text
        plt.text(center_line[int(len(center_line)/2), 0], center_line[int(len(center_line)/2), 1], str(lanelet["id"]), color=color, fontsize=font_size)

    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.xlim((0, x_lim))
    plt.ylim((0, y_lim))
    plt.title("Map Visualization")

    # Save figure as pdf
    if save_to_pdf:
        plt.tight_layout() # Set the layout to be tight to minimize white space
        plt.savefig(file_name, format="pdf", bbox_inches="tight")
        
    plt.show()


# Parse the XML file
def get_map_data(**kwargs):
    xml_file_path = kwargs.get("xml_file_path", "assets/cpm_lab_map.xml")
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
    IS_VISUALIZE = False
    if IS_VISUALIZE:
        visualize_map(lanelets)
        
    return map_data


if __name__ == "__main__":
    map_data = get_map_data()
    print(map_data)

# Example of how to use 
# map_data["lanelets"][4] # Access all the information about the lanelet with ID 5
# map_data["lanelets"][4]["left_boundary"] # Access the coordinates of the left boundary of the lanelet with ID 5
# map_data["intersection_info"][1] # Access the information about the second part of the intersection