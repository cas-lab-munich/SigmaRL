from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from tensordict import TensorDict
import torch

from utilities.helper_training import Parameters


# Scientific plotting
import scienceplots # Do not remove (https://github.com/garrettj403/SciencePlots)
plt.rcParams.update({'figure.dpi': '100'}) # Avoid DPI problem (https://github.com/garrettj403/SciencePlots/issues/60)
plt.style.use(['science','ieee']) # The science + ieee styles for IEEE papers (can also be one of 'ieee' and 'science' )
# print(plt.style.available) # List all available style

def evaluate_outputs(out_td: TensorDict, parameters: Parameters):
    """This function evaluates the test results presented as a tensordict."""    

    path_eval_out_td = parameters.where_to_save + parameters.mode_name + "_out_td.pth"        
    path_ref_path = parameters.where_to_save + parameters.mode_name + "_ref_path.pth"       
     
    if parameters.is_save_eval_results:
        # Save the input TensorDict
        torch.save(out_td, path_eval_out_td)
        
    # Set up the figure and subplots
    fig = plt.plot()
        

    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 1], w_pad=2) # left, bottom, right, top in normalized (0,1) figure coordinates

    # Save figure
    if parameters.is_save_eval_results:
        path_save_eval_fig = parameters.where_to_save + parameters.mode_name + "_eval.pdf"
        plt.savefig(path_save_eval_fig, format="pdf", bbox_inches="tight")
        print(f"All files are saved under {parameters.where_to_save}.")

    # plt.show()