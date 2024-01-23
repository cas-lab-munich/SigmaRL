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

def evaluate_outputs(out_td: TensorDict, parameters: Parameters, agent_width: float = 0.1, ref_paths: torch.Tensor = None):
    """This function evaluates the test results presented as a tensordict."""    

    path_eval_out_td = parameters.where_to_save + parameters.mode_name + "_out_td.pth"        
    path_ref_path = parameters.where_to_save + parameters.mode_name + "_ref_path.pth"        
    if parameters.is_save_eval_results:
        # Save the input TensorDict
        torch.save(out_td, path_eval_out_td)
        torch.save(ref_paths, path_ref_path)
    env_index = 0
    
    pos_traj = out_td.get(("agents","info","pos"))[env_index].squeeze(1)
    vel_traj = out_td.get(("agents","info","vel"))[env_index].squeeze(1)
    
    deviation_from_ref_path = out_td.get(("agents","info","deviation_from_ref_path"))[env_index].squeeze((1,2))
    
    deviation_mean_relative = deviation_from_ref_path / agent_width
    deviation_from_ref_path_mean = deviation_from_ref_path.mean()
    print(f"Mean deviation={deviation_from_ref_path_mean} m.")

    # Adjust parameters for plotting for different scenarios 
    if "path_tracking" in parameters.scenario_name:
        subgraph_width_ratio = [2, 1]
        # Color bar
        cb_left_margin = 0.1
        cb_bottom_margin = 0.1
        cb_width = subgraph_width_ratio[0] / sum(subgraph_width_ratio) - 2 * cb_left_margin
        cb_height = 0.02
        if "line" in parameters.path_tracking_type:
            figsize=(12, 4)
            cb_bottom_margin = 0.3
            cb_position_shape = [cb_left_margin, cb_bottom_margin, cb_width, cb_height]
        elif "sine" in parameters.path_tracking_type:
            figsize=(12, 6)
            cb_left_margin = 0.27
            cb_bottom_margin = 0.82
            cb_width = 0.35
            cb_position_shape = [cb_left_margin, cb_bottom_margin, cb_width, cb_height]
        else:
            figsize=(12, 6) # Default
    else:
        subgraph_width_ratio = [2, 1] # Default
        
    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': subgraph_width_ratio})
        

    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 1], w_pad=2) # left, bottom, right, top in normalized (0,1) figure coordinates

    # First subplot for position-velocity plot
    segments = torch.stack([pos_traj[:-1], pos_traj[1:]], dim=1)
    velocity_magnitude = vel_traj.norm(dim=1)
    print(f"Mean velocity={velocity_magnitude.mean()} m/s.")
    
    
    norm = Normalize(vmin=0, vmax=velocity_magnitude.max())
    lc = LineCollection(segments, cmap='Greys', norm=norm)
    lc.set_array(velocity_magnitude)
    lc.set_linewidth(2)
    ax1.add_collection(lc)
    

    ax1.autoscale()
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel(r'$x$ [m]', fontsize=12)
    ax1.set_ylabel(r'$y$ [m]', fontsize=12)
    ax1.set_title("Position with velocity color map", fontsize=14)
    ax1.grid(True)
    if ref_paths is not None:
        ref_path_line, = ax1.plot(ref_paths[0,:,0], ref_paths[0,:,1], "b--", label="Reference path")

    # Creating the legend
    ax1.legend(handles=[lc, ref_path_line], labels=["Actual trajectory", "Reference path"], loc="upper right")
    # ax1.legend(handles=[lc, rp], labels=["Actual trajectory", "Reference path"], loc="upper right")

    # Color bar for the first subplot    
    cbar_ax = fig.add_axes(cb_position_shape)  # (left, bottom, width, height)
    cbar = plt.colorbar(lc, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$v$ [m/s]', fontsize=9)

    is_violinplot = True
    if is_violinplot:
        # Create a violin plot
        violinplot = ax2.violinplot(deviation_mean_relative, showmeans=False, showmedians=False, showextrema=False)
        for pc in violinplot['bodies']:
            pc.set_facecolor('grey')
            pc.set_edgecolor('black')

        # Create a box plot inside the violin plot
        boxplot = ax2.boxplot(deviation_mean_relative, patch_artist=True, notch=True, medianprops=dict(color="black"))

        # Customization of the box plot to match the violin plot
        for patch in boxplot['boxes']:
            patch.set_facecolor('white')

    else:
        # Second subplot for deviation box plot
        boxplot = ax2.boxplot(deviation_mean_relative, patch_artist=True)
        for median in boxplot['medians']:
            median.set_color('black')

    # ax2.set_title("Deviation from Reference Path", fontsize=14)
    ax2.set_ylabel("Deviation (relative to agent width)", fontsize=12)
    
    # Round up to the first decimal place
    deviation_mean_relative_max_round_up = torch.ceil(deviation_mean_relative.max() * 10) / 10

    
    ax2.set_ylim([0, deviation_mean_relative_max_round_up])
    # Turn off xticks 
    ax2.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    # Customization of the box plot to match the violin plot
    for patch in boxplot['boxes']:
        patch.set_facecolor('white')

    # Save figure
    if parameters.is_save_eval_results:
        path_save_eval_fig = parameters.where_to_save + parameters.mode_name + "_eval.pdf"
        plt.savefig(path_save_eval_fig, format="pdf", bbox_inches="tight")
        print(f"All files are saved under {parameters.where_to_save}.")

    # plt.show()