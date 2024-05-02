#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import typing
from typing import List

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Box
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation

from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

import os
import sys

# from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
# !Important: Add project root to system path if you want to run this file directly
script_dir = os.path.dirname(__file__) # Directory of the current script
project_root = os.path.abspath(os.path.join(script_dir, '..', '..')) # Project root directory
if project_root not in sys.path:
    sys.path.append(project_root)

from utilities.kinematic_bicycle import KinematicBicycle

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """
        Kinematic bicycle model example scenario
        """
        self.n_agents = kwargs.get("n_agents", 2)
        width = kwargs.get("width", 0.1)  # Agent width
        l_f = kwargs.get(
            "l_f", 0.1
        )  # Distance between the front axle and the center of gravity
        l_r = kwargs.get(
            "l_r", 0.1
        )  # Distance between the rear axle and the center of gravity
        max_steering_angle = kwargs.get(
            "max_steering_angle", torch.tensor(35.0, device=device, dtype=torch.float32).deg2rad()
        )
        max_speed = kwargs.get(
            "max_speed", torch.tensor(0.5, device=device, dtype=torch.float32)
        )

        # Make world
        world = World(batch_dim, device, substeps=10, collision_force=500)

        for i in range(self.n_agents):
            if i == 0:
                # Use the kinematic bicycle model for the first agent
                agent = Agent(
                    name=f"bicycle_{i}",
                    shape=Box(length=l_f + l_r, width=width),
                    collide=True,
                    render_action=True,
                    max_speed=max_speed,
                    u_range=[1, max_steering_angle],
                    u_multiplier=[1, 1],
                    dynamics=KinematicBicycle(
                        world,
                        width=width,
                        l_f=l_f,
                        l_r=l_r,
                        max_steering_angle=max_steering_angle,
                        integration="euler",  # one of "euler", "rk4"
                    ),
                )
            else:
                agent = Agent(
                    name=f"holo_rot_{i}",
                    shape=Box(length=l_f + l_r, width=width),
                    collide=True,
                    render_action=True,
                    u_range=[1, 1, 1],
                    u_multiplier=[1, 1, 0.001],
                    dynamics=HolonomicWithRotation(),
                )

            world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=0.1,
            x_bounds=(-1, 1),
            y_bounds=(-1, 1),
        )

    def reward(self, agent: Agent):
        return torch.zeros(self.world.batch_dim)

    def observation(self, agent: Agent):
        observations = [
            agent.state.pos,
            
            agent.state.vel,
        ]
        return torch.cat(
            observations,
            dim=-1,
        )

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Agent rotation
        for agent in self.world.agents:
            color = Color.BLACK.value
            line = rendering.Line(
                (0, 0),
                (0.1, 0),
                width=1,
            )
            xform = rendering.Transform()
            xform.set_rotation(agent.state.rot[env_index])
            xform.set_translation(*agent.state.pos[env_index])
            line.add_attr(xform)
            line.set_color(*color)
            geoms.append(line)

        return geoms


# ... and the code to run the simulation.
if __name__ == "__main__":
    scenario = Scenario()
    render_interactively(
        scenario,
        control_two_agents=True,
        width=0.1,
        l_f=0.1,
        l_r=0.1,
        display_info=True,
    )
