# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Features:
# x Depth sensor
# - Generate JSON
# - Random Images?
# - Grid search
# - Reformat into class


import numpy as np
import matplotlib.pyplot as plt

import habitat_sim
from habitat_sim.utils.data.pose_extractor import TopdownView


# Helper function to render observations from the stereo agent
def render(sim):
    obs = sim.step("turn_right")
    plt.imsave("./pano.png", obs["color_sensor"])
    plt.imsave("./depth.png", obs["depth_sensor"])


def main():

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = (
        "./scenes/apartment_1.glb"
    )

    color_sensor = habitat_sim.sensor.EquirectangularSensorSpec()
    color_sensor.uuid = "color_sensor"
    color_sensor.resolution = [512, 1024]

    depth_sensor = habitat_sim.sensor.EquirectangularSensorSpec()
    depth_sensor.uuid = "depth_sensor"
    depth_sensor.resolution = [512, 1024]
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [color_sensor, depth_sensor]

    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(backend_cfg, [agent_config]))
    height = 1.5 * habitat_sim.geo.UP
    view = TopdownView(sim, 1.5).topdown_view
    color_sensor.position = height

    render(sim)
    sim.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    args = parser.parse_args()
    main()
