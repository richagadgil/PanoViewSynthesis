# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import matplotlib.pyplot as plt

import habitat_sim


# Helper function to render observations from the stereo agent
def render(sim):
    obs = sim.step("turn_right")
    img = obs["equirectangular_sensor"]
    plt.imsave("./pano.png", img)


def main():

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = (
        "./scenes/skokloster-castle.glb"
    )

    equirectangular_sensor = habitat_sim.sensor.EquirectangularSensorSpec()
    equirectangular_sensor.uuid = "equirectangular_sensor"
    equirectangular_sensor.resolution = [512, 1024]
    equirectangular_sensor.position = 1.5 * habitat_sim.geo.UP

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [equirectangular_sensor]

    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(backend_cfg, [agent_config]))

    render(sim)
    sim.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    args = parser.parse_args()
    main()
