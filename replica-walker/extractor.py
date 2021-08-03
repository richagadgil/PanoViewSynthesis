from typing import List, Tuple 

import numpy as np
from numpy import  float32, ndarray
from quaternion import quaternion

from habitat_sim import registry as registry
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from habitat_sim.utils.data import ImageExtractor, PoseExtractor
# import math
# import pdb

@registry.register_pose_extractor(name="closest_point_extractor_3d")
class ClosestPointExtractor3d(PoseExtractor):
    def __init__(self, 
    topdown_views: List[Tuple[str, str, Tuple[float32, float32, float32]]], 
    meters_per_pixel: float = 0.1
    ) -> None:
        super().__init__(topdown_views, meters_per_pixel)

    def extract_poses(
        self, view: ndarray, fp: str
    ) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]]:
        # Determine the physical spacing between each camera position
        height, width = view.shape
        dist = min(height, width) // 10  # We can modify this to be user-defined later

        # Create a grid of camera positions
        n_gridpoints_width, n_gridpoints_height = (
            width // dist - 1,
            height // dist - 1,
        )

        # Exclude camera positions at invalid positions
        gridpoints = []
        
        thresh = 0.8
        reduced_view = gaussian_filter(view.astype(float32), sigma=2)
        
        reduced_view[reduced_view >= thresh] = 1
        reduced_view[reduced_view < thresh] = 0
        reduced_view = reduced_view.astype(int)
        # Use these to compare difference
        # plt.imsave("./view.png", view)
        # plt.imsave("./reduced.png", reduced_view)
        for h in range(n_gridpoints_height):
            for w in range(n_gridpoints_width):
                point = (dist + h * dist, 0, dist + w * dist)
                if self._valid_point(*point, reduced_view):
                    gridpoints.append(point)

        # Generate a pose for each side of the cubemap
        # eye = np.array([15,10,0]).astype(int)
        # lap = eye + np.array([0,10,0]).astype(int)
        poses = []
        # [(tuple(eye), tuple(lap), fp), (eye, lap, fp)]
        # TODO: Vary the y position by some value
        for point in gridpoints:
            points_of_interest = self._panorama_extraction(point, view, dist)
            poses.extend([(point, poi, fp) for poi in points_of_interest])
        # Returns poses in 3D cartesian coordinate system
        return poses

    def _convert_to_scene_coordinate_system(
        self,
        poses: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]],
        ref_point: Tuple[float32, float32, float32],
    ) -> List[Tuple[Tuple[int, int], quaternion, str]]:
        # Convert from topdown map coordinate system to that of the scene
        start_point = np.array(ref_point)

        for i, pose in enumerate(poses):
            pos, look_at_point, filepath = pose

            new_pos = start_point + np.array(pos) * self.meters_per_pixel
            new_lap = start_point + np.array(look_at_point) * self.meters_per_pixel
            cam_normal = new_lap - new_pos
            new_rot = self._compute_quat(cam_normal)
            new_pos_t: Tuple[int, int, int] = tuple(new_pos)
            poses[i] = (new_pos_t, new_rot, filepath)

        return poses

    def _panorama_extraction(
        self, point: Tuple[int, int, int], view: ndarray, dist: int
    ) -> List[Tuple[int, int]]:
        in_bounds_of_topdown_view = lambda row, col: 0 <= row < len(
            view
        ) and 0 <= col < len(view[0])
        neighbor_dist = dist // 2
        # TODO: reorganize these points to just take cube faces
        neighbors = [
            (point[0] - neighbor_dist, point[1], point[2]),
            (point[0] + neighbor_dist, point[1], point[2]),
            (point[0], point[1] - neighbor_dist, point[2]),
            (point[0], point[1] + neighbor_dist, point[2]),
            (point[0], point[1], point[2] - neighbor_dist ),
            (point[0], point[1], point[2] + neighbor_dist)
        ]

        return neighbors

def lookAt(eye, center, up):
    F = center - eye
    f = normalize(F)
    s = np.cross(f, normalize(up))
    u = np.cross(normalize(s), f)
    M = np.eye(4)
    M[0, 0:3] = s
    M[1, 0:3] = u
    M[2, 0:3] = -f

    T = np.eye(4)
    T[0:3, 3] = -eye
    return M@T

def normalize(vec):
    return vec / np.linalg.norm(vec)


# For viewing the extractor output
def display_sample(sample):
    img = sample["rgba"]
    depth = sample["depth"]
    semantic = sample["semantic"]

    arr = [img, depth, semantic]
    titles = ["rgba", "depth", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()


scene_filepath = "scenes/apartment_1.glb"

extractor = ImageExtractor(
    scene_filepath,
    img_size=(512, 512),
    output=["rgba", "depth", "semantic"],
    pose_extractor_name="closest_point_extractor_3d"

)

# Use the list of train outputs instead of the default, which is the full list
# of outputs (test + train)
extractor.set_mode('train')

# Index in to the extractor like a normal python list
sample = extractor[0]
plt.imsave(f"./reference/face.png", sample["rgba"])

# Or use slicing
# samples = extractor[1:4]
# for sample in samples:
#     display_sample(sample)
# cubemap_i = 0
# cubemap_i = min(cubemap_i, len(extractor )//6)
# start_i =cubemap_i * 6
# for i ,sample in enumerate(extractor[start_i:start_i + 6]):
#     plt.imsave(f"./reference/face_{i}.png", sample["rgba"])

# Close the extractor so we can instantiate another one later
# (see close method for detailed explanation)
extractor.close()