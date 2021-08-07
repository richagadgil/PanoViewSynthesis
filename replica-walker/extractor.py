from typing import List, Tuple

import numpy as np
from numpy import float32, ndarray
from quaternion import quaternion
import quaternion as qt

from habitat_sim import registry as registry
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

from habitat_sim.utils.data import ImageExtractor, PoseExtractor
# import math
import pdb


@registry.register_pose_extractor(name="cube_map_extractor")
class CubeMapExtractor(PoseExtractor):
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
        # We can modify this to be user-defined later
        dist = min(height, width) // 10
        cam_height = 3
        # Create a grid of camera positions
        n_gridpoints_width, n_gridpoints_height = (
            width // dist - 1,
            height // dist - 1,
        )

        gridpoints = []

        reduced_view = binary_erosion(view, iterations=3)
        # Use these to compare difference between original and eroded maps
        # plt.imsave("./view.png", view)
        # plt.imsave("./reduced.png", reduced_view)
        for h in range(n_gridpoints_height):
            for w in range(n_gridpoints_width):
                point = (dist + h * dist, dist + w * dist)
                if self._valid_point(*point, reduced_view):
                    gridpoints.append(point)
        # Generate a pose for each side of the cubemap
        # eye = np.array([15,10,0]).astype(int)
        # lap = eye + np.array([0,10,0]).astype(int)
        poses = []
        # [(tuple(eye), tuple(lap), fp), (eye, lap, fp)]
        for row, col in gridpoints:
            position = (col, cam_height, row)
            points_of_interest = self._panorama_extraction(
                position, view, dist)
            poses.extend([(position, poi, fp) for poi in points_of_interest])
        # Returns poses in 3D cartesian coordinate system
        return poses

    def _convert_to_scene_coordinate_system(
        self,
        poses: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]],
        ref_point: Tuple[float32, float32, float32],
    ) -> List[Tuple[Tuple[int, int], quaternion, str]]:
        # Convert from topdown map coordinate system to that of the scene
        start_point = np.array(ref_point)
        converted_poses = []
        left = Rotation.from_euler("y", -89, degrees=True)
        right = Rotation.from_euler("y", 89, degrees=True)
        back = Rotation.from_euler("y", 179, degrees=True)
        up = Rotation.from_euler("x", 89, degrees=True)
        down = Rotation.from_euler("x", -89, degrees=True)
        front = Rotation.from_euler("x", 0, degrees=True)
        rots = [left, right, up, down, back, front]
        rots = [qt.from_rotation_matrix(
            rot.as_matrix()) for rot in rots]
        for i, pose in enumerate(poses):
            pos, look_at_point, filepath = pose

            new_pos = start_point + np.array(pos) * self.meters_per_pixel
            new_lap = start_point + \
                np.array(look_at_point) * self.meters_per_pixel
            displacement = new_lap - new_pos

            rot = lookAt(np.array([0, 0, 0]),
                         displacement, np.array([0, 1, 0]))
            print(Rotation.from_matrix(rot[:3, :3]).as_euler(
                seq="xyz", degrees=True))
            rot = rots[i % len(rots)]
            # print(new_pos)
            converted_poses.append(
                (new_pos, rot, filepath))

        return converted_poses

    def _panorama_extraction(
        self, point: Tuple[int, int, int], view: ndarray, dist: int
    ) -> List[Tuple[int, int]]:
        neighbor_dist = 1
        # TODO: reorganize these points to just take cube faces
        neighbors = [
            (point[0] - neighbor_dist, point[1], point[2]),  # left
            (point[0] + neighbor_dist, point[1], point[2]),  # right
            (point[0], point[1] - neighbor_dist, point[2]),  # down
            (point[0], point[1] + neighbor_dist, point[2]),  # up
            (point[0], point[1], point[2] - neighbor_dist),  # backward
            (point[0], point[1], point[2] + neighbor_dist)  # forward
        ]

        return neighbors


def lookAt(eye, center, up):
    F = center - eye
    f = normalize(F)
    if abs(f[1]) > 0.99:
        f = up * np.sign(f[1])
        u = np.array([0, 0, 1])
        s = np.cross(f, u)
    else:
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
    output=["rgba"],
    pose_extractor_name="cube_map_extractor",
    shuffle=False

)

# Use the list of train outputs instead of the default, which is the full list
# of outputs (test + train)
# extractor.set_mode('train')

# Index in to the extractor like a normal python list
# sample = extractor[0]
# plt.imsave(f"./reference/face.png", sample["rgba"])

# Or use slicing
# samples = extractor[1:4]
# for sample in samples:
#     display_sample(sample)
cubemap_i = 0
# cubemap_i = min(cubemap_i, len(extractor )//6)
start_i = cubemap_i * 6
for i, sample in enumerate(extractor[0:6]):
    plt.imsave(f"./tmp/face_{i}.png", sample["rgba"])

extractor.close()
