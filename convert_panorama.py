
import numpy as np
from PIL import Image
import py360convert
import os
from os.path import join



def convert_cubemaps_to_equirectangular():
    """
    Converts the depth and frame cubemaps located in the cube_map directory
    in each scene of cube_dir to equirectangular images.
    """
    OUTPUT_SIZE = (5120, 2560) # width, height
    scenes = [d for d in os.listdir("cube_dir") if os.path.isdir(join("cube_dir", d))]
    print("")
    for i, scene in enumerate(scenes):
        print(f"\rConverting scene {i + 1}/{len(scenes)} - {scene}", end="")
        data_path = join("cube_dir", scene, "00")
        spherical_panorama_dir = join(data_path, "spherical_panorama")
        # Ensure spherical_panorama directory exists
        if not os.path.exists(spherical_panorama_dir):
            os.mkdir(spherical_panorama_dir)
        # Convert depth and frame maps to equirectangular
        for cube_map in ["cube_00_depth.png", "cube_00_frame.png"]: 
            with Image.open(join(data_path, "cube_map", cube_map)) as im:
                output = py360convert.c2e(np.asarray(im), OUTPUT_SIZE[1], OUTPUT_SIZE[0])
                spherical_image = Image.fromarray(output.astype(np.uint8))
                spherical_image_name = cube_map.replace("cube", "sphere")
                spherical_image.save(join(spherical_panorama_dir, spherical_image_name))
                spherical_image.close()


if __name__ == "__main__":
    convert_cubemaps_to_equirectangular()
