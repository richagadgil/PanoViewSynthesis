from OpenGL.GL import *
from OpenGL.GLUT import *
import time

import numpy as np
import matplotlib.pyplot as plt
import os
import re

from rendertools import *

from imageio import imwrite

import json

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cylinder_path')
    parser.add_argument('test_path')
    args = parser.parse_args()

    width = 640
    height = 480
    fovy = 90

    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_3_2_CORE_PROFILE)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow('window')
    glutHideWindow(window)

    min_depth = 1.
    max_depth = 100.
    depths = 1./np.linspace(1./max_depth, 1./min_depth, 32, endpoint=True)

    regexr = '\/(\w+)\/(\w+).png'
    folder, image_name = re.findall(regexr, args.test_path)[0]
    target_json_path = os.path.join(args.test_path.split(
        folder)[0], f'{folder}_json', image_name.split('_frame')[0] + '.json')

    with open(target_json_path) as f:
        target_json = json.load(f)

    meshes = [Cylinder(bottom=-1*depth, top=1*depth, radius=depth,
                       texturepath=args.cylinder_path) for i, depth in enumerate(depths)]

    renderer = Renderer(meshes, width=width, height=height,
                        offscreen=True)

    eye = np.array(target_json["eye"])
    target = np.array(target_json["target"])
    up = np.array(target_json["up"])

    view_matrix = lookAt(eye, target, up)
    proj_matrix = perspective(fovy, width/height, 0.1, 1000.0)
    mvp_matrix = proj_matrix@view_matrix

    image = renderer.render(mvp_matrix)
    imwrite('rendered_frame.png', image)

    sys.exit(0)
