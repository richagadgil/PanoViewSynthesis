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
import tensorflow as tf

import argparse

def compute_sigma(pred_disp,points):
    height, width = pred_disp.shape
    
    sigmas = []

    for i in range(0, width, height):
        square_pd = pred_disp[:, i:i+height]
        square_ad = points[:, i:i+height]

        pd = tf.math.log(square_pd)
        p_depths = tf.where(~tf.math.is_finite(pd), tf.zeros_like(pd), pd)

        ad = tf.math.log(tf.linalg.inv(square_ad))
        a_depths = tf.where(~tf.math.is_finite(ad), tf.zeros_like(ad), ad)

        s = tf.math.exp(tf.reduce_mean(p_depths + a_depths))
        sigmas.append(s)

    print(sigmas)
    print(tf.reduce_mean(sigmas))
    print()
    return tf.reduce_mean(sigmas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cylinder_path')
    parser.add_argument('test_path')
    args = parser.parse_args()

    width = 1280
    height = 1280
    fovy = 90

    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_3_2_CORE_PROFILE)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow('window')
    glutHideWindow(window)

    min_depth = 1.
    max_depth = 100.
    depths = 1./np.linspace(1./max_depth, 1./min_depth,
                            32, endpoint=True)
    print(depths)

    regexr = '\/(\w+)\/(\w+).png'
    folder, image_name = re.findall(regexr, args.test_path)[0]
    target_json_path = os.path.join(args.test_path.split(
        folder)[0], f'{folder}_json', image_name.split('_frame')[0] + '.json')

    with open(target_json_path) as f:
        target_json = json.load(f)

    # Scale Depths 
    predicted_depth = imread(os.path.join(args.cylinder_path, 'predicted_depth.png')).astype('float32')[:,:,0]
    print(predicted_depth)
    actual_depth = imread(os.path.join(args.cylinder_path, 'actual_depth.png')).astype('float32')
    print(actual_depth)
    #sigma = compute_sigma(predicted_depth, actual_depth) 
    #print(sigma.numpy())
    #depths = [i*sigma.numpy() for i in depths]
    print(depths)
    meshes = [Sphere(radius=depth, width_segments=200, height_segments=200,
                         texturepath=os.path.join(args.cylinder_path, 'layers/layer_%d.png' % i)) for i, depth in enumerate(depths)]

    renderer = Renderer(meshes, width=width, height=height,
                        offscreen=True)

    eye = np.array([target_json["eye"]]) * 0.5
    target = np.array(target_json["target"])
    up = np.array(target_json["up"])

    view_matrix = lookAt(eye, target, up)
    proj_matrix = perspective(fovy, width/height, 0.1, 1000.0)
    mvp_matrix = proj_matrix@view_matrix

    produced_image = renderer.render(mvp_matrix)
    imwrite('no_sigma_produced_frame.png', produced_image)

    target_image = imread(args.test_path)

    # Calculate MSE HERE
    mse = ((produced_image - target_image)**2).mean(axis=None)
    print('MSE: ', mse)
    sys.exit(0)


# python evaluation.py cube_dir/room_0/ cube_dir/room_0/00/cube_images/cube_00_01_frame.png
