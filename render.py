from OpenGL.GL import *
from OpenGL.GLUT import *
import time

import numpy as np
import matplotlib.pyplot as plt
import os

from rendertools import *

from imageio import imwrite

import cv2
from scipy.spatial.transform import Rotation

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--mode',default='cylindrical')
    parser.add_argument('--offscreen',action='store_true')
    parser.add_argument('--wobble',action='store_true')
    parser.add_argument('--render',default='mpi') # mpi, depth, plain, plaindisp
    args = parser.parse_args()
    
    width = 1920
    height = 1080 
    fovy = 90
        
    scale = 3.0
    render_radius = 0.1
    wobble = 1/3
    wobble_rate = 10

    glutInit()
    glutInitDisplayMode(GLUT_RGBA|GLUT_3_2_CORE_PROFILE|GLUT_DOUBLE)
    glutInitWindowSize(width,height)
    glutInitWindowPosition(0,0)
    window = glutCreateWindow('window')
    if args.offscreen:
        glutHideWindow(window)
    
    min_depth = 1.
    max_depth = 100.
    depths = 1./np.linspace(1./max_depth,1./min_depth,32,endpoint=True)

    #min_depth = 2.
    #max_depth = 6.
    #depths = 1./np.linspace(1./max_depth,1./min_depth,192,endpoint=True)
    #min_depth = 1.1999999
    #max_depth = 6.280106
    #depths = 1./np.linspace(1./max_depth,1./min_depth,16,endpoint=True)

    # nerf
    #min_depth = 0.1
    #max_depth = 150.
    #depths = 1./np.linspace(1./(1.+max_depth),1./(1.+min_depth),32,endpoint=True)
    #depths = depths[:13]
     
    if args.render == 'depth':
        texturepath = os.path.join(args.path,'input.png')
        disparitypath = os.path.join(args.path,'disparity_map.png')
        #meshes = [Cylinder(bottom=-2.5*scale,top=2.5*scale,radius=scale,texturepath=texturepath,disparitypath=disparitypath,nvertsegments=64)]
        meshes = [DepthCylinder(height=5*scale,radius=scale,texturepath=texturepath,disparitypath=disparitypath,nsegments=360,nvertsegments=63)]
    elif args.render == 'plain' or args.render == 'plaindisp':
        if args.render == 'plain':
            texturepath = os.path.join(args.path,'input.png')
        else:
            texturepath = os.path.join(args.path,'disparity_map.png')
        meshes = [Cylinder(bottom=-2.5*1000,top=2.5*1000,radius=1000,texturepath=texturepath)]
    elif args.render == 'mpi':
        meshes = [Cylinder(bottom=-2.5*scale*depth,top=2.5*scale*depth,radius=scale*depth,texturepath=os.path.join(args.path,'layer_%d.png'%i)) for i,depth in enumerate(depths)]
        #else:
            #meshes = [Plane(depth=depth,texturepath=os.path.join(args.path,'layer%d.png'%i)) for i,depth in enumerate(depths)]
    
    #renderer = Renderer(meshes,width=width,height=height,disparity=args.disparity,offscreen=args.offscreen)
    renderer = Renderer(meshes,width=width,height=height,offscreen=args.offscreen)
    
    if args.offscreen:
        writer = cv2.VideoWriter('render.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, (width,height))
        #os.makedirs('frames',exist_ok=True)
        
        thetas = np.linspace(0,2*np.pi,1200,endpoint=False)
        #thetas = thetas[:120]
        if args.wobble:
            thetas = np.linspace(0,2*np.pi,120,endpoint=False)

        for i,t in enumerate(thetas):
            print(i,len(thetas))

            # make wobble
            #sw,cw = np.sin(t*wobble_rate),np.cos(t*wobble_rate)
            #eye = np.array([sw*wobble,0,-cw*render_radius])
            #target = np.array([0,cw*wobble,-render_radius*2])

            # rotate eye and target
            #R = Rotation.from_euler('y',-t).as_matrix()
            #eye = R@eye
            #target = R@target

            if args.wobble:
                eye = np.array([np.sin(t)*wobble,np.cos(t)*wobble,np.sin(t)*wobble])
                up = np.array([0,1,0])
                #target = eye+np.array([0,0,-1])
                target = np.array([0,0,10])
            else:
                eye = np.array([np.sin(t*wobble_rate)*wobble,0,0])
                up = np.array([0,1,0])
                target = eye+np.array([np.sin(t),0,-np.cos(t)])
        
            #eye = np.array([s*render_radius+sw*wobble,sw*wobble,-c*render_radius])
            #target = eye + np.array([s,0,-c])
            up = np.array([0,1,0])
            view_matrix = lookAt(eye,target,up)
            proj_matrix = perspective(fovy, width/height, .1, 10000.0)
            mvp_matrix = proj_matrix@view_matrix

            image = renderer.render(mvp_matrix)
            #imwrite('frames/frame%06d.png'%i,image)
            writer.write(image[:,:,::-1])

        sys.exit(0)

    time0 = time.time()
    def do_render():
        t = (time.time() - time0)/10
        #eye = np.array([np.sin(t*10)/3,0,0])*.4
        if args.mode == 'plane':
            eye = np.array([np.sin(t*10),0,0])
            up = np.array([0,1,0])
            target = eye+np.array([0,0,-1])
        else:
            if args.wobble:
                eye = np.array([np.sin(t*wobble_rate)*wobble,np.cos(t*wobble_rate)*wobble,np.sin(t*wobble_rate)*wobble])
                up = np.array([0,1,0])
                #target = eye+np.array([0,0,1])
                target = np.array([0,0,10])
            else:
                eye = np.array([np.sin(t*wobble_rate)*wobble,0,0])
                up = np.array([0,1,0])
                target = eye+np.array([np.sin(t),0,-np.cos(t)])

        view_matrix = lookAt(eye,target,up)
        proj_matrix = perspective(fovy, width/height, 0.1, 10000.0)
        mvp_matrix = proj_matrix@view_matrix

        renderer.render(mvp_matrix)

        glutSwapBuffers()
    
    def keyboard(key,x,y):
        print(key)
        ch = key.decode("utf-8")
        print(ch)
        if key == GLUT_KEY_LEFT:
            eye[0] -= .1
        elif key == GLUT_KEY_RIGHT:
            eye[0] += .1
        print(eye)

    glutDisplayFunc(do_render)
    glutIdleFunc(do_render)
    glutKeyboardFunc(keyboard)
    glutMainLoop()

