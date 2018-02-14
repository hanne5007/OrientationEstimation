#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/7/18 5:47 PM 

@author: Hantian Liu
"""

import numpy as np
import math
from utils_plot import cart2sph, sph2cart, sph2plane, pixel2sph
from scipy import io
import os, cv2
from sync import sync_to_cam
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


camfolder = "./data/cam"
imufolder = "./data/imu"
viconfolder = "./data/vicon"

scale=500
canvassize=np.array([1800,3600,3])

first_still=300
last_still=200

def interpolate_points(x, y, im):
	size = np.shape(im)
	a = size[0] - 1
	b = size[1] - 1

	f_x = np.floor(x).astype(np.int64)
	c_x = np.ceil(x).astype(np.int64)
	f_x[f_x < 0] = 0
	c_x[c_x >= b] = b

	f_y = np.floor(y).astype(np.int64)
	c_y = np.ceil(y).astype(np.int64)
	f_y[f_y < 0] = 0
	c_y[c_y >= a] = a

	f_h = y - f_y
	c_h = 1 - f_h
	f_w = x - f_x
	c_w = 1 - f_w

	interp_val = im[f_x, f_y] * c_h * c_w + im[c_x, f_y] * c_h * f_w + \
				 im[f_x, c_y] * f_h * c_w + im[c_x, c_y] * f_h * f_w
	return interp_val


def warp(im, R, canvas):
	"""

	:param im: m*n*3
	:param R: 3*3
	:param canvas: M*N*3
	:return: canvas with warped im: M*N*3
	"""
	h,w,unused1=np.shape(im)
	r=np.arange(h)
	c=np.arange(w)
	py, pz = np.meshgrid(c, r)
	py=py.flatten()
	pz=pz.flatten()
	az, al, r=pixel2sph(py, pz)
	x, y, z=sph2cart(az, al, r)
	pts_num=len(x)
	pts=np.zeros([3, pts_num])
	pts[0,:]=x[np.newaxis,:]
	pts[1, :] = y[np.newaxis, :]
	pts[2, :] = z[np.newaxis, :]

	pts_rot=np.dot(R,pts)
	x_rot=pts_rot[0,:].flatten()
	y_rot=pts_rot[1,:].flatten()
	z_rot=pts_rot[2,:].flatten()
	theta_rot, phi_rot, r_rot=cart2sph(x_rot, y_rot, z_rot)
	xnew, ynew = sph2plane(theta_rot, phi_rot, r_rot)

	xpos=xnew*scale+canvassize[0]/2
	ypos=ynew*scale+canvassize[1]/2
	xpos=xpos.astype('int64')
	ypos=ypos.astype('int64')
	canvas[xpos, ypos, 0] = im[pz, py, 0]
	canvas[xpos, ypos, 1] = im[pz, py, 1]
	canvas[xpos, ypos, 2] = im[pz, py, 2]
	return canvas


if __name__ == '__main__':
	panorama=np.zeros(canvassize)

	#cv2.namedWindow('test', cv2.WINDOW_NORMAL)
	#cv2.imshow('test', panorama.astype(np.uint8))
	#cv2.waitKey(3)
	total=len(os.listdir(imufolder))
	total=8
	for i in range(8, total + 1):
		dataname=os.path.join(camfolder, "cam" + str(i) + '.mat')
		if not os.path.isfile(dataname):
			continue

		camdata = io.loadmat(dataname)
		cam_ts = camdata['ts']
		cam_ts = cam_ts[0, :]
		cam = camdata['cam']

		vicon = io.loadmat(os.path.join(viconfolder, "viconRot" + str(i) + '.mat'))
		gt_ts = vicon['ts']
		gt_ts = gt_ts[0, :]
		gt = vicon['rots']

		imu = io.loadmat(os.path.join(imufolder, "imuRaw" + str(i) + '.mat'))
		imu_ts = imu['ts']
		imu_ts = imu_ts[0, :]
		imuraw = imu['vals']

		gt_new=sync_to_cam(gt_ts, cam_ts, gt)

		videoname='Video'+'Vicon'+str(i)+'.mp4'
		video = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), \
								15, (canvassize[1], canvassize[0]))
		cv2.namedWindow(videoname, cv2.WINDOW_NORMAL)
		frame_num=len(cam_ts)
		end_frame=frame_num-last_still
		start_frame=first_still
		for frame in range(start_frame, end_frame):
			print('adding frame '+str(frame))
			panorama=warp(cam[:,:,:,frame], gt_new[:,:,frame], panorama)
			panorama=panorama.astype('uint8')

			#panorama=np.fliplr(panorama)
			cv2.imshow(videoname, panorama)
			key = cv2.waitKey(100)
			video.write(panorama)
		video.release()
		cv2.destroyAllWindows()
		#plt.imshow(panorama)
		#plt.show()


