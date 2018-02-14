#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/7/18 8:04 PM 

@author: Hantian Liu
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from UKF import UKF, init
from fromRaw import getAcc, getAngularVelocity
from utils import quart2mat
from panorama import warp
from noFilter import rotationMatrixToEulerAngles, onlyAcc, onlyGyro
import numpy as np
from scipy import io
import os, transforms3d, cv2, math
from sync import sync_to_cam, sync_gt


############################
## MODIFY THESE VARIABLES ##
############################
camfolder = "./data/cam"
imufolder = "./data/imu"
viconfolder = "./data/vicon"
max_dataset_num=13

vicon_exists=False

show_rpy_plots=False
show_imu_only_rpy_plots=True

show_panorama=True
show_panorama_from_vicon=True

first_still=0
last_still=0
############################


#pixel scale up for panorama
scale=500
#final panorama canvas size
canvassize=np.array([1800,3600,3])
# tolerance for the magnitude of valid acc measurements
epsilon=0.008

def showAllResults():
	#total = #len(os.listdir(imufolder))
	for datanum in range(11, max_dataset_num + 1):
		print('Data number '+str(datanum))
		imuname=os.path.join(imufolder, "imuRaw" + str(datanum) + '.mat')
		if not os.path.isfile(imuname):
			continue
		imu = io.loadmat(imuname)
		imu_ts = imu['ts']
		imu_ts = imu_ts[0, :]
		imuraw = imu['vals']

		if vicon_exists:
			vicon = io.loadmat(os.path.join(viconfolder, "viconRot" + str(datanum) + '.mat'))
			gt_ts = vicon['ts']
			gt_ts = gt_ts[0, :]
			gt = vicon['rots']
			gt_synced, imu_synced, imu_ts = sync_gt(gt_ts, imu_ts, gt, imuraw)
			r_gt = []
			p_gt = []
			y_gt = []
		else:
			imu_synced=imuraw

		print('converting raw data to values')
		ox, oy, oz = getAngularVelocity(imu_synced)
		ax, ay, az = getAcc(imu_synced)
		R=[]
		r_my = []
		p_my = []
		y_my = []

		r_a = []
		p_a = []
		y_a = []
		r_g = []
		p_g = []
		y_g = []
		q0 = np.zeros([4, 1])
		mu, sigma = init()
		print('running UKF')
		for i in range(len(imu_ts)):
			if abs(ax[i] ** 2 + ay[i] ** 2 + az[i] ** 2 - 1) > epsilon:
				acc_valid = False
			else:
				acc_valid = True

			mu, sigma, q_ukf = UKF(ox, oy, oz, imu_ts, i, mu, sigma, acc_valid, ax, ay, az)
			my_mat = quart2mat(q_ukf)
			R.append(my_mat)
			rr, pp, yy = transforms3d.euler.mat2euler(my_mat, axes = 'szyx')
			r_my.append(rr)
			p_my.append(pp)
			y_my.append(yy)
			if vicon_exists:
				r, p, y = transforms3d.euler.mat2euler(gt_synced[:, :, i], axes = 'szyx')
				r_gt.append(r)
				p_gt.append(p)
				y_gt.append(y)

			if show_imu_only_rpy_plots:
				rrr, ppp, yyy, q0 = onlyGyro(ox, oy, oz, imu_ts, i, q0)
				r_g.append(rrr)
				p_g.append(ppp)
				y_g.append(yyy)
				rrrr, pppp, yyyy = onlyAcc(ax, ay, az, i)
				r_a.append(rrrr)
				p_a.append(pppp)
				y_a.append(yyyy)

		if show_rpy_plots:
			if vicon_exists:
				r_gt_mat = np.asarray(r_gt)
				p_gt_mat = np.asarray(p_gt)
				y_gt_mat = np.asarray(y_gt)
				r_my_mat = np.asarray(r_my)
				p_my_mat = np.asarray(p_my)
				y_my_mat = np.asarray(y_my)
				fig = plt.figure()
				ax1 = fig.add_subplot(311)
				ax1.set_ylabel('yaw angle')
				ax1.plot(imu_ts, r_gt_mat, 'r', label = 'Vicon')
				ax1.plot(imu_ts, r_my_mat, 'g', label = 'UKF')
				ax2 = fig.add_subplot(312)
				ax2.set_ylabel('pitch angle')
				ax2.plot(imu_ts, p_gt_mat, 'r', label = 'Vicon')
				ax2.plot(imu_ts, p_my_mat, 'g', label = 'UKF')
				ax3 = fig.add_subplot(313)
				ax3.set_ylabel('roll angle')
				ax3.set_xlabel('time stamp')
				ax3.plot(imu_ts, y_gt_mat, 'r', label = 'Vicon')
				ax3.plot(imu_ts, y_my_mat, 'g', label = 'UKF')

				if show_imu_only_rpy_plots:
					r_g_mat = np.asarray(r_g)
					p_g_mat = np.asarray(p_g)
					y_g_mat = np.asarray(y_g)
					r_a_mat = np.asarray(r_a)
					p_a_mat = np.asarray(p_a)
					y_a_mat = np.asarray(y_a)
					ax1.plot(imu_ts, r_g_mat, 'b', label = 'Gyro')
					ax1.plot(imu_ts, r_a_mat, 'c', label = 'Acc')
					ax2.plot(imu_ts, p_g_mat, 'b', label = 'Gyro')
					ax2.plot(imu_ts, p_a_mat, 'c', label = 'Acc')
					ax3.plot(imu_ts, y_g_mat, 'b', label = 'Gyro')
					ax3.plot(imu_ts, y_a_mat, 'c', label = 'Acc')
					ax1.legend(loc = 'upper right')
					ax2.legend(loc = 'upper right')
					ax3.legend(loc = 'upper right')
					fig.suptitle('Euler angles for dataset no.' + str(datanum))
					plt.show()
				else:
					ax1.legend(loc = 'upper right')
					ax2.legend(loc = 'upper right')
					ax3.legend(loc = 'upper right')
					fig.suptitle('Euler angles for dataset no.' + str(datanum))
					plt.show()
			else:
				r_my_mat = np.asarray(r_my)
				p_my_mat = np.asarray(p_my)
				y_my_mat = np.asarray(y_my)
				fig = plt.figure()
				ax1 = fig.add_subplot(311)
				ax1.set_ylabel('yaw angle')
				ax1.plot(imu_ts, r_my_mat, 'g', label = 'UKF')
				ax2 = fig.add_subplot(312)
				ax2.set_ylabel('pitch angle')
				ax2.plot(imu_ts, p_my_mat, 'g', label = 'UKF')
				ax3 = fig.add_subplot(313)
				ax3.set_ylabel('roll angle')
				ax3.set_xlabel('time stamp')
				ax3.plot(imu_ts, y_my_mat, 'g', label = 'UKF')

				if show_imu_only_rpy_plots:
					r_g_mat = np.asarray(r_g)
					p_g_mat = np.asarray(p_g)
					y_g_mat = np.asarray(y_g)
					r_a_mat = np.asarray(r_a)
					p_a_mat = np.asarray(p_a)
					y_a_mat = np.asarray(y_a)
					ax1.plot(imu_ts, r_g_mat, 'b', label = 'Gyro')
					ax1.plot(imu_ts, r_a_mat, 'c', label = 'Acc')
					ax2.plot(imu_ts, p_g_mat, 'b', label = 'Gyro')
					ax2.plot(imu_ts, p_a_mat, 'c', label = 'Acc')
					ax3.plot(imu_ts, y_g_mat, 'b', label = 'Gyro')
					ax3.plot(imu_ts, y_a_mat, 'c', label = 'Acc')
					ax1.legend(loc = 'upper right')
					ax2.legend(loc = 'upper right')
					ax3.legend(loc = 'upper right')
					fig.suptitle('Euler angles for dataset no.' + str(datanum))
					plt.show()
				else:
					ax1.legend(loc = 'upper right')
					ax2.legend(loc = 'upper right')
					ax3.legend(loc = 'upper right')
					fig.suptitle('Euler angles for dataset no.' + str(datanum))
					plt.show()

		if show_panorama:
			panorama = np.zeros(canvassize)
			pano = np.zeros(canvassize)

			camname = os.path.join(camfolder, "cam" + str(datanum) + '.mat')
			if not os.path.isfile(camname):
				continue
			camdata = io.loadmat(camname)
			cam_ts = camdata['ts']
			cam_ts = cam_ts[0, :]
			cam = camdata['cam']

			if vicon_exists and show_panorama_from_vicon:
				gt_new = sync_to_cam(gt_ts, cam_ts, gt)
				frame_num = len(cam_ts)
				print('Adding frames to panorama from vicon data')
				#fig2 = plt.figure()
				videoname = 'Video' + 'Vicon' + str(datanum) + '.mp4'
				video = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), \
										15, (canvassize[1], canvassize[0]))
				cv2.namedWindow(videoname, cv2.WINDOW_NORMAL)

				for frame in range(first_still, frame_num-last_still):
					panorama = warp(cam[:, :, :, frame], gt_new[:, :, frame], panorama)
					panorama = panorama.astype('uint8')

					cv2.imshow(videoname, panorama)
					key = cv2.waitKey(100)
					video.write(panorama)
				video.release()
				cv2.destroyAllWindows()
				'''
				ax=fig2.add_subplot(111)
				ax.imshow(panorama)
				fig2.suptitle('Panorama from vicon for dataset no.' + str(datanum))
				plt.show()
				'''

			R_len=np.shape(R)[0]
			R_reshape=np.zeros([3,3,R_len])
			for eachR in range(R_len):
				R_reshape[:,:,eachR]=R[eachR]
			R_new = sync_to_cam(imu_ts, cam_ts, R_reshape)
			frame_num = len(cam_ts)
			print('Adding frames to panorama from IMU data')
			videoname = 'Video' + 'IMU' + str(datanum) + '.mp4'
			video = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), \
									15, (canvassize[1], canvassize[0]))
			cv2.namedWindow(videoname, cv2.WINDOW_NORMAL)

			for frame in range(first_still, frame_num - last_still):
				pano = warp(cam[:, :, :, frame], R_new[:, :, frame], pano)
				pano = pano.astype('uint8')

				cv2.imshow(videoname, pano)
				key = cv2.waitKey(100)
				video.write(pano)
			video.release()
			cv2.destroyAllWindows()
			'''
			fig3 = plt.figure()
			ax = fig3.add_subplot(111)
			ax.imshow(pano)
			fig3.suptitle('Panorama from IMU for dataset no.' + str(datanum))
			plt.show()			
			'''


if __name__ == '__main__':
	showAllResults()