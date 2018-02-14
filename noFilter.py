#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/9/18 6:31 PM 

@author: Hantian Liu
"""

import numpy as np
from utils import vec2quart, quart2vec, qmultiply, quart2mat, vec2Skew
from fromRaw import getAcc, getAngularVelocity
from sync import sync_gt, sync_to_cam
import math, os
from scipy import io
import transforms3d
from UKF import UKF, init
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


camfolder = "./data/cam"
imufolder = "./data/imu"

vicon_exists=True
if vicon_exists:
	viconfolder = "./data/vicon"

show_rpy_plots=True
show_panorama=True
show_panorama_from_vicon=True

scale=500
canvassize=np.array([1800,3600,3])
# tolerance for the magnitude of valid acc measurements
epsilon=0.01


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
	#assert (isRotationMatrix(R))

	sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

	singular = sy < 1e-6

	if not singular:
		x = math.atan2(R[2, 1], R[2, 2])
		y = math.atan2(-R[2, 0], sy)
		z = math.atan2(R[1, 0], R[0, 0])
	else:
		x = math.atan2(-R[1, 2], R[1, 1])
		y = math.atan2(-R[2, 0], sy)
		z = 0

	return x, y, z


def onlyGyro(omgx, omgy, omgz, ts, i, qprev):
	"""

	:param omgx: n needs data at timestamp k-1
	:param omgy: n
	:param omgz: n
	:param imu_ts: n
	:param i: timestamp k
	:return: r, p, y, at timestamp k
	"""
	if i==0:
		wk = np.zeros([3, 1])
		wk[0, :] = omgx[i]
		wk[1, :] = omgy[i]
		wk[2, :] = omgz[i]
		q=vec2quart(wk)
		R = quart2mat(q)
		r, p, y = transforms3d.euler.mat2euler(R, axes = 'szyx') #rotationMatrixToEulerAngles(R)
		return r,p,y,q

	dt = ts[i] - ts[i - 1]

	wk=np.zeros([3,1])
	wk[0,:]=omgx[i-1]
	wk[1,:]=omgy[i-1]
	wk[2,:]=omgz[i-1]
	#q_curr=vec2quart(wk)

	qdelta = np.zeros([4, 1])
	angle = np.linalg.norm(wk) * dt
	if angle == 0:
		axis = np.ones([3, 1])
	else:
		axis = wk / np.linalg.norm(wk)
	qdelta[0, :] = math.cos(angle / 2)
	qdelta[1:, :] = axis * math.sin(angle / 2)

	q = qmultiply(qprev, qdelta)
	R=quart2mat(q)
	r, p, y = transforms3d.euler.mat2euler(R, axes = 'szyx')#rotationMatrixToEulerAngles(R)
	return r,p,y,q

def onlyAcc(accx, accy, accz, i):
	"""
	R: from body to world R*a=g
	:param accx: n needs data at timestamp k
	:param accy: n
	:param accz: n
	:return: r, p, y at timestamp k
	"""
	g_body=np.array([[accx[i], accy[i], accz[i]]])
	g_body=g_body/np.linalg.norm(g_body)
	g_world=np.array([[0,0,1]])
	#R*g=a from world to body

	v=np.cross(g_body, g_world)
	#v=np.cross(g_world, g_body)
	#sin = np.linalg.norm(v)
	#v=v/sin
	cos=sum(g_world.flatten()*g_body.flatten())
	#cos=cos/np.linalg.norm(g_body)/np.linalg.norm(g_world)

	vskew=vec2Skew(v)
	#R=np.identity(3)+vskew*sin+np.dot(vskew,vskew)*(1-cos)
	R = np.identity(3) + vskew + np.dot(vskew, vskew) * (1/(1+cos))
	#print(np.dot(R, R.transpose()))

	r, p, y = transforms3d.euler.mat2euler(R, axes = 'szyx')#rotationMatrixToEulerAngles(R)
	return r,p,y



if __name__ == '__main__':

	total = len(os.listdir(imufolder))
	for datanum in range(1, total + 1):
		print('Data number '+str(datanum))
		imu = io.loadmat(os.path.join(imufolder, "imuRaw" + str(datanum) + '.mat'))
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

		r_a=[]
		p_a=[]
		y_a=[]
		r_g=[]
		p_g=[]
		y_g=[]

		r_my = []
		p_my = []
		y_my = []

		mu, sigma = init()
		q0=np.zeros([4,1])
		print('running UKF')
		for i in range(len(imu_ts)):
			if abs(ax[i] ** 2 + ay[i] ** 2 + az[i] ** 2 - 1) > epsilon:
				acc_valid = False
			else:
				acc_valid = True
			mu, sigma, q_ukf = UKF(ox, oy, oz, imu_ts, i, mu, sigma, acc_valid, ax, ay, az)
			my_mat = quart2mat(q_ukf)
			R.append(my_mat)
			rr, pp, yy = rotationMatrixToEulerAngles(my_mat)#transforms3d.euler.mat2euler(my_mat, axes = 'szyx')
			r_my.append(rr)
			p_my.append(pp)
			y_my.append(yy)

			rrr, ppp, yyy, q0=onlyGyro(ox, oy, oz, imu_ts, i, q0)
			r_g.append(rrr)
			p_g.append(ppp)
			y_g.append(yyy)
			rrrr, pppp, yyyy = onlyAcc(ax, ay, az, i)
			r_a.append(rrrr)
			p_a.append(pppp)
			y_a.append(yyyy)

			if vicon_exists:
				r, p, y = rotationMatrixToEulerAngles(gt_synced[:,:,i])#transforms3d.euler.mat2euler(gt_synced[:, :, i], axes = 'szyx')
				r_gt.append(r)
				p_gt.append(p)
				y_gt.append(y)

		if show_rpy_plots:
			if vicon_exists:
				r_gt_mat = np.asarray(r_gt)
				p_gt_mat = np.asarray(p_gt)
				y_gt_mat = np.asarray(y_gt)
				r_my_mat = np.asarray(r_my)
				p_my_mat = np.asarray(p_my)
				y_my_mat = np.asarray(y_my)

				r_g_mat = np.asarray(r_g)
				p_g_mat = np.asarray(p_g)
				y_g_mat = np.asarray(y_g)
				r_a_mat = np.asarray(r_a)
				p_a_mat = np.asarray(p_a)
				y_a_mat = np.asarray(y_a)

				fig = plt.figure()
				ax1 = fig.add_subplot(311)
				ax1.set_ylabel('roll angle')
				ax1.plot(imu_ts, r_gt_mat, 'r', label = 'Vicon')
				ax1.plot(imu_ts, r_my_mat, 'g', label = 'UKF')
				ax1.plot(imu_ts, r_g_mat,'b', label='Gyro')
				ax1.plot(imu_ts, r_a_mat, 'c', label='Acc')
				ax2 = fig.add_subplot(312)
				ax2.set_ylabel('pitch angle')
				ax2.plot(imu_ts, p_gt_mat, 'r', label = 'Vicon')
				ax2.plot(imu_ts, p_my_mat, 'g', label = 'UKF')
				ax2.plot(imu_ts, p_g_mat, 'b', label = 'Gyro')
				ax2.plot(imu_ts, p_a_mat, 'c', label = 'Acc')
				ax3 = fig.add_subplot(313)
				ax3.set_ylabel('yaw angle')
				ax3.set_xlabel('time stamp')
				ax3.plot(imu_ts, y_gt_mat, 'r', label = 'Vicon')
				ax3.plot(imu_ts, y_my_mat, 'g', label = 'UKF')
				ax3.plot(imu_ts, y_g_mat, 'b', label = 'Gyro')
				ax3.plot(imu_ts, y_a_mat, 'c', label = 'Acc')
				ax1.legend(loc = 'upper right')
				ax2.legend(loc = 'upper right')
				ax3.legend(loc = 'upper right')
				fig.suptitle('Euler angles for dataset no.'+str(datanum))
				plt.show()