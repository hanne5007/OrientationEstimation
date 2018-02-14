#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/3/18 4:54 PM 

@author: Hantian Liu
"""

import numpy as np
from fromRaw import getAcc, getAngularVelocity
from utils import qmultiply, vec2quart, quart2vec, vec2Skew, qconj, quartmean, quart2mat
import math
import transforms3d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb

#Q-process noise covariance
sigmaq=100
sigmaw=100
#R-measurement noise covariance
sq=10
sw=10
#Sigma points covariance initilization
alpha=0.01
n=6
# tolerance for the magnitude of valid acc measurements
epsilon=0.01

Q = np.identity(6)
Q[0:3, :] = Q[0:3, :] * sigmaq
Q[3:, :] = Q[3:, :] * sigmaw

R = np.identity(6)
R[0:3, :] = R[0:3, :] * sq
R[3:, :] = R[3:, :] * sw

def init():
	mu = np.zeros([7, 1])
	mu[0:1, :] = 1
	#q=np.zeros([4,1])
	#q[0,:]=1
	sigma = np.identity(6) * alpha
	return mu, sigma

def cov(X, Y):
	"""

	:param X: n*m
	:param Y: n*m
	:return: covariance n*n
	"""
	r, c= np.shape(X)
	Y=Y.transpose()
	#covmat=np.zeros([r,r])
	#for i in range(c):
	#	covmat=covmat+np.dot(X[:,i:i+1],Y[i:i+1,:])
	covmat=np.dot(X, Y)
	covmat=covmat/c
	return covmat


def process(x1, noise, dt):
	"""

	:param x1: 7*1
	:param noise: 7*1
	:return:
	"""
	x2 = np.zeros([7,1])
	omg=x1[4:,:]+noise[4:,:]

	#qdelta=vec2quart(x1[4:,:]*dt)

	qdelta = np.zeros([4, 1])
	v=x1[4:,:]
	angle = np.linalg.norm(v)*dt
	if angle == 0:
		axis = np.ones([3, 1])
	else:
		axis = v / np.linalg.norm(v)
	qdelta[0, :] = math.cos(angle / 2)
	qdelta[1:, :] = axis * math.sin(angle / 2)

	q=qmultiply(qmultiply(x1[0:4,:],noise[0:4,:]),qdelta)
	#pdb.set_trace()
	x2[4:,:]=omg
	x2[0:4,:]=q
	return x2

def measurement(x, noise):
	"""

	:param x: 7*1
	:param noise: 6*1
	:return: 6*1 acc 3*1 then omg 3*1
	"""
	z=np.zeros([6,1])
	omg=x[4:,:]+noise[3:,:]

	qg_world = np.array([[0], [0], [0], [1]])
	qg_body=qmultiply(qmultiply(qconj(x[0:4,:]),qg_world), x[0:4,:])
	# qg is vector quartenion
	# 0-scalar
	# vector part== corresponding vector
	z[0:3,:]=qg_body[1:4,:]+noise[0:3,:]
	z[3:,:]=omg
	return z

def update(mu, K, v):
	"""

	:param mu: 7*1
	:param K: 6*6
	:param v: 6*1
	:return: 7*1
	"""
	mu_new=np.zeros([7,1])
	Kv=np.dot(K, v)
	qKv=vec2quart(Kv[0:3,:])
	q=qmultiply(mu[0:4,:], qKv)
	mu_new[0:4,:]=q
	mu_new[4:,:]=Kv[3:,:]+mu[4:,:]
	return mu_new

def UKF(omgx, omgy, omgz, gt_ts, i_data, mu, sigma, is_acc, accx, accy, accz):
	nonoise=np.zeros([7, 1])
	nonoise[0,:]=1

	dt=gt_ts[i_data]-gt_ts[i_data-1]

	S = np.linalg.cholesky(sigma + Q)
	ST=S.transpose() #TODO
	S=S*0.8+ST*0.2
	W = np.ones([n, 2 * n])
	W[:, 0:n] = np.sqrt(2 * n) * S
	W[:, n:] = -np.sqrt(2 * n) * S

	# sigma points
	X = np.zeros([7, 2 * n])
	for i in range(2 * n):
		X[0:4, i:i + 1] = qmultiply(mu[0:4,:], vec2quart(W[0:3, i:i+1]))
		X[4:, i:i + 1] = W[3:, i:i + 1] + mu[4:, :]
	'''
	sp=np.zeros([3,2*n])
	for i in range(2*n):
		sp[:,i:i+1]=quart2vec(X[0:4,i:i+1])
	print('mean',np.mean(sp, axis=1))
	print('cov', cov(sp, sp))
	'''

	# transformed sigma points
	Y=np.zeros(np.shape(X))
	for i in range(2*n):
		Y[:,i:i+1]=process(X[:,i:i+1], nonoise, dt)


	#Prediction step
	#print('predict!')
	mu_bar=np.zeros([7,1])
	mu_bar[0:4,:]=quartmean(Y, mu[0:4,:])
	Wprime=np.zeros([n, 2*n])
	omgmean=np.mean(Y[4:,:], axis=1)
	mu_bar[4:,:]=omgmean[:,np.newaxis]
	for i in range(2*n):
		Wprime[0:3, i:i+1]=quart2vec(qmultiply(Y[0:4,i:i+1],qconj(mu_bar[0:4,:])))
		Wprime[3:,i:i+1]=Y[4:,i:i+1]-mu_bar[4:,:]
	sigma_bar=cov(Wprime, Wprime)

	Z = np.zeros([n, 2*n])
	for i in range(2 * n):
		Z[:, i:i + 1] = measurement(Y[:, i:i + 1], np.zeros([6,1]))
	z_bar = np.mean(Z, axis = 1)
	z_bar = z_bar[:, np.newaxis]
	Zdiff=Z - np.tile(z_bar, (1, 2 * n))
	sigma_xz=cov(Wprime, Zdiff)
	sigma_zz=cov(Zdiff, Zdiff)
	sigma_vv=sigma_zz+R

	#true measurement
	#print('measure!')
	meas=np.zeros([6,1])
	if is_acc:
		meas[0:1,:]=accx[i_data]
		meas[1:2,:]=accy[i_data]
		meas[2:3,:]=accz[i_data]
	meas[3:4,:]=omgx[i_data]
	meas[4:5, :] = omgy[i_data]
	meas[5:, :] = omgz[i_data]

	#inno
	z=meas
	v=z-z_bar

	#Kalman gain
	K=np.dot(sigma_xz, np.linalg.inv(sigma_vv))

	#Update step
	#print('update')
	mu_update=update(mu_bar, K, v)
	sigma_update=sigma_bar - np.dot(np.dot(K, sigma_vv), K.transpose())

	mu=mu_update
	sigma=sigma_update
	q=mu[0:4,:]

	return mu, sigma, q


if __name__ == '__main__':

	gt=np.load('gt.npy')
	imu_ts=np.load('ts.npy')
	imuraw=np.load('imuraw.npy')
	ox,oy,oz=getAngularVelocity(imuraw)
	ax,ay,az=getAcc(imuraw)

	#q=[]
	r_my=[]
	p_my=[]
	y_my=[]
	#q_gt=[]
	r_gt=[]
	p_gt=[]
	y_gt=[]
	num=len(imu_ts)
	#num=100

	mu, sigma=init()
	for i in range(num):
		if abs(ax[i]**2+ay[i]**2+az[i]**2-1)>epsilon:
			acc_valid=False
		else:
			acc_valid=True

		mu, sigma, q_ukf=UKF(ox,oy,oz, imu_ts,i, mu, sigma, acc_valid, ax, ay, az)
		#q.append(q_ukf)
		my_mat=quart2mat(q_ukf)
		rr, pp, yy=transforms3d.euler.mat2euler(my_mat,axes='szyx')
		r_my.append(rr)
		p_my.append(pp)
		y_my.append(yy)


		r,p,y=transforms3d.euler.mat2euler(gt[:,:,i],axes='szyx')
		r_gt.append(r)
		p_gt.append(p)
		y_gt.append(y)
		#q_gt_i=transforms3d.euler.euler2quat(r,p,y,axes='szyx')
		#q_gt.append(q_gt_i)
	'''
	q_mat=np.asarray(q)
	q_mat=q_mat[:,:,0]
	q_gt_mat=np.asarray(q_gt)
	fig = plt.figure()
	ax1 = fig.add_subplot(221)
	ax1.plot(imu_ts[0,500:500+num],q_gt_mat[:,0],'r', imu_ts[0,500:500+num], q_mat[:,0],'b')
	ax2 = fig.add_subplot(222)
	ax2.plot(imu_ts[0,500:500+num], q_gt_mat[:,1], 'r', imu_ts[0,500:500+num], q_mat[:,1], 'b')
	ax3 = fig.add_subplot(223)
	ax3.plot(imu_ts[0,500:500+num], q_gt_mat[:,2], 'r', imu_ts[0,500:500+num], q_mat[:,2], 'b')
	ax4 = fig.add_subplot(224)
	ax4.plot(imu_ts[0,500:500+num], q_gt_mat[:,3], 'r', imu_ts[0,500:500+num], q_mat[:,3], 'b')
	'''
	r_gt_mat=np.asarray(r_gt)
	p_gt_mat = np.asarray(p_gt)
	y_gt_mat = np.asarray(y_gt)

	r_my_mat = np.asarray(r_my)
	p_my_mat = np.asarray(p_my)
	y_my_mat = np.asarray(y_my)

	ts=np.arange(0,num)
	fig=plt.figure()
	ax1=fig.add_subplot(311)
	ax1.set_title('roll angle')
	ax1.plot(ts, r_gt_mat,'r',label='Vicon')
	ax1.plot(ts, r_my_mat, 'g', label='UKF')
	ax2 = fig.add_subplot(312)
	ax2.set_title('pitch angle')
	ax2.plot(ts, p_gt_mat, 'r', label='Vicon')
	ax2.plot(ts, p_my_mat, 'g', label='UKF')
	ax3 = fig.add_subplot(313)
	ax3.set_title('yaw angle')
	ax3.plot(ts, y_gt_mat, 'r', label='Vicon')
	ax3.plot(ts, y_my_mat, 'g', label='UKF')
	fig.suptitle('Q'+str(sigmaq)+' '+str(sigmaw)+'\n'+'R'+str(sq)+' '+str(sw))
	ax1.legend(loc='upper right')
	ax2.legend(loc='upper right')
	ax3.legend(loc='upper right')
	plt.show()

