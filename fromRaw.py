#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/3/18 3:37 PM 

@author: Hantian Liu
"""
import numpy as np
from utils import vec2Skew
import math
import transforms3d

#imuraw=np.load('imuraw.npy')
#gt=np.load('gt.npy')

def getAngularVelocity(imuraw):
	"""

	:param imuraw:
	:return: rad/sec
	"""
	sensitivity=3.33 #0.83 amplified or not?
	#bias=1.23/3.3*1023
	bias = np.mean(imuraw[:, 0:150], axis = 1)
	bias=bias[:, np.newaxis]
	value = (imuraw- bias)* 3300 / 1023 * np.pi / 180 / sensitivity
	omgx=value[4]
	omgy=value[5]
	omgz=value[3]
	return omgx,omgy,omgz

def getAcc(imuraw):
	"""

	:param imuraw:
	:return: unit: g
	"""
	sensitivity = 360 #300 for 3V
	#bias=1.5/3.3*1023
	bias = np.mean(imuraw[:, 0:150], axis = 1) - np.array([0,0,1,0,0,0])*sensitivity*1023/3300
	bias = bias[:, np.newaxis]
	value = (imuraw-bias) * 3300 / 1023 / sensitivity #*9.81
	accx=-value[0]
	accy=-value[1]
	accz=value[2]
	return accx, accy, accz

def findRotBias(accx, accy, accz, gt):
	"""

	:param accx: n
	:param accy: n
	:param accz: n
	:param gt: 3*3*n
	:return:
	"""
	n=len(accx)
	mat=[]
	g_world = np.array([[0], [0], [1]])
	for i in range(n):
		mat.append(np.dot(gt[:,:,i].transpose(), g_world))




if __name__ == '__main__':
	#R=getRot(-1,3,2)
	#
	#print(np.dot(R, np.array([[-1],[3],[2]])))
	imuraw = np.load('imuraw.npy')
	ax, ay, az = getAcc(imuraw[:,100])
	R=getRot(ax, ay, az)
	print(ax)
	print(ay)
	print(az)
	print(R)
