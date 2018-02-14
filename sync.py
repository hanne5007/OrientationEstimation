#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/6/18 10:16 PM 

@author: Hantian Liu
"""

import numpy as np

def sync_gt(gt_ts, imu_ts, gt, imu):
	"""

	:param gt_ts: n
	:param imu_ts: m
	:param gt: gt 3*3*n
	:param imu: imu 6*m
	:return: gt_new 3*3*min(n, m)
	         imu_new 6*min(n,m)
	         ts imu ts 0 to min(n,m)
	"""
	n=len(gt_ts)
	m=len(imu_ts)
	if n<m:
		ts=imu_ts[:n]
		imu_new=imu[:,:n]
		gt_new=np.zeros([3,3,n])
		for i in range(n):
			diff=gt_ts-imu_ts[i]
			diff=abs(diff)
			ind=np.where(diff==min(diff))
			ind=ind[0][0]
			gt_new[:,:,i:i+1]=gt[:,:,ind:ind+1]
	elif n>=m:
		ts=imu_ts[:m]
		gt_new=gt[:,:,:m]
		gt_new = np.zeros([3, 3, m])
		imu_new=np.zeros([6,m])
		imu_new = imu[:, :m]
		for i in range(m):
			diff=gt_ts-imu_ts[i]
			diff=abs(diff)
			ind=np.where(diff==min(diff))
			ind=ind[0][0]
			#imu_new[:,i:i+1]=imu[:,ind:ind+1]
			gt_new[:, :, i:i + 1] = gt[:, :, ind:ind + 1]
	return gt_new, imu_new, ts


def sync_to_cam(R_ts, cam_ts, R):
	"""

	:param R_ts: n
	:param cam_ts: m
	:param R: 3*3*n
	:return: R_new 3*3*m
	"""
	m=len(cam_ts)
	R_new=np.zeros([3,3,m])
	for i in range(m):
		diff=R_ts-cam_ts[i]
		diff=abs(diff)
		ind=np.where(diff==min(diff))
		ind=ind[0][0]
		R_new[:,:, i:i+1]=R[:,:, ind:ind+1]
	return R_new