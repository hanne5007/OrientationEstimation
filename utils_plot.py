#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/6/18 5:36 PM 

@author: Hantian Liu
"""

import numpy as np
import math

Hfov=math.pi/3
Vfov=math.pi/4

def sph2cart(theta, phi, r):
	"""

	:param theta: azimuth n
	:param phi: altitude n
	:param r: radius n
	:return: x n, y n, z n
	"""
	rcos = r * np.cos(phi)
	x = rcos * np.cos(theta)
	y = rcos * np.sin(theta)
	z = r * np.sin(phi)
	return x, y, z

def cart2sph(x, y, z):
	"""

	:param x: n
	:param y: n
	:param z: n
	:return: theta: azimuth n
			phi: altitude n
			radius n
	"""
	r=np.sqrt(x**2+y**2+z**2)
	hxy=np.sqrt(x**2+y**2)
	phi = np.arctan2(z, hxy)
	theta = np.arctan2(y, x)
	return theta, phi, r

def pixel2sph(px, py):
	"""

	:param px: col n
	:param py: row n
	:return: theta: azimuth n
			phi: altitude n
			radius n
	"""
	xnum=np.max(px)+1
	ynum=np.max(py)+1
	r=1
	theta=px*Hfov/xnum-Hfov/2
	theta=-theta
	phi=py*Vfov/ynum-Vfov/2
	phi=-phi
	return theta, phi, r

def sph2plane(theta, phi, r):
	"""

	:param theta: azimuth n
	:param phi: altitude n
	:param r: radius n
	:return: x row n,  y col n
	"""
	x=-phi
	y=-theta
	return x, y


if __name__ == '__main__':
	t,p,r=cart2sph(1,2,3)
	x,y,z=sph2cart(t,p,r)
	print(x)
	print(y)
	print(z)