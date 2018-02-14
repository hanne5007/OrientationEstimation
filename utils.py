#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/3/18 4:22 PM 

@author: Hantian Liu
"""
import numpy as np
import pdb
import math
import transforms3d

def qmultiply(x,y):
	"""

	:param x: 4*1
	:param y: 4*1
	:return: 4*1 quartenion
	"""
	z=np.zeros([4,1])
	xv=x[1:4,:]
	yv=y[1:4,:]
	xvt=xv.transpose()
	yvt=yv.transpose()
	z[0,:]=x[0,:]*y[0,:]-sum(xv.flatten()*yv.flatten())
	#pdb.set_trace()
	z[1:4,:]=x[0,:]*yv+y[0,:]*xv+np.cross(xvt,yvt).transpose()
	return z


def vec2Skew(v):
	"""
	turn a vector into its skew-symmetric matrix
	:param v:
	:return:
	"""
	v=v[0,:]
	skew=np.array([[0, -v[2], v[1]],\
				   [v[2], 0, -v[0]],\
				   [-v[1], v[0], 0]])
	return skew


def vec2quart(v):
	"""

	:param v: 3*1
	:return: quartenion 4*1
	"""
	q=np.zeros([4,1])
	angle=np.linalg.norm(v)
	if angle==0:
		axis=np.ones([3,1])
	else:
		axis=v/angle
	q[0,:]=math.cos(angle/2)
	q[1:,:]=axis*math.sin(angle/2)
	#print('if unit???')
	#print(np.linalg.norm(q))
	return q

def quart2vec(q):
	"""

	:param q: 4*1
	:return: vector 3*1
	"""
	norm=np.linalg.norm(q)
	if norm!=1:
		#print('automatically normalize quartenion')
		q=q/np.linalg.norm(q)
	#halfangle=math.acos(q[0,:])
	c=q[0,:]
	s=np.linalg.norm(q[1:,:])
	halfangle=math.atan2(s, c)
	#print(halfangle0)
	#print(halfangle)
	if halfangle==0:
		return np.zeros([3,1])
	v=q[1:,:]/math.sin(halfangle)
	v=v*2*halfangle
	return v

def qconj(q):
	"""

	:param q: 4*1
	:return: 4*1
	"""
	qnew=np.zeros([4,1])
	qnew[0,:]=q[0,:]
	qnew[1:,:]=-q[1:,:]
	return qnew


def quartmean(q, q0):
	"""

	:param q: 4*n
	:param q0: 4*1
	:return: mean quartenion 4*1
	"""
	epsilon=1e-6 #TODO
	maxiter=100
	avgnorm=1
	norm = np.linalg.norm(q0)
	if norm != 1 and norm!=0:
		#print('automatically normalize quartenion')
		q0 = q0 / np.linalg.norm(q0)
	unused4, n= np.shape(q)
	#equart=np.zeros([4,n])
	evec=np.zeros([3,n])
	qprev=q0
	iteration=0
	#print('quartenion mean iterating...')
	while avgnorm>epsilon and iteration<maxiter:
		for i in range(n):
			equart=qmultiply(q[:,i:i+1], qconj(qprev))
			evec[:,i:i+1]=quart2vec(equart)
		evec_avg=np.mean(evec, axis=1)
		avgnorm=np.linalg.norm(evec_avg)
		evec_avg=evec_avg[:, np.newaxis]
		equart_avg=vec2quart(evec_avg)
		#print(np.linalg.norm(equart_avg))
		#print(np.linalg.norm(qprev))
		qcurr=qmultiply(equart_avg, qprev)
		qprev=qcurr
		#print(avgnorm)
		iteration=iteration+1
	#print('total iteration:' + str(iteration))

	return qcurr

def quart2mat(q):
	"""

	:param q: 4*1
	:return: 3*3 SO(3)
	"""
	q0=q[0,:]
	q1=q[1,:]
	q2=q[2,:]
	q3=q[3,:]
	R=np.zeros([3,3])
	R[0,0]=q0**2+q1**2-q2**2-q3**2
	R[0,1]=2*(q1*q2-q0*q3)
	R[0,2]=2*(q0*q2+q1*q3)

	R[1,0]=2*(q1*q2+q0*q3)
	R[1,1]=q0**2-q1**2+q2**2-q3**2
	R[1,2]=2*(q2*q3-q0*q1)

	R[2,0]=2*(q1*q3-q0*q2)
	R[2,1]=2*(q0*q1+q2*q3)
	R[2,2]=q0**2-q1**2-q2**2+q3**2
	return R

if __name__ == '__main__':
	q=np.array([[3],[3],[9],[2]])
	q=q/np.linalg.norm(q)

	mymat=quart2mat(q)
	print(mymat)

	r,p,y=transforms3d.euler.quat2euler(q)
	mat=transforms3d.euler.euler2mat(r,p,y)
	print(mat)

