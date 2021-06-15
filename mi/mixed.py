#Copyright Weihao Gao, UIUC
import logging
import os.path

from .model import mlp,mlp_try
from .data import Two_Random
import scipy.spatial as ss
from scipy.special import digamma
from math import log
import torch
import torch.utils.data
import numpy.random as nr
import numpy as np
from torch.nn import MSELoss
from .util import logger

#Mixed_KSG Algorithm
def Mixed_KSG(x,y,k=5):
	'''
		Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
		Using *Mixed-KSG* mutual information estimator

		Input: x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
		y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
		k: k-nearest neighbor parameter

		Output: one number of I(X;Y)
	'''

	#assert len(x)==len(y), "Lists should have same length"
	#assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	if x.ndim == 1:
		x = x.reshape((N,1))
	dx = len(x[0])
	if y.ndim == 1:
		y = y.reshape((N,1))
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)

	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

	knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
	ans = 0

	for i in range(N):
		kp, nx, ny = k, k, k
		if knn_dis[i] == 0:
			kp = len(tree_xy.query_ball_point(data[i],1e-15,p=float('inf')))
			nx = len(tree_x.query_ball_point(x[i],1e-15,p=float('inf')))
			ny = len(tree_y.query_ball_point(y[i],1e-15,p=float('inf')))
		else:
			nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
			ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
		ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny))/N
	return ans


#Partitioning Algorithm
def Partition(x,y,k=0,numb=8):
	#assert len(x)==len(y), "Lists should have same length"
	N = len(x)
	if x.ndim == 1:
		x = x.reshape((N,1))
	dx = len(x[0])
	if y.ndim == 1:
		y = y.reshape((N,1))
	dy = len(y[0])

	minx = np.zeros(dx)
	miny = np.zeros(dy)
	maxx = np.zeros(dx)
	maxy = np.zeros(dy)
	for d in range(dx):
		minx[d], maxx[d] = x[:,d].min()-1e-15, x[:,d].max()+1e-15
	for d in range(dy):
		miny[d], maxy[d] = y[:,d].min()-1e-15, y[:,d].max()+1e-15

	freq = np.zeros((numb**dx+1,numb**dy+1))
	for i in range(N):
		index_x = 0
		for d in range(dx):
			index_x *= dx
			index_x += int((x[i][d]-minx[d])*numb/(maxx[d]-minx[d]))
		index_y = 0
		for d in range(dy):
			index_y *= dy
			index_y += int((y[i][d]-miny[d])*numb/(maxy[d]-miny[d]))
		freq[index_x][index_y] += 1.0/N
	freqx = [sum(t) for t in freq]
	freqy = [sum(t) for t in freq.transpose()]
	
	ans = 0
	for i in range(numb**dx):
		for j in range(numb**dy):
			if freq[i][j] > 0:
				ans += freq[i][j]*log(freq[i][j]/(freqx[i]*freqy[j]))
	return ans

#Noisy KSG Algorithm
def Noisy_KSG(x,y,k=5,noise=0.01):
	#assert len(x)==len(y), "Lists should have same length"
	#assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	if x.ndim == 1:
		x = x.reshape((N,1))
	dx = len(x[0])
	if y.ndim == 1:
		y = y.reshape((N,1))
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)
	
	if noise > 0:
		data += nr.normal(0,noise,(N,dx+dy))

	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

	knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
	ans = 0

	for i in range(N):
		nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
		ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
		ans += (digamma(k) + log(N) - digamma(nx) - digamma(ny))/N
	return ans

#Original KSG estimator
def KSG(x,y,k=5):
	#assert len(x)==len(y), "Lists should have same length"
	#assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	if x.ndim == 1:
		x = x.reshape((N,1))
	dx = len(x[0])
	if y.ndim == 1:
		y = y.reshape((N,1))
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)
	
	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

	knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
	ans = 0

	for i in range(N):
		nx = len(tree_x.query_ball_point(x[i],knn_dis[i]+1e-15,p=float('inf')))-1
		ny = len(tree_y.query_ball_point(y[i],knn_dis[i]+1e-15,p=float('inf')))-1
		ans += (digamma(k) + log(N) - digamma(nx) - digamma(ny))/N
	return ans


def transform_linear(x, k):
	return x

def transform_super(x, k):
	return x**(k+1)

def transform_poly(x, k):
	return np.concatenate([x**(i+1) for i in range(k)],axis=1)

def transform_log(x, k):
	return np.log(x)

def transform_exp(x, k):
	return np.exp(x)

def transform_log_poly(x, k):
	return np.concatenate([(np.log(x))**(i+1) for i in range(k)],axis=1)

def F_gaussian(x, y, x_test=None, y_test=None, transform_func=transform_poly, k=1, key=None, x_label=None, id=None):
	# F_informaiton from x to y (I_F(x \to y)
	if x_test is None:
		x_test = x
		y_test = y
	N = len(x)
	N_test = len(x_test)
	x = x.reshape((N, 1))
	y = y.reshape((N, 1))
	x_test = x_test.reshape((N_test, 1))
	y_test = y_test.reshape((N_test, 1))
	model = mlp_try(k).cuda()
	opt = torch.optim.Adam(model.parameters(), lr=0.0004)
	x = transform_func(x, k)
	dataloader = torch.utils.data.DataLoader(dataset = Two_Random(x,y), batch_size = 256, shuffle = True)
	test_dataloader = torch.utils.data.DataLoader(dataset = Two_Random(x_test,y_test), batch_size = 256, shuffle = True)

	epoch = 400
	checkpoint_file = "{}_{}_{}_{}.pth".format(key,k, x_label, id)
	model.train()
	best_loss, best_epoch = float("inf"), 0
	critereon = MSELoss()
	for i in range(epoch):
		train_loss = 0.0
		for x_,y_ in dataloader:
			x_ = x_.cuda().float()
			y_ = y_.cuda().float()
			opt.zero_grad()
			output = model(x_)
			loss = critereon(output, y_)
			# logger.info("loss",loss)
			train_loss += loss.item()
			loss.backward()
			opt.step()
		logger.info("Epoch: {}\tLoss: {}".format(i, train_loss))
		if train_loss > best_loss:
			if i - best_epoch > 10:
				logger.info("out of patience!")
				break
		if train_loss <= best_loss:
			best_loss = train_loss
			best_epoch = i
			torch.save(model.state_dict(), checkpoint_file)

	if not os.path.isfile(checkpoint_file):
		return 0,0
	model = mlp_try(k)
	model.load_state_dict(torch.load(checkpoint_file))
	model.cuda().eval()

	H_y_x = 0.0
	for x_, y_ in test_dataloader:
		x_ = x_.cuda().float()
		y_ = y_.cuda().float()
		output = model(x_)
		loss = torch.pow(output - y_, 2).sum()
		H_y_x += loss.item()
	H_y_x = H_y_x / N_test
	mu = np.sum(y_test, axis=0) / N_test
	H_y = np.sum((y_test - mu) ** 2) / N_test

	H_y_x_ori = 0.0
	for x_, y_ in dataloader:
		x_ = x_.cuda().float()
		y_ = y_.cuda().float()
		output = model(x_)
		loss = torch.pow(output - y_, 2).sum()
		H_y_x_ori += loss.item()
	H_y_x_ori = H_y_x_ori / N
	mu = np.sum(y, axis=0) / N
	H_y_ori = np.sum((y - mu) ** 2) / N


	return H_y - H_y_x, H_y_ori - H_y_x_ori







