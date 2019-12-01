import os
import sys

import numpy as np
import cv2 as cv
import scipy.io as io
from collections import OrderedDict
import copy

def build_k_matrix(data):
  k= np.array([[data['alpha_x'][0,0], data['s'][0,0],data['x_0'][0,0]],
               [0, data['alpha_y'][0,0], data['y_0'][0,0]],
               [0,0,1]])
  return np.array(k)


def is_orthogonal(R):
  I = np.abs(R.dot(R.T))
  cond = np.allclose(I, np.eye(3), atol=0.001)
  return cond
  

def correct_error(R):
  u, s, vh = np.linalg.svd(R, full_matrices=True)
  R = u.dot(vh)
  return R

def compute_relative_rotation():
  base_folder = './data/'

  data = io.loadmat(base_folder + 'ex3.mat')
  
  K = build_k_matrix(data)
  K_inv = np.linalg.inv(K) 

  H_arr = [data["H1"], data["H2"]]
  for i in range(0, len(H_arr)):
    print("Computing rotation from homography for H"+str(i+1)+":")
    H = H_arr[i]
    print("Homography Matrix:")
    print(H)

    E = K_inv.dot(H)
    R = np.ones((3,3))
    R[:,0:2]= E[:,0:2]

    # Normalize R1, R2
    norm_factor = (np.linalg.norm(R[:,0]) + np.linalg.norm(R[:,1])) / 2
    R[:, 0] /= norm_factor
    R[:, 1] /= norm_factor 

    R[:,-1 ]= np.cross(R[:,0].T, R[:,1].T)
    print("Computed Rotation Matrix from Homography is:")
    print(R)

    print("Is this orthogonal?")
    if not is_orthogonal(R):
      print("NO. Making orthogonal...")
      R = correct_error(R)
      print("Test R.dot(R.T)==I again. Is it passed?:", is_orthogonal(R))
      print("Rrel for H" + str(i+1)+ " is:")
      print(R)
     

    else:
      print("YES.")
      print("then Rrel for H"+str(i+1)+" is:")
      print(R)

    print("----------------------------------------")

def compute_pose():
  base_folder = './data/'

  data = io.loadmat(base_folder + 'ex3.mat')
  
  H3_Homography = data['H3']
  x_0=data['x_0']
  y_0=data['y_0']
  alpha_x=data['alpha_x']
  alpha_y=data['alpha_y']
  s=data["s"]
  K_matrix=[[alpha_x[0][0],s[0][0],x_0[0][0]],
            [0,alpha_y[0][0],y_0[0][0]],
            [0,0,1]]
  K_inverse=np.linalg.inv(K_matrix)
  Projection_matrix=np.matmul(K_inverse,H3_Homography)
  Projection_matrix=np.matmul(Projection_matrix,K_matrix)

  r_1=Projection_matrix[:,0]
  r_2=Projection_matrix[:,1]
  mag_r1 = np.linalg.norm(r_1)
  mag_r2 = np.linalg.norm(r_2)
  lambda_=(mag_r1+mag_r2)/2
  r_1/=lambda_
  r_2/=lambda_
  r_3=np.cross(r_1,r_2)
  translation=Projection_matrix[:,2]
  translation/=lambda_
  Rotation_matrix=np.column_stack((r_1,r_2,r_3))
  print("Rotation Matrix",Rotation_matrix)
  print("translation",translation)
print("----------------------------------------")
print("Question 3")
print("----------------------------------------")
compute_relative_rotation()
print("----------------------------------------")
print("Question 4")
print("----------------------------------------")
compute_pose()