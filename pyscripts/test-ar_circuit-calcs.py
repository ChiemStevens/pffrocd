#!/usr/bin/env python3
print("importing pffrocd...")
import pffrocd # helper functions
print("pffrocd imported!")
import numpy as np
import random
import time
import quantization as qt
import sys


pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_int_scen_simd'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"
pffrocd.OUTPUT_FILE_NAME = f"/home/chiem/pffrocd"
NUMPY_DTYPE = np.float32

# get two embeddings of different people
x = pffrocd.get_embedding("/home/chiem/pffrocd/lfw/Adrian_McPherson/Adrian_McPherson_0001.jpg", dtype=NUMPY_DTYPE)
y = pffrocd.get_embedding("/home/chiem/pffrocd/lfw/Adrian_McPherson/Adrian_McPherson_0002.jpg", dtype=NUMPY_DTYPE)
z = pffrocd.get_embedding("/home/chiem/pffrocd/lfw/Aaron_Peirsol/Aaron_Peirsol_0001.jpg", dtype=NUMPY_DTYPE)

# now quantize before normalizing
# x = x / np.linalg.norm(x)
# y = y / np.linalg.norm(y)
print(pffrocd.get_cos_dist_numpy(x,y))
x1 = qt.scalar_quantisation_percentile_og(x)
y1 = qt.scalar_quantisation_percentile_og(y)
print(pffrocd.get_cos_dist_numpy(x1,y1))

x = qt.scalar_quantisation_percentile(x)
y = qt.scalar_quantisation_percentile(y)
print("x: ", x)
print("y: ", y)
print(pffrocd.get_cos_dist_numpy(x,y))

x = np.array(x, dtype=NUMPY_DTYPE)
y = np.array(y, dtype=NUMPY_DTYPE)

# SFace calculations

share0, share1 = pffrocd.create_shares(x, NUMPY_DTYPE, True)
share0prime, share1prime = pffrocd.create_shares(y, NUMPY_DTYPE, True)

# share0 = np.array(share0, dtype=np.uint8)
# share1 = np.array(share1, dtype=np.uint8)
# share0prime = np.array(share0prime, dtype=np.uint8)
# share1prime = np.array(share1prime, dtype=np.uint8)
x = np.array(x, dtype=np.uint8)
y = np.array(y, dtype=np.uint8)
print("x: ", x)
print("y: ", y)
output = pffrocd.run_sfe_improved(x, y, y_0=share0, y_1=share1, x_0=share0prime, x_1=share1prime)
print(output.stdout)

print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))
print(1-np.dot(x,y))
# the dot product written out
sum = 0
for i in range(0, len(x)):
    sum+=np.uintc(x[i])*np.uintc(y[i])
print(sum)
