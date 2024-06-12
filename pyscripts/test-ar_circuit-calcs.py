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
x = qt.quantize_tensor(x)
y = qt.quantize_tensor(y)
# I want to add 128 to every value in x to make it positive
print(x)
x = np.array(x, dtype=NUMPY_DTYPE)
y = np.array(y, dtype=NUMPY_DTYPE)

# SFace calculations

share0, share1 = pffrocd.create_shares(x, NUMPY_DTYPE, True)
print("SHARES 0: ", share0)
print("SHARES 1: ", share1)
share0prime, share1prime = pffrocd.create_shares(y, NUMPY_DTYPE, True)

# share0 = np.array(share0, dtype=np.uintc)
# share1 = np.array(share1, dtype=np.uintc)
# share0prime = np.array(share0prime, dtype=np.uintc)
# share1prime = np.array(share1prime, dtype=np.uintc)

output = pffrocd.run_sfe_improved(x, y, y_0=share0, y_1=share1, x_0=share0prime, x_1=share1prime)
print(output.stdout)

print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))
print(1-np.dot(x,y))
# the dot product written out
sum = 0
for i in range(0, len(x)):
    sum+=x[i]*y[i]
print(sum)
