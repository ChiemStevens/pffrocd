#!/usr/bin/env python3
print("importing pffrocd...")
import pffrocd # helper functions
print("pffrocd imported!")
import numpy as np
import random
import time
import sys


pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_int_scen_simd'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"
pffrocd.OUTPUT_FILE_NAME = f"/home/chiem/pffrocd/ABY/build/bin"
NUMPY_DTYPE = np.float32

# get two embeddings of different people
x = pffrocd.get_embedding("/home/chiem/pffrocd/lfw/Adrian_McPherson/Adrian_McPherson_0001.jpg", dtype=NUMPY_DTYPE)
y = pffrocd.get_embedding("/home/chiem/pffrocd/lfw/Adrian_McPherson/Adrian_McPherson_0002.jpg", dtype=NUMPY_DTYPE)

x = x / np.linalg.norm(x)
share0, share1 = pffrocd.create_shares(x, dtype=NUMPY_DTYPE)

y = y / np.linalg.norm(y)
share0prime, share1prime = pffrocd.create_shares(y, dtype=NUMPY_DTYPE)

output = pffrocd.run_sfe_improved(x, y, y_0=share0, y_1=share1, x_0=share0prime, x_1=share1prime)

print(output.stdout)

print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))