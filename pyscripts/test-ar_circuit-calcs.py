#!/usr/bin/env python3
print("importing pffrocd...")
import pffrocd # helper functions
print("pffrocd imported!")
import numpy as np
import random
import time
import quantization as qt
import sys

def cosine_similarity(v1, v2):
    # Compute the cosine similarity
    return 1- np.dot(v1, v2)

def evaluate_quantization(vector1, vector2, quantization_func):
    # Compute the cosine similarity before quantization
    #x = x / np.linalg.norm(x)
    before_quantization = cosine_similarity(vector1, vector2)

    # Quantize the vectors
    quantized_vector1 = quantization_func(vector1)
    quantized_vector2 = quantization_func(vector2)
    print(quantized_vector1)

    # Compute the cosine similarity after quantization
    after_quantization = cosine_similarity(quantized_vector1, quantized_vector2)

    # Return the cosine similarities before and after quantization
    return before_quantization, after_quantization

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
x = x / np.linalg.norm(x)
y = y / np.linalg.norm(y)
print("cosine distance: ", cosine_similarity(x, y))

# multiply each item in x and y (which are np arrays) by 1000
x = x * 1000
y = y * 1000
print(x)
print(y)

print("cosine distance: ", 1-(np.dot(x, y)/1000))

# before, after = evaluate_quantization(x,y,qt.scalar_quantisation_percentile)
# print("BEFORE QUANTIZATION: ", before)
# print("AFTER QUANTIZATION: ", after)

# x = qt.scalar_quantisation_percentile(x)
# y = qt.scalar_quantisation_percentile(y)

# share0, share1 = pffrocd.create_shares(np.array(x, dtype=NUMPY_DTYPE), NUMPY_DTYPE, True)
# share0prime, share1prime = pffrocd.create_shares(np.array(y, dtype=NUMPY_DTYPE), NUMPY_DTYPE, True)

# print("og share1 prime", share0prime)
# share0 = np.array(share0, dtype=np.uint32)
# share1 = np.array(share1, dtype=np.uint32)
# share0prime = np.array(share0prime, dtype=np.uint32)
# print(share0prime)
# share1prime = np.array(share1prime, dtype=np.uint32)
# x = np.array(x, dtype=np.uint32)
# y = np.array(y, dtype=np.uint32)
# output = pffrocd.run_sfe_improved(x, y, y_0=share0, y_1=share1, x_0=share0prime, x_1=share1prime)
# print(output.stdout)

# print("NUMPY COS_DIST:")
# print(pffrocd.get_cos_dist_numpy(x,y))
# print(np.dot(x,y))
# # the dot product written out
# sum = 0
# for i in range(0, len(x)):
#     #print(f"x[{i}]: {x[i]} * y[{i}]: {y[i]} = {x[i]*y[i]}")
#     sum+=x[i]*y[i]
# print(sum)
