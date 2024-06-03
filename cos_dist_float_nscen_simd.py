#!/usr/bin/env python3
import pffrocd # helper functions
import numpy as np

pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_float_nscen_simd'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"

# get two embeddings of different people
x,y=pffrocd.get_two_random_embeddings(False)


print("EMBEDDING X:")
print(x)

print("EMBEDDING Y:")
print(y)

print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))

x = x / np.linalg.norm(x)
y = y / np.linalg.norm(y)

print("EMBEDDING X:")
print(x)

print("EMBEDDING Y:")
print(y)

print("Normalize COS_DIST:")
print(1 - np.dot(x, y))


output = pffrocd.run_sfe(x, y)




print(output.stdout)