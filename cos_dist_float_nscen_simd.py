#!/usr/bin/env python3
import pffrocd # helper functions
import numpy as np

pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_float_scen_simd_32'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"
pffrocd.OUTPUT_FILE_NAME = f"output_{pffrocd.EXECUTABLE_NAME}.txt"

# get two embeddings of different people
x,y=pffrocd.get_two_random_embeddings(False)
share0, share1 = pffrocd.create_shares(y, dtype=np.float32)

# Print out the cosine distance for verificaiton before the normalization happens
print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))

# Normalize the embeddings
x = x / np.linalg.norm(x)
y = y / np.linalg.norm(y)

# Print out the cosine distance for verificaiton after the normalization happens
print("Normalize COS_DIST:")
print(1 - np.dot(x, y))

# Run the circuit
output = pffrocd.run_sfe(x, y, y_0=share0, y_1=share1)


# get two embeddings of different people
# print("Getting two random embeddings...")
# x, y = pffrocd.get_two_random_embeddings(False)
# print("got the embeddings!")

# r = pffrocd.generate_nonce(y)

# y1 = r

# y0 = pffrocd.fxor(y, y1)


# output = pffrocd.run_sfe(x, y, y_0=y0, y_1=y1)

# print(output.stdout)

# Print the output
print(output.stdout)