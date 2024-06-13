#!/usr/bin/env python3
print("importing pffrocd...")
import pffrocd # helper functions
print("pffrocd imported!")
import numpy as np
import random
import time
import quantization as qt
import sys
import os

# def evaluate_quantization(vector1, vector2, quantization_func):
#     # Compute the cosine similarity before quantization
#     #x = x / np.linalg.norm(x)
#     before_quantization = cosine_similarity(vector1, vector2)

#     # Quantize the vectors
#     quantized_vector1 = quantization_func(vector1)
#     quantized_vector2 = quantization_func(vector2)
#     print(quantized_vector1)

#     # Compute the cosine similarity after quantization
#     after_quantization = cosine_similarity(quantized_vector1, quantized_vector2)

#     # Return the cosine similarities before and after quantization
#     return before_quantization, after_quantization

# pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
# pffrocd.EXECUTABLE_NAME = 'cos_dist_int_scen_simd'
# pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"
# pffrocd.OUTPUT_FILE_NAME = f"/home/chiem/pffrocd"
NUMPY_DTYPE = np.float32

def cosine_similarity(v1, v2):
    # Compute the cosine similarity
    return 1- np.dot(v1, v2)

def get_two_random_images(same_person):
    """Get two random embeddings of either the same person or two different people out of all the images available"""
    people = [p for p in os.listdir('lfw') if os.path.isdir(os.path.join('lfw', p))] # list of all people that have images
    people_with_multiple_images = [p for p in people if len([img for img in os.listdir(os.path.join("lfw", p)) if img != '.DS_Store']) > 1]  # list of people with more than one image in folder
    img1, img2 = None, None # face embeddings
    while img1 is None or img2 is None: # try until the chosen images have detectable faces
        try:
            if same_person:
                # same person should have more than one image (we might still end up choosing the same image of that person with prob 1/n, but that's ok)
                person1 = random.choice(people_with_multiple_images)
                person2 = person1
            else:
                # two persons chosen should be different
                person1 = random.choice(people)
                person2 = random.choice([p for p in people if p != person1])
            # get two random images
            img1 = f"lfw/{person1}/{random.choice(os.listdir(f'lfw/{person1}'))}"
            img2 = f"lfw/{person2}/{random.choice(os.listdir(f'lfw/{person2}'))}"
        except Exception as e:
            # failed to detect faces in images, try again
            # print(e)
            pass

    return img1,img2

img1, img2 = get_two_random_images(True)
x = pffrocd.get_embedding(img1, dtype=np.float32)
y = pffrocd.get_embedding(img2, dtype=np.float32)
x = x / np.linalg.norm(x)
y = y / np.linalg.norm(y)
# multiply each item in x and y (which are np arrays) by 65535
max_value = np.iinfo(np.uint8).max
print("before max: ", x[0])
x = x * 1000
y = y * 1000
print("after max: ", x[0])

# now convert x and y to uint16
x = np.array(x, dtype=np.uint8)
y = np.array(y, dtype=np.uint8)
print("after max uint: ", x[0])
# Compute the cosine similarity
def cosine_similarity(v1, v2):
    print(np.dot(v1, v2))
    return 1 - (np.dot(v1, v2) / 100)

print("cosine distance uint8: ", cosine_similarity(x, y))

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
