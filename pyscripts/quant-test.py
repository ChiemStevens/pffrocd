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
import pandas as pd

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

def cosine_similarity(v1, v2):
    # Compute the cosine similarity
    return 1- np.dot(v1, v2)

def quant_uint32_cos_sim(x,y):
    x = x * 1000
    y = y * 1000
    x = np.array(x, dtype=np.uint32)
    y = np.array(y, dtype=np.uint32)
    return 1 - (np.dot(x, y) / 1000000)

def quant_uint16_cos_sim(x,y):
    x = x * 100
    y = y * 100
    x = np.array(x, dtype=np.uint16)
    y = np.array(y, dtype=np.uint16)
    return 1 - (np.dot(x, y) / 10000)

def quant_uint8_cos_sim(x,y):
    x = x * 1000
    y = y * 1000
    x = np.array(x, dtype=np.uint8)
    y = np.array(y, dtype=np.uint8)
    return 1 - (np.dot(x, y) / 1000)

# create a dataframe with pandas with columns for each cosine similarity
df = pd.DataFrame(columns=['cosine_numpy', 'cosine_uint32', 'cosine_uint16', 'cosine_uint8'])
for i in range(100):
    img1, img2 = get_two_random_images(True)
    x = pffrocd.get_embedding(img1, dtype=np.float32)
    y = pffrocd.get_embedding(img2, dtype=np.float32)
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    print("cosine distance: ", cosine_similarity(x, y))
    print("cosine distance uint32: ", quant_uint32_cos_sim(x, y))
    print("cosine distance uint16: ", quant_uint16_cos_sim(x, y))
    print("cosine distance uint8: ", quant_uint8_cos_sim(x, y))

    # append values to the dataframe
    df = df.append({'cosine_numpy': cosine_similarity(x, y), 'cosine_uint32': quant_uint32_cos_sim(x, y), 'cosine_uint16': quant_uint16_cos_sim(x, y), 'cosine_uint8': quant_uint8_cos_sim(x, y)}, ignore_index=True)

# save the dataframe to a csv file
df.to_csv('cosine_similarity.csv')