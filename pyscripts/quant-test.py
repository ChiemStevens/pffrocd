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

x,y = pffrocd.get_two_random_embeddings(False)
print(x)
print(y)