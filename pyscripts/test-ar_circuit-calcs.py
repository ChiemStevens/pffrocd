# party_0 = {x0-r0, x1-r1, x2-r2};
# party_1 = {r0, r1, r2};

# output :
# Result = x0 * x1 * x2;
def get_cos_dist_numpy(x, y):
    return 1 - np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

def get_cos_dist_nom(x, y):
    return 1 - np.dot(x, y)

import pffrocd # helper functions
import numpy as np

# get two random int vectors
x = pffrocd.get_embedding("/home/chiem/pffrocd/lfw/German_Khan/German_Khan_0001.jpg", np.float32)
y = pffrocd.get_embedding("/home/chiem/pffrocd/lfw/Gina_Centrello/Gina_Centrello_0001.jpg", np.float32)

share0, share1 = pffrocd.create_shares(x, np.float32)
share0prime, share1prime = pffrocd.create_shares(y, np.float32)

# get the shares back
result = np.float32(share0+share1)
result_prime = np.float32(share0prime + share1prime)

# cosine distance between the shares
result_cosine = get_cos_dist_numpy(result, result_prime)
print(result_cosine)

print(get_cos_dist_numpy(x,y))
