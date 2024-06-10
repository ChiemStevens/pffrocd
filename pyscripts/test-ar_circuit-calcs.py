# party_0 = {x0-r0, x1-r1, x2-r2};
# party_1 = {r0, r1, r2};

# output :
# Result = x0 * x1 * x2;


import pffrocd # helper functions
import numpy as np

# get two random int vectors
x, y = pffrocd.get_two_random_embeddings(False)

share0, share1 = pffrocd.create_shares(x, np.float32)
share0prime, share1prime = pffrocd.create_shares(y, np.float32)

# get the shares back
result = share0 - share1
result_prime = share0prime - share1prime

# cosine distance between the shares
result_cosine = pffrocd.cosine_distance(result, result_prime)
