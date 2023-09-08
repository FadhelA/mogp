import torch

from tqdm import tqdm
import numpy as np
import sys

sys.path.append("../")
from simulationmodel import FFNN
from sampling_utils import lam_sampler, sample_finite_GBFRY

# Save file name
save_name = "samples_shallow_large_gbfry.npy"

# Create a FFNN with the given dimensions
input_size = 1
num_hidden = 0
output_size = 2

x = torch.ones(size=(input_size,))

p_list = [100, 500, 1000, 2000]
n_samples = 50000
n_samples_horseshoe = 5000
kappa = 1  # np.sqrt(2)
dist_list = ['gbfry', 'invgamma', 'beta', 'bernoulli', 'horseshoe', 'iid',
             'gbfry_heavy_light', 'gbfry_light_light', 'gbfry_light_heavy', 'gbfry_heavy_heavy'
            ]
dist_list = [
             'gbfry_heavy_light', 'gbfry_light_light', 'gbfry_light_heavy', 'gbfry_heavy_heavy'
            ]

samples = dict()

for t_p, p in enumerate(p_list):
    print("Width {}/{}".format(t_p+1, len(p_list)))
    samples[p] = dict()
    # Initialize the architecture
    nn = FFNN(input_size, num_hidden, p, output_size)
    nn.eval()
    for t, dist in enumerate(dist_list):
        print("Distribution {} / {}".format(t + 1, len(dist_list)))

        n_s = n_samples_horseshoe if dist == 'horseshoe' else n_samples
        samples[p][dist] = np.zeros((n_s, output_size))
        for s in tqdm(range(n_s)):
            # lam_rvs is a function to sample the variances
            lam_rvs = lam_sampler(p, dist)
            # this initialises the weights of the nn
            nn.init_weights(lam_rvs, kappa=kappa, sigma_b=0)

            samples[p][dist][s, :] = nn(x).detach().numpy()

np.save(save_name, samples)
