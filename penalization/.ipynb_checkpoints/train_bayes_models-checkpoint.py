import sys
import os

sys.path.append('../')

from tqdm import tqdm


from torchvision import transforms, datasets


import torch
import torch.nn as nn

from simulationmodel import BayesFFNN, BayesLinear
from sampling_utils import *
from fast_mnist import *



# Create FFNN models with the given dimensions
input_size = 28 * 28
num_hidden = 3
output_size = 10

epochs = 50
p_list = [500, 1000, 2000]

device = 'cuda'

# Loading / Dowloading MNIST data
#train = datasets.MNIST("../", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)

#test = datasets.MNIST("../", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#testset = torch.utils.data.DataLoader(test, batch_size=100, shuffle=True)

train = FastMNIST("../", train=True, download=True, device=device)
trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)

test = FastMNIST("../", train=False, download=True, device=device)
testset = torch.utils.data.DataLoader(test, batch_size=100)

path = 'results_h_3_corrected'
if not os.path.isdir(path):
    os.makedirs(path)
    

for r in range(1, 6):
    for p in p_list:

        normal_ffnn = BayesFFNN(input_size, num_hidden, p, output_size)
        iid_rvs = lam_dist(p, 'iid')
        # this initialises the weights of the nn
        normal_ffnn.init_weights(iid_rvs, kappa=1, sigma_b=1)

        horseshoe_ffnn = BayesFFNN(input_size, num_hidden, p, output_size)
        horseshoe_rvs = lam_dist(p, 'horseshoe')
        # this initialises the weights of the nn
        horseshoe_ffnn.init_weights(iid_rvs, kappa=1, sigma_b=1)
        horseshoe_ffnn.set_prior(horseshoe_rvs)


        gbfry_light_ffnn = BayesFFNN(input_size, num_hidden, p, output_size)
        light_rvs = GBFRYInit(alpha=0.2, tau=5)
        # this initialises the weights of the nn
        gbfry_light_ffnn.init_weights(iid_rvs, kappa=1, sigma_b=1)
        gbfry_light_ffnn.set_prior(light_rvs)

        gbfry_mid_ffnn = BayesFFNN(input_size, num_hidden, p, output_size)
        mid_rvs = GBFRYInit(alpha=0.5, tau=5)
        # this initialises the weights of the nn
        gbfry_mid_ffnn.init_weights(iid_rvs, kappa=1, sigma_b=1)
        gbfry_mid_ffnn.set_prior(mid_rvs)
        
        gbfry_heavy_ffnn = BayesFFNN(input_size, num_hidden, p, output_size)
        heavy_rvs = GBFRYInit(alpha=0.8,  tau=5)
        # this initialises the weights of the nn
        gbfry_heavy_ffnn.init_weights(iid_rvs, kappa=1, sigma_b=1)
        gbfry_heavy_ffnn.set_prior(heavy_rvs)

        # Train the models
        def train_model(simple_net, epochs, trainset, testset):
            optimizer = torch.optim.Adam(params=simple_net.parameters(), lr=1e-2)
            simple_net.to(device)
            simple_net.double()
            for e in tqdm(range(epochs)):

                for g in optimizer.param_groups:
                    if e < 10:
                        g['lr'] = 1e-2 
                    elif e < 20:
                        g['lr'] = 5e-3
                    else:
                        g['lr'] = 1e-3

                simple_net.train()
                for X, y in trainset:
                    X = X.to(device)
                    y = y.to(device)

                    simple_net.zero_grad()
                    pred_dist = simple_net(X.view(-1, 28 * 28))

                    l = nn.CrossEntropyLoss(reduction="mean")
                    log_prior_term = simple_net.log_prior()/60000/5
                    loss = l(pred_dist, y)-log_prior_term
                    loss.backward()
                    optimizer.step()

                print("Training loss = ", loss)

                simple_net.eval()
                accurate = 0
                total = 0
                with torch.no_grad():
                    for X, y in testset:
                        X = X.to(device)
                        y = y.to(device)

                        pred_dist = simple_net(X.view(-1, 28 * 28))
                        pred_y = torch.argmax(pred_dist, dim=-1)
                        accurate += torch.sum(pred_y == y).to('cpu').numpy()
                        total += y.shape[0]
                print("Accuracy on test set: ", round(accurate / total * 100, 1))


        #train_model(normal_ffnn, epochs, trainset, testset)
        #torch.save(normal_ffnn, '{}/normal_ffnn_{}_run_{}.net'.format(path, p, r))

        #train_model(gbfry_light_ffnn, epochs, trainset, testset)
        #torch.save(gbfry_light_ffnn, '{}/gbfry_light_ffnn_{}_run_{}.net'.format(path, p, r))
        
        train_model(gbfry_mid_ffnn, epochs, trainset, testset)
        torch.save(gbfry_mid_ffnn, '{}/gbfry_mid_ffnn_{}_run_{}.net'.format(path, p, r))

        #train_model(horseshoe_ffnn, epochs, trainset, testset)
        #torch.save(horseshoe_ffnn, '{}/horseshoe_ffnn_{}_run_{}.net'.format(path, p, r))


        #train_model(gbfry_heavy_ffnn, epochs, trainset, testset)
        #torch.save(gbfry_heavy_ffnn, '{}/gbfry_heavy_ffnn_{}_run_{}.net'.format(path, p, r))
    
    


