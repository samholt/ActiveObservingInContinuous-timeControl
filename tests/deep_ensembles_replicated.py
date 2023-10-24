import numpy as np
import random
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
SEED_CORE = 9
WEIGHTS_INITIALIZATION_STD=0.5
torch.manual_seed(SEED_CORE)
random.seed(SEED_CORE)
np.random.seed(SEED_CORE)
torch.use_deterministic_algorithms(True)

plt.rc('font', size=12)

# Not changing the permutation for each individual trained ensemble makes no noticeable impact on difference

# Function definitions

def seed_all(seed):
    """
    Set the torch, numpy, and random module seeds based on the seed
    specified in config.
    """
    # Set the seeds using the shifted seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=100):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim),
        )

        # for m in self.linear_tanh_stack.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)

        for m in self.linear_tanh_stack.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=WEIGHTS_INITIALIZATION_STD)

    def forward(self, x):
        return self.linear_tanh_stack(x)

class GaussianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=100):
        super(GaussianMLP, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim * 2),
        )

        # for m in self.linear_tanh_stack.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)

        for m in self.linear_tanh_stack.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=WEIGHTS_INITIALIZATION_STD)

    def forward(self, x):
        out = self.linear_tanh_stack(x)
        mean, variance = torch.split(out, self.output_dim, dim=1)
        variance = torch.nn.Softplus()(variance) + 1e-6
        return mean, variance

class EnsembleGaussianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=100, ensemble_size=5):
        super(EnsembleGaussianMLP, self).__init__()
        self.ensemble_size = ensemble_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.models = []
        for _ in range(ensemble_size):
            self.models.append(GaussianMLP(input_dim, output_dim, hidden_units))

    def forward(self, x):
        means, variances = [], []
        for model in self.models:
            mean, variance = model(x)
            means.append(mean)
            variances.append(variance)
        means = torch.stack(means)
        variances = torch.stack(variances)
        mean = means.mean(dim=0)
        variance = (variances + means**2).mean(dim=0) - mean**2
        return mean, variance

def gaussian_NLL(y, mean, variance):
    return (torch.log(variance) + ((y - mean)**2 / variance)).mean()

def generate_adversarial_example(model, Xi_natural, yi_natrual, epsilon, X_range) -> torch.Tensor:
    """
    fast gradient sign method
    """
    Xip = Xi_natural.detach()
    yip = yi_natrual.detach()
    Xip.requires_grad_()
    with torch.enable_grad():
        mean, variance = model(Xip)
        loss = gaussian_NLL(yip, mean, variance)
    grad = torch.autograd.grad(loss, [Xip])[0]
    Xip = Xip.detach() + epsilon * X_range * torch.sign(grad.detach())
    return Xip

# Core functions to test

def plot_one_mse_model(gt_X, gt_y, X, y, config):
    seed_all(config.seed)
    # One MSE model
    model = MLP(1,1)
    optim = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(config.epochs):
        optim.zero_grad()
        loss = loss_fn(y, model(X))
        if epoch == 0:
            print('initial loss: ', loss.item())
        loss.backward()
        optim.step()
    print('final loss: ',loss.item())

    # plt.plot(gt_X, gt_y, 'b-', label='ground truth: $y=x^3$')
    # plt.plot(X, y,'or', label='data points')
    # plt.grid()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.savefig('./tests/figs/gt.png')
    # plt.clf()

    plt.plot(gt_X, gt_y, 'b-', label='ground truth: $y=x^3$')
    plt.plot(X, y,'or', label='data points')
    plt.plot(gt_X, model(gt_X).detach().numpy(), label='MLP (MSE)', color='grey')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('./tests/figs/single_mlp.png')
    plt.clf()


def plot_ensemble_size_mse_models(gt_X, gt_y, X, y, config):
    seed_all(config.seed)
    # 5 MSE models
    models = []
    permutation = torch.randperm(X.size()[0])
    for i in range(config.ensemble_size):
        model = MLP(1,1)
        optim = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(config.epochs):
            optim.zero_grad()
            loss = loss_fn(y[permutation], model(X[permutation]))
            if epoch == 0:
                print(f'Network {i} initial loss: {loss.item()}')
            loss.backward()
            optim.step()
        print(f'Network {i} final loss: {loss.item()}')
        models.append(model)

    plt.plot(gt_X, gt_y, 'b-', label='ground truth: $y=x^3$')
    plt.plot(X, y,'or', label='data points')
    ys = []
    for net in models:
        ys.append(net(torch.tensor(gt_X).float()).clone().detach())
    ys = torch.stack(ys)
    mean = ys.mean(dim=0)
    std = ys.std(dim=0)
    plt.plot(gt_X, mean, label='MLP (MSE)', color='grey')
    plt.fill_between(gt_X.view(100,), (mean-3* std).view(100,), (mean+3*std).view(100,),color='grey',alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Figure 1 (left) from the paper')
    plt.legend()
    plt.savefig('./tests/figs/ensemble_size_mse_models.png')
    plt.clf()


def plot_one_gaussian_model(gt_X, gt_y, X, y, config):
    seed_all(config.seed)
    # One NLL model
    model = GaussianMLP(1,1)
    optim = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    permutation = torch.randperm(X.size()[0])
    Xi = X[permutation]
    yi = y[permutation]

    for epoch in range(config.epochs):
        optim.zero_grad()
        mean, variance = model(Xi)
        loss = gaussian_NLL(yi, mean, variance)
        if epoch == 0:
            print('initial loss: ', loss.item())
        loss.backward()
        optim.step()
    print('final loss: ',loss.item())

    mean, variance = model(gt_X)
    mean, std = mean.detach(), torch.sqrt(variance.detach())
    plt.plot(gt_X, gt_y, 'b-', label='ground truth: $y=x^3$')
    plt.plot(X, y,'or', label='data points')
    plt.plot(gt_X, mean, label='GMLP (NNL)', color='grey')
    plt.fill_between(gt_X.view(100,), (mean-3*std).view(100,), (mean+3*std).view(100,),color='grey',alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('./tests/figs/one_gaussian_model.png')
    plt.clf()


def plot_one_gaussian_model_with_adversarial_loss(gt_X, gt_y, X, y, config):
    seed_all(config.seed)
    # One NLL model with adversarial_loss
    model = GaussianMLP(1,1)
    optim = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    permutation = torch.randperm(X.size()[0])
    Xi = X[permutation]
    yi = y[permutation]

    for epoch in range(config.epochs):
        optim.zero_grad()
        mean, variance = model(Xi)
        loss = gaussian_NLL(Xi, mean, variance)
        # Adversarial example gen
        Xi_perturbed = generate_adversarial_example(model, Xi, yi, config.epsilon, config.X_range)
        mean, variance = model(Xi_perturbed)
        adversarial_loss = gaussian_NLL(yi, mean, variance)
        total_loss = (1 - config.adversarial_lambda) * loss + config.adversarial_lambda * adversarial_loss
        if epoch == 0:
            print('initial loss: ', total_loss.item())
        total_loss.backward()
        optim.step()
    print('final loss: ',loss.item())

    mean, variance = model(gt_X)
    mean, std = mean.detach(), torch.sqrt(variance.detach())
    plt.plot(gt_X, gt_y, 'b-', label='ground truth: $y=x^3$')
    plt.plot(X, y,'or', label='data points')
    plt.plot(gt_X, mean, label='GMLP (NNL) with adversarial_loss', color='grey')
    plt.fill_between(gt_X.view(100,), (mean-3*std).view(100,), (mean+3*std).view(100,),color='grey',alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('./tests/figs/one_gaussian_model_with_adversarial_loss.png')
    plt.clf()


def plot_ensemble_size_gaussian_model(gt_X, gt_y, X, y, config):
    seed_all(config.seed)
    # Ensemble GaussianMLP NLL models
    ensemble_model = EnsembleGaussianMLP(1,1, ensemble_size=config.ensemble_size)
    permutation = torch.randperm(X.size()[0])

    for i in range(ensemble_model.ensemble_size):
        model = ensemble_model.models[i]
        optim = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
        Xi = X[permutation]
        yi = y[permutation]
        for epoch in range(config.epochs):
            Xi.requires_grad = True
            optim.zero_grad()
            mean, variance = model(Xi)
            loss = gaussian_NLL(yi, mean, variance)
            total_loss = loss
            if epoch == 0:
                print(f'Network {i} initial loss: {total_loss.item()}')
            total_loss.backward()
            optim.step()
        print(f'Network {i} final loss: {total_loss.item()}')
        ensemble_model.models[i] = model

    mean, variance = ensemble_model(gt_X)
    mean, std = mean.detach(), torch.sqrt(variance.detach())
    plt.plot(gt_X, gt_y, 'b-', label='ground truth: $y=x^3$')
    plt.plot(X, y,'or', label='data points')
    plt.plot(gt_X, mean, label='Ensemble GMLP (NNL)', color='grey')
    plt.fill_between(gt_X.view(100,), (mean-3*std).view(100,), (mean+3*std).view(100,),color='grey',alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('./tests/figs/ensemble_size_gaussian_model.png')
    plt.clf()


    plt.plot(gt_X, gt_y, 'b-', label='ground truth: $y=x^3$')
    plt.plot(X, y,'or', label='data points')
    for i, net in enumerate(ensemble_model.models):
        mean, variance = net(gt_X)
        mean, std = mean.detach(), torch.sqrt(variance.detach())
        plt.plot(gt_X, mean, label=f'GMLP (NNL) {i+1}', alpha=0.5)
        plt.fill_between(gt_X.view(100,), (mean-3*std).view(100,), (mean+3*std).view(100,),alpha=0.1)
    plt.title('Outputs of the network in the ensemble')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('./tests/figs/ensemble_size_gaussian_model_individual.png')
    plt.clf()

def plot_ensemble_size_gaussian_model_with_adversarial_loss_v1(gt_X, gt_y, X, y, config):
    seed_all(config.seed)
    # Ensemble GaussianMLP NLL models with adversarial_loss
    ensemble_model = EnsembleGaussianMLP(1,1, ensemble_size=config.ensemble_size)
    permutation = torch.randperm(X.size()[0])

    for i in range(ensemble_model.ensemble_size):
        model = ensemble_model.models[i]
        optim = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
        Xi = X[permutation]
        yi = y[permutation]
        for epoch in range(config.epochs):
            Xi.requires_grad = True
            optim.zero_grad()
            mean, variance = model(Xi)
            loss = gaussian_NLL(yi, mean, variance)
            grad = torch.autograd.grad(loss, [Xi], retain_graph=True)[0]
            perturbed_X = Xi.detach() + config.epsilon * config.X_range * grad.detach().sign()
            mean, variance = model(perturbed_X)
            adversarial_loss = gaussian_NLL(yi, mean, variance)
            # total_loss = loss + adversarial_loss
            total_loss = (1 - config.adversarial_lambda) * loss + config.adversarial_lambda * adversarial_loss
            if epoch == 0:
                print(f'Network {i} initial loss: {total_loss.item()}')
            total_loss.backward()
            optim.step()
        print(f'Network {i} final loss: {total_loss.item()}')
        ensemble_model.models[i] = model

    mean, variance = ensemble_model(gt_X)
    mean, std = mean.detach(), torch.sqrt(variance.detach())
    plt.plot(gt_X, gt_y, 'b-', label='ground truth: $y=x^3$')
    plt.plot(X, y,'or', label='data points')
    plt.plot(gt_X, mean, label='Ensemble GMLP (NNL)', color='grey')
    plt.fill_between(gt_X.view(100,), (mean-3*std).view(100,), (mean+3*std).view(100,),color='grey',alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('./tests/figs/ensemble_size_gaussian_model_with_adversarial_loss_v1.png')
    plt.clf()

def plot_ensemble_size_gaussian_model_with_adversarial_loss_v2(gt_X, gt_y, X, y, config):
    seed_all(config.seed)
    # Ensemble GaussianMLP NLL models with adversarial_loss v2
    ensemble_model = EnsembleGaussianMLP(1,1, ensemble_size=config.ensemble_size)
    permutation = torch.randperm(X.size()[0])

    for i in range(ensemble_model.ensemble_size):
        model = ensemble_model.models[i]
        optim = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
        Xi = X[permutation]
        yi = y[permutation]
        for epoch in range(config.epochs):
            optim.zero_grad()
            mean, variance = model(Xi)
            loss = gaussian_NLL(yi, mean, variance)
            # Adversarial example gen
            Xi_perturbed = generate_adversarial_example(model, Xi, yi, config.epsilon, config.X_range)
            mean, variance = model(Xi_perturbed)
            adversarial_loss = gaussian_NLL(yi, mean, variance)
            total_loss = (1 - config.adversarial_lambda) * loss + config.adversarial_lambda * adversarial_loss
            if epoch == 0:
                print(f'Network {i} initial loss: {total_loss.item()}')
            total_loss.backward()
            optim.step()
        print(f'Network {i} final loss: {total_loss.item()}')
        ensemble_model.models[i] = model

    mean, variance = ensemble_model(gt_X)
    mean, std = mean.detach(), torch.sqrt(variance.detach())
    plt.plot(gt_X, gt_y, 'b-', label='ground truth: $y=x^3$')
    plt.plot(X, y,'or', label='data points')
    plt.plot(gt_X, mean, label='Ensemble GMLP (NNL)', color='grey')
    plt.fill_between(gt_X.view(100,), (mean-3*std).view(100,), (mean+3*std).view(100,),color='grey',alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('./tests/figs/ensemble_size_gaussian_model_with_adversarial_loss_v2.png')
    plt.clf()

# Main method
if __name__ == '__main__':
    seed_all(SEED_CORE)
    dirpath = Path('./tests/figs/')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    dirpath.mkdir()

    config = dotdict(dict(epochs = 500, # 40 original paper
                    samples = 20,
                    seed =  SEED_CORE,
                    learning_rate = 0.1,
                    epsilon = 0.01, # 0.01
                    X_range = 8, # Can derive this from the data
                    ensemble_size = 5,
                    adversarial_lambda = 0.5)) # [0,1]

    X = torch.FloatTensor(config.samples, 1).uniform_(-4, 4)
    y = X**3 + torch.normal(mean=0, std=3, size=(config.samples, 1))

    # Ground truth data
    gt_X = torch.linspace(-6, 6, 100).view(100, 1)
    gt_y = gt_X**3

    plot_one_mse_model(gt_X.detach().clone(), gt_y.detach().clone(), X.detach().clone(), y.detach().clone(), config)
    plot_ensemble_size_mse_models(gt_X.detach().clone(), gt_y.detach().clone(), X.detach().clone(), y.detach().clone(), config)
    plot_one_gaussian_model(gt_X.detach().clone(), gt_y.detach().clone(), X.detach().clone(), y.detach().clone(), config)
    plot_one_gaussian_model_with_adversarial_loss(gt_X.detach().clone(), gt_y.detach().clone(), X.detach().clone(), y.detach().clone(), config)
    plot_ensemble_size_gaussian_model(gt_X.detach().clone(), gt_y.detach().clone(), X.detach().clone(), y.detach().clone(), config)
    plot_ensemble_size_gaussian_model_with_adversarial_loss_v1(gt_X.detach().clone(), gt_y.detach().clone(), X.detach().clone(), y.detach().clone(), config)
    plot_ensemble_size_gaussian_model_with_adversarial_loss_v2(gt_X.detach().clone(), gt_y.detach().clone(), X.detach().clone(), y.detach().clone(), config)

    print('fin.')


