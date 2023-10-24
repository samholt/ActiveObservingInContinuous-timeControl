# Probabilistic ensemble model
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
WEIGHTS_INITIALIZATION_STD=0.5
VARIANCE_EPS = 1e-6
TORCH_PRECISION = torch.float32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()

class GaussianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=128,
                model_activation='tanh',
                state_std=None,
                model_initialization='xavier',
                ):
        super(GaussianMLP, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.register_buffer("state_std", torch.tensor(state_std, dtype=TORCH_PRECISION))

        if model_activation == 'tanh':
            activation = nn.Tanh
        elif model_activation == 'silu':
            activation = nn.SiLU
        else:
            raise NotImplementedError

        self.stack = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            activation(),
            nn.Linear(hidden_units, hidden_units),
            activation(),
            nn.Linear(hidden_units, hidden_units),
            activation(),
            nn.Linear(hidden_units, output_dim * 2),
        )
        
        for m in self.stack.modules():
            if isinstance(m, nn.Linear):
                if model_initialization=='xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif model_initialization=='normal':
                    nn.init.normal_(m.weight, mean=0, std=WEIGHTS_INITIALIZATION_STD)

    def forward(self, x):
        out = self.stack(x)
        mean, log_var = torch.split(out, self.output_dim, dim=1)
        mean = mean * self.state_std
        log_var = log_var + 2.0 * torch.log(self.state_std)
        return mean, log_var
    
class ProbabilisticEnsemble(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_units=500,
                 ensemble_size=5,
                 encode_obs_time=False,
                 state_mean=None,
                 state_std=None,
                 action_mean=None,
                 action_std=None,
                 normalize=False,
                 normalize_time=False,
                 dt=0.05,
                 model_activation='tanh',
                 model_initialization='xavier',
                 model_pe_use_pets_log_var=True,
                 discrete=False,
                 ):
        super(ProbabilisticEnsemble, self).__init__()
        assert ensemble_size >= 1, "Ensemble size must be at least one"
        self.encode_obs_time = encode_obs_time
        self.output_dim = state_dim
        self.normalize = normalize
        self.normalize_time = normalize_time
        self.ensemble_size = ensemble_size
        self.discrete = discrete
        if not discrete:
            input_dim = state_dim + action_dim + 1
        else:
            input_dim = state_dim + action_dim
        self.register_buffer("state_mean", torch.tensor(state_mean, dtype=TORCH_PRECISION))
        self.register_buffer("state_std", torch.tensor(state_std, dtype=TORCH_PRECISION))
        self.register_buffer("action_mean", torch.tensor(action_mean, dtype=TORCH_PRECISION))
        self.register_buffer("action_std", torch.tensor(action_std, dtype=TORCH_PRECISION))
        self.register_buffer("dt", torch.tensor(dt, dtype=TORCH_PRECISION))
        self.model_pe_use_pets_log_var = model_pe_use_pets_log_var
        if model_pe_use_pets_log_var:
            self.max_logvar = nn.Parameter(torch.ones(1, state_dim, dtype=TORCH_PRECISION) / 2.0)
            self.min_logvar = nn.Parameter(-torch.ones(1, state_dim, dtype=TORCH_PRECISION) * 10.0)


        self.models = nn.ModuleList([GaussianMLP(input_dim,
                                                state_dim,
                                                hidden_units=hidden_units,
                                                model_activation=model_activation,
                                                state_std=state_std,
                                                model_initialization=model_initialization) for _ in range(ensemble_size)])
        
    def _forward_ensemble_separate(self, in_batch_obs, in_batch_action, ts_pred):
        if self.normalize:
            batch_obs = (in_batch_obs - self.state_mean) / self.state_std
            batch_action = (in_batch_action - self.action_mean) / self.action_std
            if self.normalize_time:
                ts_pred = (ts_pred / (self.dt*8.0))
        else:
            batch_obs = in_batch_obs
            batch_action = in_batch_action / 3.0
        if not self.discrete:
            sa_in = torch.cat((batch_obs, batch_action, ts_pred), axis=1)
        else:
            sa_in = torch.cat((batch_obs, batch_action), axis=1)
        outputs = [self.models[j](sa_in) for j in range(self.ensemble_size)]
        means = torch.stack([o[0] for o in outputs])
        if self.model_pe_use_pets_log_var:
            log_variances = torch.stack([o[1] for o in outputs])
            log_variances = self.max_logvar - F.softplus(self.max_logvar - log_variances)
            log_variances = self.min_logvar + F.softplus(log_variances - self.min_logvar)
            return means, log_variances
        else:
            raise NotImplementedError
            variances = torch.stack([torch.nn.Softplus()(o[1]) + VARIANCE_EPS for o in outputs])
            return means, variances

    def forward(self, in_batch_obs, in_batch_action, ts_pred):
        if self.model_pe_use_pets_log_var:
            mean_for_each_network, log_variance_for_each_network = self._forward_ensemble_separate(in_batch_obs, in_batch_action, ts_pred)
            mean = mean_for_each_network.mean(dim=0)
            variance = (torch.exp(log_variance_for_each_network) + torch.square(mean_for_each_network)).mean(dim=0) - torch.square(mean)
            return mean, variance
        else:
            mean_for_each_network, variance_for_each_network = self._forward_ensemble_separate(in_batch_obs, in_batch_action, ts_pred)
            mean = mean_for_each_network.mean(dim=0)
            variance = (variance_for_each_network + torch.square(mean_for_each_network)).mean(dim=0) - torch.square(mean)
            return mean, variance