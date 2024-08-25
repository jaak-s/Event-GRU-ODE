import torch
import math
import numpy as np

from torch import Tensor

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm  = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def grad_max(model):
    total_max = 0.0
    for p in model.parameters():
        param_max  = p.grad.data.max()
        if param_max > total_max:
            total_max = param_max
    return total_max

def dict2obj(d):
    return type("D", (object,), d)

class GaussianLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, delta_norm, logstd, M_obs):
        return 0.5 * ((torch.pow(delta_norm, 2) + 2 * logstd) * M_obs)

class LaplaceLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, delta_norm, logstd, M_obs):
        L = (torch.abs(delta_norm) + logstd) * M_obs
        if self.reduction == "mean":
            return L.sum() / M_obs.sum()
        if self.reduction == "sum":
            return L.sum()
        if self.reduction == "none":
            return L
        raise ValueError(f"Unknown reduction '{self.reduction}'.")

class L1LossMasked(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask):
        L = torch.abs(input - target) * mask
        return L.sum() / mask.sum()

class GRUBayes(torch.nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size, prep_hidden_size, bias=True):
        super().__init__()
        self.gru_d     = torch.nn.GRUCell((prep_hidden_size + 1) * input_size, hidden_size, bias=bias)

        ## prep layer and its initialization
        std            = math.sqrt(2.0 / (4 + prep_hidden_size))
        self.w_prep    = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden_size))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden_size))

        self.input_size  = input_size
        self.prep_hidden_size = prep_hidden_size

    def forward(self, h, gru_input, M_obs, i_obs):
        gru_input = gru_input.unsqueeze(2)
        gru_input = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is [sample x feature x prep_hidden_size]
        ## M_obs is     [sample x feature]
        gru_input = (gru_input.permute(2, 0, 1) * M_obs).permute(1, 2, 0)

        ## concatenating observation mask to gru_input
        gru_input = torch.cat([gru_input, M_obs.unsqueeze(-1)], dim=-1)
        gru_input = gru_input.view(-1, (self.prep_hidden_size + 1) * self.input_size)

        if i_obs is None:
            ## processing all
            return self.gru_d(gru_input, h)

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h

class GRUBayesSimple(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.gru_d = torch.nn.GRUCell(input_size, hidden_size, bias=bias)

    def forward(self, h, x):
        return self.gru_d(x, h)

def compute_KL_loss(mean, logstd, X_obs, M_obs, obs_noise_std=1e-2):
    obs_noise_std = torch.tensor(obs_noise_std)
    #return (gaussian_KL(mu_1 = mean, mu_2 = X_obs, logsigma_1 = logstd, logsigma_2 = torch.log(obs_noise_std))*M_obs).sum()

    ## using KL(p_obs || p_predicted)
    return (gaussian_KL(mu_2 = mean, mu_1 = X_obs, logsigma_2 = logstd, logsigma_1 = torch.log(obs_noise_std))*M_obs).sum()

def gaussian_KL(mu_1, mu_2, logsigma_1, logsigma_2):
    return logsigma_2 - logsigma_1 + (torch.exp(2*logsigma_1) + torch.pow((mu_1 - mu_2),2)) / (2 * torch.exp(2*logsigma_2))


def p_to_pred(p, X):
    """Takes in predictions p and X.
    Returns predicted mean, logstd, delta, delta_norm"""
    mean, logstd = torch.chunk(p, 2, dim=1)
    sigma        = torch.exp(logstd)

    X2           = X.clone()
    mask_na      = torch.isnan(X)
    X2[mask_na]  = mean.detach()[mask_na]

    delta        = X2 - mean
    delta_norm   = delta / sigma

    return mean, logstd, delta, delta_norm


class Discretized_GRU(torch.nn.Module):
    """
    Args:
    impute  if True feeds back p into gru, if False then autonomous ODE
    """
    ## Discretized GRU model (GRU-ODE-Bayes without ODE but with Bayes)
    #def __init__(self, input_size, hidden_size, p_hidden_size, prep_hidden_size, bias=True, cov_size=1, cov_hidden=1, classification_hidden=1, mixing=1, dropout_rate=0, impute=True):
    def __init__(self, conf):
        
        super().__init__()
        self.impute = conf.impute

        if   conf.loss == "L2":  self.loss = GaussianLoss()
        elif conf.loss == "L1":  self.loss = LaplaceLoss()
        else:  raise ValueError(f"Loss '{conf.loss}' unknown.")

        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(conf.hidden_size, p_hidden_size, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=conf.dropout_rate),
            torch.nn.Linear(conf.p_hidden_size, 2 * conf.input_size, bias=bias),
        )

        self.gru = torch.nn.GRUCell(2*conf.input_size, conf.hidden_size, bias = bias)
        self.gru_bayes = GRUBayes(conf.input_size, conf.hidden_size, conf.prep_hidden_size, bias=True)

        self.covariates_map = torch.nn.Sequential(
            torch.nn.Linear(conf.cov_size, conf.cov_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=conf.dropout_rate),
            torch.nn.Linear(conf.cov_hidden, conf.hidden_size, bias=bias),
            torch.nn.Tanh()
        )

        self.input_size = conf.input_size
        self.mixing     = conf.mixing #mixing hyperparameter for loss_1 and loss_2 aggregation.
        self.pre_forecast_weight = conf.pre_forecast_weight
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, times, time_ptr, X, M, obs_idx, dt,
                Tpred, cov,
                return_path=False):
        """
        Args:
            times      np vector of observation times
            time_ptr   start indices of data for a given time
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            obs_idx    observed patients of each datapoint (indexed within the current minibatch)
            dt         time step for Euler
            Tmax       total time
            cov        static covariates for learning the first h0
            return_path   whether to return the path of h

        Returns:
            h          hidden state at final time (Tmax)
            loss       loss of the Gaussian observations
        """
        h = self.covariates_map(cov)

        p            = self.p_model(h)
        current_time = 0.0
        counter      = 0

        loss_1a = 0 #Pre-jump loss, with Bayes updates
        loss_1b = 0 #Pre-jump loss, without Bayes updates
        loss_2  = 0 #Post-jump loss (KL between p_updated and the actual sample)

        if return_path:
            path_t = [0]
            path_p = [p]
            path_h = [h]

        assert len(times) + 1 == len(time_ptr)
        assert (len(times) == 0) or (times[-1] <= Tmax)

        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until next observation
            while current_time < (obs_time - 0.001 * dt): #0.001 * dt used for numerical consistency.
                
                if self.impute is False:
                    p = torch.zeros_like(p)
                h = self.gru(p, h)
                p = self.p_model(h)

                ## using counter to avoid numerical errors
                counter += 1
                current_time = counter * dt

            ## Reached an observation
            start = time_ptr[i]
            end   = time_ptr[i+1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = obs_idx[start:end]

            ## Calculating pre-jump loss:
            mean, logstd, delta, delta_norm = p_to_pred(p[i_obs], X_obs)
            losses    = self.loss(delta_norm, logstd, M_obs)
            use_bayes = obs_time < 0.001 * dt

            ## Using GRUBayes to update h.
            if use_bayes:
                loss_1a   = loss_1a + losses.sum()
                gru_input = torch.stack([mean, logstd, delta, delta_norm], dim=2)
                h         = self.gru_bayes(h, p, M_obs, i_obs)

                import ipdb; ipdb.set_trace()

                p      = self.p_model(h)
                mean, logstd, delta, delta_norm = p_to_pred(p[i_obs], X_obs)
                loss_2 = loss_2 + compute_KL_loss(
                                    mean   = mean,
                                    logstd = logstd,
                                    X_obs  = X_obs,
                                    M_obs  = M_obs,
                                    obs_noise_std=conf.obs_noise_std)
            else:
                loss_1b = loss_1b + losses.sum()

                if return_path:
                    path_t.append(obs_time)
                    path_p.append(p)
                    path_h.append(h)

        ## after every observation has been processed, propagating until Tmax
        while current_time < Tmax - 0.001 * dt:
            if self.impute is False:
                p = torch.zeros_like(p)
            h = self.gru(p,h)
            p = self.p_model(h)

            counter += 1
            current_time = counter * dt
            #Storing the predictions
            if return_path:
                path_t.append(current_time)
                path_p.append(p)
                path_h.append(h)

        loss = self.pre_forecast_weight*(loss_1a + self.mixing * loss_2) + loss_1b
       
        if return_path:
            return h, loss, np.array(path_t), torch.stack(path_p), torch.stack(path_h)
        else:
            return h, loss 


class DiscreteGRUBayes(torch.nn.Module):
    """
    Args:
    impute  if True feeds back p into gru, if False then autonomous ODE
    """
    ## Discretized GRU model (GRU-ODE-Bayes without ODE but with Bayes)
    def __init__(self, conf):
        
        super().__init__()
        self.impute     = conf.impute

        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(conf.hidden_size, conf.p_hidden_size, bias=True),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=conf.dropout_rate),
            torch.nn.Linear(conf.p_hidden_size, 2 * conf.input_size, bias=True),
        )

        self.gru_next  = torch.nn.GRUCell(2 * conf.input_size, conf.hidden_size, bias = True)
        self.gru_bayes = GRUBayes(conf.input_size, conf.hidden_size, conf.prep_hidden_size, bias=True)

        self.covariates_map = torch.nn.Sequential(
            torch.nn.Linear(conf.cov_size, conf.cov_hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=conf.dropout_rate),
            torch.nn.Linear(conf.cov_hidden_size, conf.hidden_size, bias=True),
            torch.nn.Tanh()
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, X, num_observed, cov):
        """
        Args:
            X              input tensor [time, batch, feature]
            num_observed   how many future time steps to generate
            cov            const features [batch, cov_feature]

        Returns:
            pre        pre-jump trajectory. Last dim: [mean, logstd, delta, delta_norm]
            post       post-jump trajectory. Last dim: [mean, logstd, delta, delta_norm]
        """
        h = self.covariates_map(cov)
        p = self.p_model(h)

        out_pre  = []
        out_post = []

        for t in range(X.shape[0]):
            Xt = X[t]

            ## do step for all hiddens with the Xi
            if self.impute is False:
                p = torch.zeros_like(p)
            h = self.gru_next(p, h)
            p = self.p_model(h)

            ## Calculating pre-jump predictions:
            pre = torch.stack(p_to_pred(p, Xt), dim=-1)
            out_pre.append(pre)

            if t < num_observed:
                ## Using GRUBayes to update h.
                mask = (1 - torch.isnan(Xt)).float()
                h    = self.gru_bayes(h, pre, M_obs = mask, i_obs = None)
                p    = self.p_model(h)
                post = torch.stack(p_to_pred(p, Xt), dim=-1)
                out_post.append(post)

        out_pre  = torch.stack(out_pre, dim=0)
        out_post = torch.stack(out_post, dim=0)
        return out_pre, out_post

class GRUODECell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, delta_t, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias        = bias

        self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        self.delta_t = delta_t

    def forward(self, x, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            x        input values
            h        hidden state (current)
            delta_t  time step

        Returns:
            Updated h
        """
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        n = torch.tanh(self.lin_xn(x) + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return h + self.delta_t * dh

class FullGRUODECell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, delta_t:float, bias=True):
        """
        The step size is given by delta_t.
        """
        super().__init__()

        self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        self.delta_t = delta_t

    def forward(self, x, h):
        """
        Executes one step with GRU-ODE for all h.

        Args:
            x        input values
            h        hidden state (current)

        Returns:
            Updated h
        """
        xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        r = torch.sigmoid(xr + self.lin_hr(h))
        z = torch.sigmoid(xz + self.lin_hz(h))
        u = torch.tanh(xh + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        ## Euler:
        return h + self.delta_t * dh

class EventGRUBayes(torch.nn.Module):
    dt : float
    """
    Args:
    """
    ## Event GRU model (GRU-ODE-Bayes without ODE but with Bayes)
    def __init__(self, conf):
        super().__init__()

        if conf.model == "gru_discrete":
            self.gru_next = torch.nn.GRUCell(1, conf.hidden_size, bias = True)
        elif conf.model == "gru_ode":
            delta_t = conf.dt / conf.ode_slower
            self.gru_next = FullGRUODECell(1, conf.hidden_size, bias = True, delta_t = delta_t)
        else:
            raise ValueError(f"Unknown model '{conf.model}'.")

        self.gru_bayes = torch.nn.GRUCell(conf.input_size, conf.hidden_size, bias=True)
        self.net_out   = torch.nn.Sequential(
            torch.nn.Linear(conf.hidden_size, conf.output_hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=conf.output_dropout),
            torch.nn.Linear(conf.output_hidden_size, conf.output_size, bias=True),
        )

        self.conf = conf
        self.dt = conf.dt

        if conf.cov_size >= 1:
            self.covariates_map = torch.nn.Sequential(
                torch.nn.Linear(conf.cov_size, conf.cov_hidden_size, bias=True),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=conf.cov_dropout),
                torch.nn.Linear(conf.cov_hidden_size, conf.hidden_size, bias=True),
                torch.nn.Tanh()
            )
        else:
            ## cov map assumes inputs is (0s)
            self.covariates_map = torch.nn.Linear(1, conf.hidden_size, bias=True)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self,
            time_uniq:  Tensor,
            time_ptr:   Tensor,
            X:          Tensor,
            sample_ids: Tensor,
            T:          float,
            covs:       Tensor):
        """
        Args:
            X              input tensor [time_sample, feature]
            num_observed   how many future time steps to generate
            covs            const features [sample, cov_feature]

        Returns:
            h_path         post-bayes trajectory of hidden values
        """
        h = self.covariates_map(covs.float())

        ## just-in-case converting sample_ids and time_ptr to int64
        time_ptr   = time_ptr.to(torch.int64)
        sample_ids = sample_ids.to(torch.int64)

        h_path = []
        
        dt = self.dt
        i  = 0

        for t in torch.arange(0.0, T, step=dt):
            while i < len(time_uniq) and time_uniq[i] <= t + 1e-5:
                ## observation data
                h2      = h.clone()
                start   = time_ptr[i]
                end     = time_ptr[i+1]
                idx     = sample_ids[start:end]

                h2[idx] = self.gru_bayes(X[start:end], h[idx])
                h  = h2
                i += 1
            h_path.append(h)

            ## propagate (all hidden) to time **t**
            p = torch.zeros(h.shape[0], 1, device=h.device)
            h = self.gru_next(p, h)

        h_path.append(h)

        h_path = torch.stack(h_path, dim=0)
        o_path = self.net_out(h_path)
        return o_path
