import os
import sys
import numpy as np
import torch
from functools import partial
import torch.nn.functional as F
from torch.cuda.amp import autocast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OP_DIR = os.path.join(BASE_DIR, 'op')
sys.path.append(OP_DIR)


class Net(torch.nn.Module):
    def __init__(self,
        model,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        pred_type       = 'EPSILON',            # Prediction type: 'eps', 'x0', or 'v'.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        amp             = False,            # Execute the underlying model at FP16 precision?
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        noise_schedule  = 'linear',
        lambda_max=10.0,
        lambda_min=-10.0,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.amp = amp
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = model
        self.noise_schedule = noise_schedule
        self.pred_type = pred_type  # 'eps', 'x0', or 'v'
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        
        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):  # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.amp and not force_fp32 and x.device.type == 'cuda') else torch.float32

        with autocast(enabled=self.amp and not force_fp32):
            c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)
            c_in = 1 / (sigma ** 2 + 1).sqrt()

            raw_output = self.model((c_in * x).to(dtype), c_noise.flatten().repeat(x.shape[0]).int(), y=class_labels, **model_kwargs)
            model_output = raw_output[0] if isinstance(raw_output, tuple) else raw_output

            assert model_output.dtype == dtype

            if self.pred_type == 'EPSILON':
                c_skip = 1
                c_out = -sigma
                denoised = c_skip * x + c_out * model_output[:, :self.img_channels].to(torch.float32)

            elif self.pred_type == 'START_X':
                denoised = model_output

            elif self.pred_type == 'VELOCITY':
                c_skip = c_in ** 2
                c_out = -sigma * c_in
                denoised = c_skip * x + c_out * model_output[:, :self.img_channels].to(torch.float32)

            else:
                raise ValueError(f"Unsupported pred_type: {self.pred_type}")

        return denoised
        

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        if self.noise_schedule == 'cosine':
            return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2
        
        elif self.noise_schedule == 'linear':
            betas = np.linspace(0.0001, 0.02, self.M + 1, dtype=np.float64)
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            return alphas_cumprod[self.M - j]
        
        elif self.noise_schedule == 'linear_logsnr':
            t = (self.M - j) / self.M
            lam = self.lambda_max + t * (self.lambda_min - self.lambda_max)
            return torch.sigmoid(lam)  
        
        else:
            raise NotImplementedError(f"unknown path type: {self.noise_schedule}")

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)


def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, **model_kwargs,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device):
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))

    with autocast(enabled=net.amp):
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
            t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
            x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

            h = t_next - t_hat
            denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels, **model_kwargs).to(torch.float64)
            
            d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
            x_prime = x_hat + alpha * h * d_cur
            t_prime = t_hat + alpha * h

            if solver == 'euler' or i == num_steps - 1:
                x_next = x_hat + h * d_cur
            else:
                assert solver == 'heun'
                denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels, **model_kwargs).to(torch.float64)
                d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
                x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next
