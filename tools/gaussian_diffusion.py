"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
from torchdiffeq import odeint
import numpy as np
import torch as th
import torch.distributed as dist
from tools.nn import mean_flat
from tools.losses import normal_kl, discretized_gaussian_log_likelihood
from tools import logger
import torch.nn.functional as F
from tools.utils import *
import torch.nn as nn

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    VELOCITY = enum.auto() # the model predicts velocity alpha_t * epsilon - sigma_t * x_0
    VECTOR = enum.auto() # the model predicts v in flow matching d_sigma_t * epsilon - d_alpha_t * x_0
    SCORE = enum.auto()  # the model predicts the score function
    
class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    
    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def get_named_beta_schedule(
    schedule_name,
    num_diffusion_timesteps,
    lambda_max=10.0,
    lambda_min=-10.0,
):
    """
    Get a pre-defined beta schedule for the given name.

    Supported:
      - "linear":     linear beta schedule
      - "cosine":     cosine alpha_bar schedule
      - "linear_logsnr": linear schedule in log-SNR space

    For linear_logsnr, we define
        lambda(t) = lambda_max + t * (lambda_min - lambda_max),
    where lambda(t) = log(alpha_t^2 / sigma_t^2),
    and alpha_bar(t) = alpha_t^2 = sigmoid(lambda(t)).
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    elif schedule_name == "linear_logsnr":
        def logsnr(t):
            return lambda_max + t * (lambda_min - lambda_max)

        def alpha_bar(t):
            return 1.0 / (1.0 + math.exp(-logsnr(t)))

        return betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar)

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_bar function,
    which defines the cumulative product of (1-beta) over time from t in [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda/function that takes t in [0,1] and
                      produces the cumulative product of (1-beta) up to t.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        args,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        device="cuda",
    ):
        self.args = args
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        self.mse_loss_weight_type=args.weight_type
        self.gamma=args.gamma
        self.learn_sigma=args.learn_sigma
        
        self.p2_gamma=args.p2_gamma
        self.p2_k=args.p2_k
        
        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas >= 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        # alphas = 1.0 - betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )


    def unpack_model_output(self, raw_output):
        """
        Some models return (pred, aux) or (pred, aux1, aux2, ...).
        For sampling, we only need the primary prediction tensor.
        """
        if isinstance(raw_output, tuple):
            return raw_output[0]
        return raw_output
    
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        
        with torch.cuda.amp.autocast(enabled=self.args.amp):
            model_output = model(x, self._scale_timesteps(t), **model_kwargs)
            
        model_output = self.unpack_model_output(model_output)
        
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, 
                                      ModelMeanType.EPSILON,
                                      ModelMeanType.VELOCITY,
                                      ]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            elif self.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_v(x_t=x, t=t, v=model_output)
                )

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        ind = t[0].cpu().numpy()
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
        
    def _predict_xstart_from_v(self, x_t, t, v):
        assert x_t.shape == v.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, t.shape) * x_t
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, t.shape) * v
        )
            
    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def sample_t(self, x_start):
        if self.args.time_dist[0] == 'uniform':
            t = th.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
        else:
            raise NotImplementedError(f"Unknown time_dist: {self.args.time_dist}")
            
        return t

    def compute_target(self, x_start, noise, t, alpha=None, sigma=None):
        if alpha is None or sigma is None:
            alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, t.shape)
            sigma = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, t.shape)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=self.q_sample(x_start, t, noise=noise), t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
            ModelMeanType.VELOCITY: alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x_start,
        }[self.model_mean_type]
        
        return target
        
    def training_losses(self, model, x_start, features=None, t=None, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        if t is None:
            t = self.sample_t(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        
        mse_loss_weight = None
        alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, t.shape)
        sigma = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
        
        mse_loss_weight = compute_mse_loss_weight(self.model_mean_type, self.mse_loss_weight_type, t, alpha, sigma, self.p2_k, self.p2_gamma)
        
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
                
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            
            raw_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            
            if isinstance(raw_output, tuple):
                model_output = raw_output[0]
                sec_out = raw_output[1] if len(raw_output) > 1 else None
            else:
                model_output = raw_output
            
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0
                    
            target = self.compute_target(x_start, noise, t, alpha, sigma)
            assert model_output.shape == target.shape == x_start.shape

            raw_mse = mean_flat((target - model_output) ** 2)
                
            terms["mse"] = mse_loss_weight * raw_mse

            if self.args.learn_align:
                assert self.gamma > 0, "Gamma must be greater than 0 for align loss"
                # proj_loss = cosine_similarity(features, sec_out)
                align_loss = compute_align_loss(features, sec_out, self.args.align_type)
                terms["align"] = align_loss                
                                
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            elif self.args.learn_align:
                 terms["loss"] = terms["mse"] + self.gamma * terms["align"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms    
        
    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

def compute_align_loss(target, output, type, temperature=0.1):
    if type == "cosine":
        return - F.cosine_similarity(target, output, dim=-1).mean()

    elif type == "mse":
        return F.mse_loss(output, target)

    elif type == "mse_l2":
        target = F.normalize(target, dim=-1)
        output = F.normalize(output, dim=-1)
        return F.mse_loss(output, target)
        # return (output - target).pow(2).sum(dim=-1).mean()

    elif type == "nt_xent":
        assert temperature > 0, "temperature must be > 0"

        N, T, D = target.shape
        B = N * T

        # [B, D]
        target = target.reshape(B, D)
        output = output.reshape(B, D)

        # L2 normalize
        target = F.normalize(target, dim=1)
        output = F.normalize(output, dim=1)

        # similarity matrix: [B, B]
        logits = torch.matmul(output, target.T) / temperature

        labels = torch.arange(B, device=logits.device)

        # symmetric NT-Xent (optional but recommended)
        loss_i = F.cross_entropy(logits, labels)
        loss_j = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss_i + loss_j)

    else:
        raise ValueError(f"Unknown align loss type: {type}.")
    
def projection_loss(z, z_tilde):
    z = F.normalize(z, dim=-1)
    z_tilde = F.normalize(z_tilde, dim=-1)
    proj_loss = -th.mean(th.sum(z * z_tilde, dim=-1))
    return proj_loss


def cosine_similarity(target, output):
    cosine_sim = F.cosine_similarity(target, output, dim=-1)
    return -cosine_sim.mean()

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

# utils
@th.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not dist.is_initialized():
        return output

    tensors_gather = [th.ones_like(tensor)
        for _ in range(th.distributed.get_world_size())]
    th.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = th.cat(tensors_gather, dim=0)
    return output


def compute_mse_loss_weight(model_mean_type, mse_loss_weight_type, t, alpha, sigma, p2_k=1.0, p2_gamma=1.0):
    snr = (alpha / sigma) ** 2
    mse_loss_weight = None

    if mse_loss_weight_type == 'constant':
        return th.ones_like(t)

    if model_mean_type.name == "EPSILON":
        if mse_loss_weight_type.startswith("min_snr_"):
            k = float(mse_loss_weight_type.split('min_snr_')[-1])
            mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).min(dim=1)[0] / snr
        elif mse_loss_weight_type.startswith("max_snr_"):
            k = float(mse_loss_weight_type.split('max_snr_')[-1])
            mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).max(dim=1)[0] / snr
        elif mse_loss_weight_type == 'lambda':
            mse_loss_weight = sigma
        elif mse_loss_weight_type == 'debias':
            mse_loss_weight = sigma / alpha
        elif mse_loss_weight_type == 'p2':
            mse_loss_weight = 1 / (p2_k + snr) ** p2_gamma
        elif mse_loss_weight_type == 'min_debias':
            mse_loss_weight = th.minimum(sigma / alpha, th.ones_like(sigma))
        elif mse_loss_weight_type == 'max_debias':
            mse_loss_weight = th.maximum(sigma / alpha, th.ones_like(sigma))

    elif model_mean_type.name == "START_X":
        if mse_loss_weight_type == 'trunc_snr':
            mse_loss_weight = th.stack([snr, th.ones_like(t)], dim=1).max(dim=1)[0]
        elif mse_loss_weight_type == 'snr':
            mse_loss_weight = snr
        elif mse_loss_weight_type == 'inv_snr':
            mse_loss_weight = 1. / snr
        elif mse_loss_weight_type.startswith("min_snr_"):
            k = float(mse_loss_weight_type.split('min_snr_')[-1])
            mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).min(dim=1)[0]
        elif mse_loss_weight_type.startswith("max_snr_"):
            k = float(mse_loss_weight_type.split('max_snr_')[-1])
            mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).max(dim=1)[0]
        elif mse_loss_weight_type == 'lambda':
            mse_loss_weight = alpha
            
    elif model_mean_type.name == "VECTOR":
        if mse_loss_weight_type == 'lambda':
            mse_loss_weight = th.ones_like(t)

    elif model_mean_type.name == "VELOCITY":
        if mse_loss_weight_type.startswith("min_snr_"):
            k = float(mse_loss_weight_type.split('min_snr_')[-1])
            mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).min(dim=1)[0] / (snr + 1)
        elif mse_loss_weight_type == 'lambda':
            mse_loss_weight = alpha * sigma

    if mse_loss_weight is None:
        raise ValueError(f"Invalid mse_loss_weight_type: {mse_loss_weight_type}")

    mse_loss_weight[snr == 0] = 1.0  # Handle edge cases
    return mse_loss_weight

            
class FlowMatching:
    def __init__(
        self,
        *,
        args,
        model_mean_type, 
        device="cuda",
    ):
        self.args = args
        self.model_mean_type = model_mean_type
        self.mse_loss_weight_type=args.weight_type   
             
        self.path_type = args.path_type      
        self.sampler_type = args.sampler_type
         
        # P2 weighting
        self.p2_gamma=args.p2_gamma
        self.p2_k=args.p2_k
        
        self.gamma=args.gamma
        self.learn_sigma=args.learn_sigma
        
    def expand_t_like_x(self, t, x):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        dims = [1] * (len(x.size()) - 1)
        return t.view(t.size(0), *dims).to(x)
    
    def float_equal(self, num1, num2, eps=1e-8):
        return abs(num1 - num2) < eps
    
    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = th.full_like(t, -1.0)  
            d_sigma_t = th.full_like(t, 1.0)  
        elif self.path_type == "cosine":
            alpha_t = th.cos(t * np.pi / 2)
            sigma_t = th.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * th.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * th.cos(t * np.pi / 2)
        elif self.path_type == "linear_logsnr":
            lambda_max = 10; lambda_min = -10.0
            lam = lambda_max + t * (lambda_min - lambda_max)
            alpha_t = th.sigmoid(0.5 * lam)
            sigma_t = th.sigmoid(-0.5 * lam)
            d_alpha_t = -10.0 * alpha_t * sigma_t
            d_sigma_t = -d_alpha_t
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def convert_model_output_to_vector(self, model_output, x_t, t):
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)

        if self.model_mean_type == ModelMeanType.START_X:
            start_x = model_output
            noise = (x_t - alpha_t * start_x) / sigma_t
            
        elif self.model_mean_type == ModelMeanType.EPSILON:
            noise = model_output
            start_x = (x_t - sigma_t * noise) / alpha_t
            
        elif self.model_mean_type == ModelMeanType.VELOCITY:
            # v = α_t * ε - σ_t * x₀ → solve ε and x₀
            start_x = (alpha_t * x_t - sigma_t * model_output) / (alpha_t**2 + sigma_t**2)
            noise = (sigma_t * x_t + alpha_t * model_output) / (alpha_t**2 + sigma_t**2)
            
        elif self.model_mean_type == ModelMeanType.VECTOR:
            return model_output
        
        else:
            raise NotImplementedError("Unsupported model_mean_type for vector")

        vector = d_alpha_t * start_x + d_sigma_t * noise
        return vector

    def convert_model_output_to_score(self, model_output, x_t, t):
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)

        if self.model_mean_type == ModelMeanType.START_X:
            start_x = model_output
            score = -(x_t - alpha_t * start_x) / (sigma_t ** 2)

        elif self.model_mean_type == ModelMeanType.EPSILON:
            noise = model_output
            score = -noise / sigma_t

        elif self.model_mean_type == ModelMeanType.VELOCITY:
            # v = α_t * ε - σ_t * x₀ → solve ε and x₀
            noise = (sigma_t * x_t + alpha_t * model_output) / (alpha_t**2 + sigma_t**2)
            score = -noise / sigma_t

        elif self.model_mean_type == ModelMeanType.VECTOR:
            # start_x = (sigma_t * model_output - d_sigma_t * x_t) / (sigma_t * d_alpha_t - alpha_t * d_sigma_t)
            noise = (d_alpha_t * x_t - alpha_t * model_output) / (sigma_t * d_alpha_t - alpha_t * d_sigma_t)
            score = -noise / sigma_t
            
        elif self.model_mean_type == ModelMeanType.SCORE:
            return model_output
        
        else:
            raise NotImplementedError("Unsupported model_mean_type for score")

        return score

    def sample_t(self, x_start):
        if self.args.time_dist[0] == 'uniform':
            t = th.rand(x_start.shape[0], device=x_start.device)
        elif self.args.time_dist[0] == 'lognorm':
            # logit-normal: z ~ N(mu, sigma^2), t = sigmoid(z)
            mu, sigma = float(self.args.time_dist[-2]), float(self.args.time_dist[-1])
            normal_samples = th.randn(x_start.shape[0], device=x_start.device) * sigma + mu
            t = th.sigmoid(normal_samples)
        else:
            raise NotImplementedError(f"Unknown time_dist: {self.args.time_dist}")
            
        return t
    
    def q_sample(self, x_start, noise, t,):
        t = self.expand_t_like_x(t, x_start)
        alpha_t, sigma_t, _, _ = self.interpolant(t)
        x_t = alpha_t * x_start + sigma_t * noise
        return x_t


    def compute_target(self, x_start, noise, t, alpha_t=None, sigma_t=None, d_alpha_t=None, d_sigma_t=None):
        if alpha_t is None or sigma_t is None or d_alpha_t is None or d_sigma_t is None:
            alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)

        alpha = self.expand_t_like_x(alpha_t, x_start)
        sigma = self.expand_t_like_x(sigma_t, x_start)
        d_alpha = self.expand_t_like_x(d_alpha_t, x_start)
        d_sigma = self.expand_t_like_x(d_sigma_t, x_start)

        return {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
            ModelMeanType.VELOCITY: alpha * noise - sigma * x_start,
            ModelMeanType.VECTOR: d_alpha * x_start + d_sigma * noise,
            ModelMeanType.SCORE: -noise / sigma,
        }[self.model_mean_type]

    #taining
    def training_losses(self, model, x_start, features=None, t=None, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        if t is None:
            t = self.sample_t(x_start)
        
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)     
                     
        x_t = self.q_sample(x_start, noise, t)
        
        terms = {}
        
        mse_loss_weight = compute_mse_loss_weight(self.model_mean_type, self.mse_loss_weight_type, t, alpha_t, sigma_t, self.p2_k, self.p2_gamma)
                
        target = self.compute_target(x_start, noise, t, alpha_t=alpha_t, sigma_t=sigma_t, d_alpha_t=d_alpha_t, d_sigma_t=d_sigma_t)
        
        raw_output = model(x_t, t, **model_kwargs)

        if isinstance(raw_output, tuple):
            model_output = raw_output[0]
            sec_out = raw_output[1] if len(raw_output) > 1 else None
        else:
            model_output = raw_output
        
        assert model_output.shape == target.shape == x_start.shape

        raw_mse = mean_flat((target - model_output) ** 2)
            
        terms["mse"] = mse_loss_weight * raw_mse 
        
        if self.args.learn_align:
            assert self.gamma > 0, "Gamma must be greater than 0 for align loss"
            # proj_loss = cosine_similarity(features, sec_out)
            align_loss = compute_align_loss(features, sec_out, self.args.align_type)
            terms["align"] = align_loss        
                  
        if self.args.learn_align:
            terms["loss"] = terms["mse"] + self.gamma * terms["align"] 
        else:
            terms["loss"] = terms["mse"]
            
        return terms
    
    
    # sampling
    def forward_model(self, model, sample_tensor, time_tensor, **model_kwargs):
        time_tensor = time_tensor.view(sample_tensor.shape[0])

        with torch.cuda.amp.autocast(enabled=self.args.amp):
            raw_output = model(sample_tensor, time_tensor, **model_kwargs)
            model_output = raw_output[0] if isinstance(raw_output, tuple) else raw_output

        return model_output


    def ode_sample(self, model, noise, device, num_steps=50, solver='dopri5', **model_kwargs):
        timesteps = th.linspace(1.0, 0.0, num_steps, device=device)

        def drift(time_scalar, sample_tensor):
            time_tensor = self.expand_t_like_x(time_scalar, sample_tensor)
            model_output = self.forward_model(model, sample_tensor, time_tensor, **model_kwargs)
            return self.convert_model_output_to_vector(model_output, sample_tensor, time_tensor)

        samples = odeint(func=drift, y0=noise, t=timesteps, method=solver, rtol=self.rtol, atol=self.atol)
        return samples[-1]


    def compute_diffusion(self, time_tensor):
        _, sigma_t, _, d_sigma_t = self.interpolant(time_tensor)
        return 2 * sigma_t * d_sigma_t


    def sde_sample(self, model, noise, device, num_steps=50, solver='heun', **model_kwargs):
        def compute_drift(sample_tensor, time_tensor, diffusion):
            model_output = self.forward_model(model, sample_tensor, time_tensor, **model_kwargs)
            score = self.convert_model_output_to_score(model_output, sample_tensor, time_tensor)
            vector = self.convert_model_output_to_vector(model_output, sample_tensor, time_tensor)
            return vector - 0.5 * diffusion * score

        timesteps = th.linspace(1.0, 0.04, num_steps, dtype=th.float64, device=device)
        timesteps = th.cat([timesteps, th.tensor([0.0], dtype=th.float64, device=device)])
        sample_tensor = noise

        with th.no_grad():
            for time_cur, time_next in zip(timesteps[:-2], timesteps[1:-1]):
                step_size = time_next - time_cur
                time_tensor = self.expand_t_like_x(time_cur, sample_tensor)
                diffusion = self.compute_diffusion(time_tensor)
                drift_cur = compute_drift(sample_tensor, time_tensor, diffusion)
                noise_term = th.sqrt(diffusion) * th.randn_like(sample_tensor) * th.sqrt(th.abs(step_size))

                if solver == 'euler':
                    sample_tensor = sample_tensor + drift_cur * step_size + noise_term

                elif solver == 'heun':
                    sample_pred = sample_tensor + drift_cur * step_size + noise_term
                    time_next_tensor = self.expand_t_like_x(time_next, sample_pred)
                    diffusion_next = self.compute_diffusion(time_next_tensor)
                    drift_next = compute_drift(sample_pred, time_next_tensor, diffusion_next)
                    sample_tensor = sample_tensor + 0.5 * (drift_cur + drift_next) * step_size + noise_term

                else:
                    raise ValueError(f"Unknown solver: {solver}")

            time_cur, time_next = timesteps[-2], timesteps[-1]
            step_size = time_next - time_cur
            time_tensor = self.expand_t_like_x(time_cur, sample_tensor)
            diffusion = self.compute_diffusion(time_tensor)
            drift_cur = compute_drift(sample_tensor, time_tensor, diffusion)

        return sample_tensor + drift_cur * step_size


    def sample(self, model, noise, device, num_steps=50, solver='heun', **model_kwargs):
        if self.sampler_type == "ode":
            return self.ode_sample(model, noise, device, num_steps, solver=solver, **model_kwargs)

        if self.sampler_type == "sde":
            return self.sde_sample(model, noise, device, num_steps, solver=solver, **model_kwargs)

        raise NotImplementedError(f"Unsupported sampler_type: {self.sampler_type}")