import torch
from tqdm import tqdm
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from tools import dist_util
from .cfg_edm import ablation_sampler, Net
from models.unet import EncoderUNetModel


class IntervalCFG(torch.nn.Module):
    def __init__(self, model, num_classes, guidance_scale=1.0, interval=(-1.0, -1.0), class_cond=True):
        super().__init__()
        self.model = model
        self.null_label = int(num_classes)
        self.guidance_scale = float(guidance_scale)
        self.interval = interval
        self.class_cond = class_cond

    def _use_cfg(self, time_value):
        if abs(self.guidance_scale - 1.0) < 1e-8:
            return False
        time_from, time_to = self.interval
        return time_from <= time_value < time_to if time_from >= 0 and time_to > time_from else True

    def _format_time(self, time_tensor, batch_size):
        if time_tensor.dim() == 0:
            return time_tensor.expand(batch_size)
        if time_tensor.numel() == 1:
            return time_tensor.reshape(1).expand(batch_size)
        return time_tensor.reshape(batch_size)

    def forward(self, sample_tensor, time_tensor, **model_kwargs):
        time_tensor = self._format_time(time_tensor, sample_tensor.shape[0])
        class_labels = model_kwargs.get("y", None)

        if not (self.class_cond and class_labels is not None and self._use_cfg(float(time_tensor.float().mean().item()))):
            return self.model(sample_tensor, time_tensor, **model_kwargs)

        assert class_labels.shape[0] == sample_tensor.shape[0], f"CFG expects label batch size {sample_tensor.shape[0]}, but got {class_labels.shape[0]}."

        cfg_kwargs = dict(model_kwargs)
        cfg_kwargs["y"] = torch.cat([class_labels, torch.full_like(class_labels, self.null_label)], dim=0)

        model_output = self.model(torch.cat([sample_tensor, sample_tensor], dim=0), torch.cat([time_tensor, time_tensor], dim=0), **cfg_kwargs)
        model_output = model_output[0] if isinstance(model_output, tuple) else model_output

        cond_output, uncond_output = model_output.chunk(2, dim=0)
        return uncond_output + self.guidance_scale * (cond_output - uncond_output)


class Classifier:
    def __init__(self, args, device, model):
        self.args = args
        self.device = device
        self.model = model
        self.classifier = self._load_classifier() if args.use_classifier else None

    def _create_classifier(self):
        attention_ds = [self.args.image_size // res for res in self.model.attention_resolutions]
        classifier_kwargs = {
            "image_size": self.model.image_size,
            "in_channels": self.args.in_chans,
            "model_channels": self.model.model_channels,
            "out_channels": self.model.num_classes,
            "num_res_blocks": self.model.num_res_blocks,
            "attention_resolutions": tuple(attention_ds),
            "channel_mult": self.model.channel_mult,
            "num_head_channels": self.model.num_head_channels,
            "use_scale_shift_norm": self.model.use_scale_shift_norm,
            "resblock_updown": self.model.resblock_updown,
            "pool": "attention",
        }
        return EncoderUNetModel(**classifier_kwargs)

    def _load_classifier(self):
        classifier = self._create_classifier()
        classifier.load_state_dict(torch.load(self.args.use_classifier, map_location="cpu"))
        classifier.to(self.device)
        classifier.eval()
        return classifier

    def cond_fn(self, x, t, y, scale=1.0):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(log_probs)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale


def sync_ema_model(eval_model):
    for param in eval_model.parameters():
        dist.broadcast(param.data, src=0)


class Sampler:
    def __init__(self, args, device, eval_model, diffusion, classifier=None):
        self.args = args
        self.device = device
        self.model = eval_model
        self.diffusion = diffusion
        self.classifier = classifier
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.args.vae}", local_files_only=True).to(self.device) if self.args.in_chans == 4 else None

        if self.vae is not None:
            self.vae.eval()
            self.vae.requires_grad_(False)
            self.vae.to(memory_format=torch.channels_last)

    def _build_cfg_model(self, num_classes):
        return IntervalCFG(self.model, num_classes, self.args.guidance_scale, self.args.interval, self.args.class_cond).eval()

    def _model_fn(self, x, t, y=None):
        return self.model(x, t, y if self.args.class_cond else None)

    def ddim_sampler(self, num_samples, sample_size, image_size, num_classes, progress_bar=False):
        self.model.eval()
        all_samples, all_labels = [], []
        world_size = dist.get_world_size() if self.args.parallel else 1

        if self.args.parallel:
            sync_ema_model(self.model)
            dist.barrier()

        if progress_bar and dist_util.is_main_process():
            pbar = tqdm(total=num_samples, desc="Generating Samples (DDIM)")

        cfg_model = self._build_cfg_model(num_classes)

        while len(all_samples) * sample_size < num_samples:
            class_labels = self._get_y_cond(sample_size, num_classes)
            sample_model = self.model if self.classifier else cfg_model

            samples = self.diffusion.ddim_sample_loop(
                sample_model if not self.classifier else self._model_fn,
                (sample_size, self.args.in_chans, image_size, image_size),
                device=self.device,
                model_kwargs={"y": class_labels} if self.args.class_cond else {},
                cond_fn=(lambda x, t, y: self.classifier.cond_fn(x, t, y, self.args.guidance_scale)) if self.classifier else None,
            )

            samples, class_labels = self._process_sample_label(samples, class_labels)
            self._gather_samples(all_samples, all_labels, samples, class_labels, world_size)

            if dist_util.is_main_process() and progress_bar:
                pbar.update(samples.shape[0] * world_size)

        return all_samples, all_labels

    def edm_sampler(self, num_samples, sample_size, image_size, num_classes, progress_bar=False):
        self.model.eval()
        all_samples, all_labels = [], []
        world_size = dist.get_world_size() if self.args.parallel else 1

        if self.args.parallel:
            sync_ema_model(self.model)
            dist.barrier()

        if progress_bar and dist_util.is_main_process():
            pbar = tqdm(total=num_samples, desc=f"Generating Samples ({self.args.solver.capitalize()})")

        cfg_model = self._build_cfg_model(num_classes)
        net = Net(model=cfg_model, img_channels=self.args.in_chans, img_resolution=image_size, label_dim=num_classes, 
                  noise_schedule=self.args.path_type, amp=self.args.amp, pred_type=self.args.mean_type).to(self.device)

        while len(all_samples) * sample_size < num_samples:
            class_labels = self._get_y_cond(sample_size, num_classes)
            latent_noise = torch.randn([sample_size, net.img_channels, net.img_resolution, net.img_resolution], device=self.device)

            samples = ablation_sampler(
                net,
                latents=latent_noise,
                num_steps=self.args.sample_steps,
                solver=self.args.solver,
                discretization=self.args.discretization,
                schedule=self.args.schedule,
                scaling=self.args.scaling,
                class_labels=class_labels,
            )

            samples, class_labels = self._process_sample_label(samples, class_labels)
            self._gather_samples(all_samples, all_labels, samples, class_labels, world_size)

            if dist_util.is_main_process() and progress_bar:
                pbar.update(samples.shape[0] * world_size)

        return all_samples, all_labels

    def flow_matching_sampler(self, num_samples, sample_size, image_size, num_classes, progress_bar=False):
        self.model.eval()
        all_samples, all_labels = [], []
        world_size = dist.get_world_size() if self.args.parallel else 1

        if self.args.parallel:
            sync_ema_model(self.model)
            dist.barrier()

        if progress_bar and dist_util.is_main_process():
            pbar = tqdm(total=num_samples, desc=f"Generating Samples ({self.args.solver.capitalize()})")

        cfg_model = self._build_cfg_model(num_classes)

        while len(all_samples) * sample_size < num_samples:
            class_labels = self._get_y_cond(sample_size, num_classes)
            latent_noise = torch.randn([sample_size, self.args.in_chans, image_size, image_size], device=self.device)

            samples = self.diffusion.sample(cfg_model, latent_noise, self.device, num_steps=self.args.sample_steps, solver=self.args.solver, y=class_labels)

            samples, class_labels = self._process_sample_label(samples, class_labels)
            self._gather_samples(all_samples, all_labels, samples, class_labels, world_size)

            if dist_util.is_main_process() and progress_bar:
                pbar.update(samples.shape[0] * world_size)

        return all_samples, all_labels

    def _get_y_cond(self, sample_size, num_classes):
        if not self.args.class_cond:
            return None

        labels = self.args.class_labels
        if labels is None:
            return torch.randint(0, num_classes, (sample_size,), device=self.device)

        assert all(isinstance(label, int) and 0 <= label < num_classes for label in labels), f"class_labels must be integers in [0, {num_classes})"
        assert len(labels) <= sample_size, f"len(class_labels) must be <= sample_size ({sample_size})"

        labels = torch.tensor(labels, device=self.device, dtype=torch.long)
        return labels[torch.randint(len(labels), (sample_size,), device=self.device)]

    def _gather_samples(self, all_samples, all_labels, samples, class_labels, world_size):
        if self.args.parallel:
            gathered_samples = [torch.zeros_like(samples) for _ in range(world_size)]
            dist.all_gather(gathered_samples, samples)
            all_samples.extend([sample_batch.cpu().numpy() for sample_batch in gathered_samples])

            if self.args.class_cond:
                gathered_labels = [torch.zeros_like(class_labels) for _ in range(world_size)]
                dist.all_gather(gathered_labels, class_labels)
                all_labels.extend([label_batch.cpu().numpy() for label_batch in gathered_labels])
            return

        all_samples.append(samples.cpu().numpy())

        if self.args.class_cond:
            all_labels.append(class_labels.cpu().numpy())

    def _process_sample_label(self, samples, class_labels=None):
        if self.vae is not None:
            with torch.no_grad(), torch.cuda.amp.autocast():
                samples = samples.to(dtype=self.vae.dtype)
                samples = self.vae.decode(samples / self.args.latent_scale).sample

        return self._inverse_normalize(samples), class_labels

    def _inverse_normalize(self, samples):
        return ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()

    def sample(self, num_samples, sample_size, image_size, num_classes, progress_bar=False):
        if self.args.model_mode == "flow":
            return self.flow_matching_sampler(num_samples, sample_size, image_size, num_classes, progress_bar)

        if self.args.model_mode == "diffusion":
            if self.args.solver == "ddim":
                return self.ddim_sampler(num_samples, sample_size, image_size, num_classes, progress_bar)
            return self.edm_sampler(num_samples, sample_size, image_size, num_classes, progress_bar)

        raise ValueError(f"Unsupported model_mode: {self.args.model_mode}")