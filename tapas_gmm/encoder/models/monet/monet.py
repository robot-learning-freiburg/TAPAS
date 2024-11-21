from dataclasses import dataclass, field
from typing import Dict

import torch
from torch import nn

from tapas_gmm.encoder.models.monet.networks import VAE, AttentionNet
from tapas_gmm.utils.select_gpu import device


@dataclass
class MONetModelConfig:
    image_channels: int = 3
    latent_dims: tuple[int, ...] = (16, 16)
    slots: dict[int, int] = field(
        default_factory=lambda: {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    )
    sigma_fg: float = 0.12
    sigma_bg: float = 0.09


class MONetModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.no_slots = len(config["slots"].keys())

        self.attention_net = AttentionNet(config["image_channels"], 1)

        for i, d in enumerate(config["latent_dims"]):
            setattr(
                self,
                "bvae" + str(i),
                VAE(
                    input_channels=config["image_channels"],
                    latent_dim=d,
                    sigma_bg=config["sigma_bg"],
                    sigma_fg=config["sigma_fg"],
                ),
            )

        # wandb config transforms keys to str -> convert back for slot_map
        self.slot_map = {int(k): v for k, v in config["slots"].items()}

    def encode(self, batch):
        # TODO: minimize redundancy between encode and forward
        shape = list(batch.shape)
        shape[1] = 1
        log_s_k = batch.new_zeros(shape).to(device)

        z_mu = []
        z_logvar = []

        for k in range(self.no_slots):
            if k != self.no_slots - 1:
                log_alpha_k, alpha_logits_k = self.attention_net(batch, log_s_k)
                log_m_k = log_s_k + log_alpha_k
                log_s_k += -alpha_logits_k + log_alpha_k

            else:
                log_m_k = log_s_k

            bvae_idx = self.slot_map[k]
            bvae = getattr(self, "bvae" + str(bvae_idx))

            z_mu_k, z_logvar_k = bvae.encode(batch, log_m_k)

            z_mu.append(z_mu_k)
            z_logvar.append(z_logvar_k)

        return torch.cat(z_mu, dim=1), torch.cat(z_logvar, dim=1)

    def forward(self, batch):
        loss_E = 0
        x_masked = 0  # Image reconstuction, ie mixture of reconstuction slots
        x_mu = []  # Means of posteriors in pixel space (after decoder)
        x_m = []  # Reconstruction components per step k (masked)
        b = []  # Exponents for the decoder loss
        m = []  # Masks
        m_tilde_logits = []

        # Initial s_k = 1: shape = (N, 1, H, W)
        # s_k is the 'scope', "an additional spatial mask updated after each
        # attention step" that "signifies the proportion of each pixel that
        # remains to be explained given all previous attention masks".
        # note that implementatation uses log_s_k, hence init as log 1 = 0
        shape = list(batch.shape)
        shape[1] = 1
        log_s_k = batch.new_zeros(shape).to(device)

        for k in range(self.no_slots):
            # Derive mask from current scope
            if k != self.no_slots - 1:
                log_alpha_k, alpha_logits_k = self.attention_net(batch, log_s_k)
                # log_alpha_k = torch.nan_to_num(log_alpha_k)
                # alpha_logits_k = torch.nan_to_num(alpha_logits_k)
                log_m_k = log_s_k + log_alpha_k
                # Compute next scope
                log_s_k += -alpha_logits_k + log_alpha_k
                # print(summarize_tensor(log_alpha_k, "log_alpha_k"))
                # print(summarize_tensor(alpha_logits_k, "alpha_logits_k"))
                # print(summarize_tensor(log_s_k, "log_s_k"))
            # "for the last step K , [...] the attention network is not applied
            # but the last scope is used directly instead, i.e. m_K=s_{K − 1}"
            else:
                log_m_k = log_s_k

            # repr matters: "two different internal β-VAEs – one applied to
            # the first slot, and the other to the remaining 5 slots"
            bvae_idx = self.slot_map[k]
            bvae = getattr(self, "bvae" + str(bvae_idx))
            # Get component and mask reconstruction, as well as the z_k
            # parameters
            m_tilde_k_logits, x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = bvae(
                batch, log_m_k, k == 0
            )
            # print(summarize_tensor(z_mu_k, "z_mu_k"))
            # print(summarize_tensor(z_logvar_k, "z_logvar_k"))
            # KLD is additive for independent distributions
            loss_E += -0.5 * (1 + z_logvar_k - z_mu_k.pow(2) - z_logvar_k.exp()).sum()

            m_k = log_m_k.exp()
            x_k_masked = m_k * x_mu_k
            # x_k_masked = torch.nan_to_num(x_k_masked)
            # print(summarize_tensor(log_m_k, "log_m_k"))
            # print(summarize_tensor(m_k, "m_k"))
            # print(summarize_tensor(x_mu_k, "x_mu_k"))
            # print(summarize_tensor(x_k_masked, "x_k_masked"))
            # print(summarize_tensor(m_tilde_k_logits, "m_tilde_k_logits"))

            # Exponents for the decoder loss
            b_k = (
                log_m_k
                - 0.5 * x_logvar_k
                - (batch - x_mu_k).pow(2) / (2 * x_logvar_k.exp())
            )
            b.append(b_k.unsqueeze(1))

            # Iteratively reconstruct the output image
            x_masked += x_k_masked

            # Get outputs for kth step
            m.append(m_k)  # * 2. - 1.)  # shift mask from [0, 1] to [-1, 1]
            x_mu.append(x_mu_k.detach().clone())
            x_m.append(x_k_masked.detach().clone())
            m_tilde_logits.append(m_tilde_k_logits)

        b = torch.cat(b, dim=1)
        m = torch.cat(m, dim=1)
        x_mu = torch.cat(x_mu, dim=1)
        x_m = torch.cat(x_m, dim=1)
        m_tilde_logits = torch.cat(m_tilde_logits, dim=1)

        return x_masked, loss_E, b, m, m_tilde_logits, x_mu, x_m
