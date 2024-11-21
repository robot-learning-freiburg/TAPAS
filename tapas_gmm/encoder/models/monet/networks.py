import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(start_dim=1)


class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=16, sigma_bg=0.09, sigma_fg=0.11):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        # TODO: adapt for 256x256. should be 8192, right?
        hidden_dim = 4096  # for 128x128 input. eg for 64x64: 1024
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels + 1, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 2 * latent_dim),  # gives mu, sigma of latent post
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim + 2, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, input_channels + 1, 1),
        )
        self._bg_logvar = 2 * torch.tensor(sigma_bg).log()
        self._fg_logvar = 2 * torch.tensor(sigma_fg).log()

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    @staticmethod
    def spatial_broadcast(z, h, w):
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, latent_dim) -> (n, latent_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension:
        # final shape = (n, latent_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def encode(self, x, log_m_k):
        """
        :param x: Input image
        :param log_m_k: Attention mask logits
        """
        params = self.encoder(torch.cat((x, log_m_k), dim=1))
        z_mu = params[:, : self.latent_dim]
        z_logvar = params[:, self.latent_dim :]

        return z_mu, z_logvar

    def forward(self, x, log_m_k, background=False):
        """
        :param x: Input image
        :param log_m_k: Attention mask logits
        :return: x_k and reconstructed mask logits
        """
        params = self.encoder(torch.cat((x, log_m_k), dim=1))
        z_mu = params[:, : self.latent_dim]
        z_logvar = params[:, self.latent_dim :]
        z = self.reparameterize(z_mu, z_logvar)
        # z = torch.nan_to_num(z)

        # "The height and width of the input to this CNN were both 8 larger
        # than the target output (i.e. image) size to arrive at the target size
        # (i.e. accommodating for the lack of padding)."
        h, w = x.shape[-2:]
        z_sb = self.spatial_broadcast(z, h + 8, w + 8)

        output = self.decoder(z_sb)
        # output = torch.nan_to_num(output)
        x_mu = output[:, : self.input_channels]
        x_logvar = self._bg_logvar if background else self._fg_logvar
        m_logits = output[:, self.input_channels :]

        # if (torch.isnan(summarize_tensor(x_mu, "x_mu")['mean']).any()):
        #     breakpoint()

        return m_logits, x_mu, x_logvar, z_mu, z_logvar


class AttentionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, resize=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(output_channels, affine=True)
        self._resize = resize

    def forward(self, *inputs):
        # batch = inputs
        # h = []
        downsampling = len(inputs) == 1
        x = inputs[0] if downsampling else torch.cat(inputs, dim=1)
        # pc = x
        x = self.conv(x)
        # ac = x
        # x = torch.nan_to_num(self.norm(x))
        # an = x
        x = skip = F.relu(x)
        if self._resize:
            x = F.interpolate(
                skip,
                scale_factor=0.5 if downsampling else 2.0,
                mode="nearest",
                recompute_scale_factor=True,
            )
        # if (torch.isnan(summarize_tensor(x, "x")['mean']).any()
        #         or torch.isnan(summarize_tensor(skip, "skip")[
        #             'mean']).any()):
        #     breakpoint()
        return (x, skip) if downsampling else x


class AttentionNet(nn.Module):
    def __init__(self, input_channels, output_channels, ngf=64):
        """
        ngf (int)       -- the number of filters in the last conv layer
        """
        super().__init__()
        self.downblock1 = AttentionBlock(input_channels + 1, ngf)
        self.downblock2 = AttentionBlock(ngf, ngf * 2)
        self.downblock3 = AttentionBlock(ngf * 2, ngf * 4)
        self.downblock4 = AttentionBlock(ngf * 4, ngf * 8)
        self.downblock5 = AttentionBlock(ngf * 8, ngf * 8, resize=False)

        self.mlp = nn.Sequential(
            nn.Linear(8 * 8 * ngf * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 8 * 8 * ngf * 8),
            nn.ReLU(),
        )

        self.upblock1 = AttentionBlock(2 * ngf * 8, ngf * 8)
        self.upblock2 = AttentionBlock(2 * ngf * 8, ngf * 4)
        self.upblock3 = AttentionBlock(2 * ngf * 4, ngf * 2)
        self.upblock4 = AttentionBlock(2 * ngf * 2, ngf)
        # no resizing occurs in the last block of each path
        self.upblock5 = AttentionBlock(2 * ngf, ngf, resize=False)

        self.output = nn.Conv2d(ngf, output_channels, 1)

    def forward(self, x, log_s_k):
        # Downsampling blocks
        x, skip1 = self.downblock1(torch.cat((x, log_s_k), dim=1))
        x, skip2 = self.downblock2(x)
        x, skip3 = self.downblock3(x)
        x, skip4 = self.downblock4(x)
        x, skip5 = self.downblock5(x)
        skip6 = skip5
        # The input to the MLP is the last skip tensor collected from the
        # downsampling path (after flattening)
        # _, skip6 = self.downblock6(x)
        # Flatten
        x = skip6.flatten(start_dim=1)
        x = self.mlp(x)
        # Reshape to match shape of last skip tensor
        x = x.view(skip6.shape)
        # Upsampling blocks
        x = self.upblock1(x, skip5)
        x = self.upblock2(x, skip4)
        x = self.upblock3(x, skip3)
        x = self.upblock4(x, skip2)
        x = self.upblock5(x, skip1)
        # Output layer
        logits = self.output(x)
        x = F.logsigmoid(logits)
        return x, logits
