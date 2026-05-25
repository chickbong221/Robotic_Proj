import torch
import torch.nn as nn


class TwisterAuxModel(nn.Module):
    """
    TWISTER-inspired Transformer sequence model used as a weak latent regularizer.

    Mirrors TWISTER's TSSM architecture on TD-MPC2's continuous latent space:
      action_mixer(z0, a_t) -> Causal Transformer -> z_{t+1} prediction

    Training signal: MSE vs stop-gradient encoder targets from real transitions.
    Not used for planning, imagination, actor, or critic training.
    """

    def __init__(self, latent_dim, action_dim, hidden_size=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        # Mirrors TWISTER action_mixer: concat(z, a) -> Linear -> Norm -> Act
        self.action_mixer = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size, eps=1e-3),
        )

        # Causal Transformer — mirrors TWISTER's TransformerNetwork with causal=True
        # Pre-norm (norm_first=True) matches TWISTER's module_pre_norm convention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Mirrors TWISTER dynamics_predictor: hidden -> next latent prediction
        self.z_predictor = nn.Linear(hidden_size, latent_dim)

    def forward(self, z0, actions):
        """
        Args:
            z0:      [B, latent_dim]      initial TD-MPC2 latent from encoder(obs[0])
            actions: [H, B, action_dim]   real actions from replay buffer

        Returns:
            z_preds: [H, B, latent_dim]   predicted future latents at steps 1..H
        """
        H, B, _ = actions.shape

        # Broadcast z0 across horizon: [H, B, latent_dim]
        z0_expanded = z0.unsqueeze(0).expand(H, -1, -1)

        # Build sequence input (z0, a_t) for each t — [B, H, latent_dim + action_dim]
        za = torch.cat([z0_expanded, actions], dim=-1).transpose(0, 1)

        # MLP mixer — [B, H, hidden_size]
        x = self.action_mixer(za)

        # Causal mask: position t only attends to 0..t
        causal_mask = nn.Transformer.generate_square_subsequent_mask(H, device=x.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        # Project to latent space and return time-first — [H, B, latent_dim]
        return self.z_predictor(x).transpose(0, 1)
