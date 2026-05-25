"""
TWISTER + TD-MPC2 integration.

TDMPC2WithTwisterAux
    Subclasses the TDMPC2 agent (imported from ManiSkill) and adds a TWISTER-style
    causal Transformer auxiliary loss on the same real replay batch that TD-MPC2
    already samples.  No TWISTER imagination / actor / critic training.

TWISTER_TDMPC2
    Thin TWISTER Model wrapper.  Overrides fit() with TD-MPC2's online training
    loop (real ManiSkill vectorized env, MPPI planning, TD-MPC2 buffer).
    All other TWISTER infrastructure — main.py entry point, save/load,
    TensorBoard logging, checkpoint management — is reused unchanged.
"""

import sys
import os

# ── Make TDMPC2 importable from its subtree inside the TWISTER repo ──────────
_TDMPC2_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'ManiSkill', 'examples', 'baselines', 'tdmpc2')
)
if _TDMPC2_DIR not in sys.path:
    sys.path.insert(0, _TDMPC2_DIR)
# ─────────────────────────────────────────────────────────────────────────────

from collections import defaultdict
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tensordict.tensordict import TensorDict

from tdmpc2 import TDMPC2 as _TDMPC2Base          # ManiSkill TDMPC2 agent
from common.buffer import Buffer                    # ManiSkill TDMPC2 buffer
from common import math as tdmpc2_math              # soft_ce, two_hot_inv
from envs import make_envs                          # ManiSkill env factory

from nnet import models                             # TWISTER Model base
from nnet.modules.twister.twister_aux import TwisterAuxModel


# ─────────────────────────────────────────────────────────────────────────────
# TDMPC2WithTwisterAux
# ─────────────────────────────────────────────────────────────────────────────

class TDMPC2WithTwisterAux(_TDMPC2Base):
    """
    Extends TD-MPC2 with a TWISTER-style auxiliary latent representation loss.

    On the same real replay batch that TD-MPC2 samples, a causal Transformer
    (mirroring TWISTER's TSSM) predicts future latents from (z0, real actions).
    MSE vs stop-gradient encoder targets is added as a small auxiliary loss.

    TD-MPC2 losses, MPPI planning, and Q/policy updates are completely unchanged.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.twister_aux = TwisterAuxModel(
            latent_dim=cfg.true_latent_dim,
            action_dim=cfg.action_dim,
            hidden_size=getattr(cfg, 'twister_hidden_size', 256),
            num_heads=getattr(cfg, 'twister_num_heads', 4),
            num_layers=getattr(cfg, 'twister_num_layers', 2),
            dropout=getattr(cfg, 'twister_dropout', 0.1),
        ).to(self.device)
        # Add alongside existing encoder/dynamics/reward/Q groups
        self.optim.add_param_group({
            'params': self.twister_aux.parameters(),
            'lr': cfg.lr,
        })

    def save(self, fp):
        torch.save({
            'model': self.model.state_dict(),
            'twister_aux': self.twister_aux.state_dict(),
        }, fp)

    def load(self, fp):
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict['model'])
        if 'twister_aux' in state_dict:
            self.twister_aux.load_state_dict(state_dict['twister_aux'])

    def update(self, buffer):
        obs, action, reward, task = buffer.sample()

        # Stop-gradient targets shared by consistency loss and TWISTER aux
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)

        self.optim.zero_grad(set_to_none=True)
        self.model.train()
        self.twister_aux.train()

        # ── Latent rollout (unchanged from TD-MPC2) ──
        zs = torch.empty(
            self.cfg.horizon + 1, self.cfg.batch_size, self.cfg.true_latent_dim,
            device=self.device,
        )
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho ** t
            zs[t + 1] = z

        # ── TD-MPC2 losses (unchanged) ────────────────────────────────────────
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += tdmpc2_math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho ** t
            for q in range(self.cfg.num_q):
                value_loss += tdmpc2_math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho ** t
        consistency_loss *= (1 / self.cfg.horizon)
        reward_loss *= (1 / self.cfg.horizon)
        value_loss *= (1 / (self.cfg.horizon * self.cfg.num_q))
        tdmpc2_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
        )

        # ── TWISTER auxiliary loss ────────────────────────────────────────────
        # z_preds: future latent predictions from (z0, real actions) via Transformer
        # next_z:  stop-gradient encoder targets, already computed under no_grad above
        z_preds = self.twister_aux(zs[0], action)           # [H, B, latent_dim]
        twister_repr_loss = F.mse_loss(z_preds, next_z)
        total_loss = tdmpc2_loss + self.cfg.twister_loss_weight * twister_repr_loss

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.twister_aux.parameters()),
            self.cfg.grad_clip_norm,
        )
        self.optim.step()

        pi_loss = self.update_pi(zs.detach(), task)
        self.model.soft_update_target_Q()

        self.model.eval()
        self.twister_aux.eval()
        return {
            'consistency_loss': float(consistency_loss.mean().item()),
            'reward_loss': float(reward_loss.mean().item()),
            'value_loss': float(value_loss.mean().item()),
            'pi_loss': pi_loss,
            'twister_repr_loss': float(twister_repr_loss.item()),
            'total_loss': float(total_loss.mean().item()),
            'grad_norm': float(grad_norm),
            'pi_scale': float(self.scale.value),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TWISTER_TDMPC2  —  TWISTER framework wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TWISTER_TDMPC2(models.Model):
    """
    TWISTER framework wrapper around TDMPC2WithTwisterAux.

    TWISTER provides: main.py entry point, save/load, TensorBoard logging.
    TD-MPC2 provides: MPPI planning, online RL on real ManiSkill vectorized env.
    TWISTER aux: causal Transformer latent regularization (weak auxiliary loss).

    fit() runs TD-MPC2's online training loop; dataset/epoch args are ignored.
    """

    def __init__(self, cfg):
        super().__init__(name='TWISTER-TDMPC2')
        self.tdmpc2_cfg = cfg

        # ── Create vectorized envs ─────────────────────────────────────────────
        # make_envs sets cfg.env_cfg.control_mode and cfg.env_cfg.env_horizon
        self.env = make_envs(cfg, cfg.num_envs,
                             record_video_path=None, is_eval=False, logger=None)
        self.eval_env = make_envs(cfg, cfg.num_eval_envs,
                                  record_video_path=None, is_eval=True, logger=None)

        # ── Infer obs_shape / action_dim / episode_length from env ────────────
        obs_space = self.env.observation_space
        if hasattr(obs_space, 'spaces'):                   # Dict obs space (rgb)
            cfg.obs_shape = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        else:                                               # Box obs space (state)
            cfg.obs_shape = tuple(obs_space.shape)
        cfg.action_dim = int(self.env.action_space.shape[-1])
        cfg.episode_length = int(self.env.max_episode_steps)

        # ── Create agent + buffer ─────────────────────────────────────────────
        self.agent = TDMPC2WithTwisterAux(cfg)
        self.buffer = Buffer(cfg)

    # ── Override compile() — TD-MPC2 manages its own optimizer ───────────────
    def compile(self, *args, **kwargs):
        self.compiled = True

    # ── Override save/load to delegate to the TDMPC2 agent ───────────────────
    def save(self, path, **kwargs):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)
        print(f'Checkpoint saved: {path}')

    def load(self, path, **kwargs):
        print(f'Loading checkpoint: {path}')
        self.agent.load(path)

    # ── Override summary ──────────────────────────────────────────────────────
    def summary(self, **kwargs):
        print(f'Model name: {self.name}')
        print(f'  TD-MPC2 world model: {self.agent.model.total_params:,} params')
        twister_params = sum(p.numel() for p in self.agent.twister_aux.parameters())
        print(f'  TWISTER aux model:   {twister_params:,} params')

    # ── Main training loop ────────────────────────────────────────────────────
    def fit(self, dataset_train=None, epochs=None, dataset_eval=None,
            callback_path=None, **kwargs):
        """
        Runs TD-MPC2's online training loop for cfg.steps total env interactions.
        dataset_train / dataset_eval / epochs from TWISTER's main.py are ignored.
        """
        cfg = self.tdmpc2_cfg
        writer = SummaryWriter(os.path.join(callback_path, 'logs')) if callback_path else None

        step, ep_idx = 0, 0
        start_time = time()
        train_metrics = {}
        vec_done = [True]
        eval_next = True
        seed_finish = False
        rollout_times = []
        obs = None
        vec_info = {}
        save_period = getattr(cfg, 'save_period', 100_000)

        while step <= cfg.steps:

            # ── Schedule evaluation ───────────────────────────────────────────
            if step % cfg.eval_freq < cfg.num_envs:
                eval_next = True

            # ── Episode boundary ──────────────────────────────────────────────
            if vec_done[0]:
                if eval_next:
                    eval_metrics = self._eval()
                    eval_metrics.update(step=step, episode=ep_idx,
                                        total_time=time() - start_time)
                    print(f'[eval step={step}]',
                          {k: f'{v:.4f}' for k, v in eval_metrics.items()
                           if isinstance(v, (int, float))})
                    if writer:
                        for k, v in eval_metrics.items():
                            if isinstance(v, (int, float)):
                                writer.add_scalar(f'eval/{k}', v, step)
                    eval_next = False

                if step > 0:
                    tds = torch.cat(self._tds, dim=1)
                    train_metrics.update(self._episode_metrics(vec_info, cfg))
                    if seed_finish:
                        time_metrics = dict(
                            rollout_time=float(np.mean(rollout_times)),
                            rollout_fps=float(cfg.num_envs / np.mean(rollout_times)),
                            step=step, episode=ep_idx,
                            total_time=time() - start_time,
                        )
                        if writer:
                            for k, v in time_metrics.items():
                                writer.add_scalar(f'time/{k}', v, step)
                        rollout_times = []
                    train_metrics.update(step=step, episode=ep_idx,
                                         total_time=time() - start_time)
                    if writer:
                        for k, v in train_metrics.items():
                            if isinstance(v, (int, float)):
                                writer.add_scalar(f'train/{k}', v, step)
                    ep_idx = self.buffer.add(tds)

                    # Periodic checkpoint
                    if callback_path and step % save_period == 0:
                        self.save(os.path.join(callback_path, f'checkpoints_step_{step}.ckpt'))

                obs, _ = self.env.reset()
                self._tds = [self._make_td(obs, cfg.num_envs, cfg.action_dim)]

            # ── Collect one environment step ──────────────────────────────────
            t0 = time()
            if step > cfg.seed_steps:
                action = self.agent.act(obs, t0=(len(self._tds) == 1), eval_mode=False)
            else:
                action = torch.from_numpy(self.env.action_space.sample())
            obs, reward, vec_terminated, vec_truncated, vec_info = self.env.step(action)
            vec_done = vec_terminated | vec_truncated
            if vec_done[0] and 'final_observation' in vec_info:
                obs = vec_info['final_observation']
            self._tds.append(self._make_td(obs, cfg.num_envs, cfg.action_dim, action, reward))
            rollout_times.append(time() - t0)

            # ── Agent update ──────────────────────────────────────────────────
            if step >= cfg.seed_steps:
                if not seed_finish:
                    seed_finish = True
                    num_updates = int(cfg.seed_steps / cfg.steps_per_update)
                    print('Pretraining agent on seed data...')
                else:
                    num_updates = max(1, int(cfg.num_envs / cfg.steps_per_update))
                for _ in range(num_updates):
                    train_metrics = self.agent.update(self.buffer)

            step += cfg.num_envs

        if callback_path:
            self.save(os.path.join(callback_path, f'checkpoints_step_{step}_final.ckpt'))
        if writer:
            writer.close()

    # ── Evaluation helper ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _eval(self):
        cfg = self.tdmpc2_cfg
        info = {}
        for _ in range(cfg.eval_episodes_per_env):
            obs, _ = self.eval_env.reset()
            done = torch.zeros(
                cfg.num_eval_envs, dtype=torch.bool,
                device='cuda' if cfg.env_type == 'gpu' else 'cpu',
            )
            t = 0
            while not done[0]:
                action = self.agent.act(obs, t0=(t == 0), eval_mode=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated | truncated
                t += 1
        return self._episode_metrics(info, cfg)

    # ── Static helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _episode_metrics(info, cfg):
        metrics = {}
        if cfg.env_type == 'gpu':
            for k, v in info.get('final_info', {}).get('episode', {}).items():
                metrics[k] = float(v.float().mean().item())
        else:
            temp = defaultdict(list)
            for fi in info.get('final_info', []):
                for k, v in fi.get('episode', {}).items():
                    temp[k].append(v)
            for k, v in temp.items():
                metrics[k] = float(np.mean(v))
        return metrics

    @staticmethod
    def _make_td(obs, num_envs, action_dim, action=None, reward=None):
        if isinstance(obs, dict):
            obs_td = TensorDict(
                {k: v.unsqueeze(1) for k, v in obs.items()}, batch_size=()
            )
        else:
            obs_td = obs.unsqueeze(1).cpu()
        if action is None:
            action = torch.full((num_envs, action_dim), float('nan'))
        if reward is None:
            reward = torch.full((num_envs,), float('nan'))
        return TensorDict(
            dict(
                obs=obs_td,
                action=action.unsqueeze(1),
                reward=reward.cpu().unsqueeze(1),
            ),
            batch_size=(num_envs, 1),
        )
