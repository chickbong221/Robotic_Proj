"""
TWISTER-TDMPC2 config for ManiSkill3.

Run:
    python main.py -c configs/tdmpc2_maniskill.py

Override env or hyperparams via environment variables:
    env_id=PickCube-v1 python main.py -c configs/tdmpc2_maniskill.py
"""

import os
import sys

# ── Ensure TDMPC2 common utilities are importable ────────────────────────────
_TDMPC2_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'ManiSkill', 'examples', 'baselines', 'tdmpc2')
)
if _TDMPC2_DIR not in sys.path:
    sys.path.insert(0, _TDMPC2_DIR)

from omegaconf import OmegaConf
from nnet.models.tdmpc2_twister import TWISTER_TDMPC2

# ─────────────────────────────────────────────────────────────────────────────
# Build config (mirrors TDMPC2's config.yaml + TWISTER aux additions)
# ─────────────────────────────────────────────────────────────────────────────

cfg = OmegaConf.create({
    # ── Environment ───────────────────────────────────────────────────────────
    'env_id':                   os.environ.get('env_id', 'PushCube-v1'),
    'obs':                      os.environ.get('obs', 'state'),
    'control_mode':             os.environ.get('control_mode', 'default'),
    'num_envs':                 int(os.environ.get('num_envs', 32)),
    'num_eval_envs':            int(os.environ.get('num_eval_envs', 4)),
    'env_type':                 os.environ.get('env_type', 'gpu'),     # 'gpu' or 'cpu'
    'include_state':            True,
    'render_mode':              'rgb_array',
    'render_size':              64,
    'eval_reconfiguration_frequency': 1,

    # ── Training ──────────────────────────────────────────────────────────────
    'steps':                    1_000_000,
    'batch_size':               256,
    'seed_steps':               5_000,
    'steps_per_update':         1,
    'buffer_size':              1_000_000,
    'seed':                     1,

    # ── Losses ────────────────────────────────────────────────────────────────
    'consistency_coef':         20.0,
    'reward_coef':              0.1,
    'value_coef':               0.1,
    'rho':                      0.5,

    # ── Optimizer ─────────────────────────────────────────────────────────────
    'lr':                       3e-4,
    'enc_lr_scale':             0.3,
    'grad_clip_norm':           20.0,
    'tau':                      0.01,

    # ── Discount ──────────────────────────────────────────────────────────────
    'discount_denom':           5,
    'discount_min':             0.95,
    'discount_max':             0.995,

    # ── Planning (MPPI) ───────────────────────────────────────────────────────
    'mpc':                      True,
    'iterations':               6,
    'num_samples':              512,
    'num_elites':               64,
    'num_pi_trajs':             24,
    'horizon':                  3,
    'min_std':                  0.05,
    'max_std':                  2.0,
    'temperature':              0.5,

    # ── Policy prior ──────────────────────────────────────────────────────────
    'log_std_min':              -10,
    'log_std_max':              2,
    'entropy_coef':             1e-4,

    # ── Critic / distributional RL ────────────────────────────────────────────
    'num_bins':                 101,
    'vmin':                     -10,
    'vmax':                     10,

    # ── Architecture (model_size=5 defaults) ─────────────────────────────────
    'model_size':               5,
    'enc_dim':                  256,
    'mlp_dim':                  512,
    'latent_dim':               512,
    'num_enc_layers':           2,
    'num_channels':             32,
    'rgb_state_enc_dim':        64,
    'rgb_state_num_enc_layers': 1,
    'rgb_state_latent_dim':     64,
    'task_dim':                 0,
    'num_q':                    5,
    'dropout':                  0.01,
    'simnorm_dim':              8,

    # ── Evaluation ────────────────────────────────────────────────────────────
    'eval_freq':                50_000,
    'eval_episodes_per_env':    2,

    # ── TWISTER auxiliary representation regularizer ──────────────────────────
    # Causal Transformer predicts future TD-MPC2 latents from (z0, real actions).
    # Set twister_loss_weight=0 to disable.
    'twister_loss_weight':      0.01,
    'twister_hidden_size':      256,
    'twister_num_heads':        4,
    'twister_num_layers':       2,
    'twister_dropout':          0.1,

    # ── Checkpoint ────────────────────────────────────────────────────────────
    'save_period':              100_000,

    # ── Derived fields (set later by make_envs / WorldModel) ─────────────────
    # obs_shape, action_dim, episode_length  → set in TWISTER_TDMPC2.__init__
    # true_latent_dim                        → set in WorldModel.__init__
    # discount                               → set in TDMPC2.__init__
    'obs_shape':                None,
    'action_dim':               None,
    'episode_length':           None,
    'true_latent_dim':          None,
    'discount':                 None,

    # ── Multi-task (disabled for ManiSkill single-task) ───────────────────────
    'multitask':                False,
    'task_dim':                 0,
    'tasks':                    None,
    'obs_shapes':               None,
    'action_dims':              None,
    'episode_lengths':          None,

    # ── Sub-configs required by make_envs / TDMPC2 internals ─────────────────
    'env_cfg': {
        'env_id':           None,   # filled by make_envs
        'control_mode':     None,
        'obs_mode':         None,
        'reward_mode':      'normalized_dense',
        'num_envs':         None,
        'sim_backend':      None,
        'env_horizon':      None,
        'partial_reset':    False,
    },
    'eval_env_cfg': {
        'env_id':               None,
        'control_mode':         None,
        'obs_mode':             None,
        'reward_mode':          'normalized_dense',
        'num_envs':             None,
        'sim_backend':          None,
        'env_horizon':          None,
        'partial_reset':        False,
        'num_eval_episodes':    None,
    },
})

# ── Derived scalars that don't need Hydra ────────────────────────────────────
cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
cfg.tasks = [cfg.env_id]

# Propagate env identity into sub-configs (mirrors parse_cfg)
cfg.env_cfg.env_id = cfg.eval_env_cfg.env_id = cfg.env_id
cfg.env_cfg.obs_mode = cfg.eval_env_cfg.obs_mode = cfg.obs
cfg.env_cfg.reward_mode = cfg.eval_env_cfg.reward_mode = 'normalized_dense'
cfg.env_cfg.num_envs = cfg.num_envs
cfg.eval_env_cfg.num_envs = cfg.num_eval_envs
cfg.env_cfg.sim_backend = cfg.eval_env_cfg.sim_backend = cfg.env_type
cfg.eval_env_cfg.num_eval_episodes = cfg.eval_episodes_per_env * cfg.num_eval_envs

# ── Create model ──────────────────────────────────────────────────────────────
# make_envs is called inside __init__ and fills obs_shape / action_dim / etc.
model = TWISTER_TDMPC2(cfg)
model.compile()

# ── Callback path (used by main.py for TensorBoard + checkpoints) ─────────────
env_tag = os.environ.get('run_name', cfg.env_id)
callback_path = os.path.join('callbacks', 'tdmpc2_maniskill', env_tag)

# ── No dataset — env is managed internally by the model ───────────────────────
# (functions.load_datasets returns None, None when these attrs are absent)
