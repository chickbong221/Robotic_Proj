import os

# Must be set before importing dm_control / mujoco-related code
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import torch
import imageio.v2 as imageio

from nnet.models.twister import TWISTER


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------

def _to_uint8_rgb(frame):
    frame = np.asarray(frame)

    # Remove batch dim if present
    if frame.ndim == 4:
        if frame.shape[0] == 1:
            frame = frame[0]
        else:
            raise ValueError(f"Unexpected 4D frame shape: {frame.shape}")

    # CHW -> HWC
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    # grayscale -> RGB
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)

    # single channel -> RGB
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)

    # RGBA -> RGB
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Frame must be HWC RGB, got shape {frame.shape}")

    frame = np.ascontiguousarray(frame)

    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            if frame.min() >= 0.0 and frame.max() <= 1.0:
                frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frame = frame.clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)

    return frame


def _obs_to_frame(obs_state):
    if torch.is_tensor(obs_state):
        frame = obs_state.detach().cpu().numpy()
    else:
        frame = np.asarray(obs_state)
    return _to_uint8_rgb(frame)


def _latest_latent_step(latent):
    out = {}
    for key, value in latent.items():
        if key == "hidden":
            out[key] = value
            continue
        if torch.is_tensor(value) and value.ndim >= 2:
            out[key] = value[:, -1:]
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Video writer  (frames are written as-is — no resizing)
# ---------------------------------------------------------------------------

def _write_mp4_ffmpeg(frames, out_path, fps, crf=18):
    processed = [_to_uint8_rgb(f) for f in frames]

    writer = imageio.get_writer(
        out_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=[
            "-crf",    str(crf),
            "-preset", "slow",
            "-tune",   "animation",
        ],
    )
    try:
        for i, frame in enumerate(processed):
            print(
                f"  writing frame {i:4d}: shape={frame.shape}, "
                f"dtype={frame.dtype}, contiguous={frame.flags['C_CONTIGUOUS']}"
            )
            writer.append_data(frame)
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_episode(model, device, ep_idx=0):
    """Roll out one episode. Returns (frames, total_reward, steps)."""
    obs = model.env_eval.reset()
    state = model.transfer_to_device(obs.state)

    prev_latent = model.transfer_to_device(
        model.rssm.initial(
            batch_size=1,
            seq_length=2,
            dtype=obs.reward.dtype,
            detach_learned=True,
        )
    )
    prev_action = model.transfer_to_device(
        torch.zeros(1, 2, model.env.num_actions, dtype=obs.reward.dtype)
    )

    hidden = (prev_latent, prev_action)

    frames = [_obs_to_frame(obs.state)]
    total_reward = 0.0
    step = 0

    while True:
        prev_latent, prev_action = hidden

        enc = model.encoder_network(
            model.preprocess_inputs(state.unsqueeze(dim=0), time_stacked=False)
        )
        enc = {k: v.unsqueeze(dim=1) for k, v in enc.items()}

        latent, _ = model.rssm(
            states=enc,
            prev_states=prev_latent,
            prev_actions=prev_action,
            is_firsts=torch.zeros(1, 1, device=state.device),
            return_att_w=False,
        )

        latent_now = _latest_latent_step(latent)
        feat = model.rssm.get_feat(latent_now).squeeze(dim=1)

        action_dist = model.policy_network(feat)
        action = action_dist.mode()

        try:
            value = model.value_network(feat)
            print(f"  [ep {ep_idx}] step={step:4d}  value={float(value.mean()):8.4f}")
        except Exception:
            pass

        latent_for_hidden = {}
        for key in prev_latent.keys():
            if key == "hidden":
                latent_for_hidden[key] = model.rssm.slice_hidden(latent["hidden"])
            else:
                new_value = latent_now[key]
                latent_for_hidden[key] = torch.cat(
                    [prev_latent[key][:, -1:], new_value], dim=1
                )

        action_for_hidden = torch.cat(
            [prev_action[:, -1:], action.unsqueeze(1)], dim=1
        )
        hidden = (latent_for_hidden, action_for_hidden)

        env_action = (
            action.argmax(dim=-1).squeeze(dim=0)
            if model.config.policy_discrete
            else action.squeeze(dim=0)
        )

        obs = model.env_eval.step(env_action)
        state = model.transfer_to_device(obs.state)

        total_reward += float(obs.reward)
        step += model.env_eval.action_repeat
        frames.append(_obs_to_frame(obs.state))

        if obs.done or step >= model.config.time_limit_eval:
            break

    return frames, total_reward, step


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def play_and_save_video(
    ckpt_path,
    env_name,
    out_path="demo.mp4",
    device="cuda" if torch.cuda.is_available() else "cpu",
    fps=60,
    load_optimizer=False,
    crf=18,
    num_episodes=1,
    separate_videos=False,
):
    """
    Run one or more evaluation episodes and save a high-quality video.
    Frames are saved at native render resolution — no resizing applied.

    Parameters
    ----------
    num_episodes    : int   – number of episodes to record.
    separate_videos : bool  – False = one combined file (default).
                              True  = one file per episode:
                              demo_ep00.mp4, demo_ep01.mp4, ...
    fps             : int   – playback frame rate (default 60).
    crf             : int   – libx264 quality. 18 = visually lossless.
                              Lower = better quality / larger file.
    """
    model = TWISTER(env_name=env_name)
    model.compile()
    model.config.load_replay_buffer_state_dict = False
    model.load(ckpt_path, load_optimizer=load_optimizer, verbose=True, strict=True)
    model = model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    all_rewards = []
    all_steps   = []
    saved_paths = []

    base, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".mp4"

    all_frames = []  # used only when separate_videos=False

    for ep in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"  Episode {ep + 1} / {num_episodes}")
        print(f"{'='*60}")

        frames, reward, steps = _run_episode(model, device, ep_idx=ep)

        all_rewards.append(reward)
        all_steps.append(steps)

        print(f"  → reward={reward:.3f}  steps={steps}  frames={len(frames)}")

        if separate_videos:
            ep_path = f"{base}_ep{ep:02d}{ext}"
            if ep_path.endswith(".gif"):
                imageio.mimsave(ep_path, [_to_uint8_rgb(f) for f in frames], fps=fps)
            else:
                _write_mp4_ffmpeg(frames, ep_path, fps=fps, crf=crf)
            print(f"  Saved: {ep_path}")
            saved_paths.append(ep_path)
        else:
            all_frames.extend(frames)

    if not separate_videos:
        if out_path.endswith(".gif"):
            imageio.mimsave(out_path, [_to_uint8_rgb(f) for f in all_frames], fps=fps)
        else:
            _write_mp4_ffmpeg(all_frames, out_path, fps=fps, crf=crf)
        print(f"\nSaved combined video : {out_path}")
        saved_paths.append(out_path)

    # ---- summary --------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Summary ({num_episodes} episode{'s' if num_episodes > 1 else ''})")
    print(f"{'='*60}")
    for i, (r, s) in enumerate(zip(all_rewards, all_steps)):
        print(f"  ep {i:02d}  reward={r:8.3f}  steps={s}")
    print(f"  ---")
    print(f"  mean reward : {np.mean(all_rewards):.3f}")
    print(f"  std  reward : {np.std(all_rewards):.3f}")
    print(f"  min  reward : {np.min(all_rewards):.3f}")
    print(f"  max  reward : {np.max(all_rewards):.3f}")
    print(f"  Video settings : fps={fps}, crf={crf}, native resolution")
    print(f"  Saved files    : {saved_paths}")

    return all_rewards, all_steps, saved_paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    play_and_save_video(
        ckpt_path="/root/projects/TWISTER/callbacks/dmc/dmc-Manipulator-bring_peg/checkpoints_epoch_50_step_249670.ckpt",
        env_name="dmc-Manipulator-bring_peg",
        out_path="demo.mp4",
        fps=60,
        crf=18,
        num_episodes=5,
        separate_videos=False,
    )