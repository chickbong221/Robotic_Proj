from nnet.envs import dm_control


class Manipulator(dm_control.DeepMindControlEnv):

    def __init__(
            self,
            task="insert_ball",
            img_size=(64, 64),
            history_frames=1,
            episode_saving_path=None,
            action_repeat=1
        ):

        valid_tasks = ["bring_ball", "bring_peg", "insert_ball", "insert_peg"]
        assert task in valid_tasks, f"task must be one of {valid_tasks}, got {task}"

        super(Manipulator, self).__init__(
            domain="manipulator",
            task=task,
            img_size=img_size,
            history_frames=history_frames,
            episode_saving_path=episode_saving_path,
            action_repeat=action_repeat
        )

        # Match the style of your Acrobot wrapper, but infer action size safely.
        try:
            self.num_actions = self.env.action_spec().shape[0]
        except Exception:
            pass