from nnet.envs import dm_control

class Acrobot(dm_control.DeepMindControlEnv):

    def __init__(
            self, 
            task="swingup", 
            img_size=(64, 64),
            history_frames=1, 
            episode_saving_path=None, 
            action_repeat=1
        ):

        assert task in ["swingup"]
        super(Acrobot, self).__init__(
            domain="acrobot", 
            task=task, 
            img_size=img_size,
            history_frames=history_frames, 
            episode_saving_path=episode_saving_path, 
            action_repeat=action_repeat
        )

        self.num_actions = 1