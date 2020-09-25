from gym.envs.registration import register

register(id='GazeboEnv-v0',
        entry_point='envs.gazebo_env_dir:GazeboEnv'
)
