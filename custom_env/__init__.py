from gym.envs.registration import register

register(
    id='custom_env/ExpWorld-v1',
    entry_point='custom_env.envs:ExpWorld1',
    
)