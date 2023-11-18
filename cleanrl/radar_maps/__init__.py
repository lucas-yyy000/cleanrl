from gymnasium.envs.registration import register
# import gymnasium as gym

register(
      id='EmptyMap-v0',
      entry_point='radar_maps.env:EmptyMap',
      max_episode_steps=300
  )


register(
      id='EmptyMap-DoubleIntegrator-v0',
      entry_point='radar_maps.env:EmptyMap_DoubleIntegrator',
      max_episode_steps=500
  )

register(
      id='RadarMap-DoubleIntegrator-v0',
      entry_point='radar_maps.env:RadarMap_DoubleIntegrator',
      max_episode_steps=1000
  )