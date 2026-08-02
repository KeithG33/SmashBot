from gymnasium.envs.registration import register

register(
     id="melee-v0",
     entry_point="smashbot.gym:SmashMeleeEnv",
     max_episode_steps=None,
     order_enforce=True,
)

register(
    id='melee-test-v0',
     entry_point='smashbot.gym:SmashMeleeTestEnv',
     max_episode_steps=None,
     order_enforce=True,
)