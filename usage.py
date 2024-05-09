from environment_definition import env, EnvironmentSettings

settings = EnvironmentSettings(
    width=100,
    height=20,
    n_agents=3,
    n_targets=3
)

myEnv = env(settings, render_mode="human")
myEnv.reset(seed=42)

ITERATIONS = 10
for i in range(ITERATIONS):
    #TODO should use the policy HERE
    actions = {agent: myEnv.action_space(agent).sample() for agent in myEnv.agents} # sample a random action for each agent
    observations, rewards, terminations, truncations, infos = myEnv.step(actions)
    myEnv.render()
    print(rewards)