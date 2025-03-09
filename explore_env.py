from env import SingleStepEnv

env = SingleStepEnv('experiments/')
print(env.reset())
print(env.step())