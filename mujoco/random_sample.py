import gym

env = gym.make('Hopper-v2')

for episode in range(10000):
    env.reset()

    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        print("observation : {} | action : {} | reward : {}".format(
                observation, action, reward))
        print("observation.shape : {} | action.shape : {}".format(
                observation.shape, action.shape))

