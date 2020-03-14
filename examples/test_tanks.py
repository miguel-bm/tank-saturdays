import time
import numpy as np
import gym
import tank_saturdays


env = gym.make('TankSaturdays-v0')
obs = env.reset()


for i in range(500):
    env.render()
    obs, reward, done, _ = env.step(np.random.randint(9),
                                    np.random.randint(9))

    time.sleep(0.2)
