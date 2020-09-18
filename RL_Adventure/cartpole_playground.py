import gym
import itertools

env_id = "CartPole-v0"
env = gym.make(env_id)
VALID_ACTIONS = [0, 1]

state = env.reset()
for eps in range(10):
    for i in itertools.count():
        env.render()
        action = int(input("请输入0或1控制左右方向:"))
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        print("next_state[{}],reward[{}]".format(next_state,reward))
        if done:
            print('episode is done in {} step!'.format(i+1))
            env.reset()
            break