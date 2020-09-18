import math, random
import itertools

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

# from IPython.display import clear_output
# import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
import os,sys
from datetime import datetime

basedir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(basedir)

from Baseutil import BasicModule, ReplayBuffer, ModelParametersCopier
def with_log(symbol):
    # --------------------------------------- log ------------------------------------------
    result_dir = './Result/'

    dirname, filename = os.path.split(__file__)
    filename = filename.rstrip('.py')
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S') + str(symbol) + filename

    log_dir = os.path.join(result_dir, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    log_dict = {}
    log_dict['filename'] = filename
    log_dict['time_str'] = time_str
    log_dict['writer'] = writer
    return log_dict, writer

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)



env_id = "CartPole-v0"
env = gym.make(env_id)




# # The epsilon decay schedule
# epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay)  # good 静态
# epsilon_by_frame = lambda frame_idx: epsilons[frame_idx]  # frame_idx从0开始


# plt.plot([epsilon_by_frame(i) for i in range(10000)])


class DQN(BasicModule):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.Dropout(),
            nn.LeakyReLU(0.25),
            nn.Linear(128, env.action_space.n)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = int(q_value.max(1)[1].data[0])
        else:
            action = random.randrange(env.action_space.n)
        return action


model = DQN(env.observation_space.shape[0], env.action_space.n)
model_target = DQN(env.observation_space.shape[0], env.action_space.n)
ModelParametersCopier(model, model_target)

if USE_CUDA:
    model = model.cuda()
    model_target = model.cuda()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, momentum=0.95)
# optimizer = torch.optim.Adam(model.parameters(),lr=0.00015)
    # Adam(model.parameters())



epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 100000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)  # good 动态
def train():
    pass


batch_size = 64
replay_memory_size = 1000000
replay_buffer = ReplayBuffer(replay_memory_size)
# num_frames = 400000
num_episodes = 300000
gamma = 0.99
C = 8000  # update Q_target in every C steps
num2print = 1000  # print statistcs in every num2print step

_, writer = with_log(symbol=env_id)
log_dict = True


replay_start_size = 50000
state = env.reset()
for frame_idx in range(1, replay_start_size + 1):
    epsilon = epsilon_by_frame(1)
    action = model.act(state, epsilon)

    next_state, reward, done, _ = env.step(action)

    replay_buffer.push(state, action, reward, next_state, done)



    if done:
        state = env.reset()
    else:
        state = next_state

sum_loss_batch = 0
best_eps_reward = 0
all_eps_reward = []
eps_N = 10  # avg(eps_reward, N)
for i_eps in range(num_episodes):

    # save model

    state = env.reset()
    loss_batch = None
    reward_eps = 0


    # num_loss = 0
    for t in itertools.count():
        epsilon = epsilon_by_frame(frame_idx-replay_start_size+1)  # 从1开始衰减epsilon
        action = model.act(state, epsilon)

        frame_idx += 1
        if (frame_idx-replay_start_size+1) % C == 0:
            print("\nupdate Q_target".center(40,'-'))
            ModelParametersCopier(model, model_target)

        next_state, reward, done, _ = env.step(action)
        reward_eps += reward
        replay_buffer.push(state, action, reward, next_state, done)

        # update statistics

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
        state_batch = Variable(torch.FloatTensor(np.float32(state_batch)))
        next_state_batch = Variable(torch.FloatTensor(np.float32(next_state_batch)), volatile=True)
        action_batch = Variable(torch.LongTensor(action_batch))
        reward_batch = Variable(torch.FloatTensor(reward_batch))
        done_batch = Variable(torch.FloatTensor(done_batch))

        q_values = model(state_batch)
        next_q_values = model_target(next_state_batch)

        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # 扩充数据维度，在0起的指定位置N加上维数为1的维度,相当于np.newaxis
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward_batch + gamma * next_q_value * (1 - done_batch)

        loss_batch_mean = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        optimizer.zero_grad()
        loss_batch_mean.backward()
        optimizer.step()

        # num_loss += 1
        sum_loss_batch += loss_batch_mean.detach().cpu().numpy()

        # statistics
        if (frame_idx-replay_start_size+1) % num2print == 0:
            print(
                "\nTraining: episode[{:0>3}/{:0>3}], Batch[{:0>3}]; BatchAvgloss: {:.4f}".format(
                    i_eps + 1, num_episodes, frame_idx - replay_start_size + 1, sum_loss_batch / num2print))

            writer.add_scalars('Loss_group', {'Train_eps_loss': sum_loss_batch / num2print}, i_eps + 1)

            sum_loss_batch = 0

        if done:
            all_eps_reward.append(reward_eps)
            if i_eps >= eps_N:
                avg_eps_reward = sum(all_eps_reward[-eps_N:])/eps_N

                print("\rTraining_eps[{}/{}], episode_reward[{}],avg_eps_reward[{}]".format(i_eps + 1,
                                                                                          num_episodes,
                                                                                          reward_eps,
                                                                                          avg_eps_reward), end="")
                sys.stdout.flush()
                writer.add_scalars('avg_reward_group', {'avg_reward': avg_eps_reward}, i_eps + 1)

            writer.add_scalars('Reward_group', {'Train_eps_Rewards': reward_eps}, i_eps + 1)
            if reward_eps > best_eps_reward:
                best_eps_reward = reward_eps
                model.save()
            break
        else: state = next_state


    # if i_eps % 200 == 0:
        # plot(i_eps, all_rewards, losses)
    # 记录一个epoch的训练avg loss
    if log_dict:
        writer.add_scalars('Loss_group', {'TrainAvgLoss': reward_eps}, i_eps + 1)
        # # 记录learning rate
        # writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)


# def test():
#     pass
# model.eval()
# for i_episode in range(20):
#     state = env.reset()
#     reward_eps_test = 0.
#     for t in range(100):
#         env.render()
#         # print(observation)
#         action = model.act(state, epsilon)
#         state, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps| eps_reward:{}".format(t+1, reward_eps_test))
#             break
#
#         # 记录Loss和准确率
#         # if log_dict:
#             # writer.add_scalars('Loss_group', {'TestAvgLoss': reward_eps}, i_eps + 1)
# env.close()


