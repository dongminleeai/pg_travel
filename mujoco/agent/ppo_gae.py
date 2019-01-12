import numpy as np
from utils.utils import *
from hparams import HyperParams as hp

def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory) # (..., 4)
    states = np.vstack(memory[:, 0]) # (..., 11)
    actions = list(memory[:, 1]) # (..., 3)
    rewards = list(memory[:, 2]) # (..., 1)
    masks = list(memory[:, 3]) # (..., 1)
    values = critic(torch.Tensor(states))

    # ----------------------------
    # step 1: get returns and GAEs and log probability of old policy
    returns, advants = get_gae(rewards, masks, values)
    mu, std, logstd = actor(torch.Tensor(states))

    old_policy = log_density(torch.Tensor(actions), mu, std, logstd)
    old_values = critic(torch.Tensor(states))

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)
    # np.arrage(n) [0 1 2 ... 2065 2066 2067]
    # print("np.arrage(n)", arr)

    # ----------------------------
    # step 2: get value loss and actor loss and update actor & critic
    # batch를 random suffling하고 mini batch를 추출
    for _ in range(10):
        np.random.shuffle(arr)
        # [ 198  181 1240 ... 1114 1639 1447 ]
        # print("np.random.shuffle(arr)", arr)

        for i in range(n // hp.batch_size): # batch_size = 64
            # 0 1 2 3 4 ~~~ 31 (2067 // 64)
            # print("i", i)
            # 0 ~ 64, 64 ~ 128, 128 ~ 192, ... , 1984 ~ 2048
            # print("hp.batch_size * i", hp.batch_size * i)
            # print("hp.batch_size * (i + 1)", hp.batch_size * (i + 1))
            batch_index = arr[hp.batch_size * i : hp.batch_size * (i + 1)]
            # [ 198  181 1240 1210 1628 1327 1966  129 1051 1464 1498 1069  961 1173
            # 1847 1768  898 1722  887  836  584 1776    4    3  646 2014 1552  771
            # 1616  658  403 2041  562 1951 1426 1450  346 1854 1758 1452 1339 1990
            # 1080 1467 1614 1109  138 1231  165  416  653  356  813 1258 1032 1594
            # 176 1310  968 1278 1269  448 1923 1311]
            # print("batch_index", batch_index)

            batch_index = torch.LongTensor(batch_index)
            # [64]
            # print("batch_index.shape", batch_index.shape)
            
            inputs = torch.Tensor(states)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            # value에 대해서도 clipping 해주려고 detach() 
            oldvalue_samples = old_values[batch_index].detach()
            
            # value function 구하기
            values = critic(inputs)
            # clipping을 사용하여 critic loss 구하기 
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -hp.clip_param, # 0.2
                                         hp.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            # 논문에서 6번 수식. surrogate loss 구하기
            # detach() : 업데이트가 되지 않도록 만듦
            loss, ratio = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)

            # 논문에서 7번 수식. surrogate loss를 clipping해서 actor loss 만들기
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - hp.clip_param,
                                        1.0 + hp.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss # 0.5 ?

            critic_optim.zero_grad()
            loss.backward(retain_graph=True) # ?
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()


def get_gae(rewards, masks, values):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    
    running_returns = 0
    previous_value = 0
    running_advants = 0

    # gamma = 0.99, lamda = 0.98
    # 뒤에서부터 구한다!
    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (hp.gamma * running_returns * masks[t])
        returns[t] = running_returns

        # 논문에서 수식 10번
        running_delta = rewards[t] + (hp.gamma * previous_value * masks[t]) - \
                                        values.data[t]
        previous_value = values.data[t]
        
        # 논문에서 수식 14번 + lambda 추가
        running_advants = running_delta + (hp.gamma * hp.lamda * \
                                            running_advants * masks[t])
        advants[t] = running_advants

    # 캐중요!!!
    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

# 논문에서 6번 수식
def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    mu, std, logstd = actor(torch.Tensor(states))
    new_policy = log_density(actions, mu, std, logstd)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    return surrogate_loss, ratio