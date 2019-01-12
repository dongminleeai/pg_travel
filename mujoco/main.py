import os
import gym
import torch
import argparse
import numpy as np
import torch.optim as optim
from model import Actor, Critic
from utils.utils import get_action, save_checkpoint
from collections import deque
from utils.running_state import ZFilter
from hparams import HyperParams as hp
from tensorboardX import SummaryWriter 

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='PPO',
                    help='select one of algorithms among Vanilla_PG,'
                         'NPG, TPRO, PPO')
parser.add_argument('--env', type=str, default="Hopper-v2",
                    help='name of Mujoco environement')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

if args.algorithm == "PG":
    from agent.vanila_pg import train_model
elif args.algorithm == "NPG":
    from agent.tnpg import train_model
elif args.algorithm == "TRPO":
    from agent.trpo_gae import train_model
elif args.algorithm == "PPO":
    from agent.ppo_gae import train_model


if __name__=="__main__":
    env = gym.make(args.env)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    # print('state size:', num_inputs) # 11
    # print('action size:', num_actions) # 3

    writer = SummaryWriter(args.logdir)

    actor = Actor(num_inputs, num_actions)
    critic = Critic(num_inputs)

    running_state = ZFilter((num_inputs,), clip=5)
    
    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr) # hp.actor_lr = 0.0003 
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr, # hp.critic_lr = 0.0003
                              weight_decay=hp.l2_rate) # hp.l2_rate = 0.001


    episodes = 0    

    for iter in range(15000):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []

        while steps < 2048: # ?
            episodes += 1
            state = env.reset()

            # state 
            # [ 1.25242342e+00  2.48001792e-03 -4.00974886e-03 -3.74310984e-03
            # 1.76107279e-03 -4.24441739e-03  4.94643485e-03  3.23042089e-04
            # -4.13514140e-03  3.71542489e-03 -3.37123480e-03]
            # running_state(state) 
            # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

            # state 
            # [ 1.25227644e+00 -4.76101915e-03 -4.70683807e-03 -4.60811685e-03
            # 4.24723880e-04 -1.29097427e-03  2.35219037e-03 -1.20900498e-03
            # 4.22748799e-03  2.70184498e-03 -1.58194091e-03]
            # running_state(state) 
            # [ 0.97938591  0.5318234   0.61519367  0.59847909 -1.63746175  0.76422659
            # 1.44379947  0.81533691  0.78033282  0.75001262 -0.78464144]

            # running_state는 input으로 들어오는 state의 scale이 일정하지 않기 때문에 사용한다.
            # 즉, state의 각 dimension을 평균 0 분산 1로 standardization하는 것이다. 
            # 따라서 모델을 저장할 때 각 dimension 마다의 평균과 분산도 같이 저장해서 
            # 테스트할 때 불러와서 사용해야 한다.
            # print("state", state)
            state = running_state(state)
            # print("running_state(state)", state)
            
            score = 0

            for _ in range(10000): # ?
                if args.render:
                    env.render()

                steps += 1
                mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                next_state, reward, done, _ = env.step(action)

                # next_state [ 1.25204271  0.00339264 -0.00230297 -0.00529779  0.00736752  0.01062921
                # -0.1001518   0.22747316  0.43029206 -0.39176969  1.4032258 ]
                # print("next_state", next_state)
                next_state = running_state(next_state)
                # [-0.70708052  0.70709582  0.70710092 -0.70710035  0.707105    0.70710611
                # -0.70710669  0.70710674  0.70710676 -0.70710676  0.70710677]
                # print("running_state(next_state)", next_state)

                if done:
                    mask = 0
                else:
                    mask = 1

                # 차곡차곡 쌓는다!
                memory.append([state, action, reward, mask])

                score += reward
                state = next_state

                if done:
                    break
            
            # for문 종료
            scores.append(score)
        
        # while문 종료
        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        writer.add_scalar('log/score', float(score_avg), iter)

        actor.train(), critic.train() # ?
        train_model(actor, critic, memory, actor_optim, critic_optim)


        if iter % 100:
            score_avg = int(score_avg)

            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'z_filter_n':running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': args,
                'score': score_avg
            }, filename=ckpt_path)
