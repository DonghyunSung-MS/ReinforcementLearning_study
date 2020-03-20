#-*-coding: utf-8-*-
import sys,os
sys.path.append(os.pardir)
from  ddpg.class_NN import *
from copy import deepcopy
from collections import deque
import csv
import time

import gym
from torch.optim import Adam
file=open('../ddpg/ddpg_ep_return_wclip.csv','w',newline='')
file2 =open('../ddpg/ddpg_loss_wclip.csv','w',newline='')
writer = csv.writer(file)
writer2 = csv.writer(file2)
env = gym.make('Pendulum-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_every = 1000
#---------------------------- Hyperparameters---------------------------------
policy_lr = 1e-3
q_lr = 1e-4

rho = 0.999 #polyak
gamma = 0.99 #discount factor

buffer_size = int(100000) #replaybuffer
batch_size = 128

goal_score = -300
epochs = 1000000

start_steps = 1000 #exploring
act_noise = 0
#--------------------------------Algorithm--------------------------------------
def ddpg(env, agent=AC_Agent):
    # seed 고정
    torch.manual_seed(0)
    np.random.seed(0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_limit = env.action_space.high[0]

    ac = agent(env, device, hidden_sizes=(128,128)).float().to(device) #main
    target_ac = deepcopy(ac).float().to(device) #target

    for p in target_ac.parameters():
        p.requires_grad_(False)

    replay_buffer = ReplayBuffer(env, buffer_size)

    def compute_loss_q(samples): #samples ndarray in tupple
        # o, a, r, o_, d
        o, a, r, o_, d = samples
        o = torch.from_numpy(o).to(device)
        a = torch.from_numpy(a).to(device)
        r = torch.from_numpy(r).to(device)
        o_ = torch.from_numpy(o_).to(device)
        d = torch.from_numpy(d).to(device)
        #print(o.dtype,a.dtype,r.dtype,o_.dtype,d.dtype)
        q = ac.critic(o,a).to(device) # Q(o,a)

        with torch.no_grad():
            q_policy_target = target_ac.critic(o_, target_ac.actor(o_)).to(device)
            #print(q_policy_target.dtype)
            backup = r + float(gamma)*(1.0-d)*q_policy_target

        loss_q = ((q-backup)**2).mean()

        return loss_q

    def compute_loss_policy(samples):
        o, a, r, o_, d = samples
        #print(o.shape, a.shape, r.shape, o_.shape, d.shape)
        #print(o.dtype, a.dtype, r.dtype, o_.dtype, d.dtype)
        #print()
        o = torch.from_numpy(o).to(device)
        a = torch.from_numpy(a).to(device)
        r = torch.from_numpy(r).to(device)
        o_ = torch.from_numpy(o_).to(device)
        d = torch.from_numpy(d).to(device)
        q_policy = ac.critic(o, ac.actor(o).to(device)).to(device)
        #print(q_policy.mean().item())

        return -q_policy.mean() # maximize q == minimize -q

    policy_optimizer = Adam(ac.actor.parameters(), lr=policy_lr)
    q_optimizer = Adam(ac.critic.parameters(), lr=q_lr)

    def update(t,samples):
        # main Q net update
        q_optimizer.zero_grad() # gradient log to zero initializer
        loss_q = compute_loss_q(samples)
        loss_q.backward() # backprop using Adam
        torch.nn.utils.clip_grad_norm_(ac.critic.parameters(),1.0)
        q_optimizer.step() # one step updating

        # freeze Q parameters
        for p in ac.critic.parameters():
            p.requires_grad_(False)

        # Policy update
        policy_optimizer.zero_grad()
        loss_policy = compute_loss_policy(samples)
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(ac.actor.parameters(),0.5) #clipping
        policy_optimizer.step()

        # unfreeze Q parameters
        for p in ac.critic.parameters():
            p.requires_grad_(True)

        if t%save_every==0:
            #print("Q loss: {:0.5f}   policy loss: {:0.5f}".format(loss_q.item(),loss_policy.item()))
            writer2.writerow([loss_q.item(),loss_policy.item()])

        # Target Q network update using  polyak averaging
        with torch.no_grad():
            for p, p_target in zip(ac.parameters(), target_ac.parameters()):
                p_target.data.mul_(rho)
                p_target.data.add_((1 - rho)*p.data)

    def get_action(o, noise_scale):
        o = torch.from_numpy(o)
        a = ac.act(o)
        gauss_noise = np.random.randn(act_dim).astype(np.float32)
        a += noise_scale * torch.from_numpy(gauss_noise).to(device)
        a = a.to("cpu")
        return np.clip(a, -act_limit, act_limit)

    o, ep_ret, ep_len = env.reset(), 0, 0

    steps = 0
    recent_return = deque(maxlen=100)

    for epoch in range(epochs):
        d = False
        o, ep_ret, ep_len = env.reset(), 0, 0

        while not d:
            steps += 1
            if steps > start_steps:
                a = get_action(o, act_noise)
            else:
                print("random")
                a = env.action_space.sample()

            o_, r , d, _ = env.step(a)

            ep_ret += r # undiscounted return
            ep_len += 1
            mask = 1.0 if d else 0.0
            #print(o,a,r,o_,d)
            replay_buffer.save_sample((o,a,r,o_,mask))

            o = o_

            if d:
                if epoch%save_every==0:
                    print("ep: {}    ep_ret: {:0.4f}    ep_len: {}".format(epoch,ep_ret.item(),ep_len))
                    writer.writerow([ep_ret.item(),ep_len])
                recent_return.append(ep_ret.item())

            if steps > batch_size:
                batch = replay_buffer.sample_batch(batch_size)
                update(epoch,batch)

            if epoch%save_every==0:
                torch.save(ac,'../ddpg/ddpg_wclip.pth')

        if np.mean(recent_return) > goal_score:
            torch.save(ac,'../ddpg/ddpg_wclip.pth')
            print("Recent returns exeed -300 .Train terminated")
            break


if __name__=='__main__':
    start = time.time()
    ddpg(env)
    end = time.time()-start
    print(end)
