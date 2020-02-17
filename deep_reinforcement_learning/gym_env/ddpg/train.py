#-*-coding: utf-8-*-
import sys,os
sys.path.append(os.pardir)
from  ddpg.class_NN import *
from copy import deepcopy

import gym
from torch.optim import Adam

env = gym.make('MountainCarContinuous-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_every = 1#1000
#---------------------------- Hyperparameters---------------------------------
policy_lr = 1e-3
q_lr = 1e-3

rho = 0.995 #polyak
gamma = 0.99 #discount factor

buffer_size = 20#int(1e6) #replaybuffer
batch_size = 10#100

update_after = 10#1000
update_every = 2#50

max_ep_len = 100#1000
steps_per_epoch = 10#4000
epochs = 100

start_steps = 2#10000
act_noise = 0.1
#--------------------------------Algorithm--------------------------------------
def ddpg(env, agent=AC_Agent):
    # seed 고정
    torch.manual_seed(0)
    np.random.seed(0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_limit = env.action_space.high[0]

    ac = agent(env, device) #main
    target_ac = deepcopy(ac) #target

    for p in target_ac.parameters():
        p.requires_grad_(False)

    replay_buffer = ReplayBuffer(env, buffer_size)

    def compute_loss_q(samples): #samples ndarray in tupple
        # o, a, r, o_, d
        o, a, r, o_, d = samples
        o = torch.from_numpy(o)
        a = torch.from_numpy(a)
        r = torch.from_numpy(r)
        o_ = torch.from_numpy(o_)
        d = torch.from_numpy(d)
        #print(o.dtype,a.dtype,r.dtype,o_.dtype,d.dtype)
        q = ac.critic(o,a) # Q(o,a)

        with torch.no_grad():
            q_policy_target = target_ac.critic(o_, target_ac.actor(o_))
            #print(q_policy_target.dtype)
            backup = r + float(gamma)*(1-d)*q_policy_target

        loss_q = ((q-backup)**2).mean()

        return loss_q

    def compute_loss_policy(samples):
        o, a, r, o_, d = samples
        o = torch.from_numpy(o)
        a = torch.from_numpy(a)
        r = torch.from_numpy(r)
        o_ = torch.from_numpy(o_)
        d = torch.from_numpy(d)
        q_policy = ac.critic(o, ac.actor(o))
        return -q_policy.mean() # maximize q == minimize -q

    policy_optimizer = Adam(ac.actor.parameters(), lr=policy_lr)
    q_optimizer = Adam(ac.critic.parameters(), lr=q_lr)

    def update(samples):
        q_optimizer.zero_grad() # gradient log to zero initializer
        loss_q = compute_loss_q(samples)
        loss_q.backward() # backprop using Adam
        q_optimizer.step() # one step updating
        # freeze Q parameters
        for p in ac.critic.parameters():
            p.requires_grad_(False)

        policy_optimizer.zero_grad()
        loss_policy = compute_loss_policy(samples)
        loss_policy.backward()
        policy_optimizer.step()
        # unfreeze Q parameters
        for p in ac.critic.parameters():
            p.requires_grad_(True)

        #polyak averaging
        with torch.no_grad():
            for p, p_target in zip(ac.parameters(), target_ac.parameters()):
                p_target.data.mul_(rho)
                p_target.data.add_((1 - rho)*p.data)

    def get_action(o, noise_scale):
        a = ac.act(o)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0

    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        o_, r , d, _ = env.step(a)
        ep_ret += r # undiscounted return
        ep_len += 1

        d = False if ep_len==max_ep_len else d

        replay_buffer.save_sample((o,a,r,o_,float(int(d))))

        o = o_

        if t>=update_after and t % update_every==0:
            print("t: {},ep_ret: {},ep_len: {}").format(t,ep_ret,ep_len)
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(batch)

        if t%save_every==0:
            torch.save(ac,'../ddpg.pth')

if __name__=='__main__':
    ddpg(env)
