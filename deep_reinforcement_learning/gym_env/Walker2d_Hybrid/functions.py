import torch
import numpy as np

def obs2feature(observation, args):
    #observation [x, z, pitch, x_dot, z_dot, pitch_dot ]
    #HyperParameter
    K_x = [0.01, 0.02]
    K_z = [0.01, 0.02]
    K_f = 0.02
    #print(observation)
    Fx = K_x[0]*(args.target_pitch - observation[2]) + K_x[1]*(-observation[5])
    Fz = K_z[0]*(args.target_height - observation[1]) + K_z[1]*(-observation[4])
    xp = observation[0] + K_f*(args.target_velocity - observation[3]) #world frame

    return np.array([Fx, Fz, xp])

def calculate_gae(masks,rewards,old_values, args):

    previous_advantage = 0
    previsous_rewards2go = 0
    target = 0

    advantages = torch.zeros_like(masks)
    rewards2go = torch.zeros_like(masks)
    old_values = old_values.squeeze(-1)

    for i in reversed(range(0,len(masks))):
        if masks[i]==0:
            target = rewards[i]

            advantages[i] = rewards[i] - old_values.data[i]
            rewards2go[i] = rewards[i]

            previous_advantage = advantages[i]
            previsous_rewards2go = rewards2go[i]
        else:
            target = rewards[i] + args.gamma*old_values.data[i+1]
            td_residual = target - old_values.data[i]

            advantages[i] = td_residual + args.gamma * args.lamda * previous_advantage
            rewards2go[i] = rewards[i] + args.gamma * previsous_rewards2go

            previous_advantage = advantages[i]
            previsous_rewards2go = rewards2go[i]

    return rewards2go, advantages

def surrogate_loss(actor, old_policy_log,
                   advantages_samples, states_samples,  actions_samples,
                   mini_batch_index):
    mu, std = actor(torch.Tensor(states_samples))
    new_policy_samples = actor.get_log_prob(actions_samples, mu, std)
    old_policy_samples = old_policy_log[mini_batch_index]
    ratio = torch.exp(new_policy_samples - old_policy_samples)
    surrogate_loss = ratio * advantages_samples

    return surrogate_loss, ratio

def update(actor, critic,
           actor_optim, critic_optim,
           trajectories, state_size, action_size,
           args):

    trajectories = np.array(trajectories)
    state_features = np.vstack(trajectories[:,0])
    actions = list(trajectories[:,1])
    rewards = list(trajectories[:,2])
    masks = list(trajectories[:,3])
    #print(masks[-1])

    actions = torch.Tensor(actions).squeeze(1)
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)

    old_values = critic(torch.Tensor(state_features))

    #Calculate Adavantage for each time step
    rewards2go, advantages = calculate_gae(masks,rewards,old_values, args)
    #print(torch.Tensor(state_features).shape, advantages.shape, actions.shape, old_values.shape)
    mu, std = actor(torch.Tensor(state_features))
    old_policy_log = actor.get_log_prob(actions, mu, std)

    criterion = torch.nn.MSELoss()

    n = len(state_features)
    arr = np.arange(n)

    #Batch training
    for _ in range(args.model_update_num):
        np.random.shuffle(arr) #remove correlation
        for i in range(n//args.batch_size):
            mini_batch_index = arr[args.batch_size*i : args.batch_size*(i+1)]
            mini_batch_index = torch.LongTensor(mini_batch_index)

            states_samples = torch.Tensor(state_features)[mini_batch_index]
            actions_samples = actions[mini_batch_index]
            advantages_samples = advantages.unsqueeze(1)[mini_batch_index]
            rewards2go_samples = rewards2go.unsqueeze(1)[mini_batch_index]

            old_values_samples = old_values[mini_batch_index].detach()

            new_values_samples = critic(states_samples)

            #Monte
            critic_loss = criterion(new_values_samples, rewards2go_samples)
            #Surrogate Loss
            actor_loss, ratio = surrogate_loss(actor, old_policy_log.detach(),
                               advantages_samples, states_samples,  actions_samples,
                               mini_batch_index)
            ratio_clipped = torch.clamp(ratio, 1-args.clip_param, 1-args.clip_param)
            actor_loss = -torch.min(actor_loss,ratio_clipped*advantages_samples).mean()

            # update actor & critic
            loss = actor_loss + 0.5 * critic_loss

            critic_optim.zero_grad()
            loss.backward(retain_graph=True)
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()
