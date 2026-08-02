""" From minimalRL/ppo-continuous.py repo"""
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import wandb

import smashbot

from smashbot.model.smash_transformer import SmashTransformer

#Hyperparameters
learning_rate  = 0.0003
gamma           = 0.9
lmbda           = 0.9
eps_clip        = 0.2
K_epoch         = 3
rollout_len    = 64
buffer_size    = 10
minibatch_size = 32

class PPO(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.data = []
        self.action_dim = 10
        self.device = 'cuda'
        self.fc1 = feature_extractor

        self.fc_mu_std = nn.LazyLinear(2*self.action_dim).to(self.device)
        self.fc_v = nn.LazyLinear(1).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0


    def pi(self, x):
        x = x.to(self.device)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1.extract_features(x))
        mu_std = self.fc_mu_std(x)
        mus = 2.0*torch.tanh(mu_std[:, :self.action_dim])
        stds = F.softplus(mu_std[:, self.action_dim:])
        return mus, stds
    
    def v(self, x):
        x = x.to(self.device)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1.extract_features(x).squeeze())
        v = self.fc_v(x)
        return v
    
    def get_action(self, x):
        mus, stds = self.pi(x)
        dist = Normal(mus, stds)
        raw_a = dist.sample()
        log_p = dist.log_prob(raw_a)

        digital = raw_a[:5]
        values, indices = torch.sort(digital, descending=True)
        valid_indices = indices[values > 0.5]
        
        # Get up to two 
        multi_hot = torch.zeros_like(digital) 
        if valid_indices.numel() > 0:
            top_indices = valid_indices[:min(2, valid_indices.size(0))]
            multi_hot[top_indices] = 1

        a = torch.cat([multi_hot, raw_a[5:]]).cpu().numpy()
        return a, raw_a, log_p 

    def get_action(self, x):
        mus, stds = self.pi(x)
        dist = Normal(mus, stds)
        raw_a = dist.sample()
        log_p = dist.log_prob(raw_a)

        # Process digital actions for batches
        digital = raw_a[..., :5]  # Assuming first 5 are digital
        values, indices = torch.sort(digital, dim=1, descending=True)
        multi_hot = torch.zeros_like(digital)
        for b in range(digital.size(0)):
            valid_indices = indices[b, values[b] > 0.5]
            top_indices = valid_indices[:2]  # Get up to top two
            multi_hot[b, top_indices] = 1

        a = torch.cat([multi_hot, raw_a[:, 5:]], dim=1).cpu().numpy()

        return a, raw_a, log_p
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        """ Greedily batches similar shaped data"""
        data_batches = []
        curr_shape = None
        curr_pshape = None
        batch_count = 0  # Initialize batch count

        s1_batch, a1_batch, r1_batch, s1_prime_batch, prob1_a_batch, done1_batch = [], [], [], [], [], []
        s2_batch, a2_batch, r2_batch, s2_prime_batch, prob2_a_batch = [], [], [], [], []

        while len(self.data) > 0:
            (s1, s2), (a1, a2), (r1, r2), (s1_prime, s2_prime), (prob1_a, prob2_a), done = self.data.pop(0)
            s_shape = len(s1)  # Assuming the shape is defined by the first dimension of s1
            sp_shape = len(s1_prime)  

            if curr_shape is None:
                curr_shape = s_shape
                curr_pshape = sp_shape
                same_shape = True
            elif curr_shape != s_shape or curr_pshape != sp_shape:
                same_shape = False
                # Save the current batch before switching to a new shape
                if batch_count > 0:  # Check if there is any data in the current batch
                    mini_batch = (
                        (torch.as_tensor(np.stack(s1_batch)), torch.as_tensor(np.stack(s2_batch))),
                        (torch.stack(a1_batch), torch.stack(a2_batch)),
                        (torch.as_tensor(np.stack(r1_batch), dtype=torch.float), torch.as_tensor(np.stack(r2_batch), dtype=torch.float)),
                        (torch.as_tensor(np.stack(s1_prime_batch)), torch.as_tensor(np.stack(s2_prime_batch))),
                        (torch.as_tensor(np.stack(prob1_a_batch)), torch.as_tensor(np.stack(prob2_a_batch))),
                        torch.as_tensor(np.stack(done1_batch), dtype=torch.float)
                    )
                    data_batches.append(mini_batch)
                    # Reset batches
                    s1_batch, a1_batch, r1_batch, s1_prime_batch, prob1_a_batch, done1_batch = [], [], [], [], [], []
                    s2_batch, a2_batch, r2_batch, s2_prime_batch, prob2_a_batch = [], [], [], [], []
                    batch_count = 0  # Reset batch count

                curr_shape = s_shape  # Update current shape for the new batch
                curr_pshape = sp_shape

            if same_shape or curr_shape == s_shape: # Same shape or new shape
                s1_batch.append(s1)
                s2_batch.append(s2)
                a1_batch.append(a1)
                a2_batch.append(a2)
                r1_batch.append(r1)
                r2_batch.append(r2)
                s1_prime_batch.append(s1_prime)
                s2_prime_batch.append(s2_prime)
                prob1_a_batch.append(prob1_a)
                prob2_a_batch.append(prob2_a)
                done1_batch.append(0 if done else 1)
                batch_count += 1

                # Save the batch if it reaches the minibatch size
                if batch_count >= minibatch_size:
                    mini_batch = (
                        (torch.as_tensor(np.stack(s1_batch)), torch.as_tensor(np.stack(s2_batch))),
                        (torch.stack(a1_batch), torch.stack(a2_batch)),
                        (torch.as_tensor(np.stack(r1_batch), dtype=torch.float), torch.as_tensor(np.stack(r2_batch), dtype=torch.float)),
                        (torch.as_tensor(np.stack(s1_prime_batch)), torch.as_tensor(np.stack(s2_prime_batch))),
                        (torch.as_tensor(np.stack(prob1_a_batch)), torch.as_tensor(np.stack(prob2_a_batch))),
                        torch.as_tensor(np.stack(done1_batch), dtype=torch.float)
                    )
                    data_batches.append(mini_batch)
                    # Reset batches
                    s1_batch, a1_batch, r1_batch, s1_prime_batch, prob1_a_batch, done1_batch = [], [], [], [], [], []
                    s2_batch, a2_batch, r2_batch, s2_prime_batch, prob2_a_batch = [], [], [], [], []
                    batch_count = 0  # Reset batch count

        # Handle any remaining data in the batch
        if batch_count > 0:
            mini_batch = (
                (torch.tensor(np.stack(s1_batch)), torch.tensor(np.stack(s2_batch))),
                (torch.stack(a1_batch), torch.stack(a2_batch)),
                (torch.tensor(np.stack(r1_batch), dtype=torch.float), torch.tensor(np.stack(r2_batch), dtype=torch.float)),
                (torch.tensor(np.stack(s1_prime_batch)), torch.tensor(np.stack(s2_prime_batch))),
                (torch.tensor(np.stack(prob1_a_batch)), torch.tensor(np.stack(prob2_a_batch))),
                torch.tensor(np.stack(done1_batch), dtype=torch.float)
            )
            data_batches.append(mini_batch)

        return data_batches
    
    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            (s1, s2), (a1, a2), (r1, r2), (s1_prime, s2_prime), (prob1_a, prob2_a), done_mask = mini_batch

            batchsize = s1.size(0)

            s_batch = torch.cat([s1, s2])
            s_prime_batch = torch.cat([s1_prime, s2_prime])

            with torch.no_grad():
                s_out = self.v(s_batch)
                s_prime_out = self.v(s_prime_batch)

                td_target1 = r1.cuda() + gamma * s_prime_out[:batchsize].squeeze() * done_mask.cuda()
                td_target2 = r2.cuda() + gamma * s_prime_out[batchsize:].squeeze() * done_mask.cuda()

                delta1 = td_target1 - s_out[:batchsize]
                delta2 = td_target2 - s_out[batchsize:]

            delta1 = delta1.cpu().numpy()
            delta2 = delta2.cpu().numpy()

            advantage_lst1, advantage_lst2 = [], []
            advantage1, advantage2 = 0.0, 0.0

            for delta_t1, delta_t2 in zip(delta1[::-1], delta2[::-1]):
                advantage1 = gamma * lmbda * advantage1 + delta_t1[0]
                advantage2 = gamma * lmbda * advantage2 + delta_t2[0]
                advantage_lst1.append([advantage1])
                advantage_lst2.append([advantage2])
            
            advantage_lst1.reverse()
            advantage_lst2.reverse()
            advantage1 = torch.tensor(advantage_lst1, dtype=torch.float)
            advantage2 = torch.tensor(advantage_lst2, dtype=torch.float)
            data_with_adv.append(((s1,s2), (a1,a2), (r1,r2), (s1_prime,s2_prime), done_mask,
                                  (prob1_a,prob2_a), (td_target1,td_target2), (advantage1,advantage2)))

        return data_with_adv

    
    def train_net(self, log=False):
        """ Train using self-play tuples of data"""
        data = self.make_batch()
        data = self.calc_advantage(data)

        for i in range(K_epoch):
            print(f"Train epoch {i}")
            epoch_loss = 0
            epoch_policy_loss = 0
            epoch_value_loss = 0
            for mini_batch in data:
                # Unpack data for both players and move to GPU
                (s1, s2), (a1, a2), (r1, r2), (s_prime1, s_prime2), done, (old_log_prob1, old_log_prob2), (td_target1, td_target2), (advantage1, advantage2) = mini_batch
                old_log_prob1, old_log_prob2 = old_log_prob1.cuda(), old_log_prob2.cuda()
                td_target1, td_target2 = td_target1.cuda(), td_target2.cuda()
                advantage1, advantage2 = advantage1.cuda(), advantage2.cuda()

                states = torch.cat([s1, s2], dim=0)
                actions = torch.cat([a1, a2], dim=0)
                old_log_probs = torch.cat([old_log_prob1, old_log_prob2], dim=0)
                td_targets = torch.cat([td_target1, td_target2], dim=0)
                advantages = torch.cat([advantage1, advantage2], dim=0)

                mus, stds = self.pi(states)
                dists = Normal(mus, stds)
                log_probs = dists.log_prob(actions)
                ratios = torch.exp(log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                loss_policy = -torch.min(surr1, surr2)
                loss_value = F.smooth_l1_loss(self.v(states).squeeze(), td_targets)

                total_loss = (loss_policy + loss_value).mean()

                # Optimize model parameters
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
                self.optimization_step += 1

                epoch_loss += total_loss.item()
                epoch_policy_loss += loss_policy.mean().item()
                epoch_value_loss += loss_value.mean().item()

            print(f"Epoch loss: {epoch_loss}")
            if log:
                wandb.log({
                    "ppo/epoch_loss": epoch_loss,
                    "ppo/epoch_policy_loss": epoch_policy_loss,
                    "ppo/epoch_value_loss": epoch_value_loss})

def main():
    model = SmashTransformer(action_dim=10, embed_dim=112, model_dim=384, nhead=24, num_layers=5, dropout=0.0)
    model = PPO(feature_extractor=model)
    model.to("cuda")

    env = gym.make(
                id='melee-v0',
                slippi_path='/home/kage/smashbot_workspace/Slippi_Online-x86_64-ExiAI.AppImage')

    for n_epi in range(40):
        s, _ = env.reset()
        score = [0,0]
        done = False
        count = 0
        while not done:
            s1, s2 = s["p1"], s["p2"]

            a, raw_a, logprob = model.get_action(torch.stack([torch.tensor(s1), torch.tensor(s2)]).float().cuda())
            raw_a1, raw_a2 =  raw_a[:raw_a.size(0)//2], raw_a[raw_a.size(0)//2:]
            log_prob1, log_prob2 = logprob[:logprob.size(0)//2], logprob[logprob.size(0)//2:]

            p1 = {"digital": a[0, :5], "analog": a[0, 5:]}
            p2 = {"digital": a[1, :5], "analog": a[1, 5:]}
            
            actions = {"p1": p1, "p2": p2}
            s_prime, r, done, _, info = env.step(actions)

            if s_prime is None:
                break

            (s1_prime, s2_prime) = (s_prime["p1"], s_prime["p2"])
            transition = ((s1,s2), (raw_a1,raw_a2), r, (s1_prime,s2_prime), (log_prob1.tolist(),log_prob2.tolist()), done)
            model.put_data(transition)

            s = s_prime
            score = [sum(x) for x in zip(score, r)]
            count += 1

        model.train_net()
        print(f"Episode: {n_epi}, Score: {score}, Count: {count}")

def run_ppo(model, num_episodes, env_params):
    env = gym.make(**env_params)
    for n_epi in range(num_episodes):
        s, _ = env.reset()
        score = [0,0]
        done = False
        count = 0
        while not done:
            s1, s2 = s["p1"], s["p2"]

            a, raw_a, logprob = model.get_action(torch.stack([torch.tensor(s1), torch.tensor(s2)]).float().cuda())
            raw_a1, raw_a2 =  raw_a[:raw_a.size(0)//2], raw_a[raw_a.size(0)//2:]
            log_prob1, log_prob2 = logprob[:logprob.size(0)//2], logprob[logprob.size(0)//2:]

            p1 = {"digital": a[0, :5], "analog": a[0, 5:]}
            p2 = {"digital": a[1, :5], "analog": a[1, 5:]}
            
            actions = {"p1": p1, "p2": p2}
            s_prime, r, done, _, info = env.step(actions)

            if s_prime is None:
                break

            (s1_prime, s2_prime) = (s_prime["p1"], s_prime["p2"])
            transition = ((s1,s2), (raw_a1,raw_a2), r, (s1_prime,s2_prime), (log_prob1.tolist(),log_prob2.tolist()), done)
            model.put_data(transition)

            s = s_prime
            score = [sum(x) for x in zip(score, r)]
            count += 1

        model.train_net()
        print(f"Episode: {n_epi}, Score: {score}, Count: {count}")

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    main()
    print("Training Done!")
