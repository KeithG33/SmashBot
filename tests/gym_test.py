import numpy as np
from smashbot.model.smash_transformer import SmashTransformer
from smashbot.data.extract_dataset import generate_input_python, parse_game_state

import torch

import melee

import gymnasium as gym 

from ppo import PPO


def main():
    NUM_EPI = 10
    model = SmashTransformer(action_dim=10, embed_dim=112, model_dim=384, nhead=24, num_layers=5, dropout=0.0)
    model = PPO(feature_extractor=model)
    model.to("cuda")

    env = gym.vector.make(
                id='melee-v0',
                asynchronous=False,
                slippi_path='/home/kage/smashbot_workspace/Slippi_Online-x86_64-ExiAI.AppImage',)
    env.shared_memory = False

    num_episode = 0
    while num_episode < NUM_EPI:
        s, _ = env.reset()
        # score = [0, 0]
        done = False
        
        while not done:
            s1, s2 = s["p1"], s["p2"]
            print
            a, raw_a, logprob = model.get_action(torch.stack([torch.tensor(s1), torch.tensor(s2)]).float().cuda())
            # raw_a1, raw_a2 = raw_a[:raw_a.size(0)//2], raw_a[raw_a.size(0)//2:]
            # log_prob1, log_prob2 = logprob[:logprob.size(0)//2], logprob[logprob.size(0)//2:]
            
            # use fixed action for testing
            a = torch.tensor([[0, 0, 0, 0, 0, 0.99, 0.99, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0.5, 0, 1, 0.5]]).float()
            p1 = {"digital": a[0, :5], "analog": a[0, 5:]}
            p2 = {"digital": a[1, :5], "analog": a[1, 5:]}
            actions = {"p1": p1, "p2": p2}


            # p1 = {"digital": a[0, :5], "analog": a[0, 5:]}
            # p2 = {"digital": a[1, :5], "analog": a[1, 5:]}
            # actions = {"p1": p1, "p2": p2}
            
            s_prime, r, done, _, info = env.step(actions)

            if s_prime is None:
                break
            
            # transition = ((s1, s2), (raw_a1, raw_a2), r, (s_prime["p1"], s_prime["p2"]), (log_prob1.tolist(), log_prob2.tolist()), done)
            # model.put_data(transition)
            
            # s = s_prime
            # score = [sum(x) for x in zip(score, r)]

main()