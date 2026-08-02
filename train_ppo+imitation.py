import os, random, time

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import gymnasium as gym
import wandb
from ppo import PPO

from imitation import sample_data, SmashBrosDataset, SequenceBatchSampler, training_round
from smashbot.model import SmashTransformer


TRAIN_ROUNDS = 200

NUM_EPISODES = 200 # RL


DATA_DIR_TRAIN = '/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/hickle_shuffle/train'
DATA_DIR_TEST = '/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/hickle_shuffle/test'

DATASET_SIZE_TRAIN = 10
DATASET_SIZE_TEST = 1
DATASET_PROCESSES = 1

BATCH_SIZE_TRAIN = 2048
BATCH_SIZE_TEST = 2048

NUM_ROUNDS = 20 # imitation
EPOCHS_PER_ROUND = 1
LOG_EVERY = 250
VALIDATION_EVERY = 10000

MODEL_SAVEPATH = '/home/kage/smashbot_workspace/SmashBot/SmashBotTransformerPPO+BC.pt'
MODEL_BESTPATH = '/home/kage/smashbot_workspace/SmashBot/SmashBotTransformerPPO+BC_best.pt'


def sample_data(directory, num_samples):
    input_files = [f.path for f in os.scandir(directory) if "inputs" in f.name]
    sampled_input_files = random.sample(input_files, num_samples)
    sampled_output_files = [f.replace("inputs", "outputs") for f in sampled_input_files]
    return list(zip(sampled_input_files, sampled_output_files))


def run_validation(model):
    sampled_test_data = sample_data(DATA_DIR_TEST, DATASET_SIZE_TEST)
    num_processes = min(DATASET_PROCESSES, DATASET_SIZE_TEST)
    t1 = time.perf_counter()
    test_dataset = SmashBrosDataset(sampled_test_data, num_processes=num_processes)
    sampler = SequenceBatchSampler(test_dataset, BATCH_SIZE_TEST)
    val_loader = DataLoader(test_dataset, batch_sampler=sampler)   
    print(f"Loaded validation dataset with {len(test_dataset)} positions - {time.perf_counter()-t1} seconds")

    model.eval()

    t1 = time.perf_counter()
    total_loss = 0.0
    total_buttons_loss = 0.0
    total_sticks_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to('cuda' if torch.cuda.is_available() else 'cpu')
            target = target.to('cuda' if torch.cuda.is_available() else 'cpu')

            pred_policy = model.fc1(input)
                        
            loss, buttons_loss, sticks_loss = model.fc1.calculate_loss(pred_policy, target)
            total_loss += loss.item() * input.size(0)  # Multiply by batch to get total
            total_buttons_loss += buttons_loss.item() * input.size(0)
            total_sticks_loss += sticks_loss.item() * input.size(0)
            total_count += input.size(0) 
            # to lower power-consumption/heat
            time.sleep(0.1)

    avg_loss = total_loss / total_count  # Compute average loss  
    avg_buttons_loss = total_buttons_loss / total_count
    avg_sticks_loss = total_sticks_loss / total_count
    return avg_loss, avg_buttons_loss, avg_sticks_loss


def training_round(model, train_loader, num_epochs=10, log_every=1000, validation_every=20_000):
    best_val_loss = 1000

    # Pytorch train stuffs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=500_000)
    grad_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs): 
        model.train()
        t1 = time.perf_counter()
        
        for i, (input, target) in enumerate(train_loader):
            input = input.float().to('cuda' if torch.cuda.is_available() else 'cpu')
            target = target.float().to('cuda' if torch.cuda.is_available() else 'cpu')

            # AMP with gradient clipping and lr scheduling
            with autocast():
                pred_policy = model.fc1(input)
                loss, buttons_loss, sticks_loss = model.fc1.calculate_loss(pred_policy, target)
            
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            scale = grad_scaler.get_scale()
            grad_scaler.update()

            skip_lr_sched = scale > grad_scaler.get_scale()
            
            if not skip_lr_sched: scheduler.step()

            if i % log_every == 0:
                # print(f"Epoch {epoch}, Iteration {i}, Loss: {loss}, Buttons Loss: {buttons_loss}, Sticks Loss: {sticks_loss}")
                wandb.log({
                    "imitation/train_loss": loss.item(),
                    "imitation/buttons_loss": buttons_loss.item(),
                    "imitation/sticks_loss": sticks_loss.item(),
                })
            
            if i % validation_every == 0 and i > 0 :
                val_loss, val_buttons_loss, val_sticks_loss = run_validation(model)
                print(f"Validation loss: {val_loss}")
            
                wandb.log({
                    "imitation/val_loss": val_loss,
                    "imitation/val_buttons_loss": val_buttons_loss,
                    "imitation/val_sticks_loss": val_sticks_loss
                    })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), MODEL_SAVEPATH)
                
                model.train()
            
            # to lower power-consumption/heat
            time.sleep(0.1)

        print(f"Epoch took {time.perf_counter()-t1} seconds ")
        torch.save(model.state_dict(), MODEL_SAVEPATH)


def run_imitation(model, num_rounds):
    for round in range(num_rounds):
        print(f"Starting round {round}")
        # build dataset 
        # randomly sample dataset_size files 
        t1 = time.perf_counter()
        sampled_train_data = sample_data(DATA_DIR_TRAIN, DATASET_SIZE_TRAIN)
        train_dataset = SmashBrosDataset(sampled_train_data, num_processes=DATASET_PROCESSES)
        sampler = SequenceBatchSampler(train_dataset, BATCH_SIZE_TRAIN)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler)
        print(f"Successfully loaded dataset with {len(train_dataset)} images - {time.perf_counter()-t1} seconds")
    
        training_round(model, train_loader, num_epochs=EPOCHS_PER_ROUND, log_every=LOG_EVERY, validation_every=VALIDATION_EVERY)


def run_ppo(model, env_params, num_episodes):
    # Initialize env
    env = gym.make(**env_params)
    
    for epi in range(num_episodes):
        s, _ = env.reset()
        score = [0, 0]
        done = False
        
        while not done:
            s1, s2 = s["p1"], s["p2"]
            a, raw_a, logprob = model.get_action(torch.stack([torch.tensor(s1), torch.tensor(s2)]).float().cuda())
            raw_a1, raw_a2 = raw_a[:raw_a.size(0)//2], raw_a[raw_a.size(0)//2:]
            log_prob1, log_prob2 = logprob[:logprob.size(0)//2], logprob[logprob.size(0)//2:]
            
            # use fixed action for testing
            a = torch.tensor([[0, 0, 0, 0, 0, 0.99, 0.99, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0.5, 0, 1, 0.5]]).float()
            p1 = {"digital": a[0, :5], "analog": a[0, 5:]}
            p2 = {"digital": a[1, :5], "analog": a[1, 5:]}
            actions = {"p1": p1, "p2": p2}

            p1 = {"digital": a[0, :5], "analog": a[0, 5:]}
            p2 = {"digital": a[1, :5], "analog": a[1, 5:]}
            actions = {"p1": p1, "p2": p2}
            
            s_prime, r, done, _, info = env.step(actions)
            if s_prime is None:
                break
            
            transition = ((s1, s2), (raw_a1, raw_a2), r, (s_prime["p1"], s_prime["p2"]), (log_prob1.tolist(), log_prob2.tolist()), done)
            model.put_data(transition)
            
            s = s_prime
            score = [sum(x) for x in zip(score, r)]
        
        model.train_net(log=False)
        print(f"Episode: {epi}, Score: {score}")


def run_duel(best_model, curr_model, env_params, num_matches=11):
    env = gym.make(**env_params)
    curr_wins = 0

    best_model = SmashTransformer(action_dim=10, embed_dim=112, model_dim=384, nhead=24, num_layers=5, dropout=0.0)
    best_model = PPO(feature_extractor=best_model).cuda()
    best_model.load_state_dict(torch.load(MODEL_BESTPATH))

    for _ in range(num_matches):
        s, _ = env.reset()
        done = False 
        while not done:
            s1, s2 = s["p1"], s["p2"]
            a_best, raw_a, logprob = best_model.get_action(torch.tensor(s1).float().cuda())
            a_curr, raw_a, logprob = curr_model.get_action(torch.tensor(s1).float().cuda())

            p1 = {"digital": a_best[0, :5], "analog": a_best[0, 5:]}
            p2 = {"digital": a_curr[0, :5], "analog": a_curr[0, 5:]}
            actions = {"p1": p1, "p2": p2}
            
            s_prime, r, done, _, info = env.step(actions)
            
            if done or s_prime is None:
                if r[1] > r[0]: 
                    curr_wins += 1
                break

    win_rate = curr_wins / num_matches
    return win_rate > 0.55  # True if curr wins more than 55% of the matches


def run_train(model, num_rounds, num_episodes):
    """ Run a ppo + imitation learning scheme to train the model"""
    env_params = {
        "id": 'melee-v0',
        "slippi_path": '/home/kage/smashbot_workspace/Slippi_Online-x86_64-ExiAI.AppImage'
    }

    torch.save(model.state_dict(), MODEL_BESTPATH)

    for i in range(TRAIN_ROUNDS):
        run_imitation(model, num_rounds)
        # run_ppo(model, env_params, num_episodes)

        # Evaluate model
        # win_rate = run_duel(MODEL_BESTPATH, model, env_params)
        # print(f"Win rate: {win_rate}")
        # if win_rate > 0.55:
        #     torch.save(model.state_dict(), MODEL_BESTPATH)
        #     print("Old best model defeated, new best model saved")


        # # Adjust supervised - RL ratio
        # if i % 20 == 0:
        #     num_rounds = max(1, num_rounds - 1)
        #     num_episodes = min(1000, num_episodes + 20)

if __name__ == "__main__":
    # wandb.init(project="smashbot")
    
    model = SmashTransformer(action_dim=10, embed_dim=112, model_dim=384, nhead=24, num_layers=5, dropout=0.0)
    model = PPO(feature_extractor=model)
    if os.path.exists(MODEL_SAVEPATH):
        print(f"Loading previous model weights from {MODEL_SAVEPATH}")
        weights = torch.load(MODEL_SAVEPATH)
        model.load_state_dict(weights)
    model = model.to("cuda")


    run_train(model, NUM_ROUNDS, NUM_EPISODES)