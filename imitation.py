import os, random, time

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F

import wandb 

from smashbot.data.dataset import SmashBrosDataset, SequenceBatchSampler, MISC_TYPE, ACTION_TYPE, PROJECTILE_TYPE, PLAYER_TYPE, NANA_TYPE
from smashbot.model.smash_transformer import SmashTransformer
from ppo import PPO

DATA_DIR_TRAIN = '/home/kage/smashbot_workspace/dataset/hickle_action_shuffle/train'
DATA_DIR_TEST = '/home/kage/smashbot_workspace/dataset/hickle_action_shuffle/test'

DATASET_SIZE_TRAIN = 35
DATASET_SIZE_TEST = 2
DATASET_PROCESSES = 1

BATCH_SIZE_TRAIN = 4096
BATCH_SIZE_TEST = 4096
 
VALIDATION_EVERY = 12_500
LOG_EVERY = 250
NUM_ROUNDS = 100
EPOCHS_PER_ROUND = 2

MODEL_SAVEPATH = 'SmashBotTransformer-prevaction.pt'


def sample_test_data(directory, num_samples):
    input_files = [f.path for f in os.scandir(directory) if "inputs" in f.name]
    sampled_input_files = random.sample(input_files, num_samples)
    sampled_output_files = [f.replace("inputs", "outputs") for f in sampled_input_files]
    return list(zip(sampled_input_files, sampled_output_files))

def sample_train_data(directory, num_samples):
    """ Use all files with seq_len > 5 for training, sample from 3, 4, 5"""
    input_files = [f for f in os.scandir(directory) if "inputs" in f.name]
    target_id_files = [f.path for f in input_files if int(f.name.split("_")[0].replace("inputs", "")) in [3,4]]
    higher_id_files = [f.path for f in input_files if int(f.name.split("_")[0].replace("inputs", "")) > 4]
    sampled_target_files = random.sample(target_id_files, num_samples)
    sampled_higher_files = random.sample(higher_id_files, k=25)
    final_sampled_inputs = sampled_target_files + sampled_higher_files
    random.shuffle(final_sampled_inputs)
    sampled_output_files = [f.replace("inputs", "outputs") for f in final_sampled_inputs]
    return list(zip(final_sampled_inputs, sampled_output_files))


def run_validation(model):
    sampled_test_data = sample_test_data(DATA_DIR_TEST, DATASET_SIZE_TEST)
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
            # time.sleep(0.03)

    avg_loss = total_loss / total_count  # Compute average loss  
    avg_buttons_loss = total_buttons_loss / total_count
    avg_sticks_loss = total_sticks_loss / total_count
    return avg_loss, avg_buttons_loss, avg_sticks_loss


def training_round(model, train_loader, num_epochs=10, log_every=1000, validation_every=20_000):
    best_val_loss = 1000

    # Pytorch train stuffs
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
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
                    "iter": i,
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
            # time.sleep(0.03)

        print(f"Epoch took {time.perf_counter()-t1} seconds ")
        torch.save(model.state_dict(), MODEL_SAVEPATH)


def run_training(num_rounds, model):
    for round in range(num_rounds):
        # build dataset 
        # randomly sample dataset_size files 
        t1 = time.perf_counter()
        sampled_train_data = sample_train_data(DATA_DIR_TRAIN, DATASET_SIZE_TRAIN)
        train_dataset = SmashBrosDataset(sampled_train_data, num_processes=DATASET_PROCESSES)
        sampler = SequenceBatchSampler(train_dataset, BATCH_SIZE_TRAIN)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler)
        print(f"Round {round}: loaded dataset with {len(train_dataset)} states - {time.perf_counter()-t1} seconds")

        training_round(model, train_loader, num_epochs=EPOCHS_PER_ROUND, log_every=LOG_EVERY, validation_every=VALIDATION_EVERY)
        del train_dataset
        del sampler
        del train_loader



if __name__ == "__main__":
    model = SmashTransformer(action_dim=10, embed_dim=112, model_dim=504, nhead=24, num_layers=5, dropout=0.0)
    # model = SmashTransformer(action_dim=10, model_dim=384, nhead=24, num_layers=5, dropout=0.0)
    model = PPO(feature_extractor=model)
    model = model.cuda()
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    if os.path.exists(MODEL_SAVEPATH):
        model.load_state_dict(torch.load(MODEL_SAVEPATH))
        print("Loaded model weights from previous training session")

    wandb.init(project="smashbot")
    # wandb.init(project="smashbot", id='s1j26a2k', resume='must')

    run_training(NUM_ROUNDS, model) 
    # run_validation(model)
