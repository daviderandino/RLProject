import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from PPO_train_test import PPO_train_adr, PPO_test
import pandas as pd
from tqdm.auto import tqdm 
import os

"""
Utility functions for performing grid search over hyperparameters.
"""



class ADRGridSearch:
    def __init__(
        self,
        train_env_id,
        test_env_id,
        obj_adr_range,
        reward_threshold_range,
        check_freq_range,
        seed,
        timesteps,
        architectures,
    ):
        self.train_env_id = train_env_id
        self.test_env_id = test_env_id
        self.obj_adr_range = obj_adr_range
        self.reward_threshold_range = reward_threshold_range
        self.check_freq_range = check_freq_range
        self.seed = seed
        self.timesteps = timesteps
        self.architectures = architectures
        
        self.obj_adr_values = np.arange(obj_adr_range[0], obj_adr_range[1] + obj_adr_range[2], obj_adr_range[2])
        self.reward_threshold_values = np.arange(reward_threshold_range[0], reward_threshold_range[1] + reward_threshold_range[2], reward_threshold_range[2])
        self.check_freq_values = np.arange(check_freq_range[0], check_freq_range[1] + check_freq_range[2], check_freq_range[2])

    def run_search(self):
        """ """

        combinations = list(
            product(
                self.obj_adr_values,
                self.reward_threshold_values,
                self.check_freq_values,
                self.architectures,
            )
        )
        
        pbar = tqdm(combinations, desc="Total Progress", unit="exp")
        
        for obj_adr, reward_threshold, check_freq, arch in pbar:
            print(f"Running experiment with Objective ADR: {obj_adr}, Reward threshold: {reward_threshold}, Check freq: {check_freq}")

            model_name = f"ppo_source_adr_{obj_adr:.2f}_{reward_threshold}_{check_freq}_{arch}_{self.seed}"

            # serve per skippare se l'esperimento esiste già
            if os.path.exists("grid_search_results.csv"):
                existing_results = pd.read_csv("grid_search_results.csv")
                if model_name in existing_results["model_id"].values:
                    continue
            
            pbar.set_description(f"Processing {model_name}")

            PPO_train_adr(
                train_env_id=self.train_env_id,
                model_name=model_name,
                lr=3e-4,
                lr_scheduler_type="constant",
                steps=self.timesteps,
                starting_adr_range=0.05,
                objective_adr_range=obj_adr,
                increase_rate=0.05,
                reward_to_check=reward_threshold,
                check_frequency=int(check_freq),
                net_size=arch,
                seed=self.seed,
            )

            mean_reward, std_reward = PPO_test(self.test_env_id, model_name)

            new_result = {
                "model_id": model_name,
                "architecture": arch,
                "obj_adr_range": obj_adr,
                "reward_threshold": reward_threshold,
                "check_frequency": check_freq,
                "test_mean_reward": mean_reward,
                "test_std_reward": std_reward,
                "success": 1 if mean_reward >= 1500 else 0,
            }
            
            df_new = pd.DataFrame([new_result])
            
            file_exists = os.path.isfile("grid_search_results.csv")
            
            df_new.to_csv("grid_search_results.csv", 
                          mode='a', 
                          index=False, 
                          header=not file_exists)
            
            pbar.set_postfix({"last_reward": f"{mean_reward:.0f}"})

