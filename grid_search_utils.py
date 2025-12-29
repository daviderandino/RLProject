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
        increase_rate,
        learning_rate,
        lr_scheduler_type,
        starting_adr_range,
    ):
        self.train_env_id = train_env_id
        self.test_env_id = test_env_id
        self.obj_adr_range = obj_adr_range
        self.reward_threshold_range = reward_threshold_range
        self.check_freq_range = check_freq_range
        self.seed = seed
        self.timesteps = timesteps
        self.architectures = architectures
        self.increase_rate = increase_rate
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.starting_adr_range = starting_adr_range
        
        
        self.obj_adr_values = np.arange(obj_adr_range[0], obj_adr_range[1] + obj_adr_range[2], obj_adr_range[2])
        self.reward_threshold_values = np.arange(reward_threshold_range[0], reward_threshold_range[1] + reward_threshold_range[2], reward_threshold_range[2])
        self.check_freq_values = np.arange(check_freq_range[0], check_freq_range[1] + check_freq_range[2], check_freq_range[2])
        self.increase_values = np.arange(increase_rate[0], increase_rate[1] + increase_rate[2], increase_rate[2])

    def run_search(self):
        """ """

        combinations = list(
            product(
                self.obj_adr_values,
                self.reward_threshold_values,
                self.check_freq_values,
                self.architectures,
                self.increase_values
            )
        )
        
        pbar = tqdm(combinations, desc="Total Progress", unit="exp")
        
        for obj_adr, reward_threshold, check_freq, arch, increase_rate in pbar:
            print(f"Running experiment with Objective ADR: {obj_adr}, Increase rate: {increase_rate}, Reward threshold: {reward_threshold}, Check freq: {check_freq}")

            model_name = f"ppo_source_adr_{increase_rate:.2f}_{obj_adr:.2f}_{reward_threshold}_{check_freq}_{arch}_{self.seed}"

            # serve per skippare se l'esperimento esiste già
            if os.path.exists("grid_search_results.csv"):
                existing_results = pd.read_csv("grid_search_results.csv")
                if model_name in existing_results["model_id"].values:
                    continue
            
            pbar.set_description(f"Processing {model_name}")

            PPO_train_adr(
                train_env_id=self.train_env_id,
                model_name=model_name,
                lr=self.learning_rate,
                lr_scheduler_type=self.lr_scheduler_type,
                steps=self.timesteps,
                starting_adr_range=self.starting_adr_range,
                objective_adr_range=obj_adr,
                increase_rate=increase_rate,
                reward_to_check=reward_threshold,
                check_frequency=int(check_freq),
                net_size=arch,
                seed=self.seed,
            )

            mean_reward, std_reward = PPO_test(self.test_env_id, model_name)

            new_result = {
                "model_id": model_name,
                "architecture": arch,
                "lr_scheduler_type": self.lr_scheduler_type,
                "learning_rate": self.learning_rate,
                "starting_adr_range": self.starting_adr_range,
                "increase_rate": increase_rate,
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

