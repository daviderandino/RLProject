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
        csv_name,
        train_env_id,
        test_env_id,
        obj_adr_range,
        reward_threshold,
        check_freq,
        timesteps,
        architecture,
        increase_rate,
        learning_rate,
        lr_scheduler_type,
        starting_adr_range,
        seed
    ):  
        self.csv_name = csv_name
        self.train_env_id = train_env_id
        self.test_env_id = test_env_id

        # --- Hyperparameters ---
        # Fixed hypers
        self.increase_rate = increase_rate
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.starting_adr_range = starting_adr_range
        self.seed = seed
        self.timesteps = timesteps
        
        # Grid search hypers
        self.architectures = architecture
        self.obj_adr_ranges = obj_adr_range
        self.reward_thresholds = reward_threshold
        self.check_freqs = check_freq


    def run_search(self):
        combinations = list(
            product(
                self.obj_adr_ranges,
                self.reward_thresholds,
                self.check_freqs,
                self.architectures
            )
        )
        
        pbar = tqdm(combinations, desc="Total Progress", unit="exp")
        
        for obj_adr, reward_threshold, check_freq, arch in pbar:

            model_name = f"{self.train_env_id}_adr_{self.lr_scheduler_type}_{obj_adr}_{reward_threshold}_{self.increase_rate}_{check_freq}_{arch}_{self.seed}"

            # skippa se l'esperimento esiste già
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
                increase_rate=self.increase_rate,
                reward_to_check=reward_threshold,
                check_frequency=int(check_freq),
                net_size=arch,
                seed=self.seed,
            )

            mean_reward, std_reward, mean_len, std_len = PPO_test(self.test_env_id, model_name)

            new_result = {
                "model_id": model_name,
                "lr_scheduler_type": self.lr_scheduler_type,
                "obj_adr_range": obj_adr,
                "reward_threshold": reward_threshold,
                "increase_rate": self.increase_rate,
                "check_frequency": check_freq,
                "architecture": arch,
                "learning_rate": self.learning_rate,
                "starting_adr_range": self.starting_adr_range,
                "test_mean_reward": mean_reward,
                "test_std_reward": std_reward,
                "test_mean_len": mean_len,
                "test_std_len": std_len,
                "success": 1 if mean_reward >= 1500 else 0,
            }
            
            df_new = pd.DataFrame([new_result])
            
            file_exists = os.path.isfile(self.csv_name)
            
            df_new.to_csv(
                self.csv_name, 
                mode='a', # append (non sovrascrive)
                index=False, 
                header=not file_exists # scrive l'header solo se il file non esisteva
            )
            
            pbar.set_postfix({"last_reward": f"{mean_reward:.0f}"})
