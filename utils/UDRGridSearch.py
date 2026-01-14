import numpy as np
from itertools import product
from PPO_train_test import PPO_train_udr, PPO_test
import pandas as pd
from tqdm.auto import tqdm 
import os

"""
Utility functions for performing grid search over hyperparameters.
"""


class UDRGridSearch:
    def __init__(
        self,
        csv_name,
        train_env_id,
        test_env_id,
        udr_range,
        timesteps,
        architecture,
        learning_rate,
        lr_scheduler_type,
        seed
    ):  
        self.csv_name = csv_name
        self.train_env_id = train_env_id
        self.test_env_id = test_env_id

        # --- Hyperparameters ---
        # Fixed hypers
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.seed = seed
        self.timesteps = timesteps
        
        # Grid search hypers
        self.architectures = architecture
        self.udr_ranges = udr_range


    def run_search(self):
        combinations = list(
            product(
                self.udr_ranges,
                self.architectures
            )
        )
        
        pbar = tqdm(combinations, desc="Total Progress", unit="exp")
        
        for udr_range, arch in pbar:

            model_name = f"{self.train_env_id}_udr_{self.lr_scheduler_type}_{udr_range}_{arch}_{self.seed}"

            # skippa se l'esperimento esiste già
            if os.path.exists(self.csv_name):
                existing_results = pd.read_csv(self.csv_name)
                if model_name in existing_results["model_id"].values:
                    continue
            
            pbar.set_description(f"Processing {model_name}")

            PPO_train_udr(
                train_env_id=self.train_env_id,
                model_name=model_name,
                lr=self.learning_rate,
                lr_scheduler_type=self.lr_scheduler_type,
                steps=self.timesteps,
                udr_range=udr_range,
                net_size=arch,
                seed=self.seed
            )

            mean_reward, std_reward, mean_len, std_len = PPO_test(self.test_env_id, model_name)

            new_result = {
                "model_id": model_name,
                "lr_scheduler_type": self.lr_scheduler_type,
                "udr_range": udr_range,
                "architecture": arch,
                "learning_rate": self.learning_rate,
                "test_mean_reward": mean_reward,
                "test_std_reward": std_reward,
                "test_mean_len": mean_len,
                "test_std_len": std_len,
                "success": 1 if mean_reward >= 3000 else 0,
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
