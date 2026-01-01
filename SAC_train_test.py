from pathlib import Path
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from env.custom_hopper import *
from env.custom_walker import *
from adr_callback import ADRCallback
from utils.schedulers import get_lr_scheduler
from utils.plot import plot_training_results


def SAC_train_adr(
        train_env_id,
        model_name,
        lr=3e-4,           
        lr_scheduler_type="constant",
        steps=300_000,
        starting_adr_range=0.05,
        objective_adr_range=0.5,
        increase_rate=0.05,
        reward_to_check=1500,
        check_frequency=80_000,
        net_size="large",        # SAC beneficia di reti grandi
        seed=None
    ):
    """
    Addestra un modello SAC usando Automatic Domain Randomization.
    """

    print(f"\n--- Training on {train_env_id} using SAC + ADR ---")
    
    if seed is not None:
        set_random_seed(seed)

    log_dir = "logs_sac" 
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(
        train_env_id,
        enable_randomization=True,
        uniform_randomization_range=starting_adr_range
    )
    
    env = Monitor(env, filename=f"{log_dir}/monitor.csv")

    # SAC usa "pi" (Policy) e "qf" (Q-Function). PPO usava "vf".
    # Se lasci "vf" qui, SAC darà errore o ignorerà il parametro.
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], qf=[128, 128]))
    if net_size == "large":
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    elif net_size == "small":
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))

    adr_callback = ADRCallback(
        eval_env_id=train_env_id,
        check_freq=check_frequency,             
        reward_threshold=reward_to_check,       
        increase_rate=increase_rate,            
        starting_range=starting_adr_range,
        max_range=objective_adr_range,          
        seed=seed
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=get_lr_scheduler(lr_scheduler_type, initial_lr=lr),
        policy_kwargs=policy_kwargs,
        
        # PARAMETRI CRITICI PER ADR + SAC
        buffer_size=100_000,    # RIDOTTO (default 1M). Aiuta a dimenticare i dati "facili" vecchi.
        batch_size=256,         # SAC ama batch grandi
        ent_coef='auto',        # Fondamentale: impara da solo quanto esplorare
        train_freq=1,           # Aggiorna ogni step
        gradient_steps=1,       # 1 update per step
        learning_starts=5_000,  # Primi step random per riempire un po' il buffer
        
        seed=seed,
        verbose=1
    )

    model.learn(
        total_timesteps=steps,
        callback=adr_callback,
        progress_bar=True
    )

    model.save(str(Path("models") / f"{model_name}"))
    env.close()

    # Plot (funziona uguale perché legge il monitor.csv standard)
    plot_training_results(log_dir, title=f"training_{model_name}", adr_stats=adr_callback.adr_history)


def SAC_test(test_env_id, model_name):
    """
    Testa un modello SAC.
    """
    print(f"--- Testing SAC on {test_env_id} ---")
    
    env = gym.make(test_env_id, enable_randomization=False)

    loaded_model = SAC.load(f"models/{model_name}")
    
    episode_rewards, episode_lengths = evaluate_policy(
        loaded_model, 
        env, 
        n_eval_episodes=50, 
        return_episode_rewards=True
    )

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_len = np.mean(episode_lengths)
    std_len = np.std(episode_lengths)

    print(f"Test Results:")
    print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Mean Steps:  {mean_len:.2f} +/- {std_len:.2f}")

    return mean_reward, std_reward