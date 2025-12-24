from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from env.custom_hopper import *
from adr_callback import ADRCallback
from utils.schedulers import get_lr_scheduler
from utils.plot import plot_training_results


# ========== PPO TRAINING FUNCTIONS ==========

def PPO_train(train_env_id, model_name, lr=3e-4, steps=800_000):
    """
    Addestra un modello PPO senza usare tecniche di randomizzazione.
    """

    print(f"\n--- Training on {train_env_id} ---")

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(
        train_env_id
    )

    env = Monitor(env, filename=f"{log_dir}/monitor.csv")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr
    )

    model.learn(
        total_timesteps=steps,
        progress_bar=True
    )

    model.save(str(Path("models") / f"{model_name}"))
    env.close()

    plot_training_results(log_dir, title=f"training_{model_name}")


def PPO_train_udr(train_env_id, model_name, lr=3e-4, lr_scheduler_type="constant", steps=800_000, udr_range=0.4):
    """
    Addestra un modello PPO.
    Impostando 'enable_udr' a True verrà usata la Uniform Domain Randomization.
    """

    print(f"\n--- Training on {train_env_id} using UDR ---")

    log_dir = "logs_udr"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(
        train_env_id,
        enable_randomization=True,
        uniform_randomization_range=udr_range
    )

    env = Monitor(env, filename=f"{log_dir}/monitor.csv")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=get_lr_scheduler(
            lr_scheduler_type,
            initial_lr=lr
        ),
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    )

    model.learn(
        total_timesteps=steps,
        progress_bar=True
    )

    model.save(str(Path("models") / f"{model_name}"))
    env.close()

    plot_training_results(log_dir, title=f"training_{model_name}")


def PPO_train_adr(train_env_id, model_name, lr=3e-4, lr_scheduler_type="constant", steps=800_000, starting_adr_range=0.05, objective_adr_range=0.5):
    """
    Addestra un modello PPO usando Automatic Domain Randomization.
    Questo avviene grazie all'utilizzo della callback, altrimenti sarebbe
    solo una UDR.
    """

    print(f"\n--- Training on {train_env_id} using ADR ---")
    
    log_dir = "logs_adr"
    os.makedirs(log_dir, exist_ok=True)

    # ambiente di training
    env = gym.make(
        train_env_id,
        enable_randomization=True,
        uniform_randomization_range=starting_adr_range
    )
    
    env = Monitor(env, filename=f"{log_dir}/monitor.csv")

    adr_callback = ADRCallback(
        eval_env_id=train_env_id,
        check_freq=80000,                       # controllo le prestazioni del robot all'edge, ogni 5k timesteps
        reward_threshold=1200,                  # reward threshold da ottenere per espandere il range della randomization
        increase_rate=0.05,                     # +5% alla volta
        starting_range=starting_adr_range,
        max_range=objective_adr_range           # range obiettivo
    )

    # si potrebbe definire una rete neurale più grande per PPO, tipo [256, 256]
    # ovvero due layer da 256 neuroni: 
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    # e passarlo a PPO(...) come parametro policy_kwargs.
    # Il motivo è che l'ADR è un compito difficile, dunque richiede una buona
    # mente (ovvero la rete).

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=get_lr_scheduler(
            lr_scheduler_type,
            initial_lr=lr
        ),
        policy_kwargs=policy_kwargs
    )

    model.learn(
        total_timesteps=steps,
        callback=adr_callback,
        progress_bar=True
    )

    model.save(str(Path("models") / f"{model_name}"))
    env.close()

    plot_training_results(log_dir, title=f"training_{model_name}", adr_stats=adr_callback.adr_history)

# ========== PPO TESTING FUNCTION ==========

def PPO_test(test_env_id, model_name):
    """
    Testa un modello PPO.
    """

    print(f"--- Testing on {test_env_id} ---")

    # carico un nuovo ambiente per la valutazione, la UDR è sempre False
    # in testing.
    env = gym.make(test_env_id, enable_randomization=False)

    loaded_model = PPO.load(str(Path("models") / f"{model_name}"))
    
    mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=50)
    
    env.close()
    
    print(f"Test mean cumulative reward {mean_reward} +/- {std_reward}")

    return mean_reward, std_reward
