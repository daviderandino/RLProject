from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from env.custom_hopper import *
from adr_callback import ADRCallback
from utils.schedulers import get_lr_scheduler
from utils.plot import plot_training_results


# +------------------------+
# | PPO TRAINING FUNCTIONS |
# + -----------------------+

def PPO_train(train_env_id, model_name, lr=3e-4, steps=800_000, seed=None):
    """
    Addestra un modello PPO senza usare tecniche di randomizzazione.
    """

    print(f"\n--- Training on {train_env_id} ---")

    if seed is not None:
        set_random_seed(seed)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(
        train_env_id
    )

    env = Monitor(env, filename=f"{log_dir}/monitor.csv")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=4096,
        batch_size=128,
        seed=seed
    )

    model.learn(
        total_timesteps=steps,
        progress_bar=True
    )

    model.save(str(Path("models") / f"{model_name}"))
    env.close()

    plot_training_results(log_dir, title=f"training_{model_name}")


def PPO_train_udr(
        train_env_id, model_name,
        lr=3e-4,
        lr_scheduler_type="constant",
        steps=800_000,
        udr_range=0.4,
        net_size="medium",
        seed=None
    ):
    """
    Addestra un modello PPO.
    Impostando 'enable_udr' a True verrà usata la Uniform Domain Randomization.
    """

    print(f"\n--- Training on {train_env_id} using UDR ---")

    if seed is not None:
        set_random_seed(seed)

    log_dir = "logs_udr"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(
        train_env_id,
        enable_randomization=True,
        uniform_randomization_range=udr_range
    )

    env = Monitor(env, filename=f"{log_dir}/monitor.csv")
    
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    if net_size == "large":
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    elif net_size == "small":
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=get_lr_scheduler(
            lr_scheduler_type,
            initial_lr=lr
        ),
        policy_kwargs=policy_kwargs,
        n_steps=4096,
        batch_size=128,
        seed=seed
    )

    model.learn(
        total_timesteps=steps,
        progress_bar=True
    )

    model.save(str(Path("models") / f"{model_name}"))
    env.close()

    plot_training_results(log_dir, title=f"training_{model_name}")


def PPO_train_adr(
        train_env_id,
        model_name,
        lr=3e-4,
        lr_scheduler_type="constant",
        steps=800_000,
        starting_adr_range=0.05,
        objective_adr_range=0.5,
        increase_rate=0.05,
        reward_to_check=1500,
        check_frequency=80_000,
        net_size="large",
        seed=None
    ):
    """
    Addestra un modello PPO usando Automatic Domain Randomization.
    Questo avviene grazie all'utilizzo della callback, altrimenti sarebbe
    solo una UDR.
    """

    print(f"\n--- Training on {train_env_id} using ADR ---")
    
    if seed is not None:
        set_random_seed(seed)

    log_dir = "logs_adr"
    os.makedirs(log_dir, exist_ok=True)

    # ambiente di training
    env = gym.make(
        train_env_id,
        enable_randomization=True,
        uniform_randomization_range=starting_adr_range
    )
    
    env = Monitor(env, filename=f"{log_dir}/monitor.csv")

    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    if net_size == "large":
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    elif net_size == "small":
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    adr_callback = ADRCallback(
        eval_env_id=train_env_id,
        check_freq=check_frequency,             # controllo le prestazioni del robot all'edge, ogni 5k timesteps
        reward_threshold=reward_to_check,       # reward threshold da ottenere per espandere il range della randomization
        increase_rate=increase_rate,                     # +5% alla volta
        starting_range=starting_adr_range,
        max_range=objective_adr_range,           # range obiettivo
        seed=seed
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=get_lr_scheduler(
            lr_scheduler_type,
            initial_lr=lr
        ),
        policy_kwargs=policy_kwargs,
        n_steps=4096,
        batch_size=128,
        seed=seed
    )

    model.learn(
        total_timesteps=steps,
        callback=adr_callback,
        progress_bar=True
    )

    model.save(str(Path("models") / f"{model_name}"))
    env.close()

    plot_training_results(log_dir, title=f"training_{model_name}", adr_stats=adr_callback.adr_history)

# +----------------------+
# | PPO TESTING FUNCTION |
# + ---------------------+

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
