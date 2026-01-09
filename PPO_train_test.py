from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from env.custom_hopper import *
from env.custom_walker import *
from adr_callback import ADRCallback
from utils.schedulers import get_lr_scheduler
from utils.plot import plot_training_results


# +------------------------+
# | PPO TRAINING FUNCTIONS |
# + -----------------------+

def PPO_train(
        train_env_id, 
        model_name, 
        lr, 
        steps, 
        seed=None
    ):
    """
    Addestra un modello PPO senza usare tecniche di randomizzazione.
    """

    print(f"\n--- Training on {train_env_id} ---")

    if seed is not None:
        set_random_seed(seed)

    os.makedirs("logs", exist_ok=True)

    env = gym.make(
        train_env_id
    )

    env = Monitor(env, filename=f"logs/monitor.csv")
    
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

    plot_training_results("logs", title=f"training_{model_name}")


def PPO_train_udr(
        train_env_id, 
        model_name,
        lr,
        lr_scheduler_type,
        steps,
        udr_range,
        net_size,
        seed=None
    ):
    """
    Addestra un modello PPO.
    """

    print(f"\n--- Training on {train_env_id} using UDR ---")

    if seed is not None:
        set_random_seed(seed)

    os.makedirs("logs_udr", exist_ok=True)

    env = gym.make(
        train_env_id,
        enable_randomization=True,
        uniform_randomization_range=udr_range
    )

    env = Monitor(env, filename=f"logs_udr/monitor.csv")
    
    if net_size == "small":
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    elif net_size == "medium":
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    else:
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

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

    plot_training_results("logs_udr", title=f"training_{model_name}")


def PPO_train_adr(
        train_env_id,
        model_name,
        lr,
        lr_scheduler_type,
        steps,
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

    os.makedirs("logs_adr", exist_ok=True)

    # ambiente di training
    env = gym.make(
        train_env_id,
        enable_randomization=True,
        uniform_randomization_range=starting_adr_range
    )
    
    env = Monitor(env, filename=f"logs_adr/monitor.csv")

    if net_size == "small":
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    elif net_size == "medium":
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    else:
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    adr_callback = ADRCallback(
        eval_env_id=train_env_id,
        check_freq=check_frequency,             # controllo le prestazioni del robot all'edge, ogni 5k timesteps
        reward_threshold=reward_to_check,       # reward threshold da ottenere per espandere il range della randomization
        increase_rate=increase_rate,            
        starting_range=starting_adr_range,
        max_range=objective_adr_range,          # range obiettivo
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

    plot_training_results("logs_adr", title=f"training_{model_name}", adr_stats=adr_callback.adr_history)

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
