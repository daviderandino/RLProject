import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import FrameStackObservation

class ADRCallback(BaseCallback):
    """
    Automatic Domain Randomization Callback.
    Versione OTTIMIZZATA per FrameStack nativo (Single Env).
    """
    def __init__(self,
        eval_env_id,
        check_freq: int,
        reward_threshold: float,
        increase_rate: float,
        starting_range: float,
        max_range: float,
        seed=None,
        verbose=1):

        super(ADRCallback, self).__init__(verbose)
        
        self.check_freq = check_freq
        self.reward_threshold = reward_threshold
        self.increase_rate = increase_rate
        self.current_range = starting_range
        self.max_range = max_range
        self.adr_history = [(0, starting_range)]

        # --- SETUP AMBIENTE DI VALUTAZIONE ---
        # Deve rispecchiare esattamente l'ambiente di training "Optimized"
        
        # 1. Creiamo l'ambiente base
        self.eval_env = gym.make(
            eval_env_id,
            enable_randomization=True,
            uniform_randomization_range=starting_range
        )
        
        if seed is not None:
            self.eval_env.reset(seed=seed)

        # 2. FRAME STACKING NATIVO
        # Fondamentale: se la rete si aspetta 4 frame in training,
        # deve riceverne 4 anche in valutazione, altrimenti crasha per "Shape Mismatch".
        self.eval_env = FrameStackObservation(self.eval_env, num_stack=4)


    def _init_callback(self) -> None:
        # Usiamo .unwrapped per bucare il wrapper FrameStack e arrivare al CustomHopper
        self.eval_env.unwrapped.set_udr_range(self.current_range)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            
            # Attiva boundary mode
            self.eval_env.unwrapped.set_boundary_mode(True)

            # Valutazione
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=10)

            # Disattiva boundary mode
            self.eval_env.unwrapped.set_boundary_mode(False)
            
            if self.verbose > 0:
                print(f"[ADR Boundary Test] Range +/- {self.current_range:.1%} -> Reward: {mean_reward:.2f} +/- {std_reward}")

            range_updated = False

            # --- LOGICA DI AGGIORNAMENTO RANGE ---
            if mean_reward >= self.reward_threshold:
                if self.current_range < self.max_range:
                    self.current_range += self.increase_rate
                    if self.current_range > self.max_range:
                        self.current_range = self.max_range
                        print(f"[ADR Objective Reached] New Range +/- {self.current_range:.1%}")
                    else:
                        print(f"[ADR Increased] New Range +/- {self.current_range:.1%}")
                    range_updated = True
                    
            elif mean_reward < self.reward_threshold * 0.5:
                range_updated = True
                self.current_range = max(0.05, self.current_range - 0.02)
                print(f"[ADR Decreased] New Range +/- {self.current_range:.1%}")
            
            if range_updated:
                # 1. Aggiorna l'ambiente di training
                # NOTA: PPO avvolge internamente l'ambiente in un DummyVecEnv anche se è singolo.
                # Per andare sul sicuro accediamo al primo ambiente della lista (.envs[0]) e poi unwrapped.
                self.training_env.envs[0].unwrapped.set_udr_range(self.current_range)
                
                # 2. Aggiorna l'ambiente di valutazione (che gestiamo noi manualmente)
                self.eval_env.unwrapped.set_udr_range(self.current_range)

                self.adr_history.append((self.num_timesteps, self.current_range))
                
        return True
    
    def _on_training_end(self) -> None:
        self.eval_env.close()