# adr_callback.py
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class ADRCallback(BaseCallback):
    """
    Automatic Domain Randomization Callback.
    Aumenta la difficoltà (udr_range) se l'agente supera la reward_threshold.
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

        # Ambiente di valutazione dedicato (Source con UDR abilitata).
        # Deve essere allineato all'ambiente di training.
        #
        # Non lo settiamo permanentement in boundary mode ma la attiviamo
        # e poi disattiviamo solo quando necessario. In tal modo, si potrebbe
        # aggiungere una evaluation ogni tanto per vedere come si comporta il
        # modello durante il training (ma non sull'edge). O magari si può usare
        # per visualizzare l'agente in video ogni tanto.
        self.eval_env = gym.make(
            eval_env_id,
            enable_randomization=True,
            uniform_randomization_range=starting_range
        )

        if seed is not None:
            self.eval_env.reset(seed=seed)


    def _init_callback(self) -> None:
        # Applichiamo il range iniziale all'ambiente di training
        self.eval_env.unwrapped.set_udr_range(self.current_range)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0: # chiamata ad ogni passo del simulatore, ogni 'check_freq' timesteps parte
            
            self.eval_env.unwrapped.set_boundary_mode(True)

            # --- VALUTAZIONE ---
            # Testiamo su un pò di episodi. Se sopravvive ai bordi, è robusto.
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=10)

            # --- DISATTIVA MODALITÀ BOUNDARY ---
            self.eval_env.unwrapped.set_boundary_mode(False)
            
            if self.verbose > 0:
                print(f"[ADR Boundary Test] Range +/- {self.current_range:.1%} -> Reward: {mean_reward:.2f} +/- {std_reward}")

            range_updated = False

            # --- LOGICA DI ESPANSIONE/RESTRIZIONE RANGE ---
            if mean_reward >= self.reward_threshold: # se l'agente ha masterato l'edge attuale
                if self.current_range < self.max_range: # se il range attuale non è ancora quello massimo
                    self.current_range += self.increase_rate
                    if self.current_range > self.max_range: # se con l'aggiornamento abbiamo superato il range massimo, clippiamo
                        self.current_range = self.max_range
                        print(f"[ADR Objective Boundary Reached] New Range +/- {self.current_range:.1%}")
                    else:
                        print(f"[ADR Boundary Increased] New Range +/- {self.current_range:.1%}")

                    range_updated = True
                    
            # (Opzionale: Restringimento se fallisce miseramente)
            elif mean_reward < self.reward_threshold * 0.5:
                range_updated = True
                self.current_range = max(0.05, self.current_range - 0.02) # riduciamo del 2%, ma mai sotto il 5%
                print(f"[ADR Boundary Decreased] New Range +/- {self.current_range:.1%}")
            
            if range_updated:
                for env in self.training_env.envs:
                    env.unwrapped.set_udr_range(self.current_range)
                self.eval_env.unwrapped.set_udr_range(self.current_range)

                self.adr_history.append((self.num_timesteps, self.current_range))
                
        return True
    
    def _on_training_end(self) -> None:
        self.eval_env.close()
