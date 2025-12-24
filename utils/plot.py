import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(log_dir, title="Training Results", adr_stats=None):
    """
    Plotta i risultati del training leggendo il monitor.csv generato da SB3.
    
    :param log_dir: Cartella dove è salvato il monitor.csv
    :param title: Titolo del grafico
    :param adr_stats: (Opzionale) Lista di tuple (timestep, range_value) proveniente dalla callback ADR
    """
    try:
        # Stable Baselines3 salva i log saltando le prime 2 righe di header
        df = pd.read_csv(f"{log_dir}/monitor.csv", skiprows=1)
    except FileNotFoundError:
        print(f"Errore: Nessun file monitor.csv trovato in {log_dir}")
        return

    # Calcolo cumulo dei timesteps per asse X
    # 'l' è la lunghezza dell'episodio in timesteps
    x = np.cumsum(df['l'].values)
    rewards = df['r'].values

    # Calcolo smoothed rewards (media mobile su 50 episodi)
    window = 50
    rewards_smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()

    # Creazione della figura
    if adr_stats:
        # --- CASO CON ADR: 2 Subplots ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Rewards
        ax1.plot(x, rewards, alpha=0.3, color='gray', label='Raw Reward')
        ax1.plot(x, rewards_smoothed, color='blue', linewidth=2, label=f'Smoothed Reward (MA {window})')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title(f'{title} - Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: ADR Range
        # Scompattiamo la storia dell'ADR
        adr_timesteps, adr_values = zip(*adr_stats)
        
        # Aggiungiamo il punto finale coincidente con l'ultimo step del training per estendere la linea
        if adr_timesteps[-1] < x[-1]:
            adr_timesteps = list(adr_timesteps) + [x[-1]]
            adr_values = list(adr_values) + [adr_values[-1]]

        # Convertiamo i valori in percentuale (0.05 -> 5%)
        adr_values_pct = np.array(adr_values) * 100

        ax2.step(adr_timesteps, adr_values_pct, where='post', color='green', linewidth=2, label='ADR Range')
        ax2.set_ylabel('Randomization Range (%)')
        ax2.set_xlabel('Timesteps')
        ax2.set_title('ADR Difficulty Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    else:
        # --- CASO SENZA ADR: 1 Plot ---
        plt.figure(figsize=(10, 5))
        plt.plot(x, rewards, alpha=0.3, color='gray', label='Raw Reward')
        plt.plot(x, rewards_smoothed, color='blue', linewidth=2, label=f'Smoothed Reward (MA {window})')
        plt.xlabel('Timesteps')
        plt.ylabel('Cumulative Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"plots/{title}.png")