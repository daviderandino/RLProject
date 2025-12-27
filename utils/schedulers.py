from typing import Callable


def get_lr_scheduler(scheduler_type: str, initial_lr: float):
    if scheduler_type == "linear":
        return _linear_schedule(initial_lr)
    elif scheduler_type == "exponential":
        return _exponential_schedule(initial_lr, decay_rate=0.1)
    else:
        return _constant_schedule(initial_lr)

# nestare permette di catturare l'initial value nella closure della funzione

def _constant_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Ritorna sempre initial_value, ignorando il progresso.
    Simula una costante ma mantenendo la firma di una funzione.
    """
    def func(progress: float) -> float:
        return initial_value
    
    return func

def _linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Decadimento lineare del learning rate (non arriva a 0)
    :param initial_value: Il valore iniziale del learning rate.
    :return: funzione che calcola il learning rate corrente.
    """

    def func(progress: float) -> float:
        # progress_remaining va da 1.0 (inizio) a 0.0 (fine)
        
        # Calcoliamo il valore finale target
        # (il learning rate decade linearmente fino al 10% del suo valore finale)
        final_value = initial_value * 0.1
        
        # La differenza che deve decadere
        decay_amount = initial_value - final_value
        
        # Formula: Base fissa + (parte variabile * progresso)
        # Quando progress è 1.0 -> final + (init - final)*1 = init
        # Quando progress è 0.0 -> final + 0 = final
        current_lr = final_value + (decay_amount * progress)
        
        return current_lr
    
    return func

def _exponential_schedule(initial_value: float, decay_rate: float) -> Callable[[float], float]:
    """
    Exponential schedule from initial_value to 0
    :param initial_value: (float)
    :return: (function)
    """
    
    def func(progress: float) -> float:
        """
        Exponential learning rate schedule.
        :param progress: (float) Progress remaining (from 1 to 0)
        :return: (float) Decayed learning rate
        """
        progress = max(progress, 0)
        return initial_value * (progress ** decay_rate)
    
    return func
