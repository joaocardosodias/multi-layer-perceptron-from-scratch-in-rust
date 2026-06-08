import optuna
import subprocess
import re
import os

# Caminho do projeto
PROJECT_DIR = "/home/cardoso/GitHub/multi-layer-perceptron-from-scratch-in-rust"

def run_training(max_lr, augment_p_keep, weight_decay, epochs, dropout_keep):
    """
    Chama o binário Rust com os hiperparâmetros e retorna a melhor acurácia de teste.
    """
    # Compilar com os parâmetros via variáveis de ambiente
    env = os.environ.copy()
    env["MAX_LR"] = str(max_lr)
    env["AUGMENT_P_KEEP"] = str(augment_p_keep)
    env["WEIGHT_DECAY"] = str(weight_decay)
    env["EPOCHS"] = str(epochs)
    env["DROPOUT_KEEP"] = str(dropout_keep)
    
    # Compilar
    result = subprocess.run(
        ["cargo", "run", "--bin", "mlp-gpu", "--release"],
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=600  # 10 minutos por run
    )
    
    # Parsear o output para encontrar "Melhor: XX.XX%"
    output = result.stdout + result.stderr
    match = re.search(r"Melhor: (\d+\.\d+)% na Epoca (\d+)", output)
    
    if match:
        best_acc = float(match.group(1))
        best_epoch = int(match.group(2))
        return best_acc, best_epoch
    else:
        # Se não encontrar, retornar um valor ruim
        print("Erro: não conseguiu parsear output")
        print(output[-500:])  # Print últimas 500 linhas
        return 0.0, 0

def objective(trial):
    # Sugerir hiperparâmetros
    max_lr = trial.suggest_float("max_lr", 1e-3, 5e-3, log=True)
    augment_p_keep = trial.suggest_float("augment_p_keep", 0.75, 0.90)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_keep = trial.suggest_float("dropout_keep", 0.85, 0.95)
    
    print(f"\n=== Trial {trial.number} ===")
    print(f"max_lr={max_lr:.6f}, augment_p_keep={augment_p_keep:.2f}, weight_decay={weight_decay:.6f}, dropout_keep={dropout_keep:.2f}")
    
    best_acc, best_epoch = run_training(
        max_lr=max_lr,
        augment_p_keep=augment_p_keep,
        weight_decay=weight_decay,
        epochs=300,  # Fixo para velocidade
        dropout_keep=dropout_keep
    )
    
    print(f"Resultado: {best_acc:.2f}% na época {best_epoch}")
    
    return best_acc

if __name__ == "__main__":
    # Criar estudo Optuna com persistência
    study = optuna.create_study(
        study_name="mlp-gpu-optimization",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        storage="sqlite:///optuna_study.db",
        load_if_exists=True
    )
    
    # Rodar 300 trials (14h / 150s = ~336 runs)
    n_trials = 300
    print(f"\n=== Iniciando {n_trials} trials ===")
    print(f"Tempo estimado: ~{n_trials * 150 / 3600:.1f} horas")
    
    # Callback para salvar progresso a cada 10 trials
    def save_progress(study, trial):
        if trial.number % 10 == 0:
            print(f"\n--- Progresso: {trial.number}/{n_trials} trials ---")
            print(f"Melhor até agora: {study.best_value:.2f}%")
            # Salvar resultados parciais
            with open("optuna_results_partial.txt", "w") as f:
                f.write(f"Melhor acurácia: {study.best_value:.2f}%\n")
                f.write(f"Melhores hiperparâmetros: {study.best_params}\n")
                f.write(f"Total de trials: {len(study.trials)}\n")
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[save_progress])
    
    # Print resultados
    print("\n=== MELHOR RESULTADO ===")
    print(f"Melhor acurácia: {study.best_value:.2f}%")
    print(f"Melhores hiperparâmetros: {study.best_params}")
    
    # Salvar resultados
    with open("optuna_results.txt", "w") as f:
        f.write(f"Melhor acurácia: {study.best_value:.2f}%\n")
        f.write(f"Melhores hiperparâmetros: {study.best_params}\n")
        f.write("\nTodos os trials:\n")
        for trial in study.trials:
            f.write(f"Trial {trial.number}: {trial.value:.2f}% - {trial.params}\n")
    
    print(f"\nResultados salvos em optuna_results.txt")
    print(f"Estudo salvo em: optuna_study.db")
