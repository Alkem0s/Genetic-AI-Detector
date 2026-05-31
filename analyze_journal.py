import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import optuna_config as config

def analyze_study(study_name, storage):
    print(f"\n{'='*20} {study_name} {'='*20}")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        trials = study.trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Sort by value (descending because we maximize fitness/accuracy)
        completed_trials.sort(key=lambda t: t.value if t.value is not None else -1.0, reverse=True)
        
        print(f"Total trials: {len(trials)}")
        print(f"Completed trials: {len(completed_trials)}")
        
        if not completed_trials:
            print("No completed trials yet.")
            return
            
        print(f"\nBest Trial: {completed_trials[0].number} with value: {completed_trials[0].value:.6f}")
        print(f"Best Params: {completed_trials[0].params}")
        
        print("\nTop 10 Completed Trials:")
        for i, t in enumerate(completed_trials[:10]):
            print(f"Rank {i+1:<2} | Trial {t.number:<4} | Value: {t.value:.6f}")
            print(f"  Params: {t.params}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error loading study {study_name}: {e}")

if __name__ == "__main__":
    try:
        storage = JournalStorage(JournalFileStorage("optuna_journal.log"))
        study_summaries = optuna.get_all_study_summaries(storage)
        
        if not study_summaries:
            print("No studies found in optuna_journal.log.")
        else:
            print("Discovered studies in journal:")
            for summary in study_summaries:
                print(f"  - {summary.study_name} (Trials: {summary.n_trials})")
                
            for summary in study_summaries:
                analyze_study(summary.study_name, storage)
    except Exception as e:
        print(f"Error loading journal storage: {e}")
