
import optuna
import optuna_config as config
from optuna.visualization import plot_param_importances
import matplotlib.pyplot as plt

def main():
    # Load the study from the SQLite database
    study_name = config.feature_weight_study_name
    storage_url = 'sqlite:///optuna_study.db'
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        print(f"Loaded study '{study_name}' with {len(study.trials)} trials.")
        
        if len(study.trials) == 0:
            print("No trials found. Please run the optimization first.")
            return

        # Plot parameter importances
        fig = plot_param_importances(study)
        
        # Try to show it
        try:
            fig.show()
        except Exception:
            print("Could not show interactive plot. Saving to 'feature_importance.png'...")
            # If interactive fails, try to save it (requires kaleido)
            try:
                fig.write_image("feature_importance.png")
                print("Saved to feature_importance.png")
            except Exception as e:
                print(f"Failed to save image: {e}")
                print("You can try running this script in a Jupyter notebook or an environment with Plotly/Kaleido installed.")

    except Exception as e:
        print(f"Error loading study: {e}")

if __name__ == "__main__":
    main()
