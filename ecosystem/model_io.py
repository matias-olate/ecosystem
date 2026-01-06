from pathlib import Path
from cobra import Model
import cobra.io


MODEL_DIR = "models"


def load_model(model_name: str, model_directory: str = MODEL_DIR, solver: str = 'gurobi') -> Model:
    '''Loads a COBRA model from an SBML file using the specified solver.'''
    path = Path(model_directory) / model_name
    model = cobra.io.read_sbml_model(path, solver=solver)

    return model 


def save_models(model_dict: dict[str, Model], model_directory: str = MODEL_DIR) -> None:
    '''Saves all COBRA models in "model_dict" to "model_directory".'''    
    output_dir = Path(model_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in model_dict.items():
        filename = output_dir / f"{model_name}.xml"
        cobra.io.write_sbml_model(model, filename)
        print(f'model {model_name} stored')

