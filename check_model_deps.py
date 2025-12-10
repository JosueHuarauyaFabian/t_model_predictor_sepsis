"""Script to check AutoGluon model dependencies for WeightedEnsemble_L3"""
from autogluon.tabular import TabularPredictor
import json

# Load predictor (bypass Python version check)
predictor = TabularPredictor.load('d:/tesis/t_model_predictor_sepsis/modelo_multimodal_clinicalbert_best', require_py_version_match=False)

print("=== Best Model ===")
print(f"Best model: {predictor.model_best}")

print("\n=== All Models ===")
all_models = predictor.model_names()
print(f"Total models: {len(all_models)}")
for model in all_models:
    print(f"  - {model}")

# Try to get model dependencies
print("\n=== WeightedEnsemble_L3 Dependencies ===")
try:
    # Access internal trainer to get model dependencies
    trainer = predictor._trainer
    model_graph = trainer.model_graph

    if 'WeightedEnsemble_L3' in model_graph:
        deps = model_graph['WeightedEnsemble_L3']
        print(f"Direct dependencies: {deps}")

        # Get transitive dependencies
        all_deps = set()
        def get_deps(model_name):
            if model_name in model_graph:
                for dep in model_graph[model_name]:
                    all_deps.add(dep)
                    get_deps(dep)

        get_deps('WeightedEnsemble_L3')
        print(f"\nAll transitive dependencies ({len(all_deps)} models):")
        for dep in sorted(all_deps):
            print(f"  - {dep}")

except Exception as e:
    print(f"Error getting dependencies: {e}")
    import traceback
    traceback.print_exc()
