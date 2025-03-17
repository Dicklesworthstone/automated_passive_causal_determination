import logging
import subprocess
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_script(script_name):
    logging.info(f"--- Running script: {script_name} ---")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Script {script_name} failed with error:\n{result.stderr}")
        sys.exit(1)
    logging.info(f"Successfully completed {script_name}.\nOutput:\n{result.stdout}")

def main():
    logging.info("==== Automated Causal Discovery Coordination Script ====")
    
    logging.info("\nSTEP 1: Data Retrieval and Preparation")
    logging.info("We start by gathering and preprocessing multiple publicly available datasets. "
                 "These datasets are chosen to have intuitive, well-understood causal relationships "
                 "so that we can test the effectiveness of our automated causal discovery methods.")
    run_script('causal_data_retrieval_and_preparation.py')

    logging.info("\nSTEP 2: Automated Passive Causal Determination")
    logging.info("Next, we apply our automated causal discovery algorithm. "
                 "We designed this step to systematically explore causal relationships by treating each column "
                 "as a potential outcome and training predictive models (Transformers or XGBoost) "
                 "on randomly sampled subsets of other columns. "
                 "Our key insight here is that true causal relationships should allow simpler "
                 "(more parsimonious) models to achieve strong predictive performance.")
    run_script('automated_passive_causal_determination.py')

    logging.info("\nSTEP 3: Final Comprehensive Analysis")
    logging.info("In this final step, we perform an extensive analysis of all the results "
                 "from our previous modeling step. We calculate confidence scores for each causal edge "
                 "based on how consistently certain inputs appeared in the simplest, best-performing models. "
                 "This allows us to derive a comprehensive Directed Acyclic Graph (DAG) that represents the "
                 "most likely causal structure among variables in our datasets.")
    run_script('automated_passive_causal_determination_final_analysis.py')

    logging.info("\n==== Causal Discovery Process Complete ====")
    logging.info("Through this pipeline, we've created a robust automated method for causal discovery. "
                 "The foundational idea was that genuinely causal relationships greatly simplify prediction, "
                 "requiring fewer parameters to achieve lower prediction errors. "
                 "By leveraging modern machine learning techniques (like transformers and gradient boosting), "
                 "combined with intelligent sampling and complexity analysis, we've developed a novel approach "
                 "to reliably infer causal relationships from observational data alone.\n"
                 "The final DAG we've constructed is transparent, interpretable, and comes with explicit confidence "
                 "scores that guide further research or intervention decisions.")

if __name__ == '__main__':
    main()
