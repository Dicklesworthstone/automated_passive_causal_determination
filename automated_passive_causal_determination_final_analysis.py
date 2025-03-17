import sqlite3
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_data(db_path='causal_analysis.db'):
    logging.info("Connecting to SQLite database.")
    conn = sqlite3.connect(db_path)
    models_df = pd.read_sql_query("SELECT * FROM models", conn)
    edges_df = pd.read_sql_query("SELECT * FROM dag_edges", conn)
    conn.close()
    logging.info(f"Loaded {len(models_df)} model records and {len(edges_df)} edge records.")
    return models_df, edges_df

def analyze_edge_confidence(edges_df, threshold=0.05):
    logging.info("Starting detailed edge confidence analysis.")
    edges_df['is_confident'] = edges_df['confidence'] >= threshold
    confident_edges = edges_df[edges_df['is_confident']].sort_values('confidence', ascending=False)
    uncertain_edges = edges_df[~edges_df['is_confident']].sort_values('confidence', ascending=False)

    logging.info(f"Identified {len(confident_edges)} confident edges and {len(uncertain_edges)} less confident edges.")
    
    logging.info("\n--- Most Confident Causal Edges ---")
    for _, row in confident_edges.iterrows():
        logging.info(f"Edge: {row['source']} → {row['target']} | Confidence: {row['confidence']:.3f}")

    if not uncertain_edges.empty:
        logging.info("\n--- Edges with Lower Confidence (below threshold) ---")
        for _, row in uncertain_edges.iterrows():
            logging.info(f"Edge: {row['source']} → {row['target']} | Confidence: {row['confidence']:.3f}")
    else:
        logging.info("\nNo edges were below the confidence threshold.")

    return confident_edges, uncertain_edges

def analyze_models(models_df):
    logging.info("Starting analysis of stored models based on combined score (parsimony & performance).")

    models_df['rank'] = models_df.groupby(['dataset', 'output'])['combined_score'].rank(pct=True)
    top_models = models_df[models_df['rank'] <= 0.02].sort_values(['dataset', 'output', 'combined_score'])

    logging.info(f"Selected top 2% models for each dataset and output variable.")
    for _, row in top_models.iterrows():
        logging.info(f"Dataset: {row['dataset']}, Output: {row['output']}, Model Type: {row['model_type']}, "
                     f"Inputs: {row['inputs']}, CV Loss: {row['cv_loss']:.4f}, Val Loss: {row['val_loss']:.4f}, "
                     f"Params: {row['params']}, Combined Score: {row['combined_score']:.4f}")

    return top_models

def infer_global_dag(top_models):
    logging.info("Inferring global DAG from top-performing models.")
    edge_counts = {}
    for _, row in top_models.iterrows():
        inputs = row['inputs'].split(',')
        output = row['output']
        for inp in inputs:
            edge = (inp, output)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    total_occurrences = sum(edge_counts.values())
    edge_confidences = {edge: count / total_occurrences for edge, count in edge_counts.items()}
    sorted_edges = sorted(edge_confidences.items(), key=lambda x: x[1], reverse=True)

    logging.info("\n--- Final Global DAG Edges Sorted by Confidence ---")
    for (source, target), confidence in sorted_edges:
        logging.info(f"{source} → {target} | Confidence: {confidence:.3f}")

    return edge_confidences

def main():
    logging.info("=== Beginning Final Causal Analysis ===")
    models_df, edges_df = load_data()

    logging.info("\nStep 1: Edge Confidence Analysis")
    confident_edges, uncertain_edges = analyze_edge_confidence(edges_df)

    logging.info("\nStep 2: Top Model Performance Analysis")
    top_models = analyze_models(models_df)

    logging.info("\nStep 3: Global DAG Inference")
    global_dag_confidences = infer_global_dag(top_models)

    logging.info("\n=== Final Analysis Complete ===")
    logging.info("The inferred DAG represents the causal structure determined by combining predictive accuracy "
                 "and model simplicity (parsimony). High-confidence edges strongly suggest true causal relationships, "
                 "whereas lower-confidence edges indicate weaker or uncertain causality that might require further investigation.")

if __name__ == '__main__':
    main()
