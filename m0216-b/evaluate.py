import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, ndcg_score
from train import CourseRecommender
import matplotlib.pyplot as plt
import seaborn as sns

class RecommenderEvaluator:
    def __init__(self):
        self.recommender = CourseRecommender()
        self.recommender.train()
        
    def evaluate_content_based_accuracy(self):
        """Evaluate content-based filtering accuracy"""
        # Load relationship data for ground truth
        relationships_df = pd.read_csv('datasets/course_relationships.csv')
        
        total_predictions = 0
        correct_predictions = 0
        ndcg_scores = []
        precision_at_k = []
        recall_at_k = []
        
        # Evaluate each course
        for course_id in self.recommender.courses_df['course_id'].unique():
            # Get actual related courses
            actual_related = set(relationships_df[
                relationships_df['source_course_id'] == course_id
            ]['target_course_id'].tolist())
            
            # Get predicted recommendations
            recommendations = self.recommender.get_recommendations(course_id, n_recommendations=5)
            predicted_related = set(recommendations['course_id'].tolist())
            
            # Calculate metrics
            if len(actual_related) > 0:
                # Precision
                precision = len(actual_related.intersection(predicted_related)) / len(predicted_related)
                precision_at_k.append(precision)
                
                # Recall
                recall = len(actual_related.intersection(predicted_related)) / len(actual_related)
                recall_at_k.append(recall)
                
                # NDCG
                y_true = [1 if c in actual_related else 0 for c in predicted_related]
                y_scores = recommendations['similarity'].tolist()
                if len(y_true) > 0 and len(y_scores) > 0:
                    ndcg = ndcg_score([y_true], [y_scores])
                    ndcg_scores.append(ndcg)
                
                total_predictions += 1
                if len(actual_related.intersection(predicted_related)) > 0:
                    correct_predictions += 1
        
        # Calculate overall metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_precision = np.mean(precision_at_k)
        avg_recall = np.mean(recall_at_k)
        avg_ndcg = np.mean(ndcg_scores)
        
        return {
            'accuracy': accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'ndcg': avg_ndcg
        }
    
    def evaluate_similarity_distribution(self):
        """Evaluate similarity score distribution"""
        all_similarities = []
        
        for course_id in self.recommender.courses_df['course_id'].unique():
            recommendations = self.recommender.get_recommendations(course_id, n_recommendations=5)
            if recommendations is not None:
                all_similarities.extend(recommendations['similarity'].tolist())
        
        return np.array(all_similarities)
    
    def plot_metrics(self, metrics):
        """Plot evaluation metrics"""
        plt.figure(figsize=(12, 6))
        
        # Plot metrics
        plt.subplot(1, 2, 1)
        metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['ndcg']]
        metrics_labels = ['Accuracy', 'Precision', 'Recall', 'NDCG']
        
        plt.bar(metrics_labels, metrics_values)
        plt.title('Recommendation System Metrics')
        plt.ylim(0, 1)
        
        # Plot similarity distribution
        plt.subplot(1, 2, 2)
        similarities = self.evaluate_similarity_distribution()
        sns.histplot(similarities, bins=20)
        plt.title('Similarity Score Distribution')
        
        plt.tight_layout()
        plt.savefig('evaluation_metrics.png')
        plt.close()

def main():
    print("Evaluating Course Recommendation System...")
    evaluator = RecommenderEvaluator()
    
    # Evaluate accuracy
    metrics = evaluator.evaluate_content_based_accuracy()
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"NDCG: {metrics['ndcg']:.3f}")
    
    # Plot metrics
    evaluator.plot_metrics(metrics)
    print("\nMetrics visualization saved as 'evaluation_metrics.png'")

if __name__ == "__main__":
    main() 