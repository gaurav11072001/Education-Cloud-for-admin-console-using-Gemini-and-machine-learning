import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class CourseRecommender:
    def __init__(self):
        self.courses_df = None
        self.skills_df = None
        self.relationships_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.course_indices = {}
        
    def load_data(self):
        """Load datasets from CSV files"""
        try:
            self.courses_df = pd.read_csv('datasets/courses.csv')
            self.skills_df = pd.read_csv('datasets/skills.csv')
            self.relationships_df = pd.read_csv('datasets/course_relationships.csv')
            
            # Create course indices mapping
            self.course_indices = {
                course_id: idx for idx, course_id in enumerate(self.courses_df['course_id'])
            }
            
            print("Data loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_features(self):
        """Prepare course features for TF-IDF"""
        # Combine relevant features for each course
        course_features = []
        
        for course_id in self.courses_df['course_id']:
            # Get course details
            course = self.courses_df[self.courses_df['course_id'] == course_id].iloc[0]
            
            # Get course skills
            skills = self.skills_df[self.skills_df['course_id'] == course_id]['skill_name'].tolist()
            
            # Combine features
            features = [
                course['technology'],
                course['difficulty'],
                course['prerequisites'],
                *skills
            ]
            
            course_features.append(' '.join(features).lower())
        
        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(course_features)
        
        print("Features prepared successfully!")
        return True
    
    def train(self):
        """Train the recommendation model"""
        if not self.load_data():
            return False
        
        if not self.prepare_features():
            return False
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        # Save the trained model
        if not os.path.exists('models'):
            os.makedirs('models')
        
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'similarity_matrix': self.similarity_matrix,
            'course_indices': self.course_indices
        }
        
        joblib.dump(model_data, 'models/course_recommender.pkl')
        print("Model trained and saved successfully!")
        return True
    
    def get_recommendations(self, course_id, n_recommendations=3):
        """Get course recommendations"""
        try:
            # Get course index
            idx = self.course_indices[course_id]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            
            # Sort courses by similarity
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top N similar courses (excluding itself)
            sim_scores = sim_scores[1:n_recommendations+1]
            
            # Get course indices
            course_indices = [i[0] for i in sim_scores]
            
            # Get recommended courses
            recommendations = self.courses_df.iloc[course_indices]
            
            # Add similarity scores
            similarities = [i[1] for i in sim_scores]
            recommendations['similarity'] = similarities
            
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return None

def test_recommendations():
    """Test the recommendation system"""
    recommender = CourseRecommender()
    if recommender.train():
        # Test recommendations for a few courses
        test_courses = recommender.courses_df['course_id'].sample(3).tolist()
        
        for course_id in test_courses:
            print(f"\nRecommendations for Course ID: {course_id}")
            course = recommender.courses_df[recommender.courses_df['course_id'] == course_id].iloc[0]
            print(f"Course: {course['title']}")
            
            recommendations = recommender.get_recommendations(course_id)
            if recommendations is not None:
                print("\nRecommended Courses:")
                for _, rec in recommendations.iterrows():
                    print(f"- {rec['title']} (Similarity: {rec['similarity']:.2f})")

if __name__ == "__main__":
    print("Training Course Recommendation System...")
    test_recommendations() 