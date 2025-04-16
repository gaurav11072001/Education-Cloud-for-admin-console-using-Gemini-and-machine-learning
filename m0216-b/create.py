import pandas as pd
import random
from datetime import datetime, timedelta
import os

def generate_technical_course_dataset():
    """Generate a comprehensive technical course dataset for training"""
    
    # Core technologies and their related concepts
    technologies = {
        'programming': {
            'topics': ['Data Structures', 'Algorithms', 'OOP', 'Web Development'],
            'frameworks': ['Python', 'Java', 'JavaScript', 'C++', 'Ruby'],
            'difficulty': ['Beginner', 'Intermediate', 'Advanced'],
            'prerequisites': ['Basic Computer Knowledge', 'Problem Solving Skills'],
            'description': 'Learn modern programming languages and software development'
        },
        'machine-learning': {  # New dedicated ML category
            'topics': ['Neural Networks', 'Deep Learning', 'NLP', 'Computer Vision'],
            'frameworks': ['TensorFlow', 'PyTorch', 'Scikit-learn', 'Keras'],
            'difficulty': ['Beginner', 'Intermediate', 'Advanced'],
            'prerequisites': ['Python', 'Mathematics', 'Statistics'],
            'description': 'Master machine learning and AI techniques'
        },
        'data-science': {
            'topics': ['Statistics', 'Data Analysis', 'Data Visualization', 'Big Data'],
            'frameworks': ['Pandas', 'NumPy', 'R', 'Matplotlib'],
            'difficulty': ['Beginner', 'Intermediate', 'Advanced'],
            'prerequisites': ['Mathematics', 'Programming Basics', 'Statistics'],
            'description': 'Learn data analysis and statistical methods'
        }
    }
    
    # Lists to store data
    course_data = []
    skill_data = []
    course_id = 1000
    
    for tech, details in technologies.items():
        for difficulty in details['difficulty']:
            # Basic course info
            course = {
                'course_id': course_id,
                'title': f"{tech.replace('-', ' ').title()} {difficulty}",
                'technology': tech,
                'difficulty': difficulty,
                'duration_weeks': random.randint(8, 16),
                'prerequisites': '|'.join(details['prerequisites']),
                'enrolled_count': random.randint(1000, 50000),
                'rating': round(random.uniform(4.0, 5.0), 1),
                'review_count': random.randint(100, 1000),
                'last_updated': (datetime.now() - timedelta(days=random.randint(0, 90))).strftime('%Y-%m-%d'),
                'price': round(random.uniform(49.99, 199.99), 2),
                'description': details['description'],
                'is_ml_related': tech in ['machine-learning', 'data-science']
            }
            course_data.append(course)
            
            # Generate skills
            for skill in details['frameworks'] + details['topics']:
                skill_data.append({
                    'course_id': course_id,
                    'skill_name': skill,
                    'is_ml_skill': tech in ['machine-learning', 'data-science']
                })
            
            course_id += 1
    
    # Create DataFrames
    courses_df = pd.DataFrame(course_data)
    skills_df = pd.DataFrame(skill_data)
    
    # Print sample data for verification
    print("\nSample courses:")
    print(courses_df[['technology', 'difficulty', 'title']].head())
    
    # Save to CSV
    os.makedirs('datasets', exist_ok=True)
    courses_df.to_csv('datasets/courses.csv', index=False)
    skills_df.to_csv('datasets/skills.csv', index=False)
    
    print(f"\nGenerated {len(courses_df)} courses")
    print(f"Generated {len(skills_df)} skills")
    
    return courses_df, skills_df

def generate_course_relationships():
    """Generate relationships between courses for recommendations"""
    
    # Read the courses data
    courses_df = pd.read_csv('datasets/courses.csv')
    skills_df = pd.read_csv('datasets/skills.csv')
    
    relationships_data = []
    
    # Create course pairs and calculate similarity
    for idx1, course1 in courses_df.iterrows():
        course1_skills = set(skills_df[skills_df['course_id'] == course1['course_id']]['skill_name'])
        
        for idx2, course2 in courses_df.iterrows():
            if course1['course_id'] != course2['course_id']:
                course2_skills = set(skills_df[skills_df['course_id'] == course2['course_id']]['skill_name'])
                
                similarity_score = 0
                
                # Same technology bonus
                if course1['technology'] == course2['technology']:
                    similarity_score += 0.4
                
                # Skill overlap
                common_skills = course1_skills & course2_skills
                similarity_score += len(common_skills) * 0.1
                
                # Prerequisites overlap
                prereq1 = set(course1['prerequisites'].split('|'))
                prereq2 = set(course2['prerequisites'].split('|'))
                common_prereqs = prereq1 & prereq2
                similarity_score += len(common_prereqs) * 0.1
                
                if similarity_score > 0.2:
                    relationships_data.append({
                        'source_course_id': course1['course_id'],
                        'target_course_id': course2['course_id'],
                        'similarity_score': round(similarity_score, 2)
                    })
    
    # Create and save relationships DataFrame
    relationships_df = pd.DataFrame(relationships_data)
    relationships_df.to_csv('datasets/course_relationships.csv', index=False)
    
    return relationships_df

if __name__ == "__main__":
    generate_technical_course_dataset()
    
    print("\nGenerating course relationships...")
    relationships_df = generate_course_relationships()
    print(f"Generated {len(relationships_df)} course relationships")
    
    print("\nDataset creation complete! Files saved in 'datasets' directory:")
    print("- courses.csv")
    print("- skills.csv")
    print("- course_relationships.csv")
