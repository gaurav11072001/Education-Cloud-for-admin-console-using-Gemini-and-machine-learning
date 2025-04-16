import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import google.generativeai as genai  # Add Gemini import
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import json
import random
import os
import traceback  # Add this for detailed error tracking
import re
import markdown
import html
from werkzeug.utils import secure_filename
from PIL import Image
import io
import cv2
import numpy as np
from PIL import ImageEnhance

import fitz  # PyMuPDF for PDF handling
from pdf2image import convert_from_path  # For converting PDF to images
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import text

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = 'your_secret_key'  # Ensure this is a secure key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///education.db'  # SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure Gemini
genai.configure(api_key='AIzaSyDpcpNypqfOPVVP7cDUUKXcNx0zsT9xWYM')
model = genai.GenerativeModel('gemini-1.5-flash-latest')
chat = model.start_chat(history=[])

db = SQLAlchemy(app)

# Get the absolute path to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define upload folder path
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'tiff', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Configure Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define Models first
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)  # Increased length for hash
    discussions = db.relationship('Discussion', backref='user', lazy=True)
    comments = db.relationship('Comment', backref='user', lazy=True)

class Discussion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    likes = db.Column(db.Integer, default=0)
    comments = db.relationship('Comment', backref='discussion', lazy=True)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    discussion_id = db.Column(db.Integer, db.ForeignKey('discussion.id'))
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Database initialization function
def init_db():
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Add indexes for performance
        if not User.__table__.indexes:
            with db.engine.connect() as conn:
                conn.execute(text('CREATE INDEX IF NOT EXISTS idx_username ON user (username)'))
        
        db.session.commit()

# Replace MEDICAL_TOPICS with EDUCATIONAL_TOPICS

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            session['username'] = username
            session['user_id'] = new_user.id
            return redirect(url_for('dashboard'))
        except Exception as e:
            print(f"Error creating user: {e}")
            db.session.rollback()
            flash('Error creating user')
            
    return render_template('register.html')
    
@app.route('/audio/<path:filename>')
def send_audio(filename):
    return send_from_directory('uploads/audio', filename)    

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        print(f"Login attempt for username: {username}")  # Debug print
        
        user = User.query.filter_by(username=username).first()
        print(f"User found: {user}")  # Debug print
        
        if user and check_password_hash(user.password, password):
            session['username'] = username
            session['user_id'] = user.id  # Store user_id in session
            print(f"Login successful for user: {username} (ID: {user.id})")  # Debug print
            return redirect(url_for('dashboard'))
        
        print("Login failed")  # Debug print
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    courses = {
        # Technical Courses
        'machine-learning': {
            'title': 'Machine Learning',
            'description': 'Learn AI and ML fundamentals',
            'icon': 'fas fa-brain',
            'url': 'https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU',
            'platform': 'YouTube',
            'type': 'technical'
        },
        'data-science': {
            'title': 'Data Science',
            'description': 'Master data analysis',
            'icon': 'fas fa-database',
            'url': 'https://www.youtube.com/watch?v=ua-CiDNNj30&list=PLWKjhJtqVAblQe2CCWqV4j_BAT8iRYdWK',
            'platform': 'YouTube',
            'type': 'technical'
        },
        'programming': {
            'title': 'Programming',
            'description': 'Build software applications',
            'icon': 'fas fa-code',
            'url': 'https://www.youtube.com/watch?v=rfscVS0vtbw&t=24s',
            'platform': 'YouTube',
            'type': 'technical'
        },
        # Non-Technical Courses
        'digital-marketing': {
            'title': 'Digital Marketing',
            'description': 'Master online marketing strategies',
            'icon': 'fas fa-bullhorn',
            'url': 'https://www.youtube.com/watch?v=nU-IIXBWlS4',
            'platform': 'YouTube',
            'type': 'non-technical'
        },
        'business': {
            'title': 'Business Management',
            'description': 'Learn business fundamentals',
            'icon': 'fas fa-briefcase',
            'url': 'https://youtu.be/T3l51Psce3c?si=F7pkDHPi3u8yzJE8',
            'platform': 'YouTube',
            'type': 'non-technical'
        },
        'design': {
            'title': 'Graphic Design',
            'description': 'Create stunning visual content',
            'icon': 'fas fa-palette',
            'url': 'https://www.youtube.com/watch?v=WONZVnlam6U',
            'platform': 'YouTube',
            'type': 'non-technical'
        }
    }
    
    return render_template('dashboard.html',
                         username=session['username'],
                         courses=courses)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')


# Add a route to serve files from uploads directory (as a backup)


# Add this route to your Flask application
@app.route('/resources')
def resources():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    resources_data = {
        'study_materials': [
            {
                'title': 'Khan Academy Mathematics',
                'description': 'Free world-class mathematics education from elementary to calculus',
                'type': 'Interactive',
                'link': 'https://www.khanacademy.org/math',
                'icon': 'fas fa-calculator'
            },
            {
                'title': 'MIT OpenCourseWare',
                'description': 'Free access to MIT course materials across multiple subjects',
                'type': 'Course Material',
                'link': 'https://ocw.mit.edu/',
                'icon': 'fas fa-university'
            },
            {
                'title': 'Coursera Learning Platform',
                'description': 'Access to thousands of courses from top universities',
                'type': 'Online Courses',
                'link': 'https://www.coursera.org/',
                'icon': 'fas fa-graduation-cap'
            }
        ],
        'video_tutorials': [
            {
                'title': 'Crash Course',
                'description': 'Educational videos covering science, history, and more',
                'duration': 'Various',
                'link': 'https://thecrashcourse.com/',
                'icon': 'fas fa-play-circle'
            },
            {
                'title': 'TED-Ed',
                'description': 'Educational animations and lessons on various topics',
                'duration': '5-10 mins',
                'link': 'https://ed.ted.com/',
                'icon': 'fas fa-lightbulb'
            },
            {
                'title': 'Codecademy',
                'description': 'Interactive coding tutorials and courses',
                'duration': 'Self-paced',
                'link': 'https://www.codecademy.com/',
                'icon': 'fas fa-code'
            }
        ],
        'practice_exercises': [
            {
                'title': 'Brilliant.org',
                'description': 'Interactive STEM courses and practice problems',
                'difficulty': 'All Levels',
                'link': 'https://brilliant.org/',
                'icon': 'fas fa-brain'
            },
            {
                'title': 'Duolingo',
                'description': 'Free language learning platform',
                'difficulty': 'Beginner to Advanced',
                'link': 'https://www.duolingo.com/',
                'icon': 'fas fa-language'
            },
            {
                'title': 'Project Euler',
                'description': 'Mathematical and programming problems',
                'difficulty': 'Challenging',
                'link': 'https://projecteuler.net/',
                'icon': 'fas fa-square-root-alt'
            }
        ]
    }
    
    return render_template('resources.html', 
                         username=session['username'],
                         resources=resources_data)

# Error handling
@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error occurred'
    }), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404

# Add these routes
@app.route('/community')
def community():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get all discussions with their comments
    discussions = Discussion.query.order_by(Discussion.created_at.desc()).all()
    return render_template('community.html', 
                         username=session['username'],
                         discussions=discussions)

@app.route('/join_discussion/<int:discussion_id>')
def join_discussion(discussion_id):
    if 'username' not in session:
        flash('Please log in first.')
        return redirect(url_for('login'))
    
    discussion = Discussion.query.get_or_404(discussion_id)
    return render_template('discussion.html',
                         username=session['username'],
                         discussion=discussion)

@app.route('/create_discussion', methods=['POST'])
def create_discussion():
    if 'username' not in session or 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Please login first'}), 401
    
    try:
        title = request.form.get('title')
        content = request.form.get('content')
        
        if not title or not content:
            return jsonify({'status': 'error', 'message': 'Title and content are required'}), 400
        
        # Use user_id from session
        user_id = session['user_id']
        
        new_discussion = Discussion(
            user_id=user_id,
            title=title,
            content=content,
            likes=0
        )
        
        db.session.add(new_discussion)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Discussion created successfully'
        })
    
    except Exception as e:
        print(f"Error creating discussion: {str(e)}")
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

def get_db():
    db = sqlite3.connect('database.db')
    db.row_factory = sqlite3.Row
    return db

@app.route('/add_comment', methods=['POST'])
def add_comment():
    if 'username' not in session or 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    try:
        data = request.get_json()
        
        if not data or 'discussion_id' not in data or 'content' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        discussion_id = data['discussion_id']
        content = data['content'].strip()
        
        if not content:
            return jsonify({'error': 'Comment cannot be empty'}), 400
            
        # Create new comment
        new_comment = Comment(
            user_id=session['user_id'],
            discussion_id=discussion_id,
            content=content
        )
        
        db.session.add(new_comment)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'username': session['username'],
            'content': content,
            'created_at': new_comment.created_at.strftime('%B %d, %Y')
        }), 201
        
    except Exception as e:
        print(f"Error adding comment: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/like_discussion/<int:discussion_id>', methods=['POST'])
def like_discussion(discussion_id):
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please login first'}), 401
    
    try:
        discussion = Discussion.query.get_or_404(discussion_id)
        discussion.likes += 1
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Discussion liked successfully',
            'likes': discussion.likes
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Add these routes to your app.py
@app.route('/course/<course_type>/<subject>')
def course(course_type, subject):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Define main courses with their details
    courses = {
        'machine-learning': {
            'title': 'Machine Learning Beginner',
            'description': 'Master machine learning and AI techniques',
            'platform': 'Coursera',
            'icon': 'fas fa-brain',
            'duration': 13,
            'enrolled': 1000,
            'rating': 4.8,
            'reviews': 500,
            'url': 'https://www.coursera.org/learn/machine-learning-introduction',
            'topics': ['AI Fundamentals', 'Neural Networks', 'Deep Learning'],
            'skills': ['TensorFlow', 'PyTorch', 'Scikit-learn']
        },
        'data-science': {
            'title': 'Data Science Beginner',
            'description': 'Learn data analysis and statistical methods',
            'platform': 'Coursera',
            'icon': 'fas fa-database',
            'duration': 13,
            'enrolled': 800,
            'rating': 4.7,
            'reviews': 400,
            'url': 'https://www.coursera.org/professional-certificates/google-data-analytics',
            'topics': ['Data Analysis', 'Statistics', 'Python'],
            'skills': ['Pandas', 'NumPy', 'R']
        },
        'programming': {
            'title': 'Programming Beginner',
            'description': 'Learn modern programming languages',
            'platform': 'Udemy',
            'icon': 'fas fa-code',
            'duration': 8,
            'enrolled': 1200,
            'rating': 4.6,
            'reviews': 600,
            'url': 'https://www.udemy.com/course/complete-python-bootcamp/',
            'topics': ['Python Basics', 'OOP', 'Web Development'],
            'skills': ['Python', 'Java', 'JavaScript']
        }
    }
    
    # Get course info or use default values
    course_info = courses.get(course_type, {
        'title': subject.replace('-', ' ').title(),
        'description': 'Comprehensive course curriculum',
        'platform': 'Coursera',
        'icon': 'fas fa-graduation-cap',
        'duration': 12,
        'enrolled': 500,
        'rating': 4.5,
        'reviews': 300,
        'url': 'https://www.coursera.org',
        'topics': ['Topic 1', 'Topic 2', 'Topic 3'],
        'skills': ['Skill 1', 'Skill 2', 'Skill 3']
    })
    
    return render_template('course.html',
                         username=session['username'],
                         course_type=course_type,
                         subject=subject,
                         course=course_info)

# Add these routes to handle enrollment and syllabus download
@app.route('/enroll/<course_type>/<subject>')
def enroll(course_type, subject):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Here you would typically handle course enrollment logic
    flash('Successfully enrolled in the course!', 'success')
    return redirect(url_for('course', course_type=course_type, subject=subject))

@app.route('/download_syllabus/<course_type>/<subject>')
def download_syllabus(course_type, subject):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Here you would typically handle syllabus download logic
    # For now, we'll just redirect back to the course page
    flash('Syllabus download started!', 'success')
    return redirect(url_for('course', course_type=course_type, subject=subject))

# Add this to your existing course content dictionary
def get_related_courses(current_course, course_type, subject, preferences=None):
    """Get personalized course recommendations"""
    try:
        # Load course data
        courses_df = pd.read_csv('datasets/courses.csv')
        skills_df = pd.read_csv('datasets/skills.csv')
        
        # Get course skills
        course_skills = skills_df[
            skills_df['course_id'] == current_course.get('course_id', 0)
        ]['skill_name'].tolist()
        
        # Get preference matches
        matches = get_preference_matches(current_course, preferences) if preferences else []
        
        recommendations = []
        seen_techs = set()
        
        # Updated genuine course URLs
        course_urls = {
            'machine-learning': {
                'Beginner': {
                    'url': 'https://www.coursera.org/learn/machine-learning-introduction',
                    'title': 'Machine Learning Beginner',
                    'skills': ['TensorFlow', 'PyTorch', 'Scikit-learn'],
                    'description': 'Master machine learning and AI techniques'
                }
            },
            'data-science': {
                'Beginner': {
                    'url': 'https://www.coursera.org/professional-certificates/google-data-analytics',
                    'title': 'Data Science Beginner',
                    'skills': ['TensorFlow', 'PyTorch', 'Scikit-learn'],
                    'description': 'Learn data analysis and statistical methods'
                }
            },
            'programming': {
                'Beginner': {
                    'url': 'https://www.coursera.org/specializations/python',
                    'title': 'Programming Beginner',
                    'skills': ['TensorFlow', 'PyTorch', 'Scikit-learn'],
                    'description': 'Learn modern programming languages and software development'
                }
            }
        }
        
        # Get base recommendations
        base_technologies = ['machine-learning', 'data-science', 'programming']
        for tech in base_technologies:
            if tech in seen_techs:
                continue
                
            tech_courses = courses_df[courses_df['technology'] == tech]
            if tech_courses.empty:
                continue
                
            course_dict = tech_courses.iloc[0].to_dict()
            difficulty = course_dict.get('difficulty', 'Beginner')
            
            # Get URL based on technology and difficulty
            tech_urls = course_urls.get(tech, {})
            url = tech_urls.get(difficulty, tech_urls.get('Beginner', '#'))
            
            formatted_rec = {
                'type': 'technical',
                'subject': tech,
                'title': course_dict['title'],
                'icon': get_course_icon('technical', tech),
                'short_desc': get_course_description('technical', tech),
                'duration': int(course_dict.get('duration_weeks', 12)),
                'similarity': 80.0,  # Default similarity score
                'skills': course_skills[:3] if course_skills else [],
                'level': difficulty,
                'matches': matches,
                'url': url,
                'target': '_blank'
            }
            
            recommendations.append(formatted_rec)
            seen_techs.add(tech)
            
            # Convert any numpy types to Python native types
            for key, value in formatted_rec.items():
                if isinstance(value, (np.integer, np.floating)):
                    formatted_rec[key] = int(value) if isinstance(value, np.integer) else float(value)
        
        return recommendations
        
    except Exception as e:
        print(f"Error in get_related_courses: {e}")
        traceback.print_exc()
        return []

def get_preference_matches(course_data, preferences):
    """Get matching preferences for course"""
    matches = []
    if not preferences:
        return matches
        
    if preferences.get('experience') == course_data.get('difficulty'):
        matches.append('Experience Level')
    if any(interest in course_data.get('technology', '').split('|') 
           for interest in preferences.get('interests', [])):
        matches.append('Interests')
    if preferences.get('time_commitment') == 'full' or \
       course_data.get('duration_weeks', 0) <= 12:
        matches.append('Time Commitment')
    return matches

def get_course_icon(course_type, technology):
    """Get consistent icon for course type"""
    icons = {
        'programming': 'fas fa-code',
        'machine-learning': 'fas fa-brain',
        'data-science': 'fas fa-database',
        'default': 'fas fa-graduation-cap'
    }
    return icons.get(technology, icons['default'])

def get_course_description(course_type, subject):
    """Return short course description"""
    descriptions = {
        'programming': 'Learn modern programming languages and software development',
        'machine-learning': 'Master machine learning and AI techniques',
        'data-science': 'Learn data analysis and statistical methods',
        'default': 'Comprehensive course curriculum'
    }
    return descriptions.get(subject, descriptions['default'])

def get_course_duration(course_type, subject):
    """Return course duration in weeks"""
    durations = {
        'programming': 12,
        'data-science': 16,
        'networking': 10,
        'app-development': 14,
        'business': 8,
        'design': 10,
        'languages': 12,
        'marketing': 8
    }
    return durations.get(subject, 12)

@app.route('/course_preferences', methods=['POST'])
def update_course_preferences():
    """Update course recommendations based on user preferences"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    print("Received preferences:", request.form)
    
    preferences = {
        'experience': request.form.get('experience'),
        'interests': request.form.getlist('interests'),
        'goal': request.form.get('goal'),
        'time_commitment': request.form.get('time_commitment')
    }
    
    session['course_preferences'] = preferences
    
    try:
        # Load courses from CSV
        courses_df = pd.read_csv('datasets/courses.csv')
        print("\nAvailable technologies:", courses_df['technology'].unique().tolist())
        
        subject = request.form.get('subject')
        experience = preferences['experience']  # Get user's selected experience level
        print(f"Looking for subject: {subject} with experience: {experience}")
        
        # Find matching courses with the selected difficulty level
        matching_courses = courses_df[
            (courses_df['technology'] == subject) & 
            (courses_df['difficulty'] == experience)
        ]
        
        if matching_courses.empty:
            # Fallback to any course in that technology if exact match not found
            matching_courses = courses_df[courses_df['technology'] == subject]
            if matching_courses.empty:
                print(f"No courses found for technology: {subject}")
                return jsonify({
                    'success': False,
                    'error': f'No courses found for {subject}'
                }), 404
        
        # Convert DataFrame row to dict and handle numpy types
        current_course = matching_courses.iloc[0].to_dict()
        current_course = {
            k: int(v) if isinstance(v, np.integer) else (
                float(v) if isinstance(v, np.floating) else v
        )}
        
        # Ensure difficulty is set
        if 'difficulty' not in current_course:
            current_course['difficulty'] = experience
        
        print(f"Found course: {current_course['title']} ({current_course['difficulty']})")
        
        recommendations = get_related_courses(
            current_course, 
            'technical',
            subject, 
            preferences
        )
        
        # Convert any remaining numpy types in recommendations
        for rec in recommendations:
            for key, value in rec.items():
                if isinstance(value, (np.integer, np.floating)):
                    rec[key] = int(value) if isinstance(value, np.integer) else float(value)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"Error in update_course_preferences: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/recommend-non-technical', methods=['GET', 'POST'])
def recommend_non_technical():
    if 'username' not in session:
        flash('Please login to access course recommendations', 'warning')
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        try:
            # Get form data and handle empty JSON
            try:
                form_data = json.loads(request.form.get('formData', '{}'))
            except json.JSONDecodeError:
                form_data = {}
            
            print("Received form data:", form_data)
            
            # Get valid options from label encoders to use as defaults
            valid_education = label_encoders['Education'].classes_[0]
            valid_gender = label_encoders['Gender'].classes_[0]
            valid_state = label_encoders['State'].classes_[0]
            valid_hobby = label_encoders['Hobby'].classes_[0]
            
            # Process user input with default values from valid options
            user_input = {
                'skill_level': int(form_data.get('skill_level', 1)),
                'duration_weeks': int(form_data.get('duration_weeks', 4)),
                'free_time_hours': int(form_data.get('free_time_hours', 10)),
                'education': str(form_data.get('education')) if form_data.get('education') else valid_education,
                'gender': str(form_data.get('gender')) if form_data.get('gender') else valid_gender,
                'state': str(form_data.get('state')) if form_data.get('state') else valid_state,
                'age': int(form_data.get('age', 25)),
                'hobby': str(form_data.get('hobby')) if form_data.get('hobby') else valid_hobby
            }
            
            print("Processing user input with defaults:", user_input)
            
            try:
                courses = predict_courses(user_input)
                if not courses:
                    return render_template('index.html', 
                                        courses=[],
                                        username=session['username'],
                                        error="No courses found matching your criteria")
                
                return render_template('index.html', 
                                    courses=courses,
                                    username=session['username'])
                
            except Exception as prediction_error:
                print(f"Error in prediction: {prediction_error}")
                traceback.print_exc()
                return render_template('index.html', 
                                    courses=[],
                                    username=session['username'],
                                    error=f"Prediction error: {str(prediction_error)}")
            
        except Exception as e:
            print(f"Error in form processing: {str(e)}")
            traceback.print_exc()
            return render_template('index.html', 
                                courses=[],
                                username=session['username'],
                                error=f"Form processing error: {str(e)}")
    
    return render_template('index.html', 
                         courses=[],
                         username=session['username'])

@app.route('/recommend-technical', methods=['GET', 'POST'])
def recommend_technical():
    if 'username' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        try:
            # Get form data and handle empty JSON
            try:
                form_data = json.loads(request.form.get('formData', '{}'))
            except json.JSONDecodeError:
                form_data = {}
            
            print("Received form data:", form_data)
            
            # Get valid options from label encoders
            valid_education = label_encoders_technical['Education'].classes_[0]
            valid_gender = label_encoders_technical['Gender'].classes_[0]
            valid_state = label_encoders_technical['State'].classes_[0]
            valid_hobby = label_encoders_technical['Hobby'].classes_[0]
            
            # Process user input with default values from valid options
            user_input = {
                'skill_level': int(form_data.get('skill_level', 1)),
                'duration_weeks': int(form_data.get('duration_weeks', 4)),
                'free_time_hours': int(form_data.get('free_time_hours', 10)),
                'education': str(form_data.get('education')) if form_data.get('education') else valid_education,
                'gender': str(form_data.get('gender')) if form_data.get('gender') else valid_gender,
                'state': str(form_data.get('state')) if form_data.get('state') else valid_state,
                'age': int(form_data.get('age', 25)),
                'hobby': str(form_data.get('hobby')) if form_data.get('hobby') else valid_hobby
            }
            
            print("Processing user input with defaults:", user_input)
            
            try:
                courses = predict_courses_technical(user_input)
                if not courses:
                    return render_template('index1.html', 
                                        courses=[],
                                        username=session['username'],
                                        error="No courses found matching your criteria")
                
                return render_template('index1.html', 
                                    courses=courses,
                                    username=session['username'])
                
            except Exception as prediction_error:
                print(f"Error in prediction: {prediction_error}")
                traceback.print_exc()
                return render_template('index1.html', 
                                    courses=[],
                                    username=session['username'],
                                    error=f"Prediction error: {str(prediction_error)}")
            
        except Exception as e:
            print(f"Error in form processing: {str(e)}")
            traceback.print_exc()
            return render_template('index1.html', 
                                courses=[],
                                username=session['username'],
                                error=f"Form processing error: {str(e)}")
    
    return render_template('index1.html', 
                         courses=[],
                         username=session['username'])

@app.route('/get_allowed_hobbies')
def get_allowed_hobbies():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
        
    try:
        allowed_hobbies = label_encoders_technical['Hobby'].classes_.tolist()
        return jsonify({'hobbies': allowed_hobbies})
    except Exception as e:
        print(f"Error getting allowed hobbies: {e}")
        return jsonify({'hobbies': ['Reading', 'Gaming', 'Sports', 'Music', 'Dancing', 
                                  'Painting', 'Writing', 'Photography', 'Cooking', 'Traveling']})

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load datasets and models
try:
    # Non-technical courses
    df = pd.read_csv('non_technical_courses_with_reviews.csv')
    model = joblib.load('trained_model.pkl')
    
    label_encoders = {}
    for col in ['Category', 'Skill Level', 'Education', 'Gender', 'State', 'Hobby']:
        label_encoders[col] = joblib.load(f'label_encoders/{col}_label_encoder.pkl')
    
    # Technical courses
    df_technical = pd.read_csv('technical_courses_with_reviews.csv')
    model_technical = joblib.load('trained_technical_model.pkl')
    
    label_encoders_technical = {}
    for col in ['Category', 'Skill Level', 'Education', 'Gender', 'State', 'Hobby']:
        label_encoders_technical[col] = joblib.load(f'{col}_label_encoder.pkl')
        
except Exception as e:
    print(f"Error loading data and models: {e}")
    traceback.print_exc()

# Course image mappings
course_images = {
    'Effective Communication Skills': 'images/3053856.jpg',
    'Project Management Essentials': 'images/projectmanagement.jpg',
    'Creative Writing Workshop': 'images/writing.jpg',
    'Mindfulness and Meditation': 'images/mindfull.jpg',
    'Financial Literacy Basics': 'images/financially.jpg',
    'Leadership Development': 'images/leadership.jpg',
    'Art of Negotiation': 'images/negotiation.jpg',
    'Time Management Strategies': 'images/time.jpg',
    'Public Speaking Mastery': 'images/public.jpg',
    'Stress Management Techniques': 'images/stress.jpg',
    'Team Building and Collaboration': 'images/team.jpg',
    'Critical Thinking and Problem Solving': 'images/leadership.jpg',
    'Personal Branding for Success': 'images/brand.jpg',
    'Introduction to Graphic Design': 'images/graphic.jpg',
    'Digital Marketing Fundamentals': 'images/marketing.jpg'
}

course_images11 = {
    'Python Programming Basics': 'images/programming.jpg',
    'Introduction to Data Science': 'images/data.jpg',
    'Machine Learning with Python': 'images/deep.jpg',
    'Cybersecurity Essentials': 'images/cyber.jpg',
    'Web Development Bootcamp': 'images/web.jpg',
    'Advanced Java Programming': 'images/programming.jpg',
    'SQL for Data Analysis': 'images/sql.jpg',
    'Deep Learning Fundamentals': 'images/deep.jpg',
    'Ethical Hacking Workshop': 'images/hack.jpg',
    'React.js for Beginners': 'images/react.jpg',
    'Introduction to Cloud Computing': 'images/web.jpg',
    'Data Visualization with Tableau': 'images/data.jpg',
    'AI for Everyone': 'images/ai.jpg',
    'Kubernetes Basics': 'images/kuber.jpg',
    'Docker for Developers': 'images/docker.jpg',
}

categories = ['Beginner', 'Intermediate', 'Advanced']

def predict_courses(user_input, num_courses=5):
    """Predicts non-technical courses based on user input"""
    try:
        # Ensure inputs are valid for encoders
        education_value = user_input.get('education')
        gender_value = user_input.get('gender')
        state_value = user_input.get('state')
        hobby_value = user_input.get('hobby')
        
        # Safety check: use try/except for each transform to catch label errors
        try:
            education_encoded = label_encoders['Education'].transform([education_value])[0]
        except (ValueError, KeyError):
            # Use the first valid value if error
            education_encoded = 0
            
        try:
            gender_encoded = label_encoders['Gender'].transform([gender_value])[0]
        except (ValueError, KeyError):
            gender_encoded = 0
            
        try:
            state_encoded = label_encoders['State'].transform([state_value])[0]
        except (ValueError, KeyError):
            state_encoded = 0
            
        try:
            hobby_encoded = label_encoders['Hobby'].transform([hobby_value])[0]
        except (ValueError, KeyError):
            hobby_encoded = 0
        
        input_data = {
            'Category': 0,
            'Skill Level': user_input.get('skill_level'),
            'Duration': int(user_input.get('duration_weeks', 4)),
            'Free Time Hours': user_input.get('free_time_hours'),
            'Education': education_encoded,
            'Gender': gender_encoded,
            'State': state_encoded,
            'Age': user_input.get('age'),
            'Hobby': hobby_encoded
        }
        
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        
        # If no courses in the database, return empty list
        if df.empty:
            print("Warning: Course database is empty")
            return []
            
        recommended_courses = df[df['Rating'] >= 3.5]
        
        # If filtering resulted in empty dataframe, use all courses
        if recommended_courses.empty:
            recommended_courses = df
            
        recommended_courses = recommended_courses.drop_duplicates(subset=['Course Title'])
        
        # If still not enough courses, just sample without frac=1
        if len(recommended_courses) < num_courses:
            recommended_courses = recommended_courses.sample(n=min(len(recommended_courses), num_courses))
        else:
            recommended_courses = recommended_courses.sample(frac=1).head(num_courses)
        
        results = []
        for _, row in recommended_courses.iterrows():
            # Process course info safely
            try:
                sentiment_score = analyzer.polarity_scores(row['Reviews'])['compound']
                sentiment = 'Positive' if sentiment_score >= 0.05 else 'Negative' if sentiment_score <= -0.05 else 'Neutral'
                
                course_title = row.get('Course Title', 'Untitled Course')
                course_title_parts = course_title.split(' - ', 1)
                course_title_main = course_title_parts[0]
                course_title_remaining = course_title_parts[1] if len(course_title_parts) > 1 else ''
                
                course_image = course_images.get(course_title_main, 'images/4530890.jpg')
                
                results.append({
                    'Course Title': course_title_main,
                    'Title Part': course_title_remaining,
                    'Category': random.choice(categories),
                    'Rating': float(row.get('Rating', 3.5)),
                    'Course Link': row.get('Course Link', '#'),
                    'Review': row.get('Reviews', 'No reviews available.'),
                    'Sentiment': sentiment,
                    'Image': course_image
                })
            except Exception as row_error:
                print(f"Error processing course row: {row_error}")
                continue
        
        if not results:
            # If still no results, create a default placeholder course
            results.append({
                'Course Title': 'Introduction to Non-Technical Skills',
                'Title Part': 'Recommended based on your interests',
                'Category': 'Beginner',
                'Rating': 4.0,
                'Course Link': '#',
                'Review': 'This foundational course covers essential non-technical skills for your career development.',
                'Sentiment': 'Positive',
                'Image': 'images/4530890.jpg'
            })
            
        # Sort by sentiment
        sentiment_order = {'Positive': 0, 'Neutral': 1, 'Negative': 2}
        results.sort(key=lambda x: sentiment_order[x['Sentiment']])
        return results
        
    except Exception as e:
        print(f"Error in predict_courses: {e}")
        traceback.print_exc()
        
        # Return a fallback course as a last resort
        return [{
            'Course Title': 'Personal Development Fundamentals',
            'Title Part': 'Recommended Starter Course',
            'Category': 'Beginner',
            'Rating': 4.5,
            'Course Link': '#',
            'Review': 'A great starting point for anyone looking to improve their non-technical skills.',
            'Sentiment': 'Positive',
            'Image': 'images/4530890.jpg'
        }]

def predict_courses_technical(user_input, num_courses=5):
    """
    Predicts multiple technical courses with personalized recommendations based on user input.
    """
    try:
        input_data = {
            'Category': 0,
            'Skill Level': user_input.get('skill_level'),
            'Duration': int(user_input.get('duration_weeks', 4)),
            'Free Time Hours': user_input.get('free_time_hours'),
            'Education': label_encoders_technical['Education'].transform([user_input.get('education')])[0],
            'Gender': label_encoders_technical['Gender'].transform([user_input.get('gender')])[0],
            'State': label_encoders_technical['State'].transform([user_input.get('state')])[0],
            'Age': user_input.get('age'),
            'Hobby': label_encoders_technical['Hobby'].transform([user_input.get('hobby')])[0]
        }
        
        input_df = pd.DataFrame([input_data])
        prediction = model_technical.predict(input_df)[0]
        
        # Filter courses based on user's skill level
        skill_level = int(user_input.get('skill_level', 1))
        if skill_level == 1:
            difficulty_keywords = ['beginner', 'basic', 'introduction', 'fundamental']
        elif skill_level == 2:
            difficulty_keywords = ['intermediate', 'advanced-beginner', 'practical']
        else:
            difficulty_keywords = ['advanced', 'expert', 'professional', 'master']
            
        # Filter based on course title and description matching skill level
        recommended_courses = df_technical[
            (df_technical['Rating'] >= 3.5) & 
            (df_technical['Course Title'].str.lower().apply(
                lambda x: any(keyword in x.lower() for keyword in difficulty_keywords)
            ))
        ]
        
        # If not enough courses found, add more based on rating
        if len(recommended_courses) < num_courses:
            additional_courses = df_technical[
                (df_technical['Rating'] >= 3.5) & 
                (~df_technical.index.isin(recommended_courses.index))
            ]
            recommended_courses = pd.concat([recommended_courses, additional_courses])
        
        # Remove duplicates and get random selection
        recommended_courses = recommended_courses.drop_duplicates(subset=['Course Title'])
        
        # Sort by rating and then take top courses
        recommended_courses = recommended_courses.sort_values('Rating', ascending=False).head(num_courses * 2)
        
        # Randomly select from top courses
        recommended_courses = recommended_courses.sample(n=min(num_courses, len(recommended_courses)))
        
        results = []
        categories = ['Beginner', 'Intermediate', 'Advanced']
        
        for _, row in recommended_courses.iterrows():
            sentiment_score = analyzer.polarity_scores(row['Reviews'])['compound']
            sentiment = 'Positive' if sentiment_score >= 0.05 else 'Negative' if sentiment_score <= -0.05 else 'Neutral'
            
            course_title_main = row['Course Title'].split(' - ')[0]
            course_title_remaining = ' - '.join(row['Course Title'].split(' - ')[1:])
            
            # Determine category based on skill level and course title
            if any(keyword in course_title_main.lower() for keyword in ['beginner', 'basic', 'introduction']):
                category = 'Beginner'
            elif any(keyword in course_title_main.lower() for keyword in ['advanced', 'expert', 'professional']):
                category = 'Advanced'
            else:
                category = 'Intermediate'
            
            course_image = course_images11.get(course_title_main, 'images/default.jpg')
            
            results.append({
                'Course Title': course_title_main,
                'Remaining Title': course_title_remaining,
                'Category': category,
                'Rating': row['Rating'],
                'Course Link': row['Course Link'],
                'Review': row['Reviews'],
                'Sentiment': sentiment,
                'Image': course_image
            })
        
        # Sort by sentiment and rating
        sentiment_order = {'Positive': 0, 'Neutral': 1, 'Negative': 2}
        results.sort(key=lambda x: (sentiment_order[x['Sentiment']], -x['Rating']))
        
        return results
        
    except Exception as e:
        print(f"Error in predict_courses_technical: {e}")
        traceback.print_exc()
        return []

@app.route('/check-model-status')
def check_model_status():
    """Debug endpoint to check model and encoder status"""
    try:
        return jsonify({
            'model_loaded': model_technical is not None,
            'encoders_loaded': list(label_encoders_technical.keys()),
            'df_loaded': not df_technical.empty,
            'sample_values': {
                'education': list(label_encoders_technical['Education'].classes_),
                'gender': list(label_encoders_technical['Gender'].classes_),
                'state': list(label_encoders_technical['State'].classes_),
                'hobby': list(label_encoders_technical['Hobby'].classes_)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/search_courses', methods=['GET', 'POST'])
def search_courses():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    search_query = ""
    search_results = []
    
    if request.method == 'POST':
        search_query = request.form.get('search_query', '').lower().strip()
    elif request.method == 'GET':
        search_query = request.args.get('q', '').lower().strip()
    
    if search_query:
        # Define course categories with multiple YouTube links by difficulty level
        course_catalog = {
            'python': {
                'title': 'Python Programming',
                'description': 'Learn Python programming from basics to advanced concepts',
                'icon': 'fas fa-code',
                'type': 'technical',
                'levels': {
                    'Beginner': [
                        {'title': 'Python for Beginners - Full Course', 'url': 'https://www.youtube.com/watch?v=_uQrJ0TkZlc', 'duration': '6 hours'},
                        {'title': 'Python Crash Course', 'url': 'https://www.youtube.com/watch?v=rfscVS0vtbw', 'duration': '4 hours'},
                        {'title': 'Learn Python in 1 Hour', 'url': 'https://www.youtube.com/watch?v=kqtD5dpn9C8', 'duration': '1 hour'},
                        {'title': 'Python Tutorial - Python for Beginners', 'url': 'https://www.youtube.com/watch?v=_Z1RKVPEUdc', 'duration': '2 hours'},
                        {'title': 'Python Full Course for Beginners', 'url': 'https://www.youtube.com/watch?v=XKHEtdqhLK8', 'duration': '12 hours'}
                    ],
                    'Intermediate': [
                        {'title': 'Intermediate Python Programming Course', 'url': 'https://www.youtube.com/watch?v=HGOBQPFzWKo', 'duration': '6 hours'},
                        {'title': 'Python Object Oriented Programming', 'url': 'https://www.youtube.com/watch?v=Ej_02ICOIgs', 'duration': '2 hours'},
                        {'title': 'Python Data Structures', 'url': 'https://www.youtube.com/watch?v=pkYVOmU3MgA', 'duration': '4 hours'},
                        {'title': 'Python Algorithms', 'url': 'https://www.youtube.com/watch?v=pkYVOmU3MgA', 'duration': '3 hours'},
                        {'title': 'Python File Handling', 'url': 'https://www.youtube.com/watch?v=Uh2ebFW8OYM', 'duration': '1 hour'}
                    ],
                    'Advanced': [
                        {'title': 'Advanced Python - Full Course', 'url': 'https://www.youtube.com/watch?v=QLTdOEn79Rc', 'duration': '6 hours'},
                        {'title': 'Python Decorators', 'url': 'https://www.youtube.com/watch?v=FsAPt_9Bf3U', 'duration': '1 hour'},
                        {'title': 'Python Design Patterns', 'url': 'https://www.youtube.com/watch?v=bsyjSW46TDg', 'duration': '3 hours'},
                        {'title': 'Async IO in Python', 'url': 'https://www.youtube.com/watch?v=t5Bo1Je9EmE', 'duration': '2 hours'},
                        {'title': 'Python Concurrency', 'url': 'https://www.youtube.com/watch?v=olYdb0DdGtM', 'duration': '3 hours'}
                    ]
                }
            },
            'java': {
                'title': 'Java Programming',
                'description': 'Master Java programming language and ecosystem',
                'icon': 'fab fa-java',
                'type': 'technical',
                'levels': {
                    'Beginner': [
                        {'title': 'Java Tutorial for Beginners', 'url': 'https://www.youtube.com/watch?v=eIrMbAQSU34', 'duration': '2 hours'},
                        {'title': 'Java Full Course for Beginners', 'url': 'https://www.youtube.com/watch?v=grEKMHGYyns', 'duration': '12 hours'},
                        {'title': 'Java Crash Course', 'url': 'https://www.youtube.com/watch?v=drQK8ciCAjY', 'duration': '3 hours'},
                        {'title': 'Java Programming for Beginners', 'url': 'https://www.youtube.com/watch?v=RRubcjpTkks', 'duration': '6 hours'},
                        {'title': 'Java in 1 Hour', 'url': 'https://www.youtube.com/watch?v=CFD9EFcNZTQ', 'duration': '1 hour'}
                    ],
                    'Intermediate': [
                        {'title': 'Intermediate Java Programming', 'url': 'https://www.youtube.com/watch?v=7i1f0Mk8D0A', 'duration': '4 hours'},
                        {'title': 'Java Collections Framework', 'url': 'https://www.youtube.com/watch?v=GdAon80-0KA', 'duration': '2 hours'},
                        {'title': 'Java Multithreading', 'url': 'https://www.youtube.com/watch?v=r_MbozD32eo', 'duration': '3 hours'},
                        {'title': 'Java IO & NIO', 'url': 'https://www.youtube.com/watch?v=5VWUUuCe_FU', 'duration': '2 hours'},
                        {'title': 'Java Exception Handling', 'url': 'https://www.youtube.com/watch?v=xNVlq9IEBEg', 'duration': '1 hour'}
                    ],
                    'Advanced': [
                        {'title': 'Advanced Java Programming', 'url': 'https://www.youtube.com/watch?v=Ae-r8hsbPUo', 'duration': '5 hours'},
                        {'title': 'Java Design Patterns', 'url': 'https://www.youtube.com/watch?v=NU_1StN5Tkk', 'duration': '3 hours'},
                        {'title': 'Java Spring Framework', 'url': 'https://www.youtube.com/watch?v=VvGjZgqojMc', 'duration': '8 hours'},
                        {'title': 'Java Microservices', 'url': 'https://www.youtube.com/watch?v=y8IQb4ofjDo', 'duration': '4 hours'},
                        {'title': 'Java Performance Tuning', 'url': 'https://www.youtube.com/watch?v=ykpZUA33fYA', 'duration': '2 hours'}
                    ]
                }
            },
            'c++': {
                'title': 'C++ Programming',
                'description': 'Learn C++ from foundation to advanced topics',
                'icon': 'fas fa-file-code',
                'type': 'technical',
                'levels': {
                    'Beginner': [
                        {'title': 'C++ Tutorial for Beginners - Full Course', 'url': 'https://www.youtube.com/watch?v=vLnPwxZdW4Y', 'duration': '4 hours'},
                        {'title': 'C++ Programming Course - Beginner to Advanced', 'url': 'https://www.youtube.com/watch?v=8jLOx1hD3_o', 'duration': '31 hours'},
                        {'title': 'C++ Crash Course', 'url': 'https://www.youtube.com/watch?v=uhFpPlMsLzY', 'duration': '3 hours'},
                        {'title': 'C++ Programming All-in-One Tutorial', 'url': 'https://www.youtube.com/watch?v=_bYFu9mBnr4', 'duration': '7 hours'},
                        {'title': 'Learn C++ in 1 Hour', 'url': 'https://www.youtube.com/watch?v=ZzaPdXTrSb8', 'duration': '1 hour'}
                    ],
                    'Intermediate': [
                        {'title': 'Intermediate C++ Programming', 'url': 'https://www.youtube.com/watch?v=sxHng1iufQE', 'duration': '3 hours'},
                        {'title': 'C++ STL', 'url': 'https://www.youtube.com/watch?v=LyGlTmaWEPs', 'duration': '6 hours'},
                        {'title': 'C++ Object-Oriented Programming', 'url': 'https://www.youtube.com/watch?v=wN0x9eZLix4', 'duration': '4 hours'},
                        {'title': 'C++ Templates', 'url': 'https://www.youtube.com/watch?v=I-hZkUa9mIs', 'duration': '2 hours'},
                        {'title': 'C++ Exception Handling', 'url': 'https://www.youtube.com/watch?v=kjEhqgmEiWY', 'duration': '1 hour'}
                    ],
                    'Advanced': [
                        {'title': 'Advanced C++ Programming', 'url': 'https://www.youtube.com/watch?v=F_vIB3yjxaM', 'duration': '6 hours'},
                        {'title': 'C++ Multithreading', 'url': 'https://www.youtube.com/watch?v=F6Ipn7gCOsY', 'duration': '3 hours'},
                        {'title': 'C++ Smart Pointers', 'url': 'https://www.youtube.com/watch?v=UOB7-B2MfwA', 'duration': '1 hour'},
                        {'title': 'C++ Memory Management', 'url': 'https://www.youtube.com/watch?v=u2V9-uqWi1M', 'duration': '2 hours'},
                        {'title': 'C++ Design Patterns', 'url': 'https://www.youtube.com/watch?v=2hmrWoYGjS0', 'duration': '4 hours'}
                    ]
                }
            },
            'react': {
                'title': 'React.js Development',
                'description': 'Build modern web applications with React',
                'icon': 'fab fa-react',
                'type': 'technical',
                'levels': {
                    'Beginner': [
                        {'title': 'React Tutorial for Beginners', 'url': 'https://www.youtube.com/watch?v=SqcY0GlETPk', 'duration': '6 hours'},
                        {'title': 'React JS Crash Course', 'url': 'https://www.youtube.com/watch?v=w7ejDZ8SWv8', 'duration': '1.5 hours'},
                        {'title': 'React JS Full Course for Beginners', 'url': 'https://www.youtube.com/watch?v=RVFAyFWO4go', 'duration': '8 hours'},
                        {'title': 'Learn React In 30 Minutes', 'url': 'https://www.youtube.com/watch?v=hQAHSlTtcmY', 'duration': '30 minutes'},
                        {'title': 'React for Absolute Beginners', 'url': 'https://www.youtube.com/watch?v=d7pyEDqBDeE', 'duration': '4 hours'}
                    ],
                    'Intermediate': [
                        {'title': 'Intermediate React', 'url': 'https://www.youtube.com/watch?v=bMknfKXIFA8', 'duration': '7 hours'},
                        {'title': 'React Hooks Tutorial', 'url': 'https://www.youtube.com/watch?v=f687hBjwFcM', 'duration': '3 hours'},
                        {'title': 'React Router Tutorial', 'url': 'https://www.youtube.com/watch?v=Law7wfdg_ls', 'duration': '1 hour'},
                        {'title': 'React Context & Hooks', 'url': 'https://www.youtube.com/watch?v=6RhOzQciVwI', 'duration': '4 hours'},
                        {'title': 'React Form Handling', 'url': 'https://www.youtube.com/watch?v=bU_eq8qyjic', 'duration': '2 hours'}
                    ],
                    'Advanced': [
                        {'title': 'Advanced React Patterns', 'url': 'https://www.youtube.com/watch?v=WV0UUcSPk-0', 'duration': '5 hours'},
                        {'title': 'React Performance Optimization', 'url': 'https://www.youtube.com/watch?v=OlVkYnVXPl0', 'duration': '2 hours'},
                        {'title': 'Redux & Advanced State Management', 'url': 'https://www.youtube.com/watch?v=CFQqnHzxQaE', 'duration': '4 hours'},
                        {'title': 'React Testing', 'url': 'https://www.youtube.com/watch?v=OVNjsIto9xM', 'duration': '3 hours'},
                        {'title': 'React TypeScript', 'url': 'https://www.youtube.com/watch?v=Z5iWr6Srsj8', 'duration': '2 hours'}
                    ]
                }
            },
            'digital marketing': {
                'title': 'Digital Marketing',
                'description': 'Master online marketing strategies and techniques',
                'icon': 'fas fa-bullhorn',
                'type': 'non-technical',
                'levels': {
                    'Beginner': [
                        {'title': 'Digital Marketing Course for Beginners', 'url': 'https://www.youtube.com/watch?v=nU-IIXBWlS4', 'duration': '4 hours'},
                        {'title': 'Digital Marketing Full Course', 'url': 'https://www.youtube.com/watch?v=x4L1MryF8bY', 'duration': '9 hours'},
                        {'title': 'Introduction to Digital Marketing', 'url': 'https://www.youtube.com/watch?v=j4n5G7tfsxI', 'duration': '2 hours'},
                        {'title': 'Digital Marketing Fundamentals', 'url': 'https://www.youtube.com/watch?v=nU-IIXBWlS4', 'duration': '4 hours'},
                        {'title': 'Digital Marketing Strategy', 'url': 'https://www.youtube.com/watch?v=ZsQ8PdmZ0_Y', 'duration': '1 hour'}
                    ],
                    'Intermediate': [
                        {'title': 'Complete SEO Course for Beginners', 'url': 'https://www.youtube.com/watch?v=xsVTqzratPs', 'duration': '3 hours'},
                        {'title': 'Social Media Marketing Full Course', 'url': 'https://www.youtube.com/watch?v=t_hFrFGPRME', 'duration': '6 hours'},
                        {'title': 'Email Marketing Course', 'url': 'https://www.youtube.com/watch?v=tClNutMjvCE', 'duration': '2 hours'},
                        {'title': 'Content Marketing Masterclass', 'url': 'https://www.youtube.com/watch?v=N1n1i3IZxL8', 'duration': '4 hours'},
                        {'title': 'Google Ads Tutorial 2023', 'url': 'https://www.youtube.com/watch?v=tJwJIVSHmpc', 'duration': '3 hours'}
                    ],
                    'Advanced': [
                        {'title': 'Advanced Digital Marketing Strategies', 'url': 'https://www.youtube.com/watch?v=0H9XwdVaH0w', 'duration': '5 hours'},
                        {'title': 'Marketing Analytics Course', 'url': 'https://www.youtube.com/watch?v=hTviHQzGH_Q', 'duration': '4 hours'},
                        {'title': 'Conversion Rate Optimization', 'url': 'https://www.youtube.com/watch?v=XUzRJUkX52g', 'duration': '2 hours'},
                        {'title': 'Advanced SEO Techniques', 'url': 'https://www.youtube.com/watch?v=GMrr6PxoY3c', 'duration': '3 hours'},
                        {'title': 'Programmatic Advertising', 'url': 'https://www.youtube.com/watch?v=iQu8SiKBmZU', 'duration': '2 hours'}
                    ]
                }
            },
            'business management': {
                'title': 'Business Management',
                'description': 'Learn effective business management principles',
                'icon': 'fas fa-briefcase',
                'type': 'non-technical',
                'levels': {
                    'Beginner': [
                        {'title': 'Business Management - Introduction', 'url': 'https://www.youtube.com/watch?v=T3l51Psce3c', 'duration': '3 hours'},
                        {'title': 'Business Administration 101', 'url': 'https://www.youtube.com/watch?v=BIjqEYvs_HA', 'duration': '5 hours'},
                        {'title': 'Introduction to Management', 'url': 'https://www.youtube.com/watch?v=AKcb_azPJws', 'duration': '2 hours'},
                        {'title': 'Business Strategy for Beginners', 'url': 'https://www.youtube.com/watch?v=EI1OrNUGh0Q', 'duration': '1 hour'},
                        {'title': 'Fundamentals of Business Management', 'url': 'https://www.youtube.com/watch?v=kBdlM6hNDAE', 'duration': '4 hours'}
                    ],
                    'Intermediate': [
                        {'title': 'Project Management Full Course', 'url': 'https://www.youtube.com/watch?v=TM_6lMv-xAU', 'duration': '6 hours'},
                        {'title': 'Financial Management for Business', 'url': 'https://www.youtube.com/watch?v=LCou9k4Zmxs', 'duration': '3 hours'},
                        {'title': 'Human Resource Management', 'url': 'https://www.youtube.com/watch?v=c8_avX9miag', 'duration': '4 hours'},
                        {'title': 'Marketing Management Course', 'url': 'https://www.youtube.com/watch?v=yKlLPICf_Eo', 'duration': '5 hours'},
                        {'title': 'Supply Chain Management', 'url': 'https://www.youtube.com/watch?v=QCf0z8lGoQA', 'duration': '2 hours'}
                    ],
                    'Advanced': [
                        {'title': 'Strategic Management Advanced Course', 'url': 'https://www.youtube.com/watch?v=TD7WSLeQtVw', 'duration': '4 hours'},
                        {'title': 'Leadership Skills Masterclass', 'url': 'https://www.youtube.com/watch?v=IVwQFGPGZss', 'duration': '3 hours'},
                        {'title': 'Business Analytics', 'url': 'https://www.youtube.com/watch?v=pQYBZJcgGu8', 'duration': '5 hours'},
                        {'title': 'Change Management', 'url': 'https://www.youtube.com/watch?v=__IlYNMdV9E', 'duration': '2 hours'},
                        {'title': 'Corporate Finance', 'url': 'https://www.youtube.com/watch?v=JecDGsm0iHI', 'duration': '6 hours'}
                    ]
                }
            },
            'graphic design': {
                'title': 'Graphic Design',
                'description': 'Create stunning visual content and master design tools',
                'icon': 'fas fa-palette',
                'type': 'non-technical',
                'levels': {
                    'Beginner': [
                        {'title': 'Graphic Design for Beginners', 'url': 'https://www.youtube.com/watch?v=WONZVnlam6U', 'duration': '5 hours'},
                        {'title': 'Adobe Photoshop for Beginners', 'url': 'https://www.youtube.com/watch?v=IyR_uYsRdPs', 'duration': '7 hours'},
                        {'title': 'Illustrator for Beginners', 'url': 'https://www.youtube.com/watch?v=Ib8UBwu3yGA', 'duration': '3 hours'},
                        {'title': 'Graphic Design Fundamentals', 'url': 'https://www.youtube.com/watch?v=YqQx75OPRa0', 'duration': '2 hours'},
                        {'title': 'Typography for Beginners', 'url': 'https://www.youtube.com/watch?v=QrNi9FmdlxY', 'duration': '1 hour'}
                    ],
                    'Intermediate': [
                        {'title': 'Intermediate Graphic Design Course', 'url': 'https://www.youtube.com/watch?v=sByzHoiYFX0', 'duration': '4 hours'},
                        {'title': 'Logo Design Masterclass', 'url': 'https://www.youtube.com/watch?v=NfkQeOSmIMY', 'duration': '6 hours'},
                        {'title': 'Adobe InDesign Course', 'url': 'https://www.youtube.com/watch?v=oFmfiPDruPE', 'duration': '2 hours'},
                        {'title': 'UI Design Course', 'url': 'https://www.youtube.com/watch?v=c9Wg6Cb_YlU', 'duration': '5 hours'},
                        {'title': 'Digital Illustration Techniques', 'url': 'https://www.youtube.com/watch?v=qidvQZgM4T4', 'duration': '3 hours'}
                    ],
                    'Advanced': [
                        {'title': 'Advanced Graphic Design Techniques', 'url': 'https://www.youtube.com/watch?v=sF_jSrBhdlg', 'duration': '6 hours'},
                        {'title': 'Motion Graphics Course', 'url': 'https://www.youtube.com/watch?v=Mtv8QptWNbg', 'duration': '8 hours'},
                        {'title': 'Adobe After Effects Masterclass', 'url': 'https://www.youtube.com/watch?v=3T0GJ-2cDFY', 'duration': '10 hours'},
                        {'title': '3D Design Fundamentals', 'url': 'https://www.youtube.com/watch?v=9xAumJRKV6A', 'duration': '5 hours'},
                        {'title': 'Branding & Identity Design', 'url': 'https://www.youtube.com/watch?v=a6bxj7qbrMs', 'duration': '4 hours'}
                    ]
                }
            },
            'content writing': {
                'title': 'Content Writing',
                'description': 'Develop professional content writing skills',
                'icon': 'fas fa-pen-fancy',
                'type': 'non-technical',
                'levels': {
                    'Beginner': [
                        {'title': 'Content Writing for Beginners', 'url': 'https://www.youtube.com/watch?v=VNr1Kb5KbLE', 'duration': '2 hours'},
                        {'title': 'Copywriting Course', 'url': 'https://www.youtube.com/watch?v=cJThMk0N51s', 'duration': '3 hours'},
                        {'title': 'Creative Writing Fundamentals', 'url': 'https://www.youtube.com/watch?v=CiVWAPMdgQY', 'duration': '1 hour'},
                        {'title': 'Business Writing Skills', 'url': 'https://www.youtube.com/watch?v=YOtzGCa6x7Y', 'duration': '2 hours'},
                        {'title': 'SEO Writing for Beginners', 'url': 'https://www.youtube.com/watch?v=NJ45MubGfvk', 'duration': '1.5 hours'}
                    ],
                    'Intermediate': [
                        {'title': 'Content Marketing Strategy', 'url': 'https://www.youtube.com/watch?v=iv-7uQ5FM84', 'duration': '4 hours'},
                        {'title': 'Storytelling for Content Creators', 'url': 'https://www.youtube.com/watch?v=KxXgsJ6ZFoY', 'duration': '2 hours'},
                        {'title': 'Technical Writing Course', 'url': 'https://www.youtube.com/watch?v=UHBH96wEjtY', 'duration': '3 hours'},
                        {'title': 'Blog Writing Masterclass', 'url': 'https://www.youtube.com/watch?v=UHBH96wEjtY', 'duration': '2.5 hours'},
                        {'title': 'Email Copywriting', 'url': 'https://www.youtube.com/watch?v=NJ45MubGfvk', 'duration': '1.5 hours'}
                    ],
                    'Advanced': [
                        {'title': 'Advanced Content Writing Techniques', 'url': 'https://www.youtube.com/watch?v=3deSrxPkq1E', 'duration': '3 hours'},
                        {'title': 'Conversion Copywriting', 'url': 'https://www.youtube.com/watch?v=UHBH96wEjtY', 'duration': '4 hours'},
                        {'title': 'UX Writing and Microcopy', 'url': 'https://www.youtube.com/watch?v=Bf-YoY_30Rc', 'duration': '2 hours'},
                        {'title': 'Content Strategy for Digital Marketing', 'url': 'https://www.youtube.com/watch?v=FQL3M1mUUXE', 'duration': '5 hours'},
                        {'title': 'Book Writing and Publishing', 'url': 'https://www.youtube.com/watch?v=68yES0UBIn4', 'duration': '6 hours'}
                    ]
                }
            }
        }
        
        # Search for matches in the course catalog
        for key, course in course_catalog.items():
            if search_query in key or search_query in course['title'].lower():
                search_results.append(course)
        
        return render_template('search_results.html',
                         username=session['username'],
                             search_query=search_query,
                         search_results=search_results)

# Add new route for chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        message = request.json.get('message')
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Get response from Gemini
        response = chat.send_message(message)
        return jsonify({
            'response': response.text,
            'success': True
        })
    except Exception as e:
        print(f"Chatbot error: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

# Start the application
if __name__ == '__main__':
    init_db()  # Initialize database with sample data
    app.run(debug=True)
