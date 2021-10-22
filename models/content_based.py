from utilities import database as db
import json
import pandas as pd


class Student:
    def __init__(self, prefs, completed_subjects, topics):
        self.id = prefs['studentId']
        self.preferences = {pref: value for pref,value in prefs.items() if pref != "interests"}
        self.interests = prefs['interests']
        self.interest_vector = [int(topic in self.interests) for topic in topics]
        self.degree = prefs.get('degree', 'none') 
        self.major = prefs.get('major', 'none')
        self.completed_subjects = completed_subjects

class Subject:
    def __init__(self, subject, tfidfs):
        self.code = subject['code']
        self.name = subject['name']
        self.topic_vector = list(tfidfs.loc[[subject['code']]].values.flatten().tolist())

class kNNStudentSubject:
    def __init__(self, students, subjects, topics, tfidfs, k=5):
        self.students = students
        self.subjects = subjects
        self.k = k
    
    def get_neighbours(self, student_id):
        student = self.students[student_id]
        interest_scores = interest_similarities(student.interest_vector, self.subjects)
        nearest_k = sorted(interest_scores, key=lambda x: x[1], reverse=True)[:self.k]
        return nearest_k

    def get_student(self, student_id):
        return  self.students[student_id]

def cosine_similarity(a, b):
    def dot_product(a, b):
        return sum([a[i]*b[i] for i in range(len(a))])

    def magnitude(v):
        return sum([x**2 for x in v])**0.5
    
    magnitude_product = magnitude(a) * magnitude(b)

    if magnitude_product:
        return dot_product(a, b) / magnitude_product 
    else: 
        return 0
    

def interest_similarities(interests, subjects):
    return [(subject.code, cosine_similarity(interests, subject.topic_vector)) for subject in subjects]

class Recommender:
    def __init__(self):
        self.tfidfs = pd.read_pickle('data/selected_topic_keyword_scores.pkl')
        self.tfidfs.index = self.tfidfs.index.map(str)
        self.subjects =  [Subject(subject, self.tfidfs) for subject in db.get_subjects()]
        self.topics = [topic['name'] for topic in db.get_topics()]
        self.completed_subjects = db.get_completed_subjects()
        self.user_preferences = {pref['studentId']: pref for pref in db.get_user_preferences()}
        self.student_ids = [s['id'] for s in db.get_students()]
        self.students = {id: Student(self.user_preferences.get(id,{}), [cs for cs in self.completed_subjects if cs['UserId']==id], self.topics) for id in self.student_ids}
            
    def get_student_recommendation(self, student_id):
        recommendations = []
        # with open('data/selected_topic_keywords.csv', 'r') as f:
            # topics = [line.strip('\n').strip() for line in f.readlines()]

            # with open('data/topic_tfidfs.json', 'r') as f:
            #     compact_tfidf = json.load(f)
            

            # subject_topic_tfidfs = pd.read_pickle('data/selected_topic_keyword_scores.pkl')
            # subject_topic_tfidfs.index = subject_topic_tfidfs.index.map(str)

            # degrees = db.get_degrees()

        kNN_student_subject = kNNStudentSubject(self.students, self.subjects, self.topics, self.tfidfs)
        neighbours = kNN_student_subject.get_neighbours(student_id)

        result = [code for (code,score) in neighbours]
        print(result)
        return result[:min(5, len(neighbours))]  