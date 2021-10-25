from os import linesep
import sys
from flask import Flask, jsonify
from models import collaborative, content_based 



myapp = Flask(__name__)

recommender = content_based.Recommender()

ratings_based_recommender = collaborative.RatingsModel()
ratings_based_recommender.train_model()

@myapp.route('/recommend/<user_id>')
def get_recommendations(user_id):
    # So dumb and ineffficent like everything else in this codebase
    build_user_ratings_file()      
    
    # Subject recommendations by comparing user with similar users
    recommendations = ratings_based_recommender.predict_ratings_for_all_users(user_id)  

    # Subject recommendations based on interest
    recommendations.extend(recommender.get_student_recommendation(user_id))
    print(recommendations)
    # Add popular subjects if required
    if len(recommendations) < 20:
        recommendations.extend(ratings_based_recommender.get_popular_subjects())

    return jsonify(recommendations)
    

def build_user_ratings_file():
    from utilities import database as db
    userratings = db.get_completed_subjects()
    lines = []
    for r in userratings:
        lines.append(f"{r['UserId']},{r['SubjectId']},{r['Score']}\n")
    with open('data/ratings-model-train-features.csv', 'w') as f:
        f.writelines(lines)
    
run_locally = len(sys.argv) > 1 and sys.argv[1] == "local"
if run_locally:
    myapp.run(
        port=8000,
        debug=True
    )