from flask import Flask, jsonify
# from models.recommend import Recommendations
from models import content_based #, rating



myapp = Flask(__name__)

recommender = content_based.Recommender()

@myapp.route('/recommend/<user_id>')
def get_recommendations(user_id):
    recommendations = recommender.get_student_recommendation(user_id)
    # Subject recommendations by comparing user with similar users
    # recommendations = kNN.collaborativeFilter.predictUser(user_id)
    # if not recommendations:
    #     # If no similar users, get popular subjects
    #     recommendations = kNN.contentSimilarity.get_similar_subjects(-1)
    # elif len(recommendations) < 50:
    #     # If there is usable user data, but not enough recommendations, get similar popular subjects
    #     recommendations.extend(kNN.contenSUmularity.get_similar_popular_subjects(user_id))

    # recommendations = 
    return jsonify(recommendations)
    
myapp.run(
    port=8000,
    debug=True
)