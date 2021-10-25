from collections import defaultdict
from os import unlink
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import knns
import pickle

class RatingsModel:
  def __init__(self):
    self.TRAIN_PATH = 'data/ratings-model-train-features.csv'
    self.TEST_PATH = 'data/ratings-model-test-features.csv'
    self.model = knns.KNNBasic()
    self.K = 10


  def load_ratings(self, dataset_label):
    reader = Reader(sep=',',rating_scale=(1,10))
    if dataset_label == 'TRAIN':
      try:          
        data = Dataset.load_from_file("data/ratings-model-features.csv", reader=reader)
        # self.train = Dataset.load_from_file(self.TRAIN_PATH, reader=reader))
        self.data = data
        self.trainset = data.build_full_trainset()
        self.subjects = list(set([x[1] for x in self.trainset.build_testset()]))
        self.users = list(set([x[0] for x in self.trainset.build_testset()]))
      except:
        print('failed to load train data')
    elif dataset_label == 'LIVE':
      # try:          
      data = Dataset.load_from_file("data/ratings-model-features.csv", reader=reader)
      # self.subjects = list(set([x[1] for x in data.build_full_trainset()()]))
      # self.users = list(set([x[0] for x in self.data.build_full_trainset()]))
      # except:
      #   print("failed to load test data")      

  def predict_ratings_for_all_users(self, user_id):
    self.load_ratings('LIVE')
    # if user_id not in self.users:
    #   return ["poo"]

    predictions = [self.model.predict(user_id, subject_id, 1) for subject_id in self.subjects]
    top_k_predictions = self.get_top_predictions(predictions, self.K)
    # return top_k_predictions[user_id]
    return [s[0] for s in top_k_predictions[user_id]]

  def get_top_predictions(self, predictions, K=10):
    user_mapped_preds = defaultdict(list)
    for user_id, subject_id, true_r, est, _ in predictions:
      user_mapped_preds[user_id].append((subject_id, est))

    for user_id, rating in user_mapped_preds.items():
      rating.sort(key=lambda x: x[1], reverse=True)

    return user_mapped_preds

  def load_model(self):
    with open('data/model.pkl', 'r') as f:
      self.model = pickle.load(f)

  def train_model(self):
    self.load_ratings('TRAIN')
    self.model.fit(self.trainset)