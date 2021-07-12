import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
np.seterr('raise')

class MovieRecommender():

    def __init__(self, ratings_path, targets_path, mode='SUBMISSION'):
        self.mode = mode
        self.ratings_df = pd.read_csv(ratings_path, sep=',')
        self.targets_df = pd.read_csv(targets_path, sep=',')
        self.n_rows = self.ratings_df.shape[0]
        # uses distinc k values and saves the results in diferents csv files
        if self.mode == 'EVALUATION': 
            self.k_values = [160, 170, 180, 190, 200]
        else:
            # uses the otimum value for k
            self.k_values = [180]

        self.structuring()
        self.normalize()
        self.create_dicts()

    def structuring(self, ):
        self.targets_df.set_index('UserId:ItemId', inplace=True)
        self.ratings_df.rename(columns={'Prediction': 'prediction',
                                        'UserId:ItemId': 'user:item', 'Timestamp': 'timestamp'},
                                inplace=True)

    # Normalizing predictions using mean-centering approach
    def normalize(self):
        self.ratings_df[['user', 'item']] = self.ratings_df['user:item'].str.split(':', expand=True)
        # calculating the users and items mean value
        self.user_means = self.ratings_df.groupby('user').sum()['prediction']/self.ratings_df.groupby('user').count()['prediction']
        self.item_means = self.ratings_df.groupby('item').sum()['prediction']/self.ratings_df.groupby('item').count()['prediction']

        # updating prediction values according users mean values
        self.ratings_df = self.ratings_df.merge(self.user_means, left_on=['user'], right_on=['user'], suffixes=['_r', '_n'])
        self.ratings_df['prediction_n'] = self.ratings_df['prediction_r'] - self.ratings_df['prediction_n']
        self.ratings_df.set_index('user:item', inplace=True)
        self.global_mean = self.ratings_df['prediction_r'].mean()

    # Creating adjacency lists
    def create_dicts(self):
        # dict with all users ratings
        self.user_ratings = defaultdict(lambda: defaultdict(int))
        for data, row in self.ratings_df.iterrows():
            user, item = data.split(':')
            self.user_ratings[user].update({item:row['prediction_n']})

        # dict with all items ratings
        self.item_ratings = defaultdict(lambda: defaultdict(int))
        for data, row in self.ratings_df.iterrows():
            user, item = data.split(':')
            self.item_ratings[item].update({user:row['prediction_n']})


    def cosine_similarity(self, u, v):
        try:
            cos_sim = np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v))
            return cos_sim
        except Exception: # when the vectors are empty or all zeros
            return 0

    def select_neighboors(self, user_id, item_id):
        target = self.item_ratings[item_id]
        neighbors = {}
        candidates = self.user_ratings[user_id].keys()
        for c in candidates:
            predictions = self.item_ratings[c]
            # items rated by both (target and candidate)
            common_items = list(set(target.keys()) & set(predictions.keys()))
            t = []
            p = []
            for item in common_items:
                t.append(target[item])
                p.append(predictions[item])
            # similarity between target item and candidate item scores
            sim = self.cosine_similarity(t, p)
            neighbors[c] = sim
        neighbors = {id: sim for id, sim in sorted(neighbors.items(), key=lambda item: item[1], reverse=True)[1:self.k]}
        return neighbors

    def score_prediction(self, neighbors, user_id, item_id):
        user_predictions = {k: self.user_ratings[user_id][k] for k in neighbors.keys()}
        sum_1 = 0
        sum_2 = 0
        # if the item doesn't have enough neighbors
        if len(neighbors) < 10:
            return self.global_mean
        for id, sim in neighbors.items():
            sum_1 += sim * user_predictions[id]
            sum_2 += np.abs(sim)
        # if have only zero values
        if sum_1 == 0 or sum_2 == 0:
            # if it's not a case of user cold-start
            if self.user_means[user_id] != 0:
                return self.user_means[user_id]
            # if it's not a case of item cold-start
            if self.item_means[item_id] != 0:
                return self.item_means[item_id]
            # in the last case, the global mean is used
            return self.global_mean
        return sum_1/sum_2 + self.user_means[user_id]
    
    def rmse(self, predictions, targets):
        return np.sqrt(np.mean((predictions-targets)**2))

    def run(self, k, startTime):
        self.k = k
        for index, _ in self.targets_df.iterrows():
            user, item = index.split(':')
            try:
                # when it`s a unknow item
                if len(self.item_ratings[item]) == 0:
                    self.targets_df.at[index, 'Prediction'] = self.user_means[user]
                # when it`s a unknow user
                elif len(self.user_ratings[user]) == 0:
                    self.targets_df.at[index, 'Prediction'] = self.item_means[item]
                else:
                    neighbors_t = self.select_neighboors(user, item)
                    result = self.score_prediction(neighbors_t, user, item)
                    self.targets_df.at[index, 'Prediction'] = result
            # when both are unknown
            except KeyError as e:
                try:
                    if 'i' in str(e):
                        self.targets_df.at[index, 'Prediction'] = self.user_means[user]
                    else:
                        self.targets_df.at[index, 'Prediction'] = self.item_means[item]
                except KeyError as e:
                    self.targets_df.at[index, 'Prediction'] = self.global_mean

        if self.mode == 'EVALUATION':
            self.targets_df.to_csv(f'results_{self.k}.csv')
            # rmse = self.rmse(self.ratings_df['prediction_r'].values, self.targets_df['Prediction'].values)
            print(f'K: {self.k} | Execution time: {datetime.now() - startTime}')
        else:
            self.targets_df.to_csv(f'results_{self.k}.csv')

if __name__ == '__main__':
    ratings_path = sys.argv[1]
    targets_path = sys.argv[2]
    recommender = MovieRecommender(ratings_path, targets_path)
    
    for k in recommender.k_values:
        startTime = datetime.now()
        recommender.run(k, startTime)