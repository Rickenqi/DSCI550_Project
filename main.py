import csv
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

import helper


def user_clustering_setup():
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    tags = pd.read_csv('data/tags.csv')
    genres = movies['genres']
    userIds = ratings['userId']
    userId_set = set()
    genres_set = set()
    for genre in genres:
        str_list = genre.split('|')
        for v in str_list:
            genres_set.add(v)
    for userId in userIds:
        userId_set.add(userId)
    kmeansObject = UserKmeansClustering(movies, ratings, list(genres_set), list(userId_set))
    return kmeansObject


def dataset_integrate():
    url = 'https://api.themoviedb.org/3/movie/'
    api_key = '?api_key=89e9f723793b7c74281b8a270fd9a238&language=en-US'
    user_group = pd.read_csv('data/user_group.csv')
    links = pd.read_csv('data/links.csv')
    movieIds = links['movieId']
    tmdbIds = links['tmdbId']
    movie_info = []
    i = 0
    print(links.shape[0])
    for i in range(links.shape[0]):
        id = tmdbIds[i]
        movieId = movieIds[i]
        response_crew = requests.get(url + str(id) + "/credits" + api_key)
        response_info = requests.get(url + str(id) + api_key)
        crew_json_dict = json.loads(response_crew.text)
        info_json_dict = json.loads(response_info.text)
        crews = []
        budget = 0
        production_company = 0
        director = 0
        try:
            crews = crew_json_dict["crew"]
        except KeyError:
            pass
        try:
            budget = info_json_dict["budget"]
        except KeyError:
            pass
        try:
            production_company = info_json_dict["production_companies"][0]["id"]
        except (IndexError, KeyError):
            pass
        for crew in crews:
            try:
                if crew['job'] == 'Director':
                    director = crew['id']
            except KeyError:
                pass
        print({"Id": movieId, "production_company": production_company,
               "director": director, "budget": budget})
        movie_info.append({"Id": movieId, "production_company": production_company,
                           "director": director, "budget": budget})
    keys = movie_info[0].keys()
    with open('data/movies_info.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(movie_info)

def user_regression_setup():
    group = pd.read_csv('data/user_group.csv')
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies_info.csv')
    # pd.set_option('display.max_columns', None)
    df = pd.merge(movies, pd.merge(ratings, group, on='userId'), on='movieId')
    df_clear = df.drop(df[(df['production_company'] == 0) | (df['director'] == 0) | (df['budget'] == 0)].index)
    print(ratings.shape, df.shape, df_clear.shape)
    return UserRandomForestRegression(df_clear)

class UserRandomForestRegression:
    def __init__(self, data):
        self.data = data
        x_v1 = ['rating', 'production_company', 'director', 'budget', 'cluster']
        x_v2 = x_v1.append(['movieId', 'userId'])
        self.x1 = pd.DataFrame(self.data, columns=['rating', 'production_company', 'director', 'budget', 'cluster'])
        self.x2 = pd.DataFrame(self.data, columns=['rating', 'production_company', 'director',
                                                   'budget', 'cluster', 'movieId', 'userId'])

    def regression_bygroup(self, input_data):
        print(input_data.columns)
        dataset = input_data.groupby('cluster')
        for i in dataset.groups:
            data = dataset.get_group(i)
            train = data.sample(frac=0.666)
            test = data.drop(train.index)
            train_y = train['rating']
            train_x = train.drop(columns=['rating', 'cluster'])
            test_y = test['rating']
            test_x = test.drop(columns=['rating', 'cluster'])
            self.regression_result(train_x, train_y, test_x, test_y, "group regression")

    def ordinary_regression(self, input_data):
        print(input_data.columns)
        train = input_data.sample(frac=0.666)
        test = input_data.drop(train.index)
        train_y = train['rating']
        train_x = train.drop(columns=['rating'])
        test_y = test['rating']
        test_x = test.drop(columns=['rating'])
        self.regression_result(train_x, train_y, test_x, test_y, "ordinary regression")

    def regression_result(self, train_x, train_y, test_x, test_y, addition_info=""):
        print(addition_info)
        regressor = RandomForestRegressor(n_estimators=200, random_state=0)
        regressor.fit(train_x, train_y)
        pred_y = regressor.predict(test_x)
        print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pred_y))
        print('Mean Squared Error:', metrics.mean_squared_error(test_y, pred_y))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pred_y)))
        importance = list(regressor.feature_importances_)
        print(importance)

    def all_regression_model_result(self):
        self.regression_bygroup(self.x1)
        self.regression_bygroup(self.x2)
        self.ordinary_regression(self.x1)
        self.ordinary_regression(self.x2)


class UserKmeansClustering:
    def __init__(self, movies, ratings, genres, userIds):
        self.X = None
        self.genre_ratings = None
        self.movies = movies
        self.ratings = ratings
        self.genres = genres
        self.userIds = userIds

    def kmeans_cluster(self, clusters_num=7):
        kmeans = KMeans(n_clusters=clusters_num)
        print(self.X)
        prediction = kmeans.fit(self.X)
        cluster_label = kmeans.labels_.tolist()
        # cluster_label = np.expand_dims(cluster_label, -1)
        # self.X = np.concatenate((self.X, cluster_label), axis=1)
        result = self.genre_ratings
        result["cluster"] = cluster_label
        result["userId"] = self.userIds
        return self.genre_ratings

    def training_data_prepare(self, genres_input=None):
        if genres_input is None:
            genres_input = ['Crime', 'Animation']
        rating_titles = []
        for genre in genres_input:
            rating_titles.append('avg_' + genre)
        genre_ratings = helper.get_genre_ratings(self.ratings, self.movies, genres_input, rating_titles)
        genre_ratings = helper.get_most_rated_movies(genre_ratings, 1000)
        genre_ratings = genre_ratings.fillna(value=0)
        X = genre_ratings[rating_titles].values
        self.genre_ratings = genre_ratings
        self.X = X


if __name__ == "__main__":
    test = user_clustering_setup()
    genres_test = ['Crime', 'Animation', 'Film-Noir', 'Fantasy', 'Sci-Fi', 'Adventure', 'Drama', 'War', 'Thriller',
                   'Documentary', 'Action', 'Comedy', 'Mystery', 'Musical', 'Romance', 'Horror']
    test.training_data_prepare(genres_test)
    next_data_set = test.kmeans_cluster(9)
    next_data_set.to_csv("data/user_group.csv", encoding='utf-8')
    print("clustering complete...")
    # dataset_integrate()
    test_regression = user_regression_setup()
    test_regression.all_regression_model_result()

