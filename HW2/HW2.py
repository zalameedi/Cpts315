"""
Programmer : Zeid Al-Ameedi
Date : 03/02/2020
Collab : Piazza forums, stackoverflow, Dr. Jana Doppa, David Henshaw
Details : We will be using the Item-Item collaborative filtering
algorithm (reccomendation system). Given a dataset of movies compiled of different movies and user ratings,
using similarity score, neighborhood set and ultimately present
top 5 movies in a lexicographic ordering for movies.
I.E. 
User-id1 movie-id1 movie-id2 movie-id3 movie-id4 movie-id5
"""

import operator
import numpy as np
import pandas as pd
import csv
from itertools import islice


class IICFilter():
    def __init__(self, infile, outfile="output.txt", n=5, user_top=5):
        self.file = infile
        self.outfile = outfile
        self.neighborhood = n
        self.user_top = user_top

    def run(self):
        print("running . . .\n")
        movie_ratings = self.read_movies
        max_m_id, max_u_id, m_ids, user_ids = self.unique_columns(movie_ratings)
        n_uniques, mt = self.eval_ratings(max_m_id, max_u_id, m_ids, movie_ratings, user_ids)
        scores = self.comp_similarity(n_uniques, m_ids, mt)
        neighbors = self.neigh_dict(m_ids, scores)
        self.cut_neighborhood(m_ids, neighbors)
        matrix = self.build_matrix(max_m_id, max_u_id, movie_ratings)
        computed = self.neighborhood_ratings(matrix, m_ids, neighbors, user_ids)
        self.isolate_recc(computed, user_ids)

    def eval_ratings(self, max_m_id, max_u_id, m_ids, movie_ratings, user_ids):
        n_uniques = np.zeros((max_m_id + 1, max_u_id + 1))
        for rating in movie_ratings:
            n_uniques[rating[1]][rating[0]] = rating[2]
        self.normalize_matrix(n_uniques, m_ids, user_ids)
        init_matrix = dict()
        for _ids in m_ids:
            init_matrix[_ids] = np.linalg.norm(n_uniques[_ids][:])
        return n_uniques, init_matrix

    @property
    def read_movies(self):
        movie_ratings = list()
        infile = open(self.file, 'r')
        lines = infile.read().splitlines()
        lines.pop(0)
        for line in lines:
            line = line.split(",")
            col_userID = int(line[0])
            col_movieID = int(line[1])
            col_movieRT = float(line[2])
            movie_ratings.append((col_userID, col_movieID, col_movieRT))
        infile.close()
        return movie_ratings

    def unique_columns(self, movie_ratings):
        user_ids = set()
        m_ids = set()
        for rating in movie_ratings:
            user_ids.add(rating[0])
            m_ids.add(rating[1])
        user_ids = sorted(user_ids)
        m_ids = sorted(m_ids)
        max_u_id = max(user_ids)
        max_m_id = max(m_ids)
        return max_m_id, max_u_id, m_ids, user_ids

    def normalize_matrix(self, dup, m_id, userids):
        for mid in m_id:
            total = 0
            for user_id in userids:
                if dup[mid][user_id] != 0:
                    total += 1
            mean = np.sum(dup[mid][:]) / total
            self.reduce_cp(mean, dup, mid, userids)

    def reduce_cp(self, avg_rating, copy, movieid, userids):
        for userid in userids:
            if copy[movieid][userid] != 0:
                copy[movieid][userid] = copy[movieid][userid] - avg_rating

    def comp_similarity(self, dup, m_id, mt):
        final_val = dict()
        for index1 in m_id:
            for index2 in m_id:
                if index2 > index1:
                    ans = np.sum(dup[index1][:] * dup[index2][:])
                    denom = mt[index1] * mt[index2]
                    if denom != 0:
                        final_val[(index1, index2)] = ans / denom
                    else:
                        final_val[(index1, index2)] = -1
        return final_val

    def neigh_dict(self, m_ids, scores):
        neighbors = dict()
        for index1 in m_ids:
            temp = list()
            for index2 in m_ids:
                if index1 < index2:
                    temp.append((index2, scores[(index1, index2)]))
                elif index1 > index2:
                    temp.append((index2, scores[(index2, index1)]))
            neighbors[index1] = sorted(temp, key=operator.itemgetter(1), reverse=True)
        return neighbors

    def cut_neighborhood(self, m_ids, neighbors):
        for mid in m_ids:
            movie_t = list()
            arg_t = list()
            temp = neighbors[mid]
            if len(temp) > self.neighborhood:
                movie_t = self.handle_t(temp, arg_t, movie_t)
            else:
                movie_t = temp
            neighbors[mid] = movie_t

    def handle_t(self, records, arg_t, t_movies):
        s_t = records[self.neighborhood - 1][1]
        for x in records:
            if x[1] > s_t:
                t_movies.append(x)
            elif x[1] == s_t:
                arg_t.append(x)
            else:
                break
        arg_t.sort(key=operator.itemgetter(0))
        top_u = t_movies + arg_t[0:(self.neighborhood - len(t_movies))]
        return top_u

    def build_matrix(self, m_mid, m_uid, movie_ratings):
        matrix = np.zeros((m_mid + 1, m_uid + 1))
        for r in movie_ratings:
            matrix[r[1]][r[0]] = r[2]
        return matrix

    def neighborhood_ratings(self, matrix, _mid, n, uid):
        _dict = dict()
        for _uid in uid:
            self.neighborhood_helper(_dict, matrix, _mid, n, _uid)
        return _dict

    def neighborhood_helper(self, _dict, matrix, mid, neighbors, uid):
        temp = list()
        for _mid in mid:
            if matrix[_mid][uid] == 0:
                numerator = 0
                denominator = 0
                for n in neighbors[_mid]:
                    if matrix[n[0]][uid] != 0:
                        numerator += n[1] * matrix[n[0]][uid]
                        denominator += n[1]
                if denominator > 0:
                    temp.append((_mid, numerator / denominator))
        _dict[uid] = sorted(temp, key=operator.itemgetter(1), reverse=True)

    def isolate_recc(self, _dict, uid):
        final_dict = dict()
        for _uid in uid:
            u_t = list()
            arg_t = list()
            temp = _dict[_uid]
            if len(temp) > self.user_top:
                threshold = temp[self.user_top - 1][1]
                self.break_t(temp, threshold, arg_t, u_t)
                arg_t.sort(key=operator.itemgetter(0))
                u_t = u_t + arg_t[0:(self.user_top - len(u_t))]
            else:
                threshold = temp[len(temp) - 1][1]
                self.tied_ratings(temp, threshold, arg_t, u_t)
                arg_t.sort(key=operator.itemgetter(0))
                u_t = u_t + arg_t
            final_dict[_uid] = u_t
        self.write_results(self.outfile, final_dict)

    def tied_ratings(self, temp, cap, arg_t, u_t):
        for r in temp:
            if r[1] > cap:
                u_t.append(r)
            else:
                arg_t.append(r)

    def break_t(self, temp, cap, arg_t, u_t):
        for rating in temp:
            if rating[1] > cap:
                u_t.append(rating)
            elif rating[1] == cap:
                arg_t.append(rating)
            else:
                break

    def write_results(self, outfile, records):
        file = open(self.outfile, 'w')
        for key, data in sorted(records.items()):
            file.write(str(key))
            for index in data:
                file.write(' ' + "{0}".format(str(index[0])))
            file.write('\n')
        file.close()


def main():
    print("Item-Item Collab Filtering algorithm beginning...")
    iicf = IICFilter("movie-lens-data\\ratings.csv")
    print("Analyzing movies...")
    iicf.run()
    print("\nDone. Please see {0} for result!".format(iicf.outfile))


if __name__ == '__main__':
    main()
