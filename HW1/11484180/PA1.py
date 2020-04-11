# Programmer : Zeid Al-Ameedi
# Dates 02-08-2020
# HW1 - 315 Finding frequent items, pairs and triples
# Program can be run from a shell script. Approximately 2 hours to complete.
# Collab : Piazza CptS315 Course &
# https://adataanalyst.com/machine-learning/apriori-algorithm-python-3-0/
# stackoverflow questions

import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations, chain



"""
Program is encapsulated into the apriori class. This will handle the phases of reading,
parsing and labeling the data. Reading through candidates and satisfying the phases
apriori run 1 usually takes a few seconds, apriori 2 will take 20min and apriori 3 will complete in 1hr 45 minutes.
The confidence is consequently calculated and then placed into an output file.

Part 2 : Identify item triples (X, Y, Z) such that the support of{X, Y, Z} 
is at least 100.  For all such triples, compute the confidence scores of the 
corresponding association rules:  (X, Y)⇒Z,  (X, Z)⇒Y,  (Y, Z)⇒X.  Sort the rules in decreasing order of confidence scores and 
list the top 5 rules in the writeup.  Order the left-hand-side pair lexicographically and breakties, 
if any, by lexicographical order of the first then the second item in the pair
"""
class Apriori():
    def __init__(self, filename, logfile, support_threshold):
        self.filename = filename
        self.logfile = logfile
        self.support_threshold = support_threshold

    def run(self):
        print("Apriori program begins...")
        data = self.get_data(self)
        d_baskets = self.parse_data(self, data)
        self.confidence_intervals(d_baskets)
        print("Program Complete. See {0} for all sorted output <confidence>".format(self.logfile))

    def confidence_intervals(self, d_baskets):
        sp, singles, triples = self.phases(d_baskets)
        pairs_confidences = self.pairs_confidence(singles, sp)
        triples_confidences = self.triples_confidence(singles, sp, triples)
        result_pairs = pairs_confidences[:5]
        result_triples = triples_confidences[:5]
        self.get_results(self, result_pairs, result_triples)
    
    def difference(self, item, subset):
        return tuple(x for x in item if x not in subset)

    def phases(self, d_baskets):
        ones = self.phase_1(d_baskets, self.support_threshold)
        print("Phase 1 complete.")
        singles = list(ones.keys())
        pairs = self.phase_2(singles, d_baskets, self.support_threshold)
        print("Phase 2 complete.")
        item_set = set(chain.from_iterable(pairs))
        triples = self.phase_3(item_set, d_baskets, self.support_threshold)
        print("Phase 3 complete.")
        return pairs, ones, triples

    @staticmethod
    def get_data(self):
        infile = open(self.filename, "r")
        data = infile.readlines()
        return data

    @staticmethod
    def parse_data(self, basket_data):
        temp_list = [[data_point for data_point in line.split()] for line in basket_data]
        return temp_list

    def phase_1(self, data_set, sup_thresh):
        print("Init Phase 1.\n")
        my_dict = {}
        self.P1_helper(data_set, my_dict)
        return self.apriori_result(my_dict, sup_thresh)

    def apriori_result(self, item_counts, sup_thresh):
        my_dict = {}
        for (k, v) in item_counts.items():
            if v >= sup_thresh:
                my_dict[k] = v
        return my_dict

    def P1_helper(self, data_set, item_counts):
        i=0
        for basket in data_set:
            for item in basket:
                #print(f'\rProgress: {i}/{len(data_set)}', end='\r')
                if item_counts.get(item):
                    item_counts[item] = item_counts[item] + 1
                else:
                    item_counts[item] = 1

    def phase_2(self, frequent_items, data_set, s):
        print("Init Phase 2.\n")
        item_counts = {}
        candidates = set(combinations(frequent_items, 2))
        self.AP2_helper(candidates, data_set, item_counts)
        return self.apriori_result(item_counts, s)

    def AP2_helper(self, candidates, data_set, item_counts):
        i = 0
        for line in data_set:
            for candidate in candidates:
                #print(f'\rProgress: {i}/{len(candidates)}', end='\r')
                if candidate[0] in line:
                    if candidate[1] in line:
                        if item_counts.get(candidate):
                            item_counts[candidate] = item_counts[candidate] + 1
                            #i += 1
                        else:
                            item_counts[candidate] = 1

    def phase_3(self, frequent_items, data_set, sup_threshold):
        print("Init Phase 3.\n")
        item_counts = {}
        line_count = 0
        triples = set(combinations(frequent_items, 3))
        self.AP3_helper(data_set, item_counts, triples)
        return self.apriori_result(item_counts, sup_threshold)

    def AP3_helper(self, data_set, item_counts, triples):
        i=0
        for line in data_set:
            for prod_set in triples:
                #print(f'\rProgress: {i}/{len(prod_set)}', end='\r')
                if prod_set[0] in line:
                    if prod_set[1] in line:
                        if prod_set[2] in line:
                            if item_counts.get(prod_set):
                                item_counts[prod_set] = item_counts[prod_set] + 1
                                #i += 1
                            else:
                                item_counts[prod_set] = 1

    @staticmethod
    def ret_key(x):
        try:
            return x[1]
        except...:
            print("Error when returning index.")

    def pairs_confidence(self, frequent_singles, frequent_pairs):
        confidences = {}
        for pair in frequent_pairs.keys():
            confidences[(pair[0], pair[1])] = frequent_pairs[pair] / frequent_singles[pair[0]]
            confidences[(pair[1], pair[0])] = frequent_pairs[pair] / frequent_singles[pair[1]]
        return self.sort_confidence(confidences)

    def sort_confidence(self, confidences):
        return sorted(confidences.items(), key=lambda x: x[1], reverse=True)
    
    def joinSet(self, itemSet, length):
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])

    def triples_confidence(self, frequent_singles, frequent_pairs, frequent_triples):
        confidences = {}
        for t in frequent_triples.keys():
            self.freq_triples_label(confidences, frequent_pairs, frequent_triples, t)
        return self.sort_confidence(confidences)

    def freq_triples_label(self, confidences, pairs, triples, t):
        bottom = pairs.get((t[0], t[1])) or pairs.get((t[1], t[0]))
        confidences[(t[0], t[1], t[2])] = triples[t] / bottom
        bottom = pairs.get((t[0], t[2])) or pairs.get((t[2], t[0]))
        confidences[(t[0], t[2], t[1])] = triples[t] / bottom
        bottom = pairs.get((t[1], t[2])) or pairs.get((t[2], t[1]))
        confidences[(t[1], t[2], t[0])] = triples[t] / bottom
    
    def getISet(self, data_iterator):
        transactionList = list()
        itemSet = set()
        for record in data_iterator:
            transaction = frozenset(record)
            transactionList.append(transaction)
            for item in transaction:
                itemSet.add(frozenset([item]))             
        return itemSet, transactionList

    @staticmethod
    def get_results(self, pairs_results, triples_results):
        infile = open(self.logfile, "w")
        infile.writelines("Output A\n")
        for ans in pairs_results:
            infile.writelines("{0} {1} {2}\n".format(ans[0][0], ans[0][1], ans[1]))
        infile.writelines("\nOutput B\n")
        for ans in triples_results:
            infile.writelines("{0} {1} {2}\n".format(ans[0][0], ans[0][1], ans[1]))


def main():
    ap = Apriori("browsingdata.txt", "output.txt", 100)
    ap.run()


if __name__ == '__main__':
    main()
