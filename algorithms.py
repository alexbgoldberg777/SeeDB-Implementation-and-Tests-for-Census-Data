from ucimlrepo import fetch_ucirepo 
from collections import defaultdict
import numpy as np
from scipy import special as kl

k = 5
n = 10
delta = 0.5

# fetch dataset 
census_income = fetch_ucirepo(id=20)

# Full dataset
data = census_income.data.features
data = data.drop(columns=['relationship'])

# Define the sets A and M based on whether they are numerical or categorical features
A = ['workclass', 'education', 'occupation', 'race', 'sex', 'native-country']
M = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Define the set of aggregate functions
F = ['sum', 'mean', 'max', 'min', 'count']

# Get all possible combinations of A, M, and F
possibilities = set()
for a in A: 
    for m in M: 
        for f in F: 
            possibilities.add((a, m, f))

# helper functions
def normalize(agg1, agg2):
    sum1 = sum(agg1)
    sum2 = sum(agg2)
    norm1 = [x / sum1 for x in agg1]
    norm2 = [x / sum2 for x in agg2]
    return norm1, norm2

def kl_div(list2, list1):
    normalized_list_1,normalized_list_2 = normalize(list1, list2)
    normalized_list_1 = [float(x) for x in normalized_list_1]
    normalized_list_2 = [float(x) for x in normalized_list_2]
    return kl.kl_div(normalized_list_1, normalized_list_2)[0]

def aggregate_values(res_1, res_2):

    joined_res = defaultdict(list)
    adjustment = 0.00000001 # arbitrarily small number so we dont divide by 0

    for item in res_1.items():
        if item[1] == 0:
            joined_res[item[0]].append(adjustment)
        else:
            joined_res[item[0]].append(item[1])

    for item in res_2.items():
        if item[1] == 0 or item[0] not in joined_res:
            joined_res[item[0]].append(adjustment)
        else:
            joined_res[item[0]].append(item[1])

    for key in joined_res.keys():
        if len(joined_res[key]) < 2:
            joined_res[key].append(adjustment)

    return [val[0] for val in joined_res.values()], [val[1] for val in joined_res.values()]

def queries(views):
    a_list = set(map(lambda x: x[0], views))
    
    queries = []

    for a_main in a_list:
        # get subset of views
        sub_views = set(filter(lambda x: a_main == x[0], views))
        l = []
        for a,m,f in sub_views:
            l.append((m, f))
        queries.append([a, l])
    
    return queries

def hoeffding_serfling(m):
    return np.sqrt(( (1-((m-1)/n)) * (2* np.log(np.log(m)) + np.log(np.pi**2/(3*delta)))/ 2*m ))

def pruneViews(views, m, utility_sums):
    interval = hoeffding_serfling(m)
    top_views = sorted(utility_sums.items(), key=lambda item: (item[1]/m)+interval, reverse=True)[:k]
    lowest_lower_bound = min(top_views, key=lambda item: item[1])[1] - interval
    for view in views.copy():
        if view not in top_views:
            if (utility_sums[view]/m) + interval < lowest_lower_bound:
                views.remove(view)
    return views

kl_sums = {poss: 0 for poss in possibilities}

# Phase based implmentation, main program
def phase(n, views):
    size = len(data) // n
    for i in range(n):
        sub_data = data[i: i+size+1]
        # Query target
        married_adults = sub_data[sub_data['marital-status'].isin(['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'])]

        # Query Reference
        unmarried_adults = sub_data[~sub_data['marital-status'].isin(['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'])]

        qs = queries(views)
        for q in qs:
            a = q[0]
            shared = q[1]

            l1 = married_adults.groupby(a)
            l2 = unmarried_adults.groupby(a)

            for m,f in shared:
                res_1 = l1[m].agg(f)
                res_2 = l2[m].agg(f)
                agg1, agg2 = aggregate_values(res_1,res_2)
                kl_sums[(a,m,f)] += kl_div(agg1, agg2)

        views = pruneViews(views, i+1, kl_sums)
    return list(map(lambda x: x[0], sorted(kl_sums.items(), key=lambda item: item[1], reverse=True)[:k]))

top_views = phase(n, possibilities)
print(top_views)