# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:01:20 2017

@author: ville

Main file for analyzing the network as a graph and make 
recommendations based on different types of connectivities and
similarities between users.

The 3 recommendation systems are:
    
    - Recommendation of users to follow next-nearest neighbours based completing open triangles (with added weighting based on usertype)
    - Recommendation of users to follow based on shared followed users
    - Recommendation of brands/trailblazers to follow based on what other users who follow the same users follow

"""

import networkx as nx
import pandas as pd
import numpy as np
import nxviz as nv
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict, Counter
import sys
import json
from time import time


def json_loader(filename):
    
    with open(filename,encoding="UTF-8") as json_file:
        data = json.load(json_file)['hits']['hits']

    users = {}
    users = {data[k]['_source']['id']: data[k]['_source']['userName'] for k in range(len(data))}

    connections = defaultdict(list)
    
    for k in range(len(data)):
        for follows in data[k]['_source']['follows']:

            connections[data[k]['_source']['userName']].append((data[k]['_source']['userName'],users[follows]))

    return users, connections

def get_recs_symmetry(n,G):
    diff = list(set(G.predecessors(n)).difference( G.successors(n)))
    print('\nUsers who follow {}, but not vice versa:\n {}'.format(n,diff))
    
    return diff


# Function to find recommendations for node n based on open triangles (next nearest neighbours k that n are not connected to)
# - Extra weight assigned if k also follows n
# - Returns an ordered array of top recommendations
def get_recs_triangles(n,G):
    recs = defaultdict(int)
    for m in G.neighbors(n):
            for k in G.neighbors(m):
                if k not in [n]+G.neighbors(n): 
                    #Recommend if n does not follow k, weighted by recommended usertype
                    if not G.has_edge(n, k): recs[k] += 1 #* W[G.node[k]['usertype']]  
                    #Add extra weight if k also follows n
                    if G.has_edge(k,n): recs[k] += 1        

    toprecs = sorted(recs,key=recs.get,reverse=True)[:5]
    print('\nTop 5 recommended users for {} based on open triangles:\n {} \n {}'.format(n,toprecs,sorted(recs.values(),reverse=True)[:5]))
    return toprecs, Counter(recs)

# Function to recommend other users ranked by the same users followed
# - Similar to open triangles but here the ranking is based on the number of shared followed users, not on the type of user
def get_recs_common_followed(user,G):
    overlap = defaultdict(list)
    nbrs = set(G.neighbors(user))
    rest = set(G.nodes()).difference(nbrs)
    rest.remove(user)
    
    t0=time()
    test = []
    for n in nbrs:
        for m in G.predecessors(n):
            if m not in G.neighbors(user)+[user]:
                test.append(m)
    
    print("done in %0.6fs" % (time() - t0))
    t1=time()
    
    for n in list(rest):
        common  = len(nbrs.intersection(G.neighbors(n)))
        if common > 0: overlap[common].append(n)
    print("done in %0.6fs" % (time() - t1))
    try:
        max_common = max(overlap.keys())
        toprecs = overlap[max_common]
    except:
        max_common = 0
        toprecs = []
        
    toprecs = sorted(Counter(test),key=Counter(test).get,reverse=True)[:5]
    print('\nTop 5 recommended users for {} based shared followed users:\n {} \n {}'.format(user,toprecs,sorted(Counter(test).values(),reverse=True)[:5]))  
    return toprecs, Counter(test)

       
#def get_recs_influencers_fansimilarity(user,G,influencers):
#    overlap = defaultdict(list)
#    
#    inf = set(influencers)
#    nbrs_inf = set(G.neighbors(user)).intersection(inf)
#    rest_fans = set(G.nodes()).difference(inf)
#    rest_fans.remove(user)
#    for n in list(rest_fans):
#        fan_nbrs_inf = set(G.neighbors(n)).intersection(inf)
#        common_inf = nbrs_inf.intersection(fan_nbrs_inf)
#        diff = list(fan_nbrs_inf.difference(common_inf))
#        
#        common  = len(common_inf)
#        if common > 0 and len(diff) > 0: overlap[common].append(diff)
#    
#    try:
#        max_common = max(overlap.keys())
#        toprecs = [item for sublist in overlap[max_common] for item in sublist]
#    except:
#        max_common=0
#        toprecs = []
#        
#    print('\nTop recommended trailblazers/brands for {} based on {} common interests:\n {}'.format(user,max_common,toprecs))
#    return toprecs


# *NOT USED* Function to find all open triangles in the graph and count the triangles with the same missing link to find all possible recommendations
# - Recommendations only for fans and trailblazers, not brands
# - Higher recommendation count if a link is part of more than one open triangle
#def find_open_triangles(G):
#    all_recommendations = defaultdict(int)
#    
#    for n, d in G.nodes(data=True):
#        for n1, n2 in combinations(G.neighbors(n), 2):
#          
#            # Check whether n1 and n2 do not have an edge
#            if not G.has_edge(n1, n2) and (G.node[n1]['usertype'] != 'brand'):
#                all_recommendations[(n1, n2)] += 1
#    
#    return all_recommendations




users, connections = json_loader('users.json')

nodes = list(users.values())

user = 'marcus'

#Create the user nodes
G = nx.DiGraph()
G.add_nodes_from(nodes)

for node in nodes:
    G.add_edges_from(connections[node])

#######################################
#Visualize and analyze the network
#######################################

#color_map = [v['color'] for u,v in G.nodes(data=True)]
#nx.draw(G,alpha=0.5,with_labels=True,node_size=100,font_size=18)
#nv.CircosPlot(G).draw()

#Network analysis : Identify key nodes in the network based on degree and betweenness centrality
dc = pd.DataFrame.from_dict(nx.degree_centrality(G),orient='index')
bc = pd.DataFrame.from_dict(nx.betweenness_centrality(G),orient='index')
dc_bc = pd.concat([dc,bc],axis=1)
dc_bc.columns = ['degree','betweenness']
dc_bc.sort_values(by='degree',inplace=True,ascending=False)

print('Degree and betweenness centralities of the network:\n')
print(dc_bc.head(10))

#Network analysis: Find maximal cliques (maximally connected subgraphs) of 4 or more nodes
#clique_size = 20
#cliques = [clique for clique in nx.find_cliques(G.to_undirected()) if len(clique) > clique_size]
#print('\nMaximal cliques with {} or more nodes:\n {}'.format(clique_size,cliques))

#######################################
# 3 different types of recommending fans and/or influencers to follow
#######################################

# Recommend users for user based on weighted proximity (open triangles)
recs_triangles,triangles_counts = get_recs_triangles(user,G)
## Recommend users for user based on similarity (highest number of common users followed)
recs_followed, followed_counts = get_recs_common_followed(user,G)

#recs_symmetry = get_recs_symmetry(user,G)

w=2
triangles_counts_reg = {k: v/w for k, v in triangles_counts.items()}
all_counts = Counter(triangles_counts_reg)+followed_counts
rec_aggr = sorted(all_counts,key=all_counts.get,reverse=True)[:5]
print('\nTop 5 recommended users for {} based on both open triangles and shared followed users:\n {} \n {}'.format(user,rec_aggr,sorted(all_counts.values(),reverse=True)[:5]))  
    


# Recommend trailblazers/brands based on common interest (highest number of common trailblazers/brands followed)
#recs_similarity_influencers = get_recs_influencers_fansimilarity(user,G,influencers)

