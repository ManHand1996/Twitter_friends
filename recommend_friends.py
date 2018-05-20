import time
import sys
import json
import os
import networkx
import numpy as np
from sklearn.metrics import silhouette_score
from twitters import get_tweepyAPI
from sklearn.externals import joblib
from classifier import NLTKBOW
from collections import defaultdict
from operator import itemgetter


def get_friends(t,user_id):
    """
        twitter API ratelimit: 15 mins/15 times
        and request again unitil wait 5 mins
        it's spend too much time...
    """
    friends_ = []
    cursor = -1
    results = None
    while cursor != 0:
        try:
            results = t.friends_ids(user_id=user_id,cursor=cursor)
            friends_.extend([ friend for friend in results[0] ])
            cursor = results[1][1]
            if len(friends_) >= 10000:
                break
            
            print("collect 5000 friends with {0} or less 5000".format(user_id))
            sys.stdout.flush()
        except tweepy.RateLimitError as e:
            if results is None:
                print("You probably reached your API limit, waiting for 5 minutes")
                sys.stdout.flush()
                time.sleep(5*60)
            else:
                raise e
        except tweepy.TweepError as e:
            break
        finally:
            time.sleep(60)
    return friends_




# calculate number of friends occurrences
def count_friends(friends):
    friend_count = defaultdict(int)
    for friend_list in friends.values():
        for friend in friend_list:
            friend_count[friend] += 1
    return friend_count


def get_friends_list(theme,proxy,model_filename):
    t = get_tweepyAPI(proxy)
    

    original_users = [] # screen_name
    tweets_text = [] 
    user_ids = {} # map screen_name userid
    
    search_results = t.search(q=theme,count=100)
    for tweet in search_results:
        if 'text' in tweet._json:
            original_users.append(tweet._json['user']['screen_name'])
            tweets_text.append(tweet._json['text'])
            user_ids[tweet._json['user']['screen_name']] = tweet._json['user']['id_str']

    model = joblib.load(model_filename)
    y_labels = model.predict(tweets_text)


    relevant_tweets_text = [ tweets_text[i] for i in range(len(tweets_text)) if y_labels[i] == 1 ]
    relevant_users = [ original_users[i] for i in range(len(tweets_text)) if y_labels[i] == 1 ]
    
    # friends = {'user_name':[followning list]}
    friends = {}
    for screen_name in set(relevant_users):
        uid = int(user_ids[screen_name])
        friends[uid] = get_friends(t,uid)
    
    return firends



def save_friends(friends,friends_filename):
    old_friends = {}
    if os.path.exists(friends_filename):
        with open(friends_file,'r') as out:
            old_friends = json.load(out)
    with open(friends_filename,'w') as out:
        old_friends.update(friends)
        json.dump(old_friends,out)

def load_friends(friends_filename):
    if os.path.exists(friends_filename):
        with open(friends_filename,'r') as out:
            friends = json.load(out)
        return friends
    return {}


# it spend too much time,adjust len(friends) == maxnums
# default : 150
def get_more_friends(friends_filename,maxnums=150):

    friends = load_friends(friends_filename)
    count_friends_ = count_friends(friends)
    best_friends = sorted(count_friends_.items(),key=itemgetter(1),reverse=True)
    for uid,count in best_friends:
        uid = str(uid)
        if len(friends) == maxnums:
            break
        if uid not in friends:
            friends[uid] = get_friends(t,uid)
            for friend in friends[uid]:
                count_friends_[friend] += 1
    return friends


def compute_similarity(friends1,friends2):
    return len(friends1 & friends2) / len(friends1 | friends2)

def create_graph(follwers,threshold=0):
    G = networkx.Graph()
    for uid1 in follwers:
        for uid2 in follwers:
            if uid1 == uid2:
                continue
            weight = compute_similarity(follwers[uid1],follwers[uid2])
            if weight >= threshold:
                G.add_node(uid1)
                G.add_node(uid2)
                G.add_edge(uid1,uid2,weight=weight)
    return G


# silhouette coefficient
def compute_silhouette(threshold,friends):
    print("Computing sihouette")
    G = create_graph(friends,threshold=threshold)
    if (len(G.nodes()) < 2):
        return -99
    sub_graphs = networkx.connected_component_subgraphs(G)
    if not (2 <= networkx.number_connected_components(G) < len(G.nodes()) -1):
        return -99
    label_dict = {}
    for i,sub_graph in enumerate(sub_graphs):
        for node in sub_graph.nodes():
            label_dict[node] = i
    labels = np.array([label_dict[node] for node in G.nodes()])
    X = networkx.to_scipy_sparse_matrix(G).todense()

    X = 1 - X
    print("Computing better threshold..")
    return silhouette_score(X,labels,metric='precomputed')

def inverted_silhouette(threshold,friends):
    # minimize() return the mini,so reverse the silhouette coefficient
    # that will return the max
    return -compute_silhouette(threshold,friends)