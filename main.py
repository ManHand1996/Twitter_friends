import os
import networkx
from matplotlib import pyplot
from scipy.optimize import minimize
from classifier import TweetClassifier
from recommend_friends import get_friends_list,get_more_friends
from recommend_friends import inverted_silhouette,save_friends,load_friends,create_graph
"""
 theme: twitter post topic
 proxy: use in tweepy.API, http proxy 

 friends_filename = "./Data/python_friends.json"
 master_graph_path = "./graphs/friends_graph.png"
 sub_graphs_path = "./graphs/python_friends_subgraph.png"
"""

def train_labels(theme,proxy,tweets_filename,labels_filename):
    """
    get more tweet_texts and make labels about theme 
    and store with 'tweets_filename' 'labels_filename'
    """
    tweetClassifier = TweetClassifier(proxy)
    # classify texts in console
    tweetClassifier.get_labels(theme=theme,tweets_filename=tweets_filename
    ,labels_filename=labels_filename)
    
def export_model(theme,tweets_filename,labels_filename):
    tweetClassifier = TweetClassifier()
    model_file = os.path.join('./model',theme+'_'+'model.pkl')
    tweetClassifier.get_model(tweets_filename,labels_filename,model_file)
    return model_file

def start_train_model(theme):
    tweets_filename = os.path.join('./Data',theme+'_tweets.json')
    labels_filename = os.path.join('./Data',theme+'_labels.json')
    # if you want get more data to train your model run:
    train_labels(theme,'127.0.0.1:8123',tweets_filename,labels_filename)
    return export_model(theme,tweets_filename,labels_filename)



def get_friends_data(theme,proxy,model_filename,friends_filename):
    """
        run at first time
        get_more_firends(fname,maxnums) maxnums default 150
    """
    friends = get_friends_list(theme,proxy,model_filename)
    friends = get_more_friends(friends_filename)
    save_friends(friends,friends_filename)
    
    friends = load_friends(friends_filename)
    return friends



# draw the master graph of friends
def draw_graph(friends,master_graph_path):
    
    print("Drawing graph.....")
    G = create_graph(friends)
    pyplot.figure(figsize=(20,20))
    pos = networkx.random_layout(G)
    networkx.draw_networkx_nodes(G,pos,node_color='g',node_size=200,linewidths=1.5)
    edgewidth = [d['weight'] for (u,v,d) in G.edges(data=True)]
    networkx.draw_networkx_edges(G,pos,width=edgewidth)
    # arg:dpi - graph size,it's a big graph
    pyplot.savefig(master_graph_path,dpi=200)
    print("finish Drawing graph.....")

def draw_sub_graphs(friends,sub_graphs_path,threshold=0):
    
    print("Drawing subgraphs.....")
    G = create_graph(friends,threshold)
    sub_graphs = networkx.connected_component_subgraphs(G)
    n_subgraphs = networkx.number_connected_components(G)
    fig = pyplot.figure(figsize=(20,(n_subgraphs*2)))
    
    
    for i ,sub_graph in enumerate(sub_graphs):
        n_nodes = len(sub_graph.nodes())
        print("sub_graphs {0} has {1} nodes".format(i,n_nodes))
        
        ax = fig.add_subplot(int(n_subgraphs/2),2,i+1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # fig.sca(ax)
        pos = networkx.random_layout(G)

        networkx.draw_networkx_nodes(G,pos,sub_graph.nodes(),ax=ax,node_size=400)
        networkx.draw_networkx_edges(G,pos,sub_graph.edges(),ax=ax)
    
    fig.savefig(sub_graphs_path,dpi=50)
    print("finish Drawing subgraphs.....")
    return G

if __name__ == '__main__':
    theme = "python"
    proxy = "127.0.0.1:8123"
    friends_filename = "./Data/python_friends.json"
    master_graph_path = "./graphs/friends_graph.png"
    sub_graphs_path = "./graphs/python_friends_subgraph.png"
    
    model_filename = start_train_model(theme)
    friends = get_friends_data(theme,proxy,model_filename,friends_filename)
    friends = {uid:friends[uid] for uid in friends if len(friends[uid]) > 0}
    friends = {uid:set(friends[uid]) for uid in friends}
    
    # get the best threshold to divide master_graphs
    result = minimize(inverted_silhouette,0.1,method='nelder-mead',args=(friends,))
    threshold = result['x'][0]
    print("threshold :{}".format(threshold))
    draw_graph(friends,master_graph_path)
    
    G = draw_sub_graphs(friends,sub_graphs_path,threshold)
    sub_graphs = networkx.connected_component_subgraphs(G)
    for i,sub_graph in enumerate(sub_graphs):
        n_nodes = sub_graph.nodes()
        print("{0} : {1} nodes".format(i,n_nodes))
