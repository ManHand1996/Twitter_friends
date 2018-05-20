# twitter_friends
> Environment
ubuntu 16.04 TLS
Python3.5  

### requriment package:  
- sklearn
- networkx
- matplotlib  
- tweepy/twitter  

### info
- 1.edit main.py:  
```  
  if __name__ == '__main__':
      theme = "python" # Tweets topic
      proxy = "127.0.0.1:8123" # if you can't access www.twitter.com
      friends_filename = "./Data/python_friends.json" # friends list of each relevant user
      master_graph_path = "./graphs/friends_graph.png"
      sub_graphs_path = "./graphs/python_friends_subgraph.png"
```
- 2.run the main.py file: python3 main.py  
- 3.you need to manually recognize the tweet which is related to the theme（about 100 tweets）
- 4.waitting some time,cause the twitter API ratelimit
- 5.if you don't want to spend too much time annotate main.py>get_friends_data(), but it's will get few data

```
  friends = get_more_friends(friends_filename)
```
### 150 friends Graph
![](https://github.com/ManHand1996/Twitter_friends/blob/master/graphs/friends_graph.png)

### result:

0 : ['93711247', '50090898', '382267114', '20536157'] nodes  
1 : ['2426422297', '812555214024740865', '1566463268'] nodes  
2 : ['130745589', '216939636'] nodes  
3 : ['15473958', '34743251', '11348282', '1451773004'] nodes  
4 : ['428333', '2097571', '759251'] nodes  
5 : ['70831441', '68746721', '775449094739197953', '118263124', '2895770934', '33836629'] nodes  
6 : ['555031989', '841437061'] nodes  
7 : ['44196397', '13298072'] nodes  

