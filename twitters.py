import tweepy
import os
import json

def get_tweepyAPI(myproxy):
	"""
		use twitter API
		create twitter APP and get necessary key and secrect
		from https://apps.twitter.com
	"""
	keys = {}
	with open('./Data/twitterAPIToken.json','r') as out:
		keys = eval(out.read())
	authorization = tweepy.OAuthHandler(keys["consumer_key"],keys["consumer_secret"])
	authorization.set_access_token(keys["access_token"],keys["access_token_secret"])
	# myproxy:local proxy port ->: 127.0.0.1:8123
	t = tweepy.API(authorization,proxy=myproxy,retry_count=1)
	return t
