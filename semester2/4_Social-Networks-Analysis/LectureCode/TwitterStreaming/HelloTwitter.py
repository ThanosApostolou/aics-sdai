import tweepy
 
# Consumer keys and access tokens, used for OAuth
consumer_key = 'put your own...'
consumer_secret = 'put your own...'
access_token = 'put your own...'
access_token_secret = 'put your own...'
 
# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
 
# Creation of the actual interface, using authentication
api = tweepy.API(auth)

# Get my profile
user = api.me()
print('Name: ' + user.name)
print('Location: ' + user.location)
print('Friends: ' + str(user.friends_count))

 
# Sample method, used to update a status
api.update_status('Data mining on social networks: Welcome Class!!!')