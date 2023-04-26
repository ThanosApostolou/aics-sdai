
import sys
import tweepy
import codecs
import MySQLdb

# Query terms

#Q = sys.argv[1:]
#Q = ["Dow Jones", "#DowJones", "#DJIA","#DJI","#DOW","$DJI","$DOW","$DJIA","$DOWI","$INDU","Nasdaq","#Nasdaq","$Nasdaq","S&P500","$S&P500","#S&P500","SP500","SPX"]
Q = ["#euro2016","euro 2016","Euro2016","Euro 2016"]
# Get these values from your application settings

CONSUMER_KEY = 'put your own...'
CONSUMER_SECRET = 'put your own...'

# Get these values from the "My Access Token" link located in the
# margin of your application details, or perform the full OAuth
# dance

ACCESS_TOKEN = 'put your own...'
ACCESS_TOKEN_SECRET = 'put your own...'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# Note: Had you wanted to perform the full OAuth dance instead of using
# an access key and access secret, you could have uses the following
# four lines of code instead of the previous line that manually set the
# access token via auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
#
# auth_url = auth.get_authorization_url(signin_with_twitter=True)
# webbrowser.open(auth_url)
# verifier = raw_input('PIN: ').strip()
# auth.get_access_token(verifier)

class CustomStreamListener(tweepy.StreamListener):
    
    def connect(self,status):
        self.host = "127.0.0.1"
        self.user = "sakis"
        self.password = "sakis"
        self.database = "stream_monitor"
        try:
            self.db = MySQLdb.connect(host=self.host,user=self.user,passwd=self.password,db=self.database)
            self.cursor = self.db.cursor()
            print "Connection to database %s succesfull!" %self.database
        except MySQLdb.Error, e:
            try:
                print "MySQL Error[%d]: %s" %(e.args[0],e.args[1])
            except IndexError:
                print "MySQL Error: %s" %str(e)
    
    def disconnect(self):
        try:
            self.cursor.close()
            self.db.close()
            print "Connection to database %s was successfully closed!" %self.database
        except MySQLdb.Error, e:
            try:
                print "MySQL Error [%d]: %s" % (e.args[0], e.args[1])
            except IndexError:
                print "MySQL Error: %s" % str(e)
    
    def insert(self,status):
        sql_string1 = "INSERT INTO data (TEXT,SCREEN_NAME,DATE,SOURCE) VALUES"
        sql_string2 = "('%s','%s','%s','%s')"%(status.text,status.author.screen_name,status.created_at,status.source)
        sql_string = sql_string1 + sql_string2
        print sql_string
        try:
            self.cursor.execute(sql_string)
            self.db.commit()
            print "Select query was successfully executed"
        except MySQLdb.Error, e:
            try:
                print "MySQL Error [%d]: %s" % (e.args[0], e.args[1])
            except IndexError:
                print "MySQL Error: %s" % str(e)
    
    def on_status(self, status):
        
        # We'll simply print some values in a tab-delimited format
        # suitable for capturing to a flat file but you could opt
        # store them elsewhere, retweet select statuses, etc.

        try:
            print "%s\t%s\t%s\t%s" % (status.text,
                                      status.author.screen_name,
                                      status.created_at,
                                      status.source,)
            self.connect(status)
            self.insert(status)
            self.disconnect()
        except Exception, e:
            print >> sys.stderr, 'Encountered Exception:', e
            pass

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True # Don't kill the stream

sys.stdout = codecs.getwriter("iso-8859-1")(sys.stdout, 'xmlcharrefreplace')

# Create a streaming API and set a timeout value of 60 seconds

streaming_api = tweepy.streaming.Stream(auth, CustomStreamListener(), timeout=60)

# Optionally filter the statuses you want to track by providing a list
# of users to "follow"

print >> sys.stderr, 'Filtering the public timeline for "%s"' % (' '.join(Q),)

streaming_api.filter(follow=None, track=Q)
