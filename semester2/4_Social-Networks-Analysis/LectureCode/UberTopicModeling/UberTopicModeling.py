# This Python class applies the LDA topic modeling technique on the cleaned version of the Uber-related dataset 
# that was generated through the utilization of the UberSentimentClassifier class.

import logging
import sys
import os
import pprint
import math
import numpy as np
import scipy.io as spi
import gensim
import nltk
import nltk.data
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models, similarities
from nltk.tokenize import WhitespaceTokenizer,WordPunctTokenizer
from nltk.stem import LancasterStemmer 

class UberTopicModeling(object):
    
    def corpus_to_list(self,corpus):
        L = []
        for element in corpus:
            L.append(element)
        return L    
    
    def list_to_np_array(self,L):
        Lnp_rows = len(L)
        max_column_indices = []
        column_indices = []
        row_values = []
        for Lt in L:
            Lt = sum(Lt,())
            Lt_size = len(Lt)
            Lt_indices = [int(Lt[i]) for i in range(0,Lt_size,2)]
            Lt_values = [Lt[i] for i in range(1,Lt_size,2)]
            #pprint.pprint(Lt_indices)
            #pprint.pprint(Lt_values)
            column_indices.append(np.array(Lt_indices,'int'))
            row_values.append(np.array(Lt_values,'float'))
            if(len(Lt_indices)>0):
                max_column_indices.append(max(Lt_indices))
            else:
                print "No Lt indices were found!"
                max_Lt_indices = 0
                max_column_indices.append(max_Lt_indices)     
        Lnp_columns = max(max_column_indices)+1
        print'Constructing an [%d]x[%d] np array' %(Lnp_rows,Lnp_columns)
        Lnp = np.zeros((Lnp_rows,Lnp_columns))
        for row_index in range(Lnp_rows):
            Lnp[row_index,column_indices[row_index]] = row_values[row_index]
        return Lnp
    
    def __init__(self,raw_corpus_source,topics_number_list,passes):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.row_corpus_source = raw_corpus_source # .txt file row-wise storing the cleaned version of the Uber dataset.
        self.topics_number_list = topics_number_list # List determining the numbers of LDA topics to be produced.
        self.topics_number = None #Initialize container storing the number of LDA topics to be produced at each batch mode step.
        self.raw_corpus = [] # Initialize container (list) for storing the raw twitter post data.
        self.purified_raw_corpus = [] #Initialize container (list) for storing the purified raw twitter post data.
        self.dictionary = None #Initialize purified corpus dictionary to None object.
        self.words_num = 0 #Initialize the number of words in the dictionary.
        self.corpus = None #Initialize (actual) corpus object.
        self.tfidf_corpus = None #Initialize tfidf corpus object.
        self.lda = None #Initialize lda transformation object.
        self.lda_corpus = None #Initialize lda-based corpus object.
        self.passes = passes # This is an internal parameter of the probabilistic topic modeling technique.
        
    def load_row_corpus(self):
        lines_count = 0
        print "Loading Uber-related posts stored in file %s" %self.row_corpus_source
        with open(self.row_corpus_source,'r') as raw_corpus_file:
            for line in raw_corpus_file:
                self.raw_corpus.append(line)
                lines_count+=1
        raw_corpus_file.close()
        print "Successfully read %d posts" %lines_count
        
    def purify_raw_corpus(self):
        print "Purifying raw posts."
        english_stops = nltk.corpus.stopwords.words("english") 
        english_stops.append('amp')
        word_tokenizer = WordPunctTokenizer()
        stemmer = LancasterStemmer()
        for post in self.raw_corpus:
            words = word_tokenizer.tokenize(post)
            # Next code line should be commented out when stemming is not to be performed. This practice results in more descriptive topic keywords.
            #purified_post = [stemmer.stem(word.lower()) for word in words if word not in english_stops and len(word)>=3]
            purified_post = [word.lower() for word in words if word not in english_stops and len(word)>=3]
            self.purified_raw_corpus.append(purified_post)
        #pprint.pprint(self.purified_raw_corpus)
        
    def build_dictionary(self):
        print "Building corpus dictionary."
        dictionary = corpora.Dictionary(self.purified_raw_corpus)
        once_ids = [tokenid for tokenid,docfreq in dictionary.dfs.iteritems() if docfreq==1]
        dictionary.filter_tokens(once_ids)
        self.dictionary = dictionary
        self.words_num = len(self.dictionary)
        print "Constructed a dictionary of %d words" %self.words_num
        
    def build_corpus(self):
        print "Building corpus."
        self.corpus = [self.dictionary.doc2bow(purified_post) for purified_post in self.purified_raw_corpus]
        
    def build_tfidf_corpus(self):
        print "Building tfidf corpus."
        tfidf = models.TfidfModel(self.corpus)
        self.tfidf_corpus = tfidf[self.corpus]
        
    def build_lda_corpus(self):
        print "Building lda corpus."
        self.lda = gensim.models.ldamodel.LdaModel(corpus=self.corpus,id2word=self.dictionary,num_topics=self.topics_number,update_every=0,passes=self.passes)
        self.lda.print_topics(10)
        self.lda_corpus = self.lda[self.tfidf_corpus]
               
    def print_lda_corpus(self):
        print "Displaying lda-based corpus"
        for row in self.lda_corpus:
            print row
        
    def export_dictionary(self):
        current_directory = os.getcwd()
        output_file_location = "\\data\\dictionary.txt"
        complete_output_file_location = current_directory + output_file_location
        output_file = open(complete_output_file_location,"w")
        dictionary = self.dictionary.token2id
        for term in dictionary:
            row_string = "%s: %d" %(term,dictionary[term])
            print >>output_file,row_string
        output_file.close()    
    
    def export_complete_lda_topics(self):
        print "Displaying lda_based topics."
        current_directory = os.getcwd()
        lda_full_topics_file_location = "\\data\\uber_lda_full" +str(self.topics_number) + "_topics.txt"
        complete_lda_full_topics_file_location = current_directory + lda_full_topics_file_location
        output_file = open(complete_lda_full_topics_file_location,"w")
        for i in range(0,self.topics_number):
            current_topic = self.lda.show_topic(i,topn=self.words_num)
            current_string = "Topic[%d] = %s" %(i,current_topic)
            print >> output_file,current_string
        output_file.close()    
    
    def export_lda_topics(self):
        current_directory = os.getcwd()
        lda_topics_file_location = "\\data\\uber_lda_" +str(self.topics_number) + "_topics.txt"
        complete_lda_topics_file_location = current_directory + lda_topics_file_location
        print "Exporting lda-based topics (topics_number = %d) to file %s" %(self.topics_number,complete_lda_topics_file_location)
        output_file = open(complete_lda_topics_file_location,"w")
        for i in range(0,self.topics_number):
            current_topic = self.lda.show_topic(i,topn=10)
            current_string = "Topic[%d] = %s" %(i,current_topic)
            print >> output_file,current_string
        output_file.close()
        
    def export_lda_vectors(self):
        current_directory = os.getcwd()
        lda_vectors_file_location = "\\data\\uber_lda_" + str(self.topics_number) + "_vectors.mat"
        complete_lda_vectors_file_location = current_directory + lda_vectors_file_location
        print "Exporting lda-based vectors (topics_number=%d) to file %s" %(self.topics_number,complete_lda_vectors_file_location)
        lda_vectors = self.corpus_to_list(self.lda_corpus)
        lda_vectors_array = self.list_to_np_array(lda_vectors)
        spi.savemat(complete_lda_vectors_file_location,dict(lda_vectors_array=lda_vectors_array))
        
    def lda_topic_modeling(self,topics_number):
        self.topics_number = topics_number
        self.build_lda_corpus()
        self.export_lda_topics()
        self.export_complete_lda_topics()
        self.export_lda_vectors()
        
    def batch_mode_lda_topic_modeling(self):
        for topics_number in self.topics_number_list:
            print "Performing LDA topic modeling operations for topics_number = %d" %topics_number
            self.lda_topic_modeling(topics_number)
                                                        
def main():
    raw_corpus_source = 'data/UBER_all.txt'
    topics_number_list = [10];
    passes = 1 #This parameter must be changed to a larger number to gain more accurate results.
    UTM = UberTopicModeling(raw_corpus_source,topics_number_list,passes)
    UTM.load_row_corpus()
    UTM.purify_raw_corpus()
    UTM.build_dictionary()
    UTM.export_dictionary()
    UTM.build_corpus()
    UTM.build_tfidf_corpus()
    UTM.batch_mode_lda_topic_modeling()
    
    #UTM.build_lda_corpus()
    #UTM.print_lda_corpus()
    #UTM.export_lda_topics()
    #UTM.export_complete_lda_topics()
    #UTM.export_lda_vectors()
    
if __name__ == "__main__":
    main()    