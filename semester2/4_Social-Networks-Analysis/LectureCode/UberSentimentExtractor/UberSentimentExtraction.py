# This Python class performs binary sentiment classification on the UBER-related dataset.
# It is important to note that the functionality provided by this class takes for granted
# that the initial data cleaning operations are already competed. Therefore, the existing 
# "data" folder contains files corresponding to the purified versions of the scored and unscored
# corpora along with the valid rows ids of the csv file storing the raw tweets.

import sys
import os
import pprint
import re
import math
import numpy as np
import scipy.io as spi
import csv
import time
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import metrics
from sklearn import cross_validation

class UberSentimentExtractor(object):
    
    def __init__(self,purified_training_source,purified_testing_source,testing_source,valid_rows_ids_source,labels_source,tf_idf_features_num):
        self.purified_training_source = purified_training_source
        self.purified_testing_source = purified_testing_source
        self.testing_source = testing_source
        self.valid_rows_ids_source = valid_rows_ids_source
        self.labels_source = labels_source
        self.tf_idf_features_num = tf_idf_features_num
    
    def load_valid_rows_ids(self):
        print "Loading valid rows ids from file %s" %self.valid_rows_ids_source
        #Re-initialize valid_rows_ids so that this step is not required to be conducted elsewhere in the code.
        self.valid_rows_ids = []
        with open(self.valid_rows_ids_source,'r') as fp:
            for line in fp:
                self.valid_rows_ids.append(int(line))
        fp.close()
        self.valid_rows_ids = np.array(self.valid_rows_ids)
        self.valid_rows_ids = self.valid_rows_ids - 2
        # The use of (-2) offset for the valid_rows_ids array is justified because:
        # (a) we need indexing a c-like array whose offset is 0
        # (b) the first line of the the UBER_all.csv file contains no actual data.
        print 'Valid rows ids were successfully loaded:'
        pprint.pprint(self.valid_rows_ids) 
    
    def load_timestamps(self):
        # Keep in mind that timestamps are loaded only from the unscored corpus.
        print 'Loading timestamps from file: %s' %self.testing_source
        # Re-initialize timestamps so that this step is not required to be conducted elsewhere.
        self.timestamps = []
        self.dates = []
        fp = open(self.testing_source,'rb')
        reader = csv.reader( fp, delimiter=';', quotechar='"', escapechar='\\' )
        for row in reader:
            timestamp_string = row[2]
            #print timestamp_string
            timestamp = time.mktime(datetime.datetime.strptime(timestamp_string, "%Y-%m-%d %H:%M:%S").timetuple())
            date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            self.timestamps.append(timestamp)
            self.dates.append(date)
        fp.close()
        self.np_timestamps = np.array(self.timestamps)
        self.np_dates = np.array(self.dates)
        self.np_timestamps = self.np_timestamps[self.valid_rows_ids]
        self.np_dates = self.np_dates[self.valid_rows_ids]
        print "timestamps were successfully loaded:"
        pprint.pprint(self.np_timestamps)
        print "dates were successfully loaded:"
        pprint.pprint(self.np_dates)    
    
    def load_purified_unscored_corpus(self):
        print "Loading purified scored corpus from file %s" %self.purified_testing_source
        # Re-initialize testing_corpus so that this step is not required to be conducted elsewhere.
        self.testing_corpus = []
        with open(self.purified_testing_source,'r') as fp:
            for line in fp:
                line = re.sub( '\n', '',line).strip()
                self.testing_corpus.append(line)
        fp.close()
        print 'Purified unscored corpus was successfully loaded.'
        #pprint.pprint(self.testing_corpus)       
    
    def load_purified_scored_corpus(self):
        print "Loading purified scored corpus from file %s" %self.purified_training_source
        # Re-initialize training_corpus so that this step is not required to be conducted elsewhere.
        self.training_corpus = []
        with open(self.purified_training_source,'r') as fp:
            for line in fp:
                line = re.sub( '\n', '',line).strip()
                self.training_corpus.append(line)
        fp.close()
        print 'Purified scored corpus was successfully loaded.'
        #pprint.pprint(self.training_corpus)  

    def load_scored_labels(self):
        print "Loading scored labels from file %s" %self.labels_source
        positive_label = 1
        negative_label = -1
        neutral_label = 0        
        self.training_labels = []
        with open(self.labels_source,'r') as fp:
            for line in fp:
                line = re.sub( '\n', '',line).strip()
                if(line=='1'):
                    self.training_labels.append(positive_label)
                elif(line=='0'):
                    self.training_labels.append(neutral_label)
                else:
                    self.training_labels.append(negative_label)
        fp.close()
        print 'Scored labels were successfully loaded.'
        self.labels_array = np.array(self.training_labels)
        pprint.pprint(self.labels_array) 

    def combine_train_test_corpuses(self):
        self.training_patterns_num = len(self.training_corpus)
        self.testing_patterns_num = len(self.testing_corpus)
        print 'Training patterns num %d' %self.training_patterns_num
        print 'Testing patterns num %d' %self.testing_patterns_num
        #self.corpus = self.training_corpus
        self.corpus = [self.training_corpus,self.testing_corpus]
        self.corpus = sum(self.corpus,[])
        print "Training and testing corpuses were successfully combined."
        #pprint.pprint(self.corpus)
        print "Final corpus length %d" %len(self.corpus)

    def tf_idf_vectorization(self):
        print "Performing tf-idf corpus vectorization with max_features = %d" %self.tf_idf_features_num
        vectorizer = TfidfVectorizer(min_df=1,max_features=self.tf_idf_features_num)
        self.tf_idf_features = vectorizer.fit_transform(self.corpus)
        self.tf_idf_features_array = self.tf_idf_features.toarray()
        pprint.pprint(self.tf_idf_features_array)
        print "TF-IDF corpus vectorization was successfully completed"

    def cross_validated_binary_svm_classification(self):
        print "Performing 10-fold cross-validation on labeled data on 2 classes:"
        labeled_patterns = self.tf_idf_features_array[0:self.training_patterns_num,:]
        labeled_labels = self.labels_array
        positive_labels_indices = np.where(labeled_labels==1)
        positive_labels_indices = positive_labels_indices[0]
        negative_labels_indices = np.where(labeled_labels==-1)
        negative_labels_indices = negative_labels_indices[0]
        positive_labeled_patterns = labeled_patterns[positive_labels_indices,:]
        negative_labeled_patterns = labeled_patterns[negative_labels_indices,:]
        positive_labeled_labels = labeled_labels[positive_labels_indices]
        negative_labeled_labels = labeled_labels[negative_labels_indices]
        print np.shape(positive_labeled_patterns)
        print np.shape(negative_labeled_patterns)
        self.labeled_patterns = np.row_stack((positive_labeled_patterns,negative_labeled_patterns))
        positive_labeled_labels = np.reshape(positive_labeled_labels,(1,np.size(positive_labeled_labels)))
        negative_labeled_labels = np.reshape(negative_labeled_labels,(1,np.size(negative_labeled_labels)))
        print np.shape(positive_labeled_labels)
        print np.shape(negative_labeled_labels)
        self.labeled_labels = np.column_stack((positive_labeled_labels,negative_labeled_labels))
        self.labeled_labels = self.labeled_labels[0]
        print np.shape(self.labeled_labels)
        #self.labeled_labels = np.transpose(self.labeled_labels)
        svm_classifier = svm.SVC(C=1,kernel='rbf',gamma=1,verbose=False)
        #n_samples = self.labeled_patterns.shape[0]
        #cv = cross_validation.ShuffleSplit(n_samples, n_iter=100,test_size=0.05, random_state=0)        
        accuracy_scores = cross_validation.cross_val_score(svm_classifier,self.labeled_patterns,self.labeled_labels,cv=10)
        print "Accuracy per fold:"
        print accuracy_scores
        print("Mean Accuracy: %0.3f (+/- %0.3f)" % (accuracy_scores.mean(), accuracy_scores.std() * 2))
        
        precision_scores = cross_validation.cross_val_score(svm_classifier,self.labeled_patterns,self.labeled_labels,cv=10,scoring='precision')
        print "Precision per fold:"
        print precision_scores
        print("Mean Precision: %0.3f (+/- %0.3f)" % (precision_scores.mean(), precision_scores.std() * 2))
        
        recall_scores = cross_validation.cross_val_score(svm_classifier,self.labeled_patterns,self.labeled_labels,cv=10,scoring='recall')
        print "Recall per fold:"
        print recall_scores
        print("Mean Recall: %0.3f (+/- %0.3f)" % (recall_scores.mean(), recall_scores.std() * 2))
        
        f_scores = cross_validation.cross_val_score(svm_classifier,self.labeled_patterns,self.labeled_labels,cv=10,scoring='f1')
        print "F-Score per fold:"
        print f_scores
        print("Mean F-Score: %0.3f (+/- %0.3f)" % (f_scores.mean(), f_scores.std() * 2))

    def set_labeled_unlabeled_patterns_and_labels(self):
        print "Setting labeled / unlabeled patterns and labels for binary classification"
        # This function utilizes code from cross_validated_binary_classification in order
        # to set the required variables for the labeled_patters, unlabeled_patterns and labeled_labels.
        labeled_patterns = self.tf_idf_features_array[0:self.training_patterns_num,:]
        self.unlabeled_patterns = self.tf_idf_features_array[self.training_patterns_num:,:]
        labeled_labels = self.labels_array
        positive_labels_indices = np.where(labeled_labels==1)
        positive_labels_indices = positive_labels_indices[0]
        negative_labels_indices = np.where(labeled_labels==-1)
        negative_labels_indices = negative_labels_indices[0]
        positive_labeled_patterns = labeled_patterns[positive_labels_indices,:]
        negative_labeled_patterns = labeled_patterns[negative_labels_indices,:]
        positive_labeled_labels = labeled_labels[positive_labels_indices]
        negative_labeled_labels = labeled_labels[negative_labels_indices]
        #print np.shape(positive_labeled_patterns)
        #print np.shape(negative_labeled_patterns)
        self.labeled_patterns = np.row_stack((positive_labeled_patterns,negative_labeled_patterns))
        positive_labeled_labels = np.reshape(positive_labeled_labels,(1,np.size(positive_labeled_labels)))
        negative_labeled_labels = np.reshape(negative_labeled_labels,(1,np.size(negative_labeled_labels)))
        #print np.shape(positive_labeled_labels)
        #print np.shape(negative_labeled_labels)
        self.labeled_labels = np.column_stack((positive_labeled_labels,negative_labeled_labels))
        self.labeled_labels = self.labeled_labels[0]
        #print np.shape(self.labeled_labels)    

    def unscored_corpus_classification(self):
        print "Performing decision value-based classification on the unscored corpus:"
        svm_classifier = svm.SVC(C=1,kernel='rbf',gamma=1,verbose=False)
        svm_classifier.fit(self.labeled_patterns,self.labeled_labels)
        self.unlabeled_decision_values = svm_classifier.decision_function(self.unlabeled_patterns)
        self.np_unlabeled_decision_values = np.array(self.unlabeled_decision_values)
        print self.np_unlabeled_decision_values.shape
        pprint.pprint(self.np_unlabeled_decision_values)
        
    def export_date_vectors(self):
        current_directory = os.getcwd()
        date_vectors_file_location = "\\data\\dates.mat"
        complete_date_vectors_file_location = current_directory + date_vectors_file_location
        np_dates_shape = self.np_dates.shape
        np_dates_rows = np_dates_shape[0]
        np_dates_array = self.np_dates
        print "Exporting %d date vectors to file %s" %(np_dates_rows,complete_date_vectors_file_location)
        spi.savemat(complete_date_vectors_file_location,dict(np_dates_array=np_dates_array))
      
    def export_sentiment_vectors(self):
        current_directory = os.getcwd()
        sentiment_vectors_file_location = "\\data\\sentiment_values.mat"
        complete_sentiment_vectors_file_location = current_directory + sentiment_vectors_file_location
        np_unlabeled_decision_values_shape = self.np_unlabeled_decision_values.shape
        np_unlabeled_decision_values_rows = np_unlabeled_decision_values_shape[0]
        np_unlabeled_decision_values_array = self.np_unlabeled_decision_values
        print "Exporting %d sentiment vectors to file %s" %(np_unlabeled_decision_values_rows,complete_sentiment_vectors_file_location)
        spi.savemat(complete_sentiment_vectors_file_location,dict(np_unlabeled_decision_values_array=np_unlabeled_decision_values_array)) 

def main():
    # SET REQUIRED CONSTANTS
    purified_training_source = 'data/cleaned_scored_corpus.txt'
    purified_testing_source = 'data/UBER_all.txt'
    testing_source = 'data/UBER_all.csv'
    valid_rows_ids_source = 'data/UBER_all_valid_rows.txt'
    labels_source = 'data/labels.txt'
    tf_idf_features_num = 400
    UBE = UberSentimentExtractor(purified_training_source,purified_testing_source,testing_source,valid_rows_ids_source,labels_source,tf_idf_features_num)
    UBE.load_valid_rows_ids()
    UBE.load_timestamps()
    UBE.load_purified_scored_corpus()
    UBE.load_purified_unscored_corpus()
    UBE.load_scored_labels()
    UBE.combine_train_test_corpuses()
    UBE.tf_idf_vectorization()
    #UBE.cross_validated_binary_svm_classification()
    UBE.set_labeled_unlabeled_patterns_and_labels()
    UBE.unscored_corpus_classification()
    UBE.export_date_vectors()
    UBE.export_sentiment_vectors()

if __name__ == "__main__":
    main()    
        