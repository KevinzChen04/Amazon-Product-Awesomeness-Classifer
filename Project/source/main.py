import numpy as np
import pandas as pd
import json
import os
import re
import string
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from keras.regularizers import l1
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.naive_bayes import MultinomialNB

print("Finished importing")

def sentimentalAnalysis(df):
  # initialize varaibles
  sol = np.zeros([8, df.shape[0]])
  sentanal = SentimentIntensityAnalyzer()

  # loop through all reviews
  for i in range(df.shape[0]):
      
    # if text is empty, then sentimental analysis=0
    if not df['reviewText'].iloc[i]:
      sol[0][i] = 0
      sol[1][i] = 0
      sol[2][i] = 0
      sol[3][i] = 0
      continue
  
    #otherwise, calculate Sentimental analysis values
    SAValue = sentanal.polarity_scores(df['reviewText'].iloc[i])
    sol[0][i] = SAValue["neg"]
    sol[1][i] = SAValue["neu"]
    sol[2][i] = SAValue["pos"]
    sol[3][i] = SAValue["compound"]

    #do sent anal on summary text
    if not df['summary'].iloc[i]:
      sol[4][i] = 0
      sol[5][i] = 0
      sol[6][i] = 0
      sol[7][i] = 0
      continue
  
    #otherwise, calculate Sentimental analysis values
    SAValue = sentanal.polarity_scores(df['summary'].iloc[i])
    sol[4][i] = SAValue["neg"]
    sol[5][i] = SAValue["neu"]
    sol[6][i] = SAValue["pos"]
    sol[7][i] = SAValue["compound"]
  return sol

def uniqueItems(df):
  #initialize Variables
  uniqueItems = 1
  curritem = df['asin'].iloc[0]

  #loop through all rows
  for i in range(df.shape[0]):

    #check if the review is unique
    if df['asin'].iloc[i] != curritem:
      curritem = df['asin'].iloc[i]
      uniqueItems = uniqueItems + 1
      #end of if statement

  return uniqueItems

def weighted_std(values, weight):
    average = np.average(values, weights=weight)
    std = np.average((values-average)**2, weights=weight)
    return np.sqrt(std)

def parseRow(votes, unixReview, reviewCount, nones, verified, negative, neutral,
             positive, compound, negativesumm, neutralsumm, positivesumm, 
             compoundsumm, weight, reviewlens, summarylens, reviews, summaries, asin):
      row = np.zeros(33, dtype=object)
      #check if votes is empty
      if(not votes):
        #Percent of Nones
        row[0] = 0

        #Max votes
        row[1] = 0

        #Average Vote
        row[2] = 0

        #Standard deviation votes
        row[3] = 0
      else:
        #Percent of Nones final Tally
        row[0] = nones / reviewCount
      
        #Max Votes
        row[1] = np.max(votes)

        #Average Vote
        row[2] = np.average(votes)

        #Standard deviation votes
        row[3] = np.std(votes)

      #Percent of verified
      row[4] = verified/reviewCount
      
      #Minimum Review Time
      row[5] = np.min(unixReview)
      
      #Maximum Review Time
      row[6] = np.max(unixReview)

      #Average review time
      row[7] = np.average(unixReview)

      #Standard Deviation of review time
      row[8] = np.std(unixReview)

      #Number of reviews
      row[9] = reviewCount

      #sentimental analysis
      row[10] = np.average(negative, weights=weight)
      row[11] = weighted_std(negative, weight)
      row[12] = np.average(neutral, weights=weight)
      row[13] = weighted_std(neutral, weight)
      row[14] = np.average(positive, weights=weight)
      row[15] = weighted_std(positive, weight)
      row[16] = np.average(compound, weights=weight)
      row[17] = weighted_std(compound, weight)
      row[18] = np.average(negativesumm, weights=weight)
      row[19] = weighted_std(negativesumm, weight)
      row[20] = np.average(neutralsumm, weights=weight)
      row[21] = weighted_std(neutralsumm, weight)
      row[22] = np.average(positivesumm, weights=weight)
      row[23] = weighted_std(positivesumm, weight)
      row[24] = np.average(compoundsumm, weights=weight)
      row[25] = weighted_std(compoundsumm, weight)

      #length of review and summary
      row[26] = np.average(reviewlens)
      row[27] = np.average(summarylens)
      row[28] = np.std(reviewlens)
      row[29] = np.std(summarylens)

      #review and summary raw text
      row[30] = reviews
      row[31] = summaries

      #asin
      row[32] = asin

      return row

def parseData(unique, SA, df, voteWeight, verifiedWeight, imageWeight):
  #add rows and columns to aggregated data
  newData = pd.DataFrame(columns=["Percent of Nones", "Max Votes", "Average Votes",
                                  "Standard Deviation Votes", "Percent of Verified", 
                                  "Minimum Review Time", "Maximum Review Time", 
                                  "Average Review Time", "Stand Deviation of Review Time",
                                  "Number of Reviews", "negative average", "negative SD", 
                                  "neutral average", "neutral SD", "positive average", 
                                  "positive SD", "compund average", "compound SD", 
                                  "negative average summ", "negative SD summ", 
                                  "neutral average summ", "neutral SD summ", "positive average summ", 
                                  "positive SD summ", "compund average summ", "compound SD summ", 
                                  "avg length of review", "avg length of summary", 
                                  "std length of review", "std length of summary", 
                                  "reviews", "summaries", "asin"],
                         index = range(unique))
  #initialize variables
  uniqueItems = 0
  itemcount = 0
  curritem = df['asin'].iloc[0]
  votes = []
  unixReview = []
  reviewCount = 0
  nones = 0
  verified = 0

  #sentimental anaylsis variables
  negative = []
  neutral = []
  positive = []
  compound = []
  negativesumm = []
  neutralsumm = []
  positivesumm = []
  compoundsumm = []
  weight = []
  reviewlens = []
  summarylens = []
  reviews = ""
  summaries = ""

  #loop through every row in data
  for i in range(df.shape[0]):    
    if (df['asin'].iloc[i] != curritem):
      itemcount=itemcount+1
      
      #append array into panda dataframe
      newData.loc[uniqueItems] = parseRow(votes, unixReview, reviewCount, nones, 
                                          verified, negative, neutral, positive, 
                                          compound, negativesumm, neutralsumm, positivesumm, 
                                          compoundsumm, weight, reviewlens, summarylens, 
                                          reviews, summaries, df['asin'].iloc[i-1])

      #reset the value of the variables
      nones = 0
      reviewCount = 0
      votes = []
      unixReview = []
      verified = 0
      negative = []
      positive = []
      neutral = []
      compound = []
      negativesumm = []
      neutralsumm = []
      positivesumm = []
      compoundsumm = []
      reviewlens = []
      summarylens = []
      reviews = ""
      summaries = ""

      #weighting
      weight = []
      uniqueItems = uniqueItems + 1
      # end of if statement
    curritem = df['asin'].iloc[i]
    reviewCount = reviewCount + 1
    currWeight = 1

    #Percent of Nones counting number of nums
    if df['vote'].iloc[i] == None:
      nones += 1
      votes.append(0)
    elif math.isnan(float(re.sub(",", "", str(df['vote'].iloc[i])))):
      nones += 1
      votes.append(0)
    else:
      #Votes
      numVotes = int(float(re.sub(",", "", str(df['vote'].iloc[i]))))
      #WEIGHTING SENTIMENT ANALYSIS

    #Count verified
    if df['verified'].iloc[i] == True:
      verified += 1
      currWeight = currWeight*verifiedWeight
      
    #Weight for image
    if df['image'].iloc[i] != None:
      currWeight = currWeight*imageWeight

    #append unixReviewTime
    unixReview.append(int(df['unixReviewTime'].iloc[i]))

    #sentimental analysis
    negative.append(SA[0][i])
    neutral.append(SA[1][i])
    positive.append(SA[2][i])
    compound.append(SA[3][i])
    negativesumm.append(SA[4][i])
    neutralsumm.append(SA[5][i])
    positivesumm.append(SA[6][i])
    compoundsumm.append(SA[6][i])
    weight.append(currWeight)

    #length of review and summary
    if df['reviewText'].iloc[i] != None:
      reviewlens.append(len(df['reviewText'].iloc[i]))
    else: 
      reviewlens.append(0)

    if df['summary'].iloc[i] != None:
      summarylens.append(len(df['summary'].iloc[i]))
    else: 
      summarylens.append(0)
    
    #review and summary raw text
    if df['reviewText'].iloc[i]:
      reviews = reviews + ' ' + df['reviewText'].iloc[i]
    if df['summary'].iloc[i]:
      summaries= summaries + ' ' + df['summary'].iloc[i]

    #end of for loop

  # parse last rows data
  newData.loc[unique - 1] = parseRow(votes, unixReview, reviewCount, nones, 
                                     verified, negative, neutral, positive, 
                                     compound, negativesumm, neutralsumm, positivesumm, 
                                     compoundsumm, weight, reviewlens, summarylens, 
                                     reviews, summaries, df['asin'].iloc[i])
                                     
  return newData



current_directory = 'source'
dataset = 'CDs_and_Vinyl'
trainData = pd.read_json(os.path.join(current_directory, dataset, 'train', 'review_training.json'))
awesometrain = pd.read_json(os.path.join(current_directory, dataset, 'train', 'product_training.json'))

testFeatures = pd.read_json(os.path.join(current_directory, dataset, 'test3', 'review_test.json'))
awesometest = pd.read_json(os.path.join(current_directory, dataset, 'test3', 'product_test.json'))

trainData['unixReviewTime'] = pd.to_numeric(trainData['unixReviewTime'])
testFeatures['unixReviewTime'] = pd.to_numeric(testFeatures['unixReviewTime'])

trainData = trainData.sort_values('asin')
testFeatures = testFeatures.sort_values('asin')

unique = uniqueItems(trainData)
trainDataSA = sentimentalAnalysis(trainData)
parsedData = parseData(unique, trainDataSA, trainData, 1, 1, 1)

unique = uniqueItems(testFeatures)
testFeaturesSA = sentimentalAnalysis(testFeatures)
parsedTest = parseData(unique, testFeaturesSA, testFeatures, 1, 1, 1)

asins = parsedTest["asin"]
awesometrain = awesometrain.drop("asin", axis=1)

scaler = MinMaxScaler()
parsedData.iloc[:,:30] = scaler.fit_transform(parsedData.iloc[:,:30])
parsedTest.iloc[:,:30] = scaler.fit_transform(parsedTest.iloc[:,:30])

parsedData = parsedData.drop("asin", axis=1)
parsedTest = parsedTest.drop("asin", axis=1)

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')
  
max_features = 10000
sequence_length = 500

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

togtrain = parsedData["reviews"]+parsedData["summaries"]
togtest = parsedTest["reviews"] + parsedTest["summaries"]

tog = pd.concat([togtrain, togtest])
vectorize_layer.adapt(tog)

trainVectorized = vectorize_layer(togtrain)
testVectorized = vectorize_layer(togtest)

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

SAtrain = Sequential([
  layers.Embedding(max_features+1, 10, input_length=sequence_length),
  layers.Dropout(0.2),
  layers.Conv1D(filters=32, kernel_size=8, activation='relu'),
  layers.MaxPooling1D(pool_size=2),
  layers.Flatten(),
  layers.Dense(10, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(1, activation='sigmoid')])

SAtrain.compile(
    optimizer = 'adam',
    loss='binary_crossentropy',
    metrics=tf.metrics.BinaryAccuracy(threshold=0.5),
)

SAtrain.fit(trainVectorized, awesometrain, validation_split=0.2, epochs=10, callbacks = es_callback)

y_train = SAtrain.predict(trainVectorized)
y_preds = SAtrain.predict(testVectorized)

clf = MultinomialNB(force_alpha=True)
tfidf = TfidfVectorizer(stop_words ='english')

transfTrain = tfidf.fit_transform(togtrain.fillna(""))
transfTest = tfidf.transform(togtest.fillna(""))

percentile = SelectPercentile(chi2, percentile=14)

selectTrain = percentile.fit_transform(transfTrain, awesometrain)
selectTest = percentile.transform(transfTest)

clf.fit(selectTrain, awesometrain)
tfidf_train = clf.predict_proba(selectTrain)
tfidf_preds = clf.predict_proba(selectTest)
tfidf_train = pd.DataFrame({ 'tf_idf': tfidf_train[:,1]}, index = parsedData.index)
tfidf_preds = pd.DataFrame({ 'tf_idf': tfidf_preds[:,1]}, index = parsedTest.index)

numX_train = pd.DataFrame(parsedData.iloc[:,:30])
numX_test = pd.DataFrame(parsedTest.iloc[:,:30])

numX_train = pd.concat([numX_train, tfidf_train, pd.Series(y_train[:,0])], axis=1)
numX_test = pd.concat([numX_test, tfidf_preds, pd.Series(y_preds[:,0])], axis=1)

model = Sequential([
    layers.Dense(50, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss="binary_crossentropy",
    metrics=['binary_accuracy']
)
numX_train = numX_train.astype(float)
numX_test = numX_test.astype(float)

model.fit(numX_train.values, awesometrain.iloc[:,0].values, validation_split=0.2, epochs=10000, callbacks = es_callback)

def round(x):
  if x > 0.5:
    return 1
  return 0

results = model.predict(numX_test)
results = np.array(list(map(round, results)))

results = pd.DataFrame(np.transpose([asins, results]), columns = ['asin', 'predictions'])

with open('results.json', 'w') as f:
  f.write(results.to_json())
