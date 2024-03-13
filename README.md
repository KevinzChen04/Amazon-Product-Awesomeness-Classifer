# Amazon Product Awesomeness Classifier

## Project Overview
The Amazon Product Awesomeness Classifier is an AI-powered classifier designed to assess product quality based on Amazon reviews. It leverages machine learning algorithms and natural language processing (NLP) methods to analyze textual data and predict product awesomeness.

## Key Features
1. **Data Collection**:
   - Ran classifier over 700k CDs and Vinyl reviews with over 50k unique products from Amazon.
   - Loaded in dataset using Pandas and Numpy
2. **Data Cleaning and Preprocessing**:
   - Extracted the following features from the non-reviewtext datapoints:
     - % of reviews with no votes, highest # of votes, average # of votes, stdev of all votes.
     - Earliest review time, latest review time, stdev of all review times.
     - % of verified reviews
     - Standardized all data points
   - Created TF-IDF sparse matrix and conducted sentiment analysis using nltk Vader
   - Performed feature selection by analyzing Fischer scores and running recursive feature selection algorithms.
     <div align="center">
        <img src="https://i.imgur.com/ykHVyYy.png" alt="Recursive Feature Selection">
        <br>
        <p>Figure 1: Example Recursive Feature Selection Flowchart</p>
    </div>
   <div align="center">
      <img src="https://i.imgur.com/cQmNj6z.png" alt="Hyperparameter Tuning">
      <br>
      <p>Figure 2: Example Recursive Feature Selection result on Hyperparameters for tuning purposes</p>
  </div>
  
3. **Model Training and Selection**:
   - Employed Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, Random Forest, Decision Tree, and Adaboost on the sentiment analysis histogram and non-reviewtext datapoints.
   - Created a custom voting classifier to compile prediction probabilities from all models.
    <div align="center">
        <br>
        <img src="https://i.imgur.com/lqNItax.png" alt="Feature Selection">
        <br>
        <p>Figure 3: Graph of F1 Score by # of Features Used for Logistic Regression</p>
    </div>

4. **Neural Networks**
   - Started with Perceptrons but the results were subpar and were discarded
   - Integrated Kera's Neural Network and tested with different layers including dense(base), bidirectional, convolution, Long short term Memory NN
    <div align="center">
        <img src="https://i.imgur.com/bhhmPOD.jpeg" alt="Neural Networks">
        <br>
        <p>Figure 4: Compilation of different neural network layer results</p>
    </div>
    
5. **Ensemble Learning**:
   - Compiled the different results from TF-IDF, neural networks, and machine learning
   - Ran a grid search to find the optimal set of weights representing each algorithm 

## Results
  - Achieved a F1-score of .60 using a Keras deep learning model on the sentiment analysis data; was unable to outperform traditional machine learning techniques
   <div align="center">
      <img src="https://i.imgur.com/Rk5AKoF.png" alt="Deep Learning Confusion Matrix">
      <br>
      <p>Figure 5: Confusion Matrix for Deep Learning Results</p>
  </div>
  - Achieved a final F1-score of 0.74, demonstrating the effectiveness of the ensemble approach and custom algorithm development.
   <div align="center">
      <br>
      <img src="https://i.imgur.com/cxveWPa.png" alt="Final Results">
      <br>
      <p>Figure 6: Confusion Matrix of Final Results</p>
  </div>
  
## Conclusion
Though there remains much room for improvement, this project showcases the potential of machine learning in enhancing product quality assessment using customer reviews. 
