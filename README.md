# Sakar's Python-Fun repo
Hello, my name is Sergei Dulikov.
This is my repo for experementing with Python for Data Analysis, Artificial Intelligence and Machine Learning.  
Currently available projects:
- 2048
- folium
- GalaxyZoo
- MLP_from_scratch
- MNIST
- Music_Recommender
- TASD
- titanic

##2048
Artificial Intelligence based on Monte-Carlo approach, aiming to score best result in the game of 2048. Both the game itself and the algorithm were implemented.
Based on: https://habr.com/ru/company/edison/blog/437118/

## folium
Project for studying python folium module for advanced plotting on maps.

## GalaxyZoo
Attempt to replicate winner's solution for Kaggle comptetion of Galaxy Zoo: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge  
Complex convolution net was built with Keras and TensorFlow.
Based on: http://benanne.github.io/2014/04/05/galaxy-zoo.html  

## MLP_from_scratch
Implemented a Multi-Layer Perceptron using only numpy library. Allows use of several activation functions, any number of dense layers with any number of perceptrons in each. Two types of loss-functions: MSE and CrossEntropy were also implemented along with the Backpropagation algorithm.

## MNIST
Attempt to approach famous task of handwritten digits recognition with MNIST dataset: http://yann.lecun.com/exdb/mnist
Used Scikit-Learn MLPClassifier and image preprocessing: cropping and deskewing.
Achieved 0.98957 accuracy on https://www.kaggle.com/c/digit-recognizer/

## Music_Recommender
Music recomendational system based on the last.fm dataset: http://files.grouplens.org/datasets/hetrec2011/ 
The main idea is use of probabilistic matrix factorization. Project implemented with PyTorch python library.
Based on: https://towardsdatascience.com/building-a-music-recommendation-engine-with-probabilistic-matrix-factorization-in-pytorch-7d2934067d4a

## TASD
Project for studying different approaches to detemine proportions of different sources of air showers in the data from Telescope Array Surface Detector.
Used XGBoost and different architectures of dense neural nets built with Keras and TensorFlow.
Project was done as an exam task for Lomonosov Moscow State University course of Machine Learning.

## titanic
Project for famous Kaggle competion Titanic: Machine Learning from Disaster https://www.kaggle.com/c/titanic/
Tried various algorithms including: Scikit-Learn MLPClassifier with different architectures, RandomForest, XGBoost, CatBoost. Best result was achieved with CatBoost:
accuracy of 0.82296, which is currently #401 on Kaggle leaderboard.
  