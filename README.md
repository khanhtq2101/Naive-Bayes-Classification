# Naive-Bayes-Classification

This project implemeted Naive Bayes Model, one of Machine Learning Algorithms based on probability, as part of my first project. 
Our API follows scikit-learn library, on account of its clarity and simplicity. More detail of scikit-learn API can be found [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes).  

Model for both categorical and continuous features data was implemented. However, combined dataframe has not been supported yet. 

## 1. Mathematics behind Naive Bayes
**The Bayes Theorem:**  
$$P(h|D) = \frac{P(D|h) P(h)}{P(D)}$$  
Where $D$ is observed data of dimension $n$ and $h$ is hypothsis. The objective is finding hypothesis with maximum probability, which derive to Maximum a posterior (MAP).  
$$h_{MAP} = \arg \max_{h} P(D|h) P(h)$$  
However, this condition is strong, especially when the dimension of data is large, $P(D|h)$ is likely to be zero. To handle this problem, Naive Bayes method assumes that the attributes are conditionally independent given hypothesis, formally:
$$P(D|h) = P(d_1, d_2, ..., d_n | h) = \prod_{i = 1}^{n}P(d_i, h)$$
We can rewrite Bayes Theorem with this assumtion:
$$P(h|D) = \frac{P(D|h) P(h)}{P(D)} = \frac{ \prod_{i = 1}^{n} P(d_i, h) P(h)} {P(D)}$$  
And finally, **Naive Bayes Classification**:  
$$h_{MAP} = \arg \max_{h} \prod_{i = 1}^{n}P(d_i, h) P(h)$$  
Above description is apply for categorical data. With the Gaussian Naive Bayes, we need another assumtion, distribution of $P(d_i|h)$ follows normal $\forall i=1...n$.

## 2. Project structure
The python module `model/naivebayes.py` contains model for categorical and numerical data, CategoricalNB and GaussianNB correspondingly. You need to import this module to your main file, in order to run this project. 
We also provide a demonstrated jupyter notebook, which you can follow as a sample to execute.

## 3. Funtions
In this section, we describe some main function of our implementation, both model for categorical and numerical data share this API. 
Since the API follows scikit-learn library, it is composed of two main funciton **fit**(X, y) and **predict**(X).  
- **fit**(X, y): Fit model according to X, y.  
  Parameters:  
     &ensp; X: array of shape (n_samples, n_features)  
     &ensp; y: array of shape (n_samples, )  
  This function will calculate prior probabilities of given data. More detailed, $P(h)$ and $P(d_i|h) \ \forall i=1...n, h \in H$ for CategoricalNB, $P(h)$ and $\mu$ and $\sigma \ \forall i=1...n, h \in H$ for Gaussian.  
- **predict**(X): Based on calculated statistics of data, this function can predict new observation.  
  Parameters:  
  &ensp; X: array of shape (n_samples, n_features)  
  Return:  
  &ensp; C: array of shape (n_samples, ) prediction for X
  
## 4. Experiments and Results
As mentioned above, we provide a demonstrated jupyter notebook as a sample and it is also our experimental execution.  
To evaluate, Naive Bayes of scikit-learn was used as standard. The evaluation was done under three popular dataset, 
[Iris](https://archive.ics.uci.edu/ml/datasets/iris), 
[Breast Cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)) 
and [Wine](https://archive.ics.uci.edu/ml/datasets/wine).  
About overall performance, our model achived 96.67%, 92.1% and 97.22% accuracy score on three datasets respectively.
The agreement ratio (similar predictions devided by total) between our model and scikit-learn are in turn 100%, 96.49% and 100%.
