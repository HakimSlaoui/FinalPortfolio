
# Machine Learning and Recommender System Projects

This repository contains three projects demonstrating fundamental machine learning concepts and techniques: **MinHashing**, **Linear Regression**, and **Collaborative Filtering**.

## Projects Overview

### 1. **MinHashing for Set Similarity**

MinHashing is an efficient technique for estimating the Jaccard similarity between sets. This implementation shows how to generate MinHash signatures for sets of data and estimate their similarity using multiple hash functions.

#### Key Concepts:
- **MinHash Signatures**: Compresses large sets into smaller, fixed-size signature vectors.
- **Jaccard Similarity**: A measure of similarity between sets, based on the proportion of shared elements.

#### Code Walkthrough:
- Data is represented as sets.
- Multiple hash functions are applied to compute the MinHash signature for each set.
- The similarity between sets is estimated by comparing the MinHash signatures.

---

### 2. **Linear Regression**

This project implements **Linear Regression** using **Gradient Descent**. It demonstrates the process of fitting a line to data points, optimizing the parameters (weights) to minimize the Mean Squared Error (MSE).

#### Key Concepts:
- **Linear Regression**: A statistical method used for predicting a target variable based on one or more features.
- **Gradient Descent**: An optimization algorithm used to minimize the cost function by updating the model parameters.
- **Mean Squared Error (MSE)**: A measure of the average squared difference between the observed actual outcomes and the predicted values.

#### Code Walkthrough:
- A synthetic dataset is generated with a linear relationship between the feature and target.
- The linear regression model is trained using gradient descent to fit a line to the data.
- The model’s performance is evaluated using **MSE** and **R² score**.

---

### 3. **User-Based Collaborative Filtering**

This project demonstrates a **Collaborative Filtering** approach for recommending items to users. We use the **User-Based Collaborative Filtering** method, which suggests items based on the preferences of similar users.

#### Key Concepts:
- **Collaborative Filtering**: A method of making recommendations based on the past interactions of users with items.
- **Cosine Similarity**: A metric used to compute similarity between users based on their ratings.
- **Rating Prediction**: Ratings are predicted using weighted averages of ratings from similar users.
- **Recommendation**: Unrated items are recommended to a user based on predicted ratings.

#### Code Walkthrough:
- A user-item matrix is created, where rows represent users, columns represent items, and values represent ratings.
- **Cosine Similarity** is computed between users based on their ratings.
- A prediction model is built to estimate ratings for items a user hasn’t rated.
- Finally, items are recommended to users based on predicted ratings for unrated items.

---

## Dependencies

The following libraries are required to run these projects:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib scikit-learn
```
---

## Running the Projects

To run each project, simply execute the corresponding Python script in your terminal or IDE.

- MinHashing: ```minhashing.py```

- Linear Regression: ```linear_regression.py```

- Collaborative Filtering: ```collaborative_filtering.py```

---

## Conclusion

These projects provide a hands-on introduction to fundamental techniques in machine learning and recommendation systems. Each implementation is designed to be simple yet illustrative of key concepts like similarity estimation, regression, and collaborative filtering.
