# Spotify Models

This repo contains the notebook that creates models for the [spotify 1M track info](./data/spotify_1M.csv.zip) based of [Kaggle](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks)
This is a continuation of the earlier [Spotify data analysis and EDA](https://github.com/dutta-sh/spotify-analysis)

We continue where we left off, by doing a quick recap of the EDA and fixing some assumptions we made earlier.

The notebook is present in this [github repo](./fixme.ipynb) as well as on [Google Colab](https://colab.research.google.com/drive/1IFE7Hq_J26Ge5zMrbBDH5YzP9Dyv9gkX?usp=sharing).

## Problem Statement

The goal here is to compare different ML models by training and testing them on the spotify data and determine which can be a better fit for predicting future outcomes.
- Since we are predicting the popularity of a track, it is going to be a regression problem
- Various supervised learning algorithms will be used to evaluate the right choice of model

## Technical Description

- set up clean data based on the EDA from the [earlier analysis](https://github.com/dutta-sh/spotify-analysis/blob/main/Capstone_EDA.ipynb)
- use the data to train the baseline model
- use the same data to train various models and compare how they perform
- look at coefficients and features that drive the decision 
- come to a quantifiable solution

## Steps

### Revisit the EDA and determine data set

- for building the code and due to CPU/GPU constraints splitting the 1M rows is necessary
  - split the data into 1K, 10K, 40K, and 100K rows and build the initial codebase
  - apply the EDA from earlier and convert `year` to `age`
  - doing some research I realized that `key, mode, time_signature and genre` are the categorical features that can be used for the model analysis
    - earlier I had picked only `genre`. Lets keep these as well
  - analyzing the possible categorical columns we find the following
  
    | Feature          | Count |
    |------------------|-------|
    | genre            | 82    |
    | key              | 12    |
    | mode             | 2     |
    | time_signature   | 5     |
        
    - genre has 82 variations, OHE is going to create a lot of features for that
    - adding this might be too costly on processing without much gain in terms of RMSE
    - 1K data set works fine with genre and all the models below.
    - however 40K rows takes forever to complete. Lets drop "genre" and try

  - finally the dataset to be used has:
    - 40K rows
    - `year` replaced by `age`
    - `track_id` and `Unnamed: 0` columns dropped
    - all numeric and categorical columns except `genre`
    - `popularity` as the target column

### Define models

#### Goal:

- Create baseline models:
  - Use numerical features only
  - Apply on Linear Regression, Ridge and Lasso
- Create advanced models:
  - Use numerical + categorical features
  - Apply on Linear Regression, Ridge, Lasso and KNN
  - Then apply on Gradient Boosting, Random Forest and SVR
- Cross validation models:
  - Apply on Linear Regression, Ridge, Lasso and KNN
  - Then apply on Gradient Boosting, Random Forest and SVR
- Ensemble techniques
  - Voting Regressor
  - Stacking Regressor

#### Steps:

- define data structure and methods to be used during evaluation 
  - this was an iterative process and the notebook has the final state only
  - this helps organize the code and make it readable while reducing the overall LOC
  - methods/data structures that:
    - holds results of models
    - prepare data and preprocessor
    - fit and predict the model and add the score to results
    - cross validate model and add score to results
    - interpret and plot top N coeffs/features
    - print results sorted by RMSE
- data preparation for Numerical columns only
- Create baseline models
- data preparation for Numerical and categorical columns
- Create advanced models
- Cross validation models
- Ensemble techniques

### Interpretation

- The Voting Regressor outperforms every individual model, confirming that ensemble averaging reduced both bias and variance in this task.
  - The Voting Regressor:
    - Averages predictions from:
        - Tree-based models (low bias, nonlinear)
        - Kernel models (smooth generalization)
        - Linear models (stable baseline)
      - Reduces:
        - Overfitting of trees
        - Underfitting of linear models
      - Captures complementary error patterns

- Tree-Based Ensembles (Strongest Individual Models)

    | Model                  | RMSE  | R²    |
    | ---------------------- | ----- | ----- |
    | Random Forest (CV)     | 13.58 | 0.273 |
    | Random Forest          | 13.62 | 0.274 |
    | Gradient Boosting (CV) | 13.64 | 0.267 |
    | Gradient Boosting      | 13.69 | 0.267 |

  - These models capture nonlinear interactions between audio features and popularity
  - CV and non-CV results are very close, indicating stable generalization
  - Trees handle:
    - Genre interactions
    - Nonlinear tempo–energy effects
    - Threshold effects (e.g., loudness, danceability)
  - Tree-based models are the strongest single learners in this dataset.
  
- Kernel & Distance-Based Models (Mid-Tier)

  | Model    | RMSE  | R²    |
  | -------- | ----- | ----- |
  | SVR (CV) | 13.80 | 0.250 |
  | SVR      | 13.83 | 0.253 |
  | KNN (CV) | 14.07 | 0.221 |
  | KNN      | 14.08 | 0.225 |

  - SVR captures smooth nonlinear trends but struggles at scale
  - KNN suffers from:
    - High dimensionality (OHE + audio features)
    - Curse of dimensionality
    - CV slightly lowers R² → mild overfitting in non-CV runs
  - These models help the ensemble but are not optimal alone.

- Linear Models (Weakest)

  | Model                       | RMSE   | R²     |
  | --------------------------- | ------ | ------ |
  | Linear / Ridge / LASSO (CV) | ~14.50 | ~0.172 |
  | Linear / Ridge / LASSO      | ~14.56 | ~0.171 |
  | Numeric-only Linear         | ~14.57 | ~0.170 |

    - Popularity is not a linear function of audio features
    - Genre effects are non-additive
    - LASSO provides sparsity but not better prediction 
    - Numeric-only models confirm that categorical features matter, but linearity limits benefit 
    - Linear models are interpretable baselines, not competitive predictors.

### Summary

Tree-based ensemble models substantially outperform linear and distance-based approaches when predicting Spotify track popularity from audio and categorical features. A Voting Regressor combining heterogeneous learners achieves the best overall performance, demonstrating that ensemble averaging effectively balances bias and variance. However, the modest R² values indicate that popularity is influenced by many external factors beyond intrinsic audio characteristics.
    