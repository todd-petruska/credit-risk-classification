# credit-risk-classification

## Overview of the Analysis

The purpose of this project is to use various techniques to train and evaluate a model based on loan risk by using a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

The dataset is comprised of loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks, total debt, and loan status to identify healthy loans/high-risk loans. 
Value counts provides the number of healthy loans, which is 75,036 and 2,500 high-risk loans. The loan status column was used for y, then the features for “X” DataFrame were created from the remaining columns.

The data was split into training and testing datasets by using train_test_split from sklearn.model and use of Logistic Regression model with ‘lbfgs’ solver and a random state of one. The predictions on the testing data labels were saved by using the testing feature data X_test and the fitted model, then evaluated to calculate the accuracy score of the model, generate a confusion matrix, and print the classification report.

The resampled training data was used to predict the logistic regression model.  The Random Over Sampler module from the imbalanced-learn library was used to resample the data and the Logistic Regression classifier with the resampled data was used to fit the model and make predictions. The model model’s performance was evaluated by calculating the accuracy score of the model, generating a confusion matrix, and printing the classification report.

## Results

Machine Learning Model 1:
* The first model classifier results: 
* Training Data Score: 99.15%
* Testing Data Score: 99.24%

* Healthy Loans:  100% (precision), 100% (recall), 100% (f1-score)
* High-Risk Loans: 87% (precision), 89% (recall), 88% (f1-score)
* Accuracy: 99%

Machine Learning Model 2:
* Balanced Accuracy Score: 99.60%
* Healthy Loans:  100% (precision), 100% (recall), 100% (f1-score)
* High-Risk Loans: 87% (precision), 100% (recall), 93% (f1-score)
* Accuracy: 100%

## Summary
The findings revealed the 2nd model’s use of Logistic Regression model with resampled training data provides a higher degree of accuracy in identifying high-risk loans f1-score by 5% with minimal impact to model’s ability to accurately identify healthy loans. 

The 2nd model is more accurate in identifying high-risk loans with greater accuracy and only errored on 2 out of 623 events, which is an improvement over the 1st model that errored 67 out of 558 events.  The 1st model is more accurate for healthy loans and errored 80 out of 18,679 events, whereas the 2nd model errored 91 out 18, 668 events.  The 2nd model is recommended, since the loss in accuracy for healthy loans is negligible when compared to the improved accuracy gained in detecting high-risk loans. 
