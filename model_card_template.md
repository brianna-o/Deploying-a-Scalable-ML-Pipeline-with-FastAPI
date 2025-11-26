# Model Card


## Model Details
- This model predicts whether an individual's annual income exceeds \$50,000 based on demographic and employment-related attributes from the UCI Census Income dataset.
- The model is a RandomForestClassifier, trained using scikit-learn as part of the "Deploying a Scalable ML Pipeline with FastAPI" project.
- The model follows an end-to-end MLOps workflow including preprocessing, training, evaluation, and deployment.
- The final model artifacts  model.pkl, encoder.pkl, and lb.pkl  are required to run inference. 

## Intended Use
- Intended for personal educational use to demonstrate competency in deploying a machine learning model with FastAPI.
- Not intended for use outside of education.

## Training Data
- The model was trained on the Census Income Dataset from the U.S. Census Bureau.
- The datset contains 32,561 rows and 15 columns/features.
- The columns are age,workclass,fnlgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours.
- The target variable ,salary, is divided between less than or equal to 50K and greater than 50K.
- The data was split with 80% of the rows going into the training set and 20% going into the test set. 

## Evaluation Data
- The model was evaluated on the 20% held-out test set using the following classification metrics:

	- Precision: The proportion of positive predictions that were correct 
 	  Model Precision: 0.7353

	- Recall: The proportion of actual positives that were correctly identified 
  	  Model Recall: 0.6378

	- F1 Score: The harmonic mean of precision and recall.
          Model F1 Score: 0.6831

## Metrics
- The model’s performance was calculated using the precision, recall, and F1 score as the metrics.
- The calculation describes  predictive accuracy and balance between false positives and false negatives.

- The models performace:
	Precision: 0.7353 
	Recall: 0.6378
	F1 Score: 0.6831

## Ethical Considerations
- The dataset should not be used for real decision making as the results could create bias since sex, race, and  marital status are involved.

## Caveats and Recommendations
- Fairness checks were not done.
- The data was extracted in 1994 so the results may differ from if the data was more recent. 
- I would recommend that fairness checks be performed on the different demographic groups. 


