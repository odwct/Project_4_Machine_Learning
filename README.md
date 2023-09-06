# Project_4_Machine_Learning

# Predicting Earthquakes Using Machine Learning

Earthquakes are natural disasters that can result in significant loss of life and property damage. The ability to forecast earthquakes has the potential to save lives and minimize damage by enabling early warning systems and preparedness measures.

## Introduction
The goal is to discover patterns and relationships within seismic data that may contribute to earthquake occurrence. The project will focus on creating a predictive model to improve seismic activity understanding.

## Challenges
- Feature Selection:  In earthquake prediction, finding the most relevant features or variables is challenging.

- Temporal and Spatial Dynamics: Geological conditions and tectonic plates interact to influence earthquake occurrence. It is challenging to understand and model these dynamics.

- Modeling Uncertainty: Prediction of earthquakes involves uncertainty. Keeping this uncertainty in mind is key to avoiding false alarms.

- Real-Time Prediction: To provide timely warnings and responses, the model should provide real-time or near-real-time predictions.

Addressing these challenges and complexities requires a multidisciplinary approach that combines expertise in geophysics, data science, and machine learning.

## Data Collection and Preprocessing
For the predictive earthquake model, I used data from The U.S. Geological Survey spanning an entire month. For the predictive analysis, I rely on this comprehensive dataset.

•	Latitud and Longitude 
•	Depth 
•	Magnitude
•	Nst (number of seismic stations), Gap (gap between adjacent stations (in degrees)), Dmin (Horizontal distance from the epicenter to the nearest station), and RMS (root-mean-square (RMS) travel time residual, in sec)
•	Magnitude Type


## Preprocessing Data
### Preprocessing Steps:
•	Filtering by Event Type: Focus on earthquakes only for better predictions.
•	Handling Missing Values: Deleting missing data in the dataset to prevent issues during model training and evaluation.
•	Data Cleaning:  Remotion of non-numeric values from the dataset for consistent and valid data points.

## Exploratory Data Analysis (EDA)
The Exploratory Data Analysis (EDA) yielded significant insights when comparing earthquake magnitude with various factors. Here are the key findings for each comparison:

•	Magnitude vs. Latitude: 
Correlation coefficient: -0.63. 
This suggests a moderately strong negative correlation, indicating that as we move towards higher latitudes, earthquakes tend to be stronger.
•	Magnitude vs. Longitude: 
Correlation coefficient: 0.61. 
This reveals a moderately strong positive correlation, implying that earthquakes are more severe in regions with certain longitudes.
•	Magnitude vs. Depth: 
Correlation coefficient: 0.43. 
This indicates a moderate positive correlation, implying     that deeper earthquakes tend to be more intense.
•	Magnitude vs. NST (number of reporting stations):
Correlation coefficient: 0.54. 
This suggests a moderately strong positive correlation, indicating that earthquakes with more reporting stations tend to have higher magnitudes.
•	Magnitude vs. Gap: 
Correlation coefficient: -0.05. 
This indicates a very weak negative correlation, suggesting that the gap between reporting stations has little impact on earthquake magnitude.

## Feature Selection
To enhance the model's performance, several feature engineering techniques were implemented:
•	One-Hot Encoding: Conversion of categorical variables, such as 'MagType,' into a numerical format using one-hot encoding. 
•	Data Split: Division of the dataset into three subsets - training, validation, and test sets.
•	Standardization: This helps with faster and more accurate training.

## Model selection:

For the model selection, we considered three different models: Linear Regression, TensorFlow Neural Network, and MLPRegressor. Each model was evaluated using key performance metrics, including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).

Here are the performance metrics for each model:

•	Linear Regression Model:
Mean Squared Error: 0.38
Root Mean Squared Error: 0.62
Mean Absolute Error: 0.49
R-squared: 0.80

•	TensorFlow Neural Network:
Mean Squared Error: 0.46
Root Mean Squared Error: 0.68
Mean Absolute Error: 0.56
R-squared: 0.77

•	MLPRegressor Model:
Mean Squared Error: 0.19
Root Mean Squared Error: 0.44
Mean Absolute Error: 0.33
R-squared: 0.90

Based on these metrics, the MLPRegressor model outperformed the other models in terms of lower MSE, RMSE, and MAE, as well as a higher R-squared value. Therefore, I selected the MLPRegressor as the best-performing model for earthquake prediction.

## MLPRegressor Model:
The model performed exceptionally well in both validation and test data sets. Low MSE, RMSE, and MAE values imply the model is closely aligned with earthquake magnitudes. R-squared values of 0.91 indicate the model explains a significant portion of earthquake magnitude variation. Models like this have strong predictive power.

## Visualizations
This analysis utilized two key visualizations: scatter plots and residual plots, which were applied to both the validation and test datasets.

Scatter plot (Validation & Test Data):
Scatter plots show the correlation between anticipated and real values. Close alignment between the dots and the anticipated line indicates accurate earthquake predictions.

![test_scatter_plot](https://github.com/odwct/Project_4_Machine_Learning/assets/126130532/87a14268-b0c8-4034-a0c3-f2368590f14d)
![res_test_scatter_plot](https://github.com/odwct/Project_4_Machine_Learning/assets/126130532/3e008c15-0527-4a52-bade-5e76adefba9f)

Residual plot (Validation & Test Data):
Residual plots show error distribution in predictions. Our validation and test data both have "bell-shaped" plots, indicating consistent errors and stable performance across scenarios.

![validation_scatter_plot](https://github.com/odwct/Project_4_Machine_Learning/assets/126130532/0a96e3b1-28f0-4a2c-823a-13f9b147077c)
![res_val_scatter_plot](https://github.com/odwct/Project_4_Machine_Learning/assets/126130532/2489362f-ff4d-4ba2-b478-ae5e1188ad4a)

## Map of predictions
The map visually represents the geographical locations of actual earthquakes (ground truth data) and the predicted earthquake locations.

![Map_plot](https://github.com/odwct/Project_4_Machine_Learning/assets/126130532/690e749c-abe6-4d7f-a178-84599d267fba)

## Conclusion
The earthquake prediction model uses machine learning technology to enhance early warning systems and risk assessments. The precision-focused approach ensures good accuracy and reliability. This work highlights the potential of machine learning in addressing real-world challenges and emphasizes ongoing research into earthquake prediction and response.

## References

ANSS Comprehensive Earthquake Catalog (COMCAT) documentation. U.S. Geological Survey. (n.d.). "https://earthquake.usgs.gov/data/comcat/index.php#nst"

Machine learning: Trying to predict a numerical value (Ronaghan). "https://srnghn.medium.com/machine-learning-trying-to-predict-a-numerical-value-8aafb9ad4d36"

Jupyter Notebook Viewer. Notebook on nbviewer. "https://nbviewer.org/github/srnghn/ml_example_notebooks/blob/master/Predicting%20Wine%20Types%20with%20Neural%20Networks.ipynb"

scikit. sklearn.neural_network.MLPClassifier. "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier"


Google Colab. Google Colaboratory. "https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb#scrollTo=qsqenuPnCaXO"

OpenAI. (2023). ChatGPT (August 3 Version) [Large language model]. "https://chat.openai.com"



