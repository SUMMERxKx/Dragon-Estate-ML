# Dragon Real Estate - Price Predictor

## Project Overview

This project focuses on predicting real estate prices using various housing attributes. The dataset consists of 506 entries, with features like crime rate, property tax, and number of rooms. The goal of this project is to use machine learning models to predict house prices based on these attributes.

## Dataset

The dataset consists of 506 instances and 14 attributes:
- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxide concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents
- **LSTAT**: Lower status of the population (percentage)
- **MEDV**: Median value of owner-occupied homes in $1000s (Target Variable)

## Project Steps

### 1. Data Loading and Exploration
- **Pandas** is used to load the dataset.
- The first few rows of data are viewed using `housing.head()`.
- Basic statistics and data types are checked using `housing.info()` and `housing.describe()`.

### 2. Data Splitting
- The dataset is split into training and testing sets using **Scikit-learn’s** `train_test_split` function with an 80-20 split.
- Stratified sampling is performed based on the `CHAS` variable using `StratifiedShuffleSplit`.

### 3. Feature Engineering
- Correlation analysis is conducted to find attributes highly correlated with the target (`MEDV`).
- New features are created, such as the **TAXRM** feature (TAX divided by RM), which is added to explore its relationship with housing prices.

### 4. Handling Missing Data
- Missing values in the `RM` (average number of rooms) column are filled using the median.
- **SimpleImputer** from **Scikit-learn** is used to handle missing values.

### 5. Feature Scaling
- **StandardScaler** is used to scale the data to ensure features have a mean of 0 and a unit variance.

### 6. Model Selection and Training
- A **Linear Regression** model is selected and trained using the preprocessed data.
- The pipeline involves data imputation and scaling, followed by training the model on the processed data.

### 7. Model Evaluation
- The model's performance is evaluated using **Root Mean Squared Error (RMSE)**.
- Predictions are compared to the actual labels for a sample of instances.

## Key Results
The model predicts housing prices reasonably well with the following RMSE on the training data:
- **Linear Regression RMSE**: 23.32

## Tools and Libraries
- **Pandas**: For data manipulation and exploration.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting graphs and visualizations.
- **Scikit-learn**: For model training, data preprocessing, and evaluation.
  
## Code Structure
- `housing = pd.read_csv("data.csv")`: Loads the dataset.
- `train_test_split`: Splits the data into training and test sets.
- `SimpleImputer`: Handles missing data by filling in the median values.
- `StandardScaler`: Scales features to a standardized range.
- `LinearRegression`: Implements the machine learning model to predict house prices.
- `mean_squared_error`: Evaluates the model’s performance using the RMSE.

## Future Improvements
- **Feature Selection**: Experiment with adding or removing features to improve model accuracy.
- **Model Tuning**: Try different machine learning algorithms such as Decision Trees or Random Forests.
- **Cross-Validation**: Implement cross-validation to better generalize the model’s performance on unseen data.

## Setup
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/dragon-real-estate.git
