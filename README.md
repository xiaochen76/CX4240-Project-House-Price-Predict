# House-Price-Predict
## CX4240 Project
## Team Members: Yandong Luo, Xiaochen Peng, Panni Wang, Hongwu Jiang


# 2. Feature Selection

When the dataset comes to be high dimensional, it could lead to many problems. Firstly, such high dimension will significantly increase the training time of the model, and dramatically increase the complexity of the model. Secondly, the model may make decisions based on the noise, and thus, cause overfitting. What's more, the meaningless redundent data will be very misleading, and could decrease the accuracy.
To achieve better performance, we firstly used two feature selection methods to cancel out the redundent features, one of the methods is to draw the correlation matrix with heatmap, the second method is to calculate the feature importance and select the top-N features according to the feature ranking.

### (1). Dataset Analysis

To better understand the relationship among the features, we have drawn the pairplots for some features, the features we selected are those we considered as very important as our first thought. The pairplots shows how "bathrooms", "bedrooms" and "sqft_living" are distributed vis-a-vis the price as well as the "grade", which means the grading of the houses by the local county. As the pairplot shown below, we could find some linear distribution between price and the features, which could be useful in our linear model.

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Feature_Plot.png)

Moreover, we draw the distribution figures of all the features as shown below, to get a feel about which features are continuous, and which are not.

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Feature_Dist.PNG)

### (2). Correlation Heat Map

To find out how each feature is correlated to our target variable "price", we have drawn the correation heat map of all the features as shown below:

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/HeatMap_ALL.PNG)

It shows that, the correlation value can be positive or negative, when the correlation is positive, it means the increasing of value in such feature will cause the target variable "price" to increase, and vice versa.

We have further drawn the most significant correlated features with target variable "price", as shown below:

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/HeatMap_select.PNG)

It is very easy to identify which features are most related to the target variable, as it shown above, the most important features from the heatmap are: bathrooms, bedrooms, floors, grade, sqft_above, sqft_living and sqft_living15.

### (3). Feature Ranking

Of course, using the corelation heap map itself, could be not representive enough, thus, we further used the feature ranking functions in each models and get the mean ranking values, to select the top-N important ones.
We have implemented five representative models to get their scores about the features, and get the mean values of them, which are linear regression, ridge, lasso, recursive feature elimination and random forest model.
As the figures shown below, the first three linear models returned same feature ranking results, the most important features are: grade, view, bathrooms, bedrooms, yr_renovated, floors and conditions.

<img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_LR.PNG" width="280"/> <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_Ridge.PNG" width="280"/><img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_Lasso.PNG" width="280"/>

However, in recursive feature elimination, we get a quite different feature ranking as shown below:

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_RFE.PNG)

While in random forest model, the feature ranking is give as below:

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_RF.PNG)

We could find that, in recursive feature elimination and random forest feature ranking, there are more features as continuous data on the top ranks, such as the area of living rooms and lots (sqft_living and sqft_lot). While in the linear model, the top ranked features are discrete data.

To get a more balanced feature ranking, we normalized the scores from each model and get the mean values, the final feature ranking is shown as below:

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_Average.PNG)

Where the top ranked features include both continuous and discrete data: sqft_living, sqft_lot as continuous data, grade, view, bedrooms and bathrooms are discrete data.


# 3. Linear Regression and Polynomial Regression

<img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Price_Bathrooms.PNG" width="280"/> <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Price_Bedrooms.PNG" width="280"/><img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Price_SqLiving.PNG" width="280"/>

As what we have shown in the dataset analysis, some features shows a classical linear relationship, while some do not have very good linear form, thus, we use both of the linear and polynomial regression, to study how the data distribution affect the linear models.

After data pre-processing as we mentioned above, there are 14 features left in the dataset, which is divided in to training and testing set. With linear regression, when the number of features is too low, it could suffer from under-fitting, and get poor performance in both training and testing, while if the features are too many, it is also possible to get over-fitting problem. To learn about the number of features in the linear regression, we have run linear and polynomial regression, based on three models (linear regression, ridge regression and lasso regression), with all the 14 features and only top-10 important features. 

In ridge regression, it shrinks the coefficients (w) by putting constraint on them, and thus, helps to reduce the model complexity and multi-collinearity. Similarly, in lasso regression, the regularization will lead to zero coefficients, which means some of the features are completely neglected for the evaluation of output, thus, lasso regression not only helps in reducing over-fitting, but also helps in feature selection.

The reason why we used three linear models is, the ridge and lasso regression are some of the simple techniques to reduce model complexity and prevent over-fitting which may result from simple linear regression. By comparing the linear regression, ridge regression and lasso regression, we can also get an insight about how the number of features affect the model performance.

### (1). ALL Features Included

As the figure shown below, where red line is the real price value, and the blue dots are the predicted price value, the first row shows the linear, lasso and ridge regression without polynomial, the second row shows when polynomial in introduced with degree equals to 2, and the third is with degree equals to 3. It shows that, with polynomial, the prediction achieves better performance, since it can help to fit in non-linear features.

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Predict_All.PNG)

### (2). Selected Top-10 Features Included

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Predict.PNG)

The figure shown above is the relation between real price and predicted price, when we only introduced the top-10 important features. The first column shows the linear, ridge and lasso regression, and the second column shows the ones with polynomial (degree is set to 2). Similarly as what we have found in the "all features included" method, the linear regression achieves best performance among all the three linear models.

### (3). Comparison and Discussion

It is clearly shown that, no matter in the "all features included" or "selected top-10 features", the linear regression achieves the best performance. While in each method, the ones with polynomial achieves better prediction.

To find out the comparison among all the methods, we have printed out the rmse values of them, as the figure shown below.

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/RMSE.PNG)







