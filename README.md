# House-Price-Predict
## CX4240 Project
## Team Members: Yandong Luo, Xiaochen Peng, Panni Wang, Hongwu Jiang


# 2. Feature Selection

When the dataset comes to be high dimensional, it could lead to many problems. Firstly, such high dimension will significantly increase the training time of the model, and dramatically increase the complexity of the model. Secondly, the model may make decisions based on the noise, and thus, cause overfitting. What's more, the meaningless redundent data will be very misleading, and could decrease the accuracy.
To achieve better performance, we firstly used two feature selection methods to cancel out the redundent features, one of the methods is to draw the correlation matrix with heatmap, the second method is to calculate the feature importance and select the top-N features according to the feature ranking.

### (1). Dataset Analysis

To better understand the relationship among the features, we have drawn the pairplots for some features, the features we selected are those we considered as very important as our first thought. The pairplots shows how "bathrooms", "bedrooms" and "sqft_living" are distributed vis-a-vis the price as well as the "grade", which means the grading of the houses by the local county. As the pairplot shown below, we could find some linear distribution between price and the features, which could be useful in our linear model.

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Feature_Plot.png)

Moreover, we draw the distribution figures of all the features as shown below, to get a feel about which features are continus, and which are not.

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
As the figures shown below, the first three linear models returned same feature ranking results,

<img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_LR.PNG" width="250"/> <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_Ridge.PNG" width="250"/><img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_Lasso.PNG" width="250"/>

