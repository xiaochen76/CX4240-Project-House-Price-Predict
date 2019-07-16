# House-Price-Predict
## CX4240 Project
## Team Members: Yandong Luo, Xiaochen Peng, Panni Wang, Hongwu Jiang


# 2. Feature Selection

When the dataset comes to be high dimensional, it could lead to many problems. Firstly, such high dimension will significantly increase the training time of the model, and dramatically increase the complexity of the model. Secondly, the model may make decisions based on the noise, and thus, cause overfitting. What's more, the meaningless redundent data will be very misleading, and could decrease the accuracy.
To achieve better performance, we firstly used two feature selection methods to cancel out the redundent features, one of the methods is to draw the correlation matrix with heatmap, the second method is to calculate the feature importance and select the top-N features according to the feature ranking.

### (1). Correlation Heat Map

To find out how each feature is correlated to our target variable "price", we have drawn the correation heat map of all the features as shown below:

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/HeatMap_ALL.PNG)

It shows that, the correlation value can be positive or negative, when the correlation is positive, it means the increasing of value in such feature will cause the target variable "price" to increase, and vice versa.

We have further drawn the most significant correlated features with target variable "price", as shown below:

![Image](https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/HeatMap_select.PNG)

