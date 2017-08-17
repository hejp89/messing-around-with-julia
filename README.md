# Messing around with Julia

Julia examples written for fun and to learn about the language.

## Logistic

This script contains an implementation of logistic regression i.e. fitting a model of the form:

<img src="http://latex2png.com/output//latex_2473c89c96d4533c61e90e64e9f9dbc2.png" alt="Logistic function" width="300"/>

The <a href="https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm">Gauss Newton algorithm</a> is used to calculate the optimial values for the &beta; parameters i.e. the &beta; values the minimise the sum of the squares of the residuals (predicted - observed)^2.

The chart below shows the example data as points and the fitted logistic model curve. The value of the curve at a particular value of x is the probability of x being in class 1, hence whether a value is predicted to be in class 0 or 1 depends on which is the most likely.

<img src="http://www.howardpaget.co.uk/blog/wp-content/uploads/2017/08/logistic-model.png"/>

## Matrix Determinant

This script contains a implementation of calculating the determinant of a matrix using the "by minors" expansion (<a href="http://mathworld.wolfram.com/Determinant.html">http://mathworld.wolfram.com/Determinant.html</a>). 
