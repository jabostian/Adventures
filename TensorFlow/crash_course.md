

### Intro
My experiences with the TensorFlow crash course.  I've collected the python code
for this class in the code directory of this repo.  There are
_**Jupyter Notebooks**_ for these as well in the notebooks directory.

https://developers.google.com/machine-learning/crash-course/

From an
[Infoworld article](https://www.infoworld.com/article/3278008/machine-learning/what-is-tensorflow-the-machine-learning-library-explained.html)
I found:

_**TensorFlow allows developers to create dataflow graphs—structures that
describe how data moves through a graph, or a series of processing nodes. Each
node in the graph represents a mathematical operation, and each connection or
edge between nodes is a multidimensional data array, or tensor.**_

### Sessions
Here my notes on the course sessions/chapters.

-----

#### Framing
**Regression model** predicts continuous values
**Classification model** predicts discrete values

-----

#### Descending into ML
When talking about a simple linear regression, might use _**y = mx + b**_.  in
Machine learning it's _**y = wx + b**_
- **w** represents weights instead of slope
- **b** represents bias instead of Y intercept

**Loss** is the error between prediction and actual

<br>
![Home prices](./crash_course/images/house_price.png)

Useful loss function is squared error:
<br>
![Squared error](./crash_course/images/squared_error.png)

When training a model, want to minimize loss across all training examples:
<br>
![L2 loss](./crash_course/images/L2_loss.png)

Goal is to minimize loss by finding the right set of weights and biases
on average, across all examples:
<br>
![Model loss](./crash_course/images/model_loss.png)

**Mean Squared Error (MSE)** id the average squared loss per example over the
whole dataset.
<br>
![Mean Squared Error](./crash_course/images/MSE.png)

MSE is commonly used in ML, but it isn't the only practical, or even the best
loss function in all cases.

-----

#### Reducing Loss
**Hyperparameters** - config settings use to tune the model during training

The derivative of the loss function (L2 loss here) with respect to the weights
and biases tells us how the loss changes for a given example
- Simple to compute, and is a convex function

Repeatedly take small steps in the direction that minimizes loss
- Negative gradient steps
- This strategy is **gradient descent**

![Gradient Descent](./crash_course/images/gradient_descent.png)

**Learning rate** is a hyperparameter that determines how large of a step to
take in the direction of the negative gradient to minimize loss.
- This strategy is **gradient descent**

![Learning rate](./crash_course/images/learning_rate.png)

If choose too large of a step, the model can diverge, and never minimize the
loss function.  In this case, reduce the step size by a large value and re-try.


-----

### References
