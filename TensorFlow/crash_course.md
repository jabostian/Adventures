

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

###Sessions
Here my notes on the course sessions/chapters.

#### Framing
**Regression model** predicts continuous values
**Classification model** predicts discrete values

#### Descending into ML
When talking about a simple linear regression, might use _**y = mx + b**_.  in
Machine learning it's _**y = wx + b**_
- **w** represents weights instead of slope
- **b** represents bias instead of Y intercept

**Loss** is the error between prediction and actual

![Home prices](./crash_course/images/house_price.png)

Useful loss function is squared error:
![Squared error](./crash_course/images/squared_error.png)

When training a model, want to minimize loss across all training examples:
![L2 loss](./crash_course/images/L2_loss.png)

Goal is to minimize loss by finding the right set of weights and biases
on average, across all examples:
![Model loss](./crash_course/images/model_loss.png)




### References
