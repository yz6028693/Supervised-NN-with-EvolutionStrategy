In this demo, I will classify the same Archimedean Spiral pattern based on two categories of points we inputted
in the Neural Network. For each point, 7 value of this point are inputted in the NN: x, y, x^2, y^2, xy, sin(x) and sin(y).
This time, rather than use TensorFlow and Stochastic gradient descent (SGD), I will create this NN by myself and use
Evolution Strategy to train this neural network.

I haven't used multiprocessing in this version, the training speed and result can already versus those in my TensorFlow
and SGD demo.

The demo looks like this:

<a><img src="Gifs&Images/SupervisedES.gif"></a>
