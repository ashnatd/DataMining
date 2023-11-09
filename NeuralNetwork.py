import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

x, y = sklearn.datasets.make_moons(200)
plt.scatter(x[:, 0], x[:, 1], c=y)
# plt.show()
x.shape
input_neurons = 2
output_neurons = 2
samples = x.shape[0]
learning_rate = 0.001
lambda_reg = 0.001

dict1 = {
    "w1": 0,
    "b1": 0,
    "w2": 0,
    "b2": 0,
}


def retrieve(dict1):
    w1 = dict1["w1"]
    b1 = dict1["b1"]
    w2 = dict1["w2"]
    b2 = dict1["b2"]
    return w1, b1, w2, b2


def forward(x, dict1):
    w1, b1, w2, b2 = retrieve(dict1)
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    a2 = np.tanh(z2)
    exp_score = np.exp(a2)
    softmax = exp_score / np.sum(exp_score, axis=1, keepdims=True)
    return z1, a1, softmax


def loss(softmax, y, dict1):
    w1, b1, w2, b2 = retrieve(dict1)
    m = np.zeros(200)
    for i, correct_index in enumerate(y):
        predicted = softmax[i][correct_index]
        m[i] = predicted
    log_prob = -np.log(m)
    loss = np.sum(log_prob)
    reg_loss = lambda_reg / 2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    loss += reg_loss
    return float(loss / y.shape[0])


def predict(dict1, x):
    w1, b1, w2, b2 = retrieve(dict1)
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (200,2)
    return np.argmax(softmax, axis=1)  # (200,)


def backpropagation(x, y, dict1, epochs):
    for i in range(epochs):
        w1, b1, w2, b2 = retrieve(dict1)
        z1, a1, probs = forward(x, dict1)  # Unpack the values correctly
        # Rest of your code...
        # a1: (200,3), probs: (200,2)
        delta3 = np.copy(probs)
        delta3[range(x.shape[0]), y] -= 1  # (200,2)
        dW2 = (a1.T).dot(delta3)  # (3,2)
        db2 = np.sum(delta3, axis=0, keepdims=True)  # (1,2)
        delta2 = delta3.dot(w2.T) * (1 - np.power(np.tanh(z1), 2))
        dW1 = np.dot(x.T, delta2)
        db1 = np.sum(delta2, axis=0)
        # Add regularization terms
        dW2 += lambda_reg * np.sum(w2)
        dW1 += lambda_reg * np.sum(w1)
        # Update Weights: W = W + (-lr*gradient) = W - lr*gradient
        w1 += -learning_rate * dW1
        b1 += -learning_rate * db1
        w2 += -learning_rate * dW2
        b2 += -learning_rate * db2
        # Update the model dictionary
        dict1 = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        # Print the loss every 50 epochs
        if i % 50 == 0:
            print("Loss at epoch {} is: {:.3f}".format(i, loss(probs, y, dict1)))

    return dict1


def init_network(input_dim, hidden_dim, output_dim):
    model = {}
    # Xavier Initialization
    w1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
    b1 = np.zeros((1, hidden_dim))
    w2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
    b2 = np.zeros((1, output_dim))
    model["w1"] = w1
    model["b1"] = b1
    model["w2"] = w2
    model["b2"] = b2
    return model


def plot_decision_boundary(predict_func):
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = predict_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors="k", cmap=plt.cm.Spectral)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()


x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = predict(np.c_[xx.ravel(), yy.ravel()])
Z = predict(dict1, np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("Decision Boundary for hidden layer size 3")

dict1 = init_network(input_dim=input_neurons, hidden_dim=3, output_dim=output_neurons)
model = backpropagation(x, y, dict1, 1000)
# plot_decision_boundary(lambda x: predict(model, x))
plot_decision_boundary(lambda x: predict(dict1, x))