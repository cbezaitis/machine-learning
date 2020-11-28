import numpy
import random
import matplotlib.pyplot


def main():
    n = 500
    n_test = 20
    sigma_eta_square = 0.05
    e = 0.001
    a = 1
    b = 1
    theta = numpy.array([[0.2], [-1], [0.9], [0.7], [-0.2]])

    # real curve
    q = numpy.linspace(0, 2, 200).reshape((200, 1))
    w = calculate_y(200, 1, q, theta)

    # training points
    x = numpy.zeros((n, 1))
    for i in range(n):
        x[i, 0] = random.uniform(0, 2)
    x = numpy.sort(x.transpose()[0])
    x = numpy.array([x]).transpose()

    # testing points
    x_test = numpy.zeros((n_test, 1))
    for i in range(n_test):
        x_test[i, 0] = random.uniform(0, 2)
    x_test = numpy.sort(x_test.transpose()[0])
    x_test = numpy.array([x_test]).transpose()

    # calculations
    y = calculate_y(n, 1, x, theta)

    y_noisy = numpy.zeros((n, 1))
    for i in range(n):
        y_noisy[i, 0] = y[i, 0] + random.gauss(0, numpy.sqrt(sigma_eta_square))

    phi = construct_phi(n, x)

    bb = 0
    aa = 0
    mean = numpy.zeros((5, 1))
    while numpy.abs(b - bb) > e and numpy.abs(a - aa) > e:
        aa = a
        bb = b
        s = calculate_sigma(a, b, phi)
        mean = calculate_mean(s, b, phi, y_noisy)
        rox = numpy.linalg.norm(mean)
        sa = numpy.power(rox, 2) + numpy.trace(s)
        temp = numpy.matmul(phi, s)
        aux = numpy.matmul(temp, phi.transpose())
        nox = numpy.linalg.norm(y - numpy.matmul(phi, mean))
        sb = numpy.power(nox, 2) + numpy.trace(aux)
        a = 4/sa
        b = n/sb
        print(1/b)
        print(1/a)

    y_prediction = numpy.array(mean[0, 0] + \
              mean[1, 0] * x_test.transpose() + mean[2, 0] * x_test.transpose() ** 2 + \
              mean[3, 0] * x_test.transpose() ** 3 + mean[4, 0] * x_test.transpose() ** 5)

    t = 1/a
    variance = numpy.zeros((n_test, 1))
    for i in range(n_test):
        variance[i, 0] = t

    # plotting of the curves
    fig, p = matplotlib.pyplot.subplots()
    p.plot(q, w, 'C0:', label='y', markersize=3)
    p.errorbar(x_test, y_prediction.transpose(), numpy.concatenate(variance), label='y_prediction', marker='x', ls=' ', c='orange',
               markersize=3)

    legend = p.legend(loc='best', shadow=False, fontsize='medium')
    legend.get_frame().set_facecolor('1')

    matplotlib.pyplot.grid(False)
    matplotlib.pyplot.xlabel('x values')
    matplotlib.pyplot.ylabel('y values')
    matplotlib.pyplot.title('Comparison of the curves')

    matplotlib.pyplot.show()


def calculate_sigma(a, b, phi):
    s = numpy.linalg.inv(a * numpy.eye(5, 5) + b * numpy.matmul(phi.transpose(), phi))
    return numpy.array(s)


def calculate_mean(s, b, phi, y):
    aux = numpy.matmul(s, phi.transpose())
    mean = b * numpy.matmul(aux, y)
    return numpy.array(mean)


def construct_phi(n, x):
    phi = numpy.ones((n, 5))
    phi[:, 1] = x.transpose()
    phi[:, 2] = x.transpose() ** 2
    phi[:, 3] = x.transpose() ** 3
    phi[:, 4] = x.transpose() ** 5
    return phi


def calculate_y(rows, columns, x, theta):
    y = numpy.ones((rows, columns))
    y[:, 0] = theta[0, 0] + \
              theta[1, 0] * x.transpose() + theta[2, 0] * x.transpose() ** 2 + \
              theta[3, 0] * x.transpose() ** 3 + theta[4, 0] * x.transpose() ** 5
    return y


main()
