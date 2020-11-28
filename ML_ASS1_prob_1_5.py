import numpy
import random
import matplotlib.pyplot


def main():
    n = 500
    n_test = 20
    sigma_eta_square = 0.05
    sigma_theta_square = 0.1
    theta_real = numpy.array([[0.2], [-1], [0.9], [0.7], [-0.2]])
    theta = numpy.array([[-10.54], [0.465], [0.0087], [-0.093], [-0.004]])

    # real curve
    q = numpy.linspace(0, 2, 200).reshape((200, 1))
    w = calculate_y(200, 1, q, theta_real)

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
    y = calculate_y(n, 1, x, theta_real)

    y_noisy = numpy.zeros((n, 1))
    for i in range(n):
        y_noisy[i, 0] = y[i, 0] + random.gauss(0, numpy.sqrt(sigma_eta_square))

    phi = construct_phi(n, x)

    mean = numpy.zeros((n_test, 1))
    for i in range(n_test):
        mean[i, 0] = calculate_mean(x_test[i, 0], theta, sigma_eta_square, sigma_theta_square, phi, y_noisy)

    variance = numpy.zeros((n_test, 1))
    for i in range(n_test):
        variance[i, 0] = calculate_variance(x_test[i, 0], sigma_eta_square, sigma_theta_square, phi)

    # plotting of the curves
    fig, p = matplotlib.pyplot.subplots()
    p.plot(q, w, 'C0:', label='y', markersize=3)
    p.errorbar(x_test, mean, numpy.concatenate(variance), label='y_prediction', marker='x', ls=' ', c='orange', markersize=3)

    legend = p.legend(loc='best', shadow=False, fontsize='medium')
    legend.get_frame().set_facecolor('1')

    matplotlib.pyplot.grid(False)
    matplotlib.pyplot.xlabel('x values')
    matplotlib.pyplot.ylabel('y values')
    matplotlib.pyplot.title('Comparison of the curves')

    matplotlib.pyplot.show()


def calculate_variance(x, sigma_eta_square, sigma_theta_square, phi):
    small_phi = construct_small_phi(x)
    a = sigma_eta_square * sigma_theta_square * small_phi
    b = sigma_eta_square * numpy.eye(5, 5)
    c = sigma_theta_square * numpy.matmul(phi.transpose(), phi)
    d = numpy.linalg.inv(b + c)
    e = numpy.matmul(a.transpose(), d)
    variance = numpy.matmul(e, small_phi)
    variance += sigma_eta_square
    return variance


def calculate_mean(x, theta, sigma_eta_square, sigma_theta_square, phi, y):
    small_phi = construct_small_phi(x)
    conditional_mean = calculate_conditional_mean(theta, sigma_eta_square, sigma_theta_square, phi, y)
    mean = numpy.matmul(small_phi.transpose(), conditional_mean)
    return mean


def calculate_conditional_mean(theta, sigma_eta_square, sigma_theta_square, phi, y):
    a = (1/sigma_theta_square) * numpy.eye(5, 5)
    b = (1/sigma_eta_square) * numpy.matmul(phi.transpose(), phi)
    c = (1/sigma_eta_square) * numpy.linalg.inv(a + b)
    d = numpy.matmul(c, phi.transpose())
    e = y - numpy.matmul(phi, theta)
    conditional_mean = numpy.matmul(d, e)
    conditional_mean += theta
    return conditional_mean


def construct_small_phi(x):
    small_phi = numpy.ones((5, 1))
    small_phi[1, :] = x.transpose()
    small_phi[2, :] = x.transpose() ** 2
    small_phi[3, :] = x.transpose() ** 3
    small_phi[4, :] = x.transpose() ** 5
    return small_phi


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
