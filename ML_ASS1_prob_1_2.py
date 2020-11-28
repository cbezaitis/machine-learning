import numpy as np
import random
import matplotlib.pyplot


def main():
    n = 20
    mean = 0
    sigma = np.sqrt(0.1)
    theta_given = np.array([[0.2], [-1], [0.9], [0.7], [-0.2]])
    experiments = 100

    # Making 20 equidistance test data 
    x = np.linspace(0., 2., n).reshape((n, 1))
    # Calculate the y from the true model
    y = calculate_y(n, 1, x, theta_given)

    # Calculating with the 2nd degree polynomial
    # Both the mean and the variance are on the same array 
    res1 = calculate_mean_variance(n, x, y, mean, sigma, 2, experiments)

    # We extract the mean of y 
    y_mean_2degree = res1[0]

    # and the variance
    y_variance_2degree = res1[1]

    # Calculating with the 10nd degree polynomial
    res2 = calculate_mean_variance(n, x, y, mean, sigma, 10, experiments)

    y_mean_10degree = res2[0]

    y_variance_10degree = res2[1]


    # TODO: Delete
    # print("y_mean is")
    # print(y_mean_2degree)
    # print("y variance tou 2 degree is ")
    # print(y_variance_2degree)


    #Plotting
    fig, (p1, p2) = matplotlib.pyplot.subplots(2, 1)

    fig.suptitle('Comparison of the curves')
    p1.plot(x, y, 'C2:', label='y')
    p1.errorbar(x, y_mean_2degree, np.concatenate(y_variance_2degree), label='y_prediction', ls='--', c='orange')

    p1.set_ylabel('2 degree polynomial')
    p1.set_xlim([0, 2])
    p1.set_ylim([-0.2, 2])

    legend = p1.legend(loc='best', shadow=False, fontsize='medium')
    legend.get_frame().set_facecolor('1')

    p2.plot(x, y, 'C2:', label='y')
    p2.errorbar(x, y_mean_10degree, np.concatenate(y_variance_10degree), label='y_prediction', ls='--', c='orange')

    legend = p2.legend(loc='best', shadow=False, fontsize='medium')
    legend.get_frame().set_facecolor('1')

    p2.set_xlabel('x')
    p2.set_ylabel('10 degree polynomial')
    p2.set_xlim([0, 2])
    p2.set_ylim([-0.2, 2])

    matplotlib.pyplot.show()


def calculate_mean_variance(n, x, y, mean, sigma, dim, exps):
    y_noise_exp = np.zeros((exps, n, 1))
    y_noise_mean = np.zeros((n, 1))
    res = np.array((n, 1))

    for i in range(exps):
        res = use_lsm(n, x, y, mean, sigma, dim) #LEAST SQUARE METHOD gia 100
        for j in range(n):
            y_noise_exp[i, j, 0] = res[j, 0]
            y_noise_mean[j, 0] += res[j, 0]
    y_noise_mean = y_noise_mean / exps

    y_noise_variance = np.zeros((n, 1))
    for i in range(n):
        for j in range(exps):
            y_noise_variance[i, 0] += (y_noise_exp[j, i, 0] - y_noise_mean[i, 0]) ** 2
    y_noise_variance = y_noise_variance/exps

    # bias^2 = MSE - variance
    bias = np.zeros(20)
    for i in range(n):
        bias[i] = np.sqrt(np.abs(y_noise_mean[i, 0] - y[i, 0]))
    print("to bias einai")
    print(bias)
    print("y_noise_variance is")
    print(np.concatenate(y_noise_variance))

    return np.array([y_noise_mean, y_noise_variance])


def use_lsm(n, x, y, mean, sigma, degree):
    y_noisy = np.zeros((n, 1))
    
    # add noise
    for i in range(n):
        y_noisy[i, 0] = y[i, 0] + random.gauss(mean, sigma)

    # depending on the degree different calculations
    if degree == 2:
        # find phi
        phi = construct_phi_2d(n, x)
        # calculate theta
        theta = calculate_theta(phi, y_noisy)
        # calculate the experimental y
        y_exp = calculate_y_2d(n, 1, x, theta)
    else:
        phi = construct_phi_10d(n, x)
        theta = calculate_theta(phi, y_noisy)
        y_exp = calculate_y_10d(n, 1, x, theta)


    # mse = calculate_mse(y_noisy, y_exp)
    # print("MSE = " + str(mse))
    
    return np.array(y_exp)


def calculate_mse(y1, y2):
    mse = 0
    for i in range(len(y1)):
        mse += (y2[i, 0] - y1[i, 0]) ** 2
    mse = mse/len(y1)
    return mse


def construct_phi_2d(rows, x):
    phi = np.ones((rows, 3))
    for j in range(1, 3):
        for i in range(20):
            phi[i, j] = x[i, 0] ** j
    return phi


def construct_phi_10d(rows, x):
    phi = np.ones((rows, 11))
    for j in range(1, 11):
        for i in range(20):
            phi[i, j] = x[i, 0] ** j
    return phi


def calculate_y(rows, columns, x, theta_real):
    y = np.ones((rows, columns))
    y[:, 0] = theta_real[0, 0] + \
              theta_real[1, 0] * x.transpose() + theta_real[2, 0] * x.transpose() ** 2 + \
              theta_real[3, 0] * x.transpose() ** 3 + theta_real[4, 0] * x.transpose() ** 5
    return y


def calculate_y_2d(rows, columns, x, theta):
    y = np.zeros((rows, columns))
    for i in range(20):
        for j in range(3):
            y[i, 0] += theta[j, 0] * (x[i, 0] ** j)
    return y


def calculate_y_10d(rows, columns, x, theta):
    y = np.zeros((rows, columns))
    for i in range(20):
        for j in range(11):
            y[i, 0] += theta[j, 0] * (x[i, 0] ** j)
    return y


def calculate_theta(phi, y):
    # theta = (phi_transposed x phi)_inverted x phi_transposed x y
    b = np.matmul(phi.transpose(), phi)
    c = np.linalg.inv(b)
    d = np.matmul(c, phi.transpose())
    u = np.matmul(d, y)
    return u


main()
