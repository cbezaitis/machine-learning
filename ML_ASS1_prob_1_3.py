import numpy
import random


def main():
    n1 = 20
    n2 = 1000
    mean = 0
    sigma = numpy.sqrt(0.1)
    theta_real = numpy.array([[0.2], [-1], [0.9], [0.7], [-0.2]])
    print("theta real = " + str(theta_real.transpose()))

    x1 = numpy.linspace(0, 2, n1).reshape((n1, 1))
    y1 = calculate_y(n1, 1, x1, theta_real)
    y1_noisy = numpy.zeros((n1, 1))
    for i in range(n1):
        y1_noisy[i, 0] = y1[i, 0] + random.gauss(mean, sigma)
    phi = construct_phi(n1, x1)
    theta_pred = calculate_theta(phi, y1_noisy, 0)
    print("theta predicted = "+str(theta_pred.transpose()))
    y1_pred = calculate_y(n1, 1, x1, theta_pred)
    mse1 = calculate_mse(y1_noisy, y1_pred)

    x2 = numpy.array([random.uniform(0, 2) for i in range(n2)]).reshape((n2, 1))
    y2 = calculate_y(n2, 1, x2, theta_real)
    y2_noisy = numpy.zeros((n2, 1))
    for i in range(n2):
        y2_noisy[i, 0] = y2[i, 0] + random.gauss(mean, sigma)
    y2_pred = calculate_y(n2, 1, x2, theta_pred)
    mse2 = calculate_mse(y2_noisy, y2_pred)

    lamda_1 = []
    lamda_2 = []
    i = 0
    while i < 1000:
        x1 = numpy.linspace(0, 2, n1).reshape((n1, 1))
        theta_rr = calculate_theta(phi, y1_noisy, i)
        y1_rr = calculate_y(n1, 1, x1, theta_rr)
        m1 = calculate_mse(y1_noisy, y1_rr)
        if mse1 > m1:
            mse1 = m1
            lamda_1.append(i)
            print("theta rr = " + str(theta_rr.transpose()))
            print(i)

        x2 = numpy.array([random.uniform(0, 2) for i in range(n2)]).reshape((n2, 1))
        y2_rr = calculate_y(n2, 1, x2, theta_rr)
        m2 = calculate_mse(y2_noisy, y2_rr)
        if mse2 > m2:
            mse2 = m2
            lamda_2.append(i)
        i += 1

    if len(lamda_1) > 0:
        print("The lamdas which minimize the MSE1 are ")
        for i in lamda_1:
            print(i)
    if len(lamda_2) > 0:
        print("The lamdas which minimize the MSE2 are ")
        for i in lamda_2:
            print(i)
    if len(lamda_1) == 0 and len(lamda_2) == 0:
        print("There is no lamda which minimizes the MSE, so lamda sould be zero")


def calculate_mse(y1, y2):
    mse = 0
    for i in range(len(y1)):
        mse = (y2[i, 0] - y1[i, 0]) ** 2
    mse = mse/len(y1)
    return mse


def construct_phi(rows, x):
    phi = numpy.ones((rows, 5))
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


def calculate_theta(phi, y, lamda):
    # theta = (phi_transposed x phi + lamdaI)_inverted x phi_transposed x y
    r = numpy.eye(5, 5)
    r = r * lamda
    a = numpy.matmul(phi.transpose(), phi)
    b = a - r
    c = numpy.linalg.inv(b)
    d = numpy.matmul(c, phi.transpose())
    u = numpy.matmul(d, y)
    return u


main()
