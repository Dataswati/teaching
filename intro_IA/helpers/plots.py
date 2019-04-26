import numpy as np

import matplotlib.pyplot as plt

def plot_imgs(X, Y, shape, list_label=None, shuffle=False):

    if shuffle:
        ind = np.random.choice(len(Y), 10)
        X = X[ind]
        Y = Y[ind]

    f, ax = plt.subplots(2, 5, figsize=(15,6))

    i = 0
    for ax_0 in ax:
        for ax_1 in ax_0:
            x_plot = X[i]
            x_plot = x_plot.reshape(shape)

            if list_label is None:
                ax_1.set_title(str(Y[i].argmax()))
            else:
                ax_1.set_title(str(list_label[Y[i].argmax()]))

            ax_1.imshow(x_plot, cmap='gray')

            ax_1.get_xaxis().set_visible(False)
            ax_1.get_yaxis().set_visible(False)

            i += 1

    plt.show()

