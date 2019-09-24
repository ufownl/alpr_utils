import re
import numpy as np
import matplotlib.pyplot as plt


def visualize(lines):
    re_batch = re.compile(".* batch_loss (\S+).*")
    re_epoch = re.compile(".* training_loss (\S+).* validating_loss (\S+).*")
    batch_loss = []
    training_loss = []
    validating_loss = []
    for line in lines:
        m = re_batch.match(line)
        if m:
            batch_loss.append(float(m.group(1)))
        m = re_epoch.match(line)
        if m:
            training_loss.append(float(m.group(1)))
            validating_loss.append(float(m.group(2)))
    plt.subplot(2, 1, 1)
    plt.plot(np.array(batch_loss), label="batch loss")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(np.array(training_loss), label="training loss")
    plt.plot(np.array(validating_loss), label="validating loss")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    lines = []
    while True:
        try:
            lines.append(input())
        except EOFError:
            break
    visualize(lines)
