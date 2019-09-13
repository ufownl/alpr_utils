import re
import numpy as np
import matplotlib.pyplot as plt


def visualize(lines):
    regex = re.compile(".* training_loss (\S+).* validating_loss (\S+).*")
    training_loss = []
    validating_loss = []
    min_ppl = float("inf")
    for line in lines:
        m = regex.match(line)
        if m:
            training_loss.append(float(m.group(1)))
            validating_loss.append(float(m.group(2)))
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
