# Copyright (c) 2019, RangerUFO
#
# This file is part of alpr_utils.
#
# alpr_utils is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpr_utils is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with alpr_utils.  If not, see <https://www.gnu.org/licenses/>.


import re
import numpy as np
import matplotlib.pyplot as plt


def visualize(lines):
    regex = re.compile("^\[Epoch ([0-9]+)  Batch ([0-9]+)\]  batch_loss (\S+).*")
    batch_x = []
    batch_loss = []
    for line in lines:
        m = regex.match(line)
        if m:
            batch_x.append((int(m.group(1)), int(m.group(2))))
            batch_loss.append(float(m.group(3)))
    batches = max(batch_x, key=lambda x: x[1])[1]
    batch_x = [epoch + batch / batches for epoch, batch in batch_x]
    regex = re.compile("^\[Epoch ([0-9]+)\]  training_loss (\S+)  validation_loss (\S+).*")
    epoch_x = []
    training_loss = []
    validation_loss = []
    for line in lines:
        m = regex.match(line)
        if m:
            epoch_x.append(int(m.group(1)))
            training_loss.append(float(m.group(2)))
            validation_loss.append(float(m.group(3)))
    plt.subplot(2, 1, 1)
    plt.plot(np.array(batch_x), np.array(batch_loss), label="batch loss")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(np.array(epoch_x), np.array(training_loss), label="training loss")
    plt.plot(np.array(epoch_x), np.array(validation_loss), label="validation loss")
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
