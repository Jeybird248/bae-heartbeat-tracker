import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('data.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(float(row[0])))
        y.append(float(row[1]))

plt.plot(x, y, color='g')
plt.xlabel('Time')
plt.ylabel('Heartbeat (bpm)')
plt.title("Bae's Heartbeat Over Time")
plt.savefig("output.jpg")
plt.show()
