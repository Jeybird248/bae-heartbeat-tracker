import matplotlib.pyplot as plt
import csv

x = []
y = []


def format_func(x):
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    seconds = int(x%60)

    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


with open('data2.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(float(row[0])))
        y.append(float(row[1]))

for indx, val in enumerate(x):
    x[indx] = format_func(val)

plt.plot(x, y, color='black')
plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_ticks([-1, 479, 959, 1439, 1919, 2399, 2879, 3359, 3839])
plt.xlabel('Time')
plt.ylabel('Heartbeat (bpm)')
plt.title("Bae's Heartbeat Over Time")
plt.savefig("output.jpg")
plt.show()
