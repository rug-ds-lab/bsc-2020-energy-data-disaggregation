import csv
from datetime import datetime, timedelta


def toIntArray(stringArray):
    result = []
    for string in stringArray:
        result.append(int(string))

    return result


f= open("../data/weatherdata_mydata.txt", "r")
f.readline()
f.readline()
data = []
for i in range(22):
    day = f.readline()
    print(day)
    high_temp = toIntArray(f.readline().split(" "))
    low_temp = toIntArray(f.readline().split(" "))
    wind = toIntArray(f.readline().split(" "))
    arr = f.readline().split(" ")
    weather = []
    for c in arr:
        if c[0] == 'r':
            weather.append(0)
        elif c[0] == 'c':
            weather.append(1)
        elif c[0] == 'm':
            weather.append(2)
        elif c[0] == 's':
            weather.append(3)

    data.append([high_temp,low_temp,wind,weather])


with open("../data/temperature_mydata.csv", "w", encoding="utf8", newline='') as csvfile:
    file = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file.writerow(["date", "high_temp", "low_temp", "wind", "weather"])
    date = datetime(2011, 5, 2)
    delta = timedelta(hours = 6)
    for d in data:
        for i in range(4):
            file.writerow([date, d[0][i], d[1][i], d[2][i], d[3][i]])
            date += delta
    csvfile.close()
















