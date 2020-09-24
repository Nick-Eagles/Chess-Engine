import csv

class Loss:
    def __init__(self, epochNums=[], isStart=[], tLoss=[], vLoss=[]):
        self.epochNums = epochNums
        self.isStart = isStart
        self.tLoss = tLoss
        self.vLoss = vLoss

    def WriteToCSV(self, epochs, overwrite=False, filename = 'visualization/costs2.csv'):
        if overwrite:
            csv_data = [["epochNum", "cost", "costType", "isStart"]]
            for i in range(len(self.epochNums)):
                csv_data.append([self.epochNums[i], self.tLoss[i], 't_cost', self.isStart[i]])
                csv_data.append([self.epochNums[i], self.vLoss[i], 'v_cost', self.isStart[i]])

            with open(filename, 'w') as costFile:
                writer = csv.writer(costFile)
                writer.writerows(csv_data)

        else:
            csv_data = []
            for i in range(epochs):
                index = i - epochs - 1
                csv_data.append([self.epochNums[index], self.tLoss[index], 't_cost', self.isStart[index]])
                csv_data.append([self.epochNums[index], self.vLoss[index], 'v_cost', self.isStart[index]])

            with open(filename, 'a') as costFile:
                writer = csv.writer(costFile)
                writer.writerows(csv_data)

    def Update(self, tValue, vValue, isStart):
        self.tLoss.append(tValue)
        self.vLoss.append(vValue)

        if len(self.epochNums) == 0:
            self.epochNums.append(0)
        elif isStart:
            self.epochNums.append(self.epochNums[-1])
        else:
            self.epochNums.append(self.epochNums[-1] + 1)

        self.isStart.append(isStart)

    def Copy(self):
        return Loss(self.epochNums.copy(), self.isStart.copy(), self.tLoss.copy(), self.vLoss.copy())
