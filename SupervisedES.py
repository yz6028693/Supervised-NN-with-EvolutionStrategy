'''
In this demo, I will classify the same Archimedean Spiral pattern based on two categories of points we inputted
in the Neural Network. 7 value of each point are inputted in the NN: x, y, x^2, y^2, xy, sin(x) and sin(y).
This time, rather than use TensorFlow and Stochastic gradient descent (SGD), I build this NN with only numpy and 
use Evolution Strategy to train this neural network.

I haven't used multiprocessing in this version, the training speed and result can already versus those in my TensorFlow
and SGD demo.

'''

import numpy as np
import matplotlib.pyplot as plt
from math import pi, exp, sin, log
from random import shuffle, randint
#import multiprocessing as mp
#from numba import jit
from matplotlib import colors

class SupervisedES(object):


    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, -y)

    def BatchCreater(self, BatchNum, TrainArray, LabelArray):
        TrainBatch = []
        LabelBatch = []
        for _ in range(BatchNum):
            num = randint(0, len(TrainArray) - 1)
            TrainBatch.append(TrainArray[num])
            LabelBatch.append(LabelArray[num])
        return [np.asarray(TrainBatch), np.asarray(LabelBatch)]


    # def Normalize(self, Array):
    #     min = float(np.amin(Array))
    #     max = float(np.amax(Array))
    #     if min == max:
    #         return Array
    #     else:
    #         return (Array - min)/(max - min)


    def OneHot(self, Class, length):
        OneHot = [0] * length
        OneHot[Class - 1] = 1
        return np.asarray(OneHot)


    def softmax(self, x):
        e_x = np.exp(x)
        sumArray = np.sum(e_x, axis=1)
        return (e_x.T / (sumArray + 0.00001)).T


    def cross_entropy(self, predictions, RealResults):
        cross_entropy = -np.sum(RealResults * np.log(predictions + 1e-9))/RealResults.shape[0]
        return cross_entropy

    # @jit
    # def cross_entropyJit(self, predictions, RealResults):
    #     sumcross_entropy = 0
    #     for row in range(RealResults.shape[0]):
    #         for col in range(RealResults.shape[1]):
    #             sumcross_entropy -= RealResults[row, col] * log(predictions[row, col] + 1e-9)
    #     cross_entropy = sumcross_entropy / RealResults.shape[0]
    #     return cross_entropy


    def InitWB(self, Shape, std): # Shape: (7, 6, 2)
        WArray1 = std * np.random.randn(*Shape[:2])
        WArray2 = std * np.random.randn(*Shape[1:])
        BArray1 = 0.1 * np.ones((Shape[1], 1))
        BArray2 = 0.1 * np.ones((Shape[2], 1))
        return WArray1, BArray1, WArray2, BArray2


    def NNoutput(self, HiddenArray1, HiddenArray2, NormalizedArray):
        HiddenLayer = np.tanh(np.dot(NormalizedArray, HiddenArray1[0]) + HiddenArray1[1].T)
        preOut = np.dot(HiddenLayer, HiddenArray2[0]) + HiddenArray2[1].T
        OutLayer = self.softmax(preOut)
        return OutLayer

    def CorrectScore(self, HiddenArray1, HiddenArray2, BatchwithLabel): # BatchwithLabel is a list N*7, N*1
        OutLayer = self.NNoutput(HiddenArray1, HiddenArray2, BatchwithLabel[0])

        # Higher cross entropy value means lower match rate and should have a lower score.
        Score = 2 / (1 + exp(self.cross_entropy(OutLayer, BatchwithLabel[1])))
        return Score


    def CorrectRatio(self, HiddenArray1, HiddenArray2, DatawithLabel): # DatawithLabel is a list N*7, N*1
        OutLayer = self.NNoutput(HiddenArray1, HiddenArray2, DatawithLabel[0])
        Classification = np.argmax(OutLayer, 1) + 1
        label  = np.argmax(DatawithLabel[1], 1) + 1
        Correct = len(np.where(Classification * label == 1)[0]) + len(np.where(Classification * label == 4)[0])
        CorrectRatio = Correct/len(DatawithLabel[0])
        return CorrectRatio


    def Classification(self, HiddenArray1, HiddenArray2, NormalizedRasterArray):
        OutLayer = self.NNoutput(HiddenArray1, HiddenArray2, NormalizedRasterArray)
        Classification = np.argmax(OutLayer, 1) + 1
        return Classification


    def get_rewards(self, ArrayList1, ArrayList2, BatchwithLabel, GradientExplorers):

        Rewards = np.zeros(GradientExplorers)

        for n in range(GradientExplorers):

            Rewards[n] = self.CorrectScore([ArrayList1[0][n, :], ArrayList1[1][n, :]], [ArrayList2[0][n, :], ArrayList2[1][n, :]], BatchwithLabel)

        NormalizedRewards = (Rewards - np.mean(Rewards))  / (np.std(Rewards) + 0.000001)

        return NormalizedRewards


    def KidsWithNoise(self, HiddenArray1, HiddenArray2, GradientExplorers, sigma):
        Wshape1 = (GradientExplorers, 7, 6)
        Wshape2 = (GradientExplorers, 6, 2)
        Bshape1 = (GradientExplorers, 6, 1)
        Bshape2 = (GradientExplorers, 2, 1)
        Wnoise1 = np.random.randn(*Wshape1)
        Wnoise2 = np.random.randn(*Wshape2)
        Bnoise1 = np.random.randn(*Bshape1)
        Bnoise2 = np.random.randn(*Bshape2)
        W1WithNoise = np.expand_dims(HiddenArray1[0], 0).repeat(GradientExplorers, axis=0) + sigma * Wnoise1
        W2WithNoise = np.expand_dims(HiddenArray2[0], 0).repeat(GradientExplorers, axis=0) + sigma * Wnoise2
        B1WithNoise = np.expand_dims(HiddenArray1[1], 0).repeat(GradientExplorers, axis=0) + sigma * Bnoise1
        B2WithNoise = np.expand_dims(HiddenArray2[1], 0).repeat(GradientExplorers, axis=0) + sigma * Bnoise2
        Array1 = [W1WithNoise, B1WithNoise, Wnoise1, Bnoise1]
        Array2 = [W2WithNoise, B2WithNoise, Wnoise2, Bnoise2]
        return Array1, Array2


    def UpdateWB(self, Array1, Array2, NormalizedRewards, alpha, sigma, GradientExplorers):
        updatedW1 = Array1[0] + alpha / (GradientExplorers * sigma) * np.dot(Array1[2].T, NormalizedRewards).T
        updatedW2 = Array2[0] + alpha / (GradientExplorers * sigma) * np.dot(Array2[2].T, NormalizedRewards).T
        updatedB1 = Array1[1] + alpha / (GradientExplorers * sigma) * np.dot(Array1[3].T, NormalizedRewards).T
        updatedB2 = Array2[1] + alpha / (GradientExplorers * sigma) * np.dot(Array2[3].T, NormalizedRewards).T
        return updatedW1, updatedB1, updatedW2, updatedB2


    def ReGroupForTraining(self, ArrayList):
        ReGroupForTrainingArray = []
        index = list(range(len(ArrayList[0])))
        for i in index:
            ReGroupForTrainingArray.append([array[i] for array in ArrayList])
        return ReGroupForTrainingArray


    def SeparateTrainingTesting(self, Array, TrainingRatio):
        RandomlizeArray = []
        index = list(range(len(Array)))
        shuffle(index)
        for i in index:
            RandomlizeArray.append(Array[i])
        TrainingArray = RandomlizeArray[: int(len(RandomlizeArray) * TrainingRatio)]
        TestingArray =  RandomlizeArray[int(len(RandomlizeArray) * TrainingRatio):]
        TrainingInput = [i[: -1] for i in TrainingArray]
        TrainingLabel = [self.OneHot(i[-1], 2) for i in TrainingArray]
        TestingInput = [i[: -1] for i in TestingArray]
        TestingLabel = [self.OneHot(i[-1], 2) for i in TestingArray]
        return TrainingInput, TrainingLabel, TestingInput, TestingLabel


    def outputArray(self, ClassificationArray):
        OutputArray = np.zeros((600, 600), dtype=np.int)
        for n in range(600):
            for m in range(600):
                classification = int(ClassificationArray[600 * n + m]) + 1
                OutputArray[n][m] = classification
        return OutputArray


    def PlotingRasterArray(self):
        NormalizedRasterArray = []
        Range = 600.0
        for y in list(range(-300, 300)):
            for x in list(range(-300, 300)):
                NormalizedRasterArray.append(
                    np.array([(x + 300.0) / Range, (-y + 300.0) / Range, ((x + 300.0) / Range) ** 2, ((-y + 300.0) / Range) ** 2,
                     ((x + 300.0) * (-y + 300.0)) / (Range ** 2), (sin(x / 50.0) + 1) / 2, (sin(-y / 50.0) + 1) / 2]))

        return np.array(NormalizedRasterArray)


    def WorkFlow(self, HiddenArray1, HiddenArray2, BatchwithLabel, DatasetwithLabel, alpha, sigma, GradientExplorers):
        Array1, Array2 = self.KidsWithNoise(HiddenArray1, HiddenArray2, GradientExplorers, sigma)
        ArrayList1 = [Array1[0], Array1[1]]
        ArrayList2 = [Array2[0], Array2[1]]
        ListForUpdate1 = HiddenArray1 + [Array1[2], Array1[3]]
        ListForUpdate2 = HiddenArray2 + [Array2[2], Array2[3]]
        NormalizedRewards = self.get_rewards(ArrayList1, ArrayList2, BatchwithLabel, GradientExplorers)
        updatedW1, updatedB1, updatedW2, updatedB2 = self.UpdateWB(ListForUpdate1, ListForUpdate2, NormalizedRewards, alpha, sigma, GradientExplorers)
        HiddenArray1 = [updatedW1, updatedB1]
        HiddenArray2 = [updatedW2, updatedB2]
        return HiddenArray1, HiddenArray2


    def TrainingandPloting(self, StopPoint, TrainingRatio, BatchSize):
        phi = [2 * i * (2 * pi / 360) for i in range(270)]
        rho1 = np.multiply(phi, 0.5)
        rho2 = np.multiply(phi, -0.5)
        x_1, y_1 = self.pol2cart(rho1, phi)
        x_2, y_2 = self.pol2cart(rho2, phi)
        Label1 = np.ones((len(phi)), dtype=np.int)
        Label2 = np.multiply(Label1, 2)

        # normalized inputs (7 inputs: x, y, x^2, y^2, xy, sin(x), sin(y))
        InputClass1 = [(x_1 + 6) / 12.0, (y_1 + 6) / 12.0, ((x_1 + 6) / 12.0) ** 2, ((y_1 + 6) / 12.0) ** 2,
                       ((x_1 + 6) * (y_1 + 6)) / 144.0, (np.sin(x_1) + 1) / 2.0, (np.sin(y_1) + 1) / 2.0, Label1]
        InputClass2 = [(x_2 + 6) / 12.0, (y_2 + 6) / 12.0, ((x_2 + 6) / 12.0) ** 2, ((y_2 + 6) / 12.0) ** 2,
                       ((x_2 + 6) * (y_2 + 6)) / 144.0, (np.sin(x_2) + 1) / 2.0, (np.sin(y_2) + 1) / 2.0, Label2]

        ReGroupForTrainingArray1 = self.ReGroupForTraining(InputClass1)
        ReGroupForTrainingArray2 = self.ReGroupForTraining(InputClass2)
        NormalizedRasterArray = self.PlotingRasterArray()
        TrainingInput, TrainingLabel, TestingInput, TestingLabel = \
            self.SeparateTrainingTesting(ReGroupForTrainingArray1 + ReGroupForTrainingArray2, TrainingRatio)
        CorrectRatio = 0.0
        InitWArray1, InitBArray1, InitWArray2, InitBArray2 = self.InitWB((7,6,2), 0.01) # 7 inputs, 6 hidden notes (1 hidden layer), 2 outputs
        HiddenArray1 = [InitWArray1, InitBArray1]
        HiddenArray2 = [InitWArray2, InitBArray2]
        DatasetwithLabel = [np.array(TrainingInput + TestingInput), np.array(TrainingLabel + TestingLabel)]
        TrainingDatawithLabel = [np.array(TrainingInput), np.array(TrainingLabel)]
        TestingDatawithLabel = [np.array(TestingInput), np.array(TestingLabel)]
        gen = 0
        while CorrectRatio < StopPoint:
            gen += 1
            BatchwithLabel = self.BatchCreater(BatchSize, TrainingInput, TrainingLabel)

            # Choose different learningRate and SearchArea (Standard deviation) for better performance

            if CorrectRatio < 0.9:
                learningRate = 0.25
                SearchArea = 0.3
                SearchWorkers = 150
            elif CorrectRatio < 0.95:
                learningRate = 0.1
                SearchArea = 0.2
                SearchWorkers = 100
            else:
                learningRate = 0.05
                SearchArea = 0.2
                SearchWorkers = 100

            HiddenArray1, HiddenArray2 = self.WorkFlow(HiddenArray1, HiddenArray2, BatchwithLabel,
                                                        TrainingDatawithLabel, learningRate, SearchArea, SearchWorkers)
            CorrectRatio =  self.CorrectRatio(HiddenArray1, HiddenArray2, DatasetwithLabel)
            if gen % 100 == 0:
                TrainingCorrectRatio = self.CorrectRatio(HiddenArray1, HiddenArray2, TrainingDatawithLabel)
                TestingCorrectRatio = self.CorrectRatio(HiddenArray1, HiddenArray2, TestingDatawithLabel)
                print('CorrectRatio: ', CorrectRatio)
                print('TrainingCorrectRatio: ', TrainingCorrectRatio)
                print('TestingCorrectRatio: ', TestingCorrectRatio)

        print('Generations: ', gen)
        print('Final NN Weights1: ', HiddenArray1[0])
        print('Final NN Bias1: ', HiddenArray1[1])
        print('Final NN Weights2: ', HiddenArray2[0])
        print('Final NN Bias2: ', HiddenArray2[1])
        ClassificationArray = self.Classification(HiddenArray1, HiddenArray2, NormalizedRasterArray)
        OutputArray = self.outputArray(ClassificationArray)
        cmap = colors.ListedColormap(['blue', 'orange'])
        plt.figure(figsize=(9, 7))
        plt.rcParams['axes.facecolor'] = (0.2, 0.2, 0.2)
        plt.margins(0.1)
        plt.subplots_adjust(right=0.8)
        im = plt.imshow(OutputArray, extent=[-6, 6, -6, 6], cmap=cmap)
        scat1 = plt.scatter(x_1, y_1, s=10, c='b', edgecolors='white', linewidth='0.5', zorder=10)
        scat2 = plt.scatter(x_2, y_2, s=10, c='coral', edgecolors='white', linewidth='0.5', zorder=10)
        plt.show()



if __name__ == '__main__':
    tool = SupervisedES()
    # Accuracy Requirement: 98.5%, Training and testing data radio: 8 to 2, Batch size: 15.
    tool.TrainingandPloting(0.985, 0.8, 15)








