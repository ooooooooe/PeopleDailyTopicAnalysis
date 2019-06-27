import jieba
import os
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

####################################################################################
# 分词类
####################################################################################
class Segmentation:
    def __init__(self, stopWordListPath, sourcePath, targetPath):
        with open(stopWordListPath, encoding='utf-8') as sw:
            stopWordList = [w.strip() for w in sw.readlines()]
        self.stopWordList = stopWordList
        self.sourcePath = sourcePath
        self.targetPath = targetPath

    @staticmethod
    def segmentSentence(sentence, stopWordList):
        segSentenceSource = jieba.cut(sentence.strip())
        segSentence = [w.strip() for w in segSentenceSource]
        outSegSentence = ""
        for word in segSentence:
            if word not in stopWordList and word.strip() != '':
                outSegSentence += word + " "
        return outSegSentence

    def segmentFiles(self):
        fileList = os.listdir(self.sourcePath)
        fileList.sort()
        for file in fileList:
            with open(os.path.join(self.sourcePath, file), 'r', encoding='utf-8') as r:
                context = ' '.join(r.readlines())
                outContext = Segmentation.segmentSentence(context, self.stopWordList)
                with open(os.path.join(self.targetPath, file.replace(' ', '')), 'w+', encoding='utf-8') as w:
                    w.write(outContext)
                    w.close()
                r.close()

class Analysis:

    @staticmethod
    def getFirst(element):
        return element[0]

    @staticmethod
    def getSecond(element):
        return element[1]

    def __init__(self, sourceTextPath, topicListPath, fileTopicPath, timeLinePath, maxDf, minDf, debug, maxIter=20, topic=5):
        self.sourceTextPath = sourceTextPath
        self.topicListPath = topicListPath
        self.fileTopicPath = fileTopicPath
        self.timeLinePath = timeLinePath
        #self.sourceVector = CountVectorizer()
        self.dataContext = []
        self.transformer = TfidfVectorizer(max_df=maxDf, min_df=minDf)
        self.countVectorizer = CountVectorizer()
        self.lda = LatentDirichletAllocation(n_topics=topic, learning_offset=50, max_iter=maxIter, random_state=0)
        self.debug = debug
        self.fileList = os.listdir(self.sourceTextPath)
        self.fileList.sort()
        self.fileRealList = []
        self.timeToFileList = {}
        self.timeToFileCount = {}
        self.timeToFileRealList = {}
        self.timeToFileRealCount = {}
        for idx, file in enumerate(self.fileList):
            fullFilePath = os.path.join(self.sourceTextPath, file)
            with open(fullFilePath, 'r', encoding='utf-8') as r:
                context = r.readlines()
                fullContext = ' '.join(context)
                self.dataContext.append(fullContext)
                if self.debug == True:
                    if idx % 1000 == 999:
                        print('Loading %d files' % (idx + 1))
        print("All files loaded, %d files total." % (len(self.fileList)))
        self.tfIdfMatrix = self.transformer.fit_transform(self.dataContext)
        self.countMatrix = self.countVectorizer.fit_transform(self.dataContext)
        #self.tfIdfMatrix = self.transformer.fit_transform(self.wordFrequencyMatrix)

    def showFrequency(self):
        featureList = self.transformer.get_feature_names()
        print("%d features.\nFeature name:"% len(featureList))
        print(featureList)
        #print("\n")

        print("Vocabulary: \n")
        print(self.transformer.vocabulary_)
        print("\n")

        print("Word frequency matrix: \n")
        print("array mode:")
        print(self.tfIdfMatrix.toarray())
        print("dense mode:")
        print(self.tfIdfMatrix.todense())

        # print("TFIDF Matrix: \n")
        # print(self.tfIdfMatrix.toarray())

    def LatentDirichletAllocation(self):
        self.lda.fit(self.countMatrix)
        self.documentTopic = self.lda.transform(self.countMatrix)

    def saveTopicWords(self, topN):
        if os.path.exists(self.topicListPath):
            print("Topic list exists, not updated.")
        featureName = self.countVectorizer.get_feature_names()
        with open(self.topicListPath, 'w+', encoding='utf-8') as w:
            for topicIdx, topic in enumerate(self.lda.components_):
                w.write("Topic #%d: " % topicIdx)
                w.write(" ".join(featureName[i] for i in topic.argsort()[: -topN - 1 : -1]) + '\n')
        dT = pd.DataFrame(self.documentTopic, index=self.fileList)
        dT.to_csv(self.fileTopicPath, encoding='utf-8')

    def extractTopic(self, topN):
        print("Extracting TOP %d topics ..." %(topN))
        documentTopicDense = self.documentTopic #.todense()

        keyTopics = np.argsort(-documentTopicDense)[:, :topN]
        dt = pd.DataFrame(keyTopics)
        return dt

    def extractTopicByProb(self, prob):
        print("Extracting main topics PERCENTAGE more than %f ..." % prob)
        documentTopicList = self.documentTopic.tolist()
        topTopicList = []
        for i in range(len(documentTopicList)):
            #iList = np.argwhere(documentTopicList[i] > prob)
            iList = []
            for j in range(len(documentTopicList[i])):
                if documentTopicList[i][j] >= prob:
                    iList.append(int(j))
            topTopicList.append(iList)
        dt = pd.DataFrame(topTopicList, index=self.fileList)
        return dt

    def timeLineAnalysis(self, importantTopicList, prob):
        print("Analysis the time line of topics ...")
        documentTopicList = self.documentTopic.tolist()
        topTopicList = []
        for i in range(len(documentTopicList)):
            realTopic = True
            for j in range(len(importantTopicList)):
                if isinstance(importantTopicList[j], int) or isinstance(importantTopicList[j], float):
                    if documentTopicList[i][importantTopicList[j]] < prob:
                        realTopic = False
                        break
                else:
                    curTopic = False
                    for idx in importantTopicList[j]:
                        if documentTopicList[i][idx] > prob:
                            curTopic = True
                    if curTopic == False:
                        realTopic = False
            if realTopic == True:
                self.fileRealList.append(self.fileList[i])
        print("Total number of files of real topic is %d of total file size %d ..." %(len(self.fileRealList), len(self.fileList)))

        for file in self.fileList:
            time = int(file.split('.')[1].split('年')[0])
            if time not in self.timeToFileList.keys():
                self.timeToFileList[time] = [file]
            else:
                self.timeToFileList[time].append(file)
        for key in self.timeToFileList.keys():
            self.timeToFileCount[key] = len(self.timeToFileList[key])

        for file in self.fileRealList:
            time = int(file.split('.')[1].split('年')[0])
            if time not in self.timeToFileRealList.keys():
                self.timeToFileRealList[time] = [file]
            else:
                self.timeToFileRealList[time].append(file)
        for key in self.timeToFileRealList.keys():
            self.timeToFileRealCount[key] = len(self.timeToFileRealList[key])

    def timeLineAnalysisBoth(self, importantTopicListOne, importantTopicListTwo, prob):
        print("Analysis the time line of topics ...")
        fileRealListOne = []
        fileRealListTwo = []
        timeToFileRealListOne = {}
        timeToFileRealListTwo = {}
        timeToFileRealCountOne = {}
        timeToFileRealCountTwo = {}
        documentTopicList = self.documentTopic.tolist()
        for i in range(len(documentTopicList)):
            realTopic = True
            for j in range(len(importantTopicListOne)):
                if isinstance(importantTopicListOne[j], int) or isinstance(importantTopicListOne[j], float):
                    if documentTopicList[i][importantTopicListOne[j]] < prob:
                        realTopic = False
                        break
                else:
                    curTopic = False
                    for idx in importantTopicListOne[j]:
                        if documentTopicList[i][idx] > prob:
                            curTopic = True
                    if curTopic == False:
                        realTopic = False
            if realTopic == True:
                fileRealListOne.append(self.fileList[i])
        print("Total number of files of real topic ONE is %d of total file size %d ..." % (
        len(fileRealListOne), len(self.fileList)))

        for i in range(len(documentTopicList)):
            realTopic = True
            for j in range(len(importantTopicListTwo)):
                if isinstance(importantTopicListTwo[j], int) or isinstance(importantTopicListTwo[j], float):
                    if documentTopicList[i][importantTopicListTwo[j]] < prob:
                        realTopic = False
                        break
                else:
                    curTopic = False
                    for idx in importantTopicListTwo[j]:
                        if documentTopicList[i][idx] > prob:
                            curTopic = True
                    if curTopic == False:
                        realTopic = False
            if realTopic == True:
                fileRealListTwo.append(self.fileList[i])
        print("Total number of files of real topic TWO is %d of total file size %d ..." % (
            len(fileRealListTwo), len(self.fileList)))
        for file in fileRealListOne:
            time = int(file.split('.')[1].split('年')[0])
            if time not in timeToFileRealListOne.keys():
                timeToFileRealListOne[time] = [file]
            else:
                timeToFileRealListOne[time].append(file)
        for key in timeToFileRealListOne.keys():
            timeToFileRealCountOne[key] = len(timeToFileRealListOne[key])

        for file in fileRealListTwo:
            time = int(file.split('.')[1].split('年')[0])
            if time not in timeToFileRealListTwo.keys():
                timeToFileRealListTwo[time] = [file]
            else:
                timeToFileRealListTwo[time].append(file)
        for key in timeToFileRealListTwo.keys():
            timeToFileRealCountTwo[key] = len(timeToFileRealListTwo[key])
        return timeToFileRealCountOne, timeToFileRealCountTwo, fileRealListOne, fileRealListTwo

    #saving Time Line and files in topic
    def saveTimeLine(self, timeLineFiles = 'inTopic/', timeLineTitle = 'title.txt', timeLineCount = 'totalCount.txt', timeLineRealCount = 'totalTopicCount.txt'):
        print("Time Line SAVING ...")
        #saving all files in REAL topic
        timeLineFileFullPath = os.path.join(self.timeLinePath, timeLineFiles)
        if os.path.exists(timeLineFileFullPath):
            shutil.rmtree(timeLineFileFullPath)
        os.mkdir(timeLineFileFullPath)
        for file in self.fileRealList:
            with open(os.path.join(self.sourceTextPath, file), 'r', encoding='utf-8') as r:
                content = ' '.join(r.readlines())
                with open(os.path.join(timeLineFileFullPath, file), 'w+', encoding='utf-8') as w:
                    w.write(content)
                    w.close()
                r.close()

        #saving file titles in topic 2
        timeLineTitleFullPath = os.path.join(self.timeLinePath, timeLineTitle)
        with open(timeLineTitleFullPath, 'w+', encoding='utf-8') as w:
            for file in self.fileRealList:
                w.write(file + '\n')
            w.close()
        #saving file count in topic 1
        timeLineCountFullPath = os.path.join(self.timeLinePath, timeLineCount)
        with open(timeLineCountFullPath, 'w+', encoding='utf-8') as w:
            w.write(' '.join([str(item[0]) + ' ' + str(item[1]) + '\n' for item in self.timeToFileCountList]))
            w.close()

        #saving file count in REAL topic
        timeLineRealCountFullPath = os.path.join(self.timeLinePath, timeLineRealCount)
        with open(timeLineRealCountFullPath, 'w+', encoding='utf-8') as w:

            w.write(' '.join([str(item[0]) + ' ' + str(item[1]) + '\n' for item in self.timeToFileRealCountList]))
            w.close()

    def saveTimeLineBoth(self, timeLineOne, timeLineTwo, fileListOne, fileListTwo, timeLineFiles='inTopic/', timeLineTitle='title.txt', timeLineCountOne='totalCountOne.txt',
                     timeLineCountTwo='totalCountTwo.txt'):
        print("Time Line SAVING ...")
        # saving all files in REAL topic
        timeLineFileFullPath = os.path.join(self.timeLinePath, timeLineFiles)
        if os.path.exists(timeLineFileFullPath):
            shutil.rmtree(timeLineFileFullPath)
        os.mkdir(timeLineFileFullPath)
        topicPathOne = os.path.join(timeLineFileFullPath, 'One')
        topicPathTwo = os.path.join(timeLineFileFullPath, 'Two')
        os.mkdir(topicPathOne)
        os.mkdir(topicPathTwo)
        for file in fileListOne:
            with open(os.path.join(self.sourceTextPath, file), 'r', encoding='utf-8') as r:
                content = ' '.join(r.readlines())
                with open(os.path.join(topicPathOne, file), 'w+', encoding='utf-8') as w:
                    w.write(content)
                    w.close()
                r.close()
        for file in fileListTwo:
            with open(os.path.join(self.sourceTextPath, file), 'r', encoding='utf-8') as r:
                content = ' '.join(r.readlines())
                with open(os.path.join(topicPathTwo, file), 'w+', encoding='utf-8') as w:
                    w.write(content)
                    w.close()
                r.close()

        # # saving file titles in REAL topic
        # timeLineTitleFullPath = os.path.join(self.timeLinePath, timeLineTitle)
        # with open(timeLineTitleFullPath, 'w+', encoding='utf-8') as w:
        #     for file in self.fileRealList:
        #         w.write(file + '\n')
        #     w.close()

        # saving file count in ALL topic
        timeLineCountFullPath = os.path.join(self.timeLinePath, timeLineCountOne)
        with open(timeLineCountFullPath, 'w+', encoding='utf-8') as w:
            w.write(' '.join([str(item[0]) + ' ' + str(item[1]) + '\n' for item in timeLineOne]))
            w.close()

        # saving file count in REAL topic
        timeLineRealCountFullPath = os.path.join(self.timeLinePath, timeLineCountTwo)
        with open(timeLineRealCountFullPath, 'w+', encoding='utf-8') as w:

            w.write(' '.join([str(item[0]) + ' ' + str(item[1]) + '\n' for item in timeLineTwo]))
            w.close()

    #static method for plotting time line of REAL & ALL files
    @staticmethod
    def loadDataPlotTimeLine(range, figPath = 'fig/time.png', timeLineCountFullPath = 'data/timeLine/totalCount.txt', timeLineTopicCountFullPath = 'data/timeLine/totalTopicCount.txt'):
        with open(timeLineCountFullPath) as f:
            timeToFileCountList = [item.split() for item in f.readlines()]
            f.close()
        with open(timeLineTopicCountFullPath) as f:
            timeToFileRealCountList = [item.split() for item in f.readlines()]
            f.close()
        keyReal = [str(item[0]) for item in timeToFileRealCountList]
        valueReal = [int(item[1]) for item in timeToFileRealCountList]
        keyTotal = [str(item[0]) for item in timeToFileCountList]
        valueTotal = [int(item[1]) for item in timeToFileCountList]
        if(len(keyTotal) != len(keyReal)):
            print("Time line of [ALL FILE] is not equal to that of [REAL FILE].")
            return
        plt.title('Number of news [%d-2018]' % range)
        plt.plot(keyTotal, valueTotal, 'r--', keyReal, valueReal, 'k')
        plt.savefig(figPath)

    #static method for plotting time line of REAL & ALL files
    @staticmethod
    def loadDataPlotTimeLineTwoRange(left, right, figPath='fig/time.png',
                                     timeLineCountFullPath='data/timeLine/totalCount.txt',
                                     timeLineTopicCountFullPath='data/timeLine/totalTopicCount.txt'):
        with open(timeLineCountFullPath) as f:
            timeToFileCountList = [item.split() for item in f.readlines()]
            f.close()
        with open(timeLineTopicCountFullPath) as f:
            timeToFileRealCountList = [item.split() for item in f.readlines()]
            f.close()
        keyReal = [str(item[0]) for item in timeToFileRealCountList if int(item[0]) >= left and int(item[0]) <= right]
        valueReal = [int(item[1]) for item in timeToFileRealCountList if int(item[0]) >= left and int(item[0]) <= right]
        keyTotal = [str(item[0]) for item in timeToFileCountList if int(item[0]) >= left and int(item[0]) <= right]
        valueTotal = [int(item[1]) for item in timeToFileCountList if int(item[0]) >= left and int(item[0]) <= right]
        if (len(keyTotal) != len(keyReal)):
            print("Time line of [ALL FILE] is not equal to that of [REAL FILE].")
            return
        # plt.title('Number of news [%d-%d]' % (left, right))
        # plt.plot(keyTotal, valueTotal, marker='o',label='All crawled articles in People\'s Daily')
        # plt.plot(keyReal, valueReal, 'k', marker='^',label='All articles containing both topics in People\'s Daily')
        fig = plt.figure()
        view1 = fig.add_subplot(111)
        view1.set_ylabel('Number of all articles for climate change coverage only')
        view1.set_xlabel('Year')
        view1.plot(keyTotal, valueTotal, marker='o', label='People\'s Daily (climate change coverage)')
        plt.ylim(0, 1500)
        for y, h in zip(keyTotal, valueTotal):
            plt.text(y, h + 0.005, '%d' % h, ha='center', va='bottom', fontsize=9)
        plt.legend(loc=1)
        view2 = view1.twinx()
        view2.set_ylabel('Number of all articles for health and climate change converage both')
        view2.plot(keyReal, valueReal, 'k', marker='^', label='People\'s Daily (health and climate change coverage)')
        plt.ylim(0, 100)
        for y, h in zip(keyReal, valueReal):
            plt.text(y, h + 0.005, '%d' % h, ha='center', va='bottom', fontsize=9)
        plt.legend(loc=4)
        # plt.xlabel('Year')
        # plt.ylabel('Number of all articles')
        plt.savefig(figPath)

    #static method for plotting time line of REAL & ALL files
    @staticmethod
    def giveDataPlotTimeLineThreeRange(figPath='fig/time.png',
                                     timeLine = [],
                                     dataOne = [],
                                     dataTwo = [],
                                     dataThree = []):
        #fig = plt.figure()
        #view1 = fig.add_subplot(111, axes_class=AA.Axes)
        view1 = host_subplot(111, axes_class=AA.Axes)
        view2 = view1.twinx()
        view3 = view1.twinx()
        plt.subplots_adjust(right=0.75)
        view1.set_ylim(0, 3500)
        view1.set_ylabel('Number of all articles for climate change coverage only')
        view1.set_xlabel('Year')
        view1.plot(timeLine, dataOne, marker='o', label='People\'s Daily (climate change coverage)')
        for y, h in zip(timeLine, dataOne):
            view1.text(y, h + 0.005, '%d' % h, ha='center', va='bottom', fontsize=9)


        view2.set_ylim(0, 100)
        view2.set_ylabel('Number of all articles for health and climate change converage both')
        view2.plot(timeLine, dataTwo, 'k', marker='^', label='People\'s Daily (health and climate change coverage)')
        ny2 = view2.get_grid_helper().new_fixed_axis
        view2.axis['right'] = ny2(loc='right', axes=view2, offset=(0, 0))
        view2.axis['right'].toggle(all=True)
        for y, h in zip(timeLine, dataTwo):
            view2.text(y, h + 0.005, '%d' % h, ha='center', va='bottom', fontsize=9)
        #plt.legend(loc=4)
        ny3=view3.get_grid_helper().new_fixed_axis
        view3.axis['right'] = ny3(loc='right', axes=view3,offset=(50, 0))
        view3.axis['right'].toggle(all=True)
        view3.plot(timeLine, dataThree, 'r', marker='x', label='People\'s Daily (health and climate change coverage after manually check)')
        view3.set_ylim(0,50)
        view3.set_ylabel('Number of all articles for health and climate change coverage both after manually check')
        for y, h in zip(timeLine, dataThree):
            view3.text(y, h + 0.005, '%d' % h, ha='center', va='bottom', fontsize=9)
        plt.legend(loc=1)
        # plt.xlabel('Year')
        # plt.ylabel('Number of all articles')
        plt.show()
        #plt.savefig(figPath)

    @staticmethod
    def giveDataPlotTimeLineTwoRange(figPath='fig/time.png',
                                     timeLine = [],
                                     dataOne = [],
                                     labelOne = "",
                                     dotOne = 'o',
                                     yLimOne = 3500,
                                     dataTwo = [],
                                     labelTwo = "",
                                     dotTwo = 'o',
                                     yLimTwo = 100):
        #fig = plt.figure()
        #view1 = fig.add_subplot(111, axes_class=AA.Axes)
        view1 = host_subplot(111, axes_class=AA.Axes)
        view1.spines['top'].set_visible(False)
        view2 = view1.twinx()
        #view3 = view1.twinx()
        plt.subplots_adjust(right=0.75)
        # plt.spines['top'].set_visible(False)
        view1.set_ylim(0, yLimOne)
        view1.set_ylabel('Number of all articles for climate change coverage only')
        view1.set_xlabel('Year')
        view1.plot(timeLine, dataOne, markersize=3, marker=dotOne, label=labelOne)
        #view1.scatter(timeLine, dataOne, label=labelOne)
        for y, h in zip(timeLine, dataOne):
            view1.text(y, h + 0.005, '%d' % h, ha='center', va='bottom', fontsize=9)


        view2.set_ylim(0, yLimTwo)
        view2.set_ylabel('Number of all articles for health and climate change converage both')
        view2.plot(timeLine, dataTwo, 'r', markersize=3, marker=dotTwo, label=labelTwo)
        #view2.spines['top'].set_visible(False)
        ny2 = view2.get_grid_helper().new_fixed_axis
        view2.axis['right'] = ny2(loc='right', axes=view2, offset=(0, 0))
        view2.axis['right'].toggle(all=True)
        for y, h in zip(timeLine, dataTwo):
            view2.text(y, h + 0.005, '%d' % h, ha='center', va='bottom', fontsize=9)
        #plt.legend(loc=4)
        # ny3=view3.get_grid_helper().new_fixed_axis
        # view3.axis['right'] = ny3(loc='right', axes=view3,offset=(50, 0))
        # view3.axis['right'].toggle(all=True)
        # view3.plot(timeLine, dataThree, 'r', marker='x', label='People\'s Daily (health and climate change coverage after manually check)')
        # view3.set_ylim(0,50)
        # view3.set_ylabel('Number of all articles for health and climate change coverage both after manually check')
        # for y, h in zip(timeLine, dataThree):
        #     view3.text(y, h + 0.005, '%d' % h, ha='center', va='bottom', fontsize=9)
        plt.legend(loc=1,frameon=False)
        # plt.xlabel('Year')
        # plt.ylabel('Number of all articles')
        plt.show()
        #plt.savefig(figPath)

    @staticmethod
    def giveDataPlotTimeLineOneRange(figPath='fig/time.png',
                                     timeLine = [],
                                     dataOne = [],
                                     labelOne = "",
                                     dotOne = 'o',
                                     yLimOne = 3500
                                     ):
        fig = plt.figure()
        view1 = fig.add_subplot(111)
        #view1 = host_subplot(111, axes_class=AA.Axes)
        #view1.spines['top'].set_visible(False)
        #plt.subplots_adjust(right=0.75)
        view1.set_ylim(0, yLimOne)
        #view1.set_ylabel('Number of all articles for climate change coverage only')
        view1.set_xlabel('Year')
        view1.plot(timeLine, dataOne, markersize=3, marker=dotOne, label=labelOne)
        #view1.scatter(timeLine, dataOne, label=labelOne)
        for y, h in zip(timeLine, dataOne):
            view1.text(y, h + 0.005, '%d' % h, ha='center', va='bottom', fontsize=9)
        plt.legend(loc=1,frameon=False)
        plt.show()

    @staticmethod
    def loadDataPlotTimeLineOneRange(left, right, figPath='fig/time.png',
                                     timeLineCountFullPath='data/timeLine/totalCount.txt'):
        with open(timeLineCountFullPath) as f:
            timeToFileCountList = [item.split() for item in f.readlines()]
            f.close()
        keyTotal = [str(item[0]) for item in timeToFileCountList if int(item[0]) >= left and int(item[0]) <= right]
        valueTotal = [int(item[1]) for item in timeToFileCountList if int(item[0]) >= left and int(item[0]) <= right]
        fig = plt.figure()
        view1 = fig.add_subplot(111)
        view1.set_xlabel('Year')
        view1.plot(keyTotal, valueTotal, marker='o', label='People\'s Daily (Searched by key words)')
        plt.ylim(0, max(valueTotal) + 200)
        for y, h in zip(keyTotal, valueTotal):
            plt.text(y, h + 0.005, '%d' % h, ha='center', va='bottom', fontsize=9)
        plt.legend(loc=1)
        plt.legend(loc=4)
        plt.savefig(figPath)

    #plot time line of REAL & ALL files AFTER dealing with all files
    def plotTimeLine(self, range, figPath = 'fig/time.png'):
        print("Plotting the time line for both types of files ...")
        timeToFileCountList = list(self.timeToFileCount.items())
        timeToFileCountList.sort(key=Analysis.getFirst)
        firstIdx = 0
        for idx, ele in enumerate(timeToFileCountList):
            if ele[0] >= range:
                firstIdx = idx
                break
        self.timeToFileCountList = timeToFileCountList[firstIdx:]

        timeToFileRealCountList = list(self.timeToFileRealCount.items())
        timeToFileRealCountList.sort(key=Analysis.getFirst)
        firstIdx = 0
        for idx, ele in enumerate(timeToFileRealCountList):
            if ele[0] >= range:
                firstIdx = idx
                break
        self.timeToFileRealCountList = timeToFileRealCountList[firstIdx:]

        keyReal = [str(item[0]) for item in self.timeToFileRealCountList]
        valueReal = [int(item[1]) for item in self.timeToFileRealCountList]
        keyTotal = [str(item[0]) for item in self.timeToFileCountList]
        valueTotal = [int(item[1]) for item in self.timeToFileCountList]
        if(len(keyTotal) != len(keyReal)):
            print("Time line of [ALL FILE] is not equal to that of [REAL FILE].")
            return
        plt.title('Number of news [%d-2018]' % range)
        plt.plot(keyTotal, valueTotal, marker='o',label='Number of articles in People\'s Daily about climate change')
        plt.plot(keyReal, valueReal, 'k', marker='^',label='Number of articles in People\'s Daily about climate change and Health')
        plt.legend(loc='best')
        plt.xlabel('Year')
        plt.ylabel('Number of articles')
        plt.savefig(figPath)
        self.saveTimeLine()

    #plot time line of REAL & ALL files AFTER dealing with all files
    def plotTimeLineBoth(self, timeLineOne, timeLineTwo, fileListOne, fileListTwo, left, right, figPath = 'fig/time.png'):
        print("Plotting the time line for both types of topics ...")
        timeToFileCountListOne = list(timeLineOne.items())
        timeToFileCountListOne.sort(key=Analysis.getFirst)
        firstIdx = 0
        for idx, ele in enumerate(timeToFileCountListOne):
            if ele[0] >= left:
                firstIdx = idx
                break
        for idx, ele in enumerate(timeToFileCountListOne):
            if ele[0] >= right:
                lastIdx = idx
                break
        timeToFileCountListOne = timeToFileCountListOne[firstIdx: lastIdx]


        timeToFileCountListTwo = list(timeLineTwo.items())
        timeToFileCountListTwo.sort(key=Analysis.getFirst)
        firstIdx = 0
        for idx, ele in enumerate(timeToFileCountListTwo):
            if ele[0] >= left:
                firstIdx = idx
                break
        for idx, ele in enumerate(timeToFileCountListTwo):
            if ele[0] >= right:
                lastIdx = idx
                break

        timeToFileCountListTwo = timeToFileCountListTwo[firstIdx:lastIdx]

        keyReal = [str(item[0]) for item in timeToFileCountListOne]
        valueReal = [int(item[1]) for item in timeToFileCountListOne]
        keyTotal = [str(item[0]) for item in timeToFileCountListTwo]
        valueTotal = [int(item[1]) for item in timeToFileCountListTwo]
        #if(len(keyTotal) != len(keyReal)):
        #    print("Time line of [ALL FILE] is not equal to that of [REAL FILE].")
        #    return
        plt.title('Number of news [%d-2018]' % left)
        plt.plot(keyTotal, valueTotal, marker='o',label='Number of articles in People\'s Daily about climate change')
        plt.plot(keyReal, valueReal, 'k', marker='^',label='Number of articles in People\'s Daily about climate change and Health')
        plt.legend(loc='best')
        plt.xlabel('Year')
        plt.ylabel('Number of articles')
        plt.savefig(figPath)
        self.saveTimeLineBoth(timeToFileCountListOne, timeToFileCountListTwo, fileListOne, fileListTwo)

    def getTfIdfMatrix(self):
        return self.tfIdfMatrix.toarray()

    def getHighLight(self, topN):
        tfIdfDense = self.tfIdfMatrix.todense()
        featureDictionary = {v: k for k, v in self.transformer.vocabulary_.items()}
        keyWords = np.argsort(-tfIdfDense)[:, :topN]
        df = pd.DataFrame(np.vectorize(featureDictionary.get)(keyWords), index=self.fileList)
        return df

    @staticmethod
    def getDistance(src, tgt):
        #for i in range(len(tgt)):
        return 0

#segementation of words
if __name__ == '__main__':
    #stopWordListPath_ = 'data/stopWord.txt'
    #sourcePath_ = 'data/text'
    #targetPath_ = 'data/segText'
    #seg = Segmentation(stopWordListPath=stopWordListPath_, sourcePath=sourcePath_, targetPath=targetPath_)
    #seg.segmentFiles()
    #Analysis.giveDataPlotTimeLineThreeRange('time.png', timeLine=range(2008, 2018 + 1), dataOne=[723, 715, 1173, 817, 719, 1015, 893, 878, 862, 688, 762], dataTwo=[19, 28, 28, 17, 12, 27, 31, 11, 19, 21, 28], dataThree=[5, 10, 6, 2, 3, 10, 8, 4, 8, 8, 10])
    #Analysis.giveDataPlotTimeLineThreeRange('time0327.png', timeLine=range(2008, 2018 + 1), dataOne=[1864, 2158, 3395, 2576, 2404, 2704, 2393, 2603, 2655, 2468, 2484], dataTwo=[19, 28, 28, 17, 12, 27, 31, 11, 19, 21, 28], dataThree=[5, 10, 6, 2, 3, 10, 8, 4, 8, 8, 10])

    #20190404
    # Analysis.giveDataPlotTimeLineTwoRange('time1.png', timeLine=range(2008, 2018 + 1), dotOne='o',dataOne=[723, 715, 1173, 817, 719, 1015, 893, 878, 862, 688, 762], labelOne='People\' Daily (climate change coverage)', yLimOne = 1500, dotTwo='o', dataTwo=[19, 28, 28, 17, 12, 27, 31, 11, 19, 21, 28], labelTwo='People\' Daily (health and climate change coverage)', yLimTwo= 100)
    # Analysis.giveDataPlotTimeLineTwoRange('time1.png', timeLine=range(2008, 2018 + 1),
    #                                       dataOne=[1864, 2158, 3395, 2576, 2404, 2704, 2393, 2603, 2655, 2468, 2484],
    #                                       labelOne='People\' Daily (searched by only "climate change" key words)', yLimOne=3500,
    #                                       dataTwo=[5, 10, 6, 2, 3, 10, 8, 4, 8, 8, 10],
    #                                       labelTwo='People\' Daily (health and climate change coverage after manually screening)', yLimTwo=50)
    Analysis.giveDataPlotTimeLineOneRange('time2.png', timeLine=range(2008, 2018 + 1),dataOne=[1864, 2158, 3395, 2576, 2404, 2704, 2393, 2603, 2655, 2468, 2484],
                                          labelOne='People\' Daily (searched by only "climate change" key words)', yLimOne=3500)