import os
import shutil
import csv
import jieba
import Util
def transDataToLabel(sourcePath='', targetPath='', topicList = []):
    if not os.path.exists(sourcePath):
        print("Source Path Not Exsit. Down.")
        return
    if os.path.exists(targetPath):
        shutil.rmtree(targetPath)
    os.mkdir(targetPath)
    fileList = os.listdir(sourcePath)
    splitTopicList = []
    for topic in topicList:
        newTopic = []
        for t in topic:
            newT = list(jieba.cut(t))
            newTopic.append(newT)
        splitTopicList.append(newTopic)

    for file in fileList:
        fileSource = os.path.join(sourcePath, file)
        fileTarget = os.path.join(targetPath, file)
        with open(fileSource, 'r') as r:
            with open(fileTarget, 'w') as w:
                content = r.readlines()
                contentList = []
                for line in content:
                    for word in line.split(' '):
                        contentList.append(word)

                labelList = []
                contentLength = len(contentList)
                for contentIdx, item in enumerate(contentList):
                    label = 0
                    for idx, topic in enumerate(splitTopicList):
                        for t in topic:
                            flag = True
                            if contentLength >= len(t) + contentIdx:
                                for tIdx, tItem in enumerate(t):
                                    if contentList[tIdx + contentIdx] != t[tIdx]:
                                        flag = False
                                if flag:
                                    label = idx + 1
                                    break
                    labelList.append(str(label))

                contentLabel = ' '.join(labelList)
                w.write(contentLabel)
                w.close()
            r.close()

def labelAnalysisTwoTopics(sourcePath = '', targetFile = '', yearFile = '', range=50):
    fileList = os.listdir(sourcePath)
    wf = open(targetFile, 'w')
    yearCount = {}
    for file in fileList:
        fileSource = os.path.join(sourcePath, file)
        with open(fileSource, 'r') as r:
            content = r.readlines()
            contentList = []
            for line in content:
                for word in line.split(' '):
                    contentList.append(word)
            topicOneList = []
            topicTwoList = []
            for idx, w in enumerate(contentList):
                if w == '1':
                    topicOneList.append(idx)
                elif w == '2':
                    topicTwoList.append(idx)
            flag = False
            for t1 in topicOneList:
                for t2 in topicTwoList:
                    if abs(int(t1) - int(t2)) <= range:
                        flag = True
            r.close()
            if flag == True:

                year = file.split('.')[1].split('年')[0]
                #print(year)
                if str(year) not in yearCount.keys():
                    #print(year)
                    yearCount[str(year)] = 1
                else:
                    yearCount[str(year)] += 1
                wf.write(file + '\n')
    wf.close()
    yearCountItems = list(yearCount.items())
    def getFirst(element):
        return element[0]

    yearCountItems.sort(key=getFirst)
    with open(yearFile, 'w') as wf:
        for item in yearCountItems:
            print(item[0])
            wf.write(str(item[0]) + ' ' + str(item[1]) + '\n')
    wf.close()
    return

def labelAnalysisOneTopic(sourcePath = '', targetFile = '', yearFile = ''):
    fileList = os.listdir(sourcePath)
    wf = open(targetFile, 'w')
    yearCount = {}
    for file in fileList:
        year = file.split('.')[1].split('年')[0]
        #print(year)
        if str(year) not in yearCount.keys():
            #print(year)
            yearCount[str(year)] = 1
        else:
            yearCount[str(year)] += 1
        wf.write(file + '\n')
    wf.close()
    yearCountItems = list(yearCount.items())
    def getFirst(element):
        return element[0]

    yearCountItems.sort(key=getFirst)
    with open(yearFile, 'w') as wf:
        for item in yearCountItems:
            print(item[0])
            wf.write(str(item[0]) + ' ' + str(item[1]) + '\n')
    wf.close()
    return

def loadCsvAnalysis(csvFilePath='', sourcePath = '', targetPath = '', prob=0.001, topicList=[]):
    if os.path.exists(targetPath):
        shutil.rmtree(targetPath)
    os.mkdir(targetPath)

    csvFile = csv.reader(open(csvFilePath, 'r'))
    for idx, file in enumerate(csvFile):
        if idx == 0:
            continue
        inTopic = True
        for topic in topicList:
            if float(file[int(topic) + 1]) < prob:
                inTopic = False
        if inTopic:
            fileSource = os.path.join(sourcePath, file[0])
            fileTarget = os.path.join(targetPath, file[0])
            with open(fileSource, 'r') as r:
                with open(fileTarget, 'w') as w:
                    content = r.readlines()
                    w.writelines(content)
        else:
            continue
def countFile(sourcePath, targetFile):
    fileList = os.listdir(sourcePath)
    wf = open(targetFile, 'w')
    yearCount = {}
    for file in fileList:
        # print(file)
        year = file.split('年')[0]
        #print(year)
        if str(year) not in yearCount.keys():
            #print(year)
            yearCount[str(year)] = 1
        else:
            yearCount[str(year)] += 1
    yearCountItems = list(yearCount.items())
    def getFirst(element):
        return element[0]

    yearCountItems.sort(key=getFirst)
    with open(targetFile, 'w') as wf:
        for item in yearCountItems:
            print(item[0])
            wf.write(str(item[0]) + ' ' + str(item[1]) + '\n')
    wf.close()
    return
if __name__ == '__main__':
    topicList = [[
            '疟疾', '腹泻', '感染', '疾病', '肺炎', '流行病',
            '公共卫生', '流行病学', '卫生保健', '卫生', '死亡率', '发病率', '营养', '疾病',
            '非传染性疾病', '传染性疾病', '传染病', '空气污染',
            '精神障碍', '发育迟缓',
            '瘟疫', '流感', '流行感冒', '治疗', '保健', '健康', '死亡'],
        ['气候变化', '全球变暖', '温室', '极端天气', '全球环境变化',
         '低碳', '可再生能源', '碳排放', '二氧化碳排放', '气候污染',
         '气候', '全球升温', '再生能源', 'CO2排放']]
    #transDataToLabel('data/V2Whole_50_0.01/timeLineWhole/inTopic/Two', 'data/V2Whole_50_0.01/segLabelWhole', topicList)

    # labelAnalysisTwoTopics('data/V2Whole_50_0.01/segLabelWhole', 'data/V2Whole_50_0.01/label/inTopic',
    #  'data/V2Whole_50_0.01/label/inTopicYear', 50)
    # transDataToLabel('data/V2Whole_50_0.01/segTextIdxWhole', 'data/V2Whole_50_0.01/segLabelWholeV1', topicList)
    # labelAnalysisTwoTopics('data/V2Whole_50_0.01/segLabelWholeV1', 'data/V2Whole_50_0.01/labelV1/inTopic',
    #                        'data/V2Whole_50_0.01/labelV1/inTopicYear', 50)
    # loadCsvAnalysis('data/V2Whole_50_0.005/fileTopicWhole.csv', 'data/V2Whole_50_0.005/segTextIdxWhole', 'data/V2Whole_50_0.005/inTopic', 0.005, [6,12])
    # labelPath = 'data/V2Whole_50_0.005/segLabelWhole'
    # if os.path.exists(labelPath):
    #     shutil.rmtree(labelPath)
    # os.mkdir(labelPath)
    # transDataToLabel('data/V2Whole_50_0.005/inTopic', 'data/V2Whole_50_0.005/segLabel', topicList)
    # labelAnalysisTwoTopics('data/V2Whole_50_0.005/segLabel', 'data/V2Whole_50_0.005/label/inTopicSplit',
    #                        'data/V2Whole_50_0.005/label/inTopicYearSplit', 50)
    # loadCsvAnalysis('data/V2Whole_50_0.005/fileTopicWhole.csv', 'data/V2Whole_50_0.005/segTextIdxWhole', 'data/V2Whole_50_0.005/inTopicOne', 0.005, [6])
    # labelAnalysisOneTopic('data/V2Whole_50_0.005/inTopicOne', 'data/V2Whole_50_0.005/label/inTopicOneSplit',
    #                        'data/V2Whole_50_0.005/label/inTopicYearOneSplit')
    countFile('data/text_whole', 'data/textWholeCount.txt')
    Util.Analysis.loadDataPlotTimeLineOneRange(2008, 2018, 'data/totalCountArticles.png', 'data/textWholeCount.txt',)