from Crawl import Crawl
from Util import Analysis
import os
from Util import Segmentation
from Model import Kmeans
####################################################################################
# 将分别属于两个主题的关键词列表中的词汇两两组合进行搜索，可以爬取到同时包含两个主题中关键词的文章
####################################################################################
def crawl():
    keyWords = [
                [
                '疟疾', '腹泻', '感染', '疾病', '肺炎', '流行病',
                '公共卫生', '流行病学', '卫生保健', '卫生', '死亡率', '发病率', '营养', '疾病',
                 '非传染性疾病', '传染性疾病', '传染病', '空气污染',
                 '精神障碍', '发育迟缓',
                 '传染', '疾患', '症', '病', '瘟疫', '流感', '流行感冒', '治疗', '保健', '健康', '死亡'],
                ['气候变化', '全球变暖', '温室', '极端天气', '全球环境变化',
                  '低碳',  '可再生能源', '碳排放', '二氧化碳排放', '气候污染',
                 '气候', '全球升温', '再生能源', 'CO2排放']]
    crawler = Crawl(keyWords)
    crawler.crawlComplexSearchMethod()

####################################################################################
# 对文章进行词频分析（已弃用）
####################################################################################
def analysisTfIdf():
    wordCountPath = 'data/wordCount.txt'
    wordCount = 'data/wordCount.txt'
    stopWordListPath = 'data/stopWord.txt'
    sourceTextPath = 'data/text'
    segSourceTextPath = 'data/segText'
    tfIdfPath = 'data/tfidf.txt'
    if not os.path.exists(segSourceTextPath):
        os.mkdir(segSourceTextPath)
    ana = Analysis(wordCountPath, stopWordListPath, sourceTextPath, segSourceTextPath, tfIdfPath)
    ana.getTfIdf()
    return

def loadDataForTrain(dataPath):
    print("Loading data...")
    with open(dataPath, 'r', encoding='utf-8') as rf:
        data = [line.split(' ') for line in rf.readlines()]
        print("Data size is %d" % (len(data)))
        feature = [[float(x) for x in item[1:]] for item in data]
        title = [item[0] for item in data]
    return feature, title

####################################################################################
# 利用TF-IDF进行Kmeans聚类，来进行无监督文本聚类（已弃用）
####################################################################################
def trainKMeans(data, leftRange = 2, rightRange = 8, title=None):
    Kmeans.multiPredict(data=data, leftRange=leftRange, rightRange=rightRange, title=title)

####################################################################################
# 利用Kmeans聚类出同时包含两种主题的文章（已弃用）
####################################################################################
def analysisKmeans(debug=False):
    keyWordPath = 'data/keyWord.csv'
    sourceTextPath = 'data/segTextIdx'
    minDf = 0.002
    maxDf = 0.6
    anaV1 = Analysis(sourceTextPath, maxDf, minDf, debug)
    anaV1.showFrequency()
    trainKMeans(anaV1.getTfIdfMatrix())
    df = anaV1.getHighLight(10)
    df.to_csv(keyWordPath, encoding='utf-8')

####################################################################################
# 利用LDA模型对所有文本进行主题抽取
#
# Parameter:
#   topicCount: 设定LDA主题抽取个数（根据需要主题的存在与否来确定选择主题的个数，保证需要的主题能在抽取目标中）
#   realTopics: 根据需要选择的文章所包含的主题来确定，
#                   需要用同一组参数对模型训练两次。
#                       第一次训练根据打印出的每个主题所包含的关键词来确定需要的关键词，
#                       第二次训练利用选择出来的主题对模型重新训练然后抽取出所需要的文章
#                   TODO：只训练一次，保存每一个模型所包含每一个主题的概率，然后直接根据概率来抽取
####################################################################################
def analysisLDA(debug=False, version='V1', sourceTextPath = 'segTextIdx', topicListPath = 'topic.txt',
                fileTopicPath = 'fileTopic.csv', fileTopTopicPath = 'fileTopTopic.csv', timeLinePath = 'timeLine',
                topicCount = 10, realTopics = [0], thredhold = 0.01):
    sourceTextPath_ = 'data/' + version + '/' + sourceTextPath
    topicListPath_ = 'data/' + version + '/' + topicListPath
    fileTopicPath_ = 'data/' + version + '/' + fileTopicPath
    fileTopTopicPath_ = 'data/' + version + '/' + fileTopTopicPath
    timeLinePath_ = 'data/' + version + '/' + timeLinePath
    if not os.path.exists(timeLinePath_):
        os.mkdir(timeLinePath_)
    minDf = 0.002
    maxDf = 0.6
    LDA = Analysis(sourceTextPath_, topicListPath_, fileTopicPath_, timeLinePath_, maxDf, minDf, debug, maxIter = 30,
                   topic = topicCount)
    LDA.LatentDirichletAllocation()
    LDA.saveTopicWords(topN=20)
    topics = LDA.extractTopicByProb(thredhold)
    topics.to_csv(fileTopTopicPath_, encoding='utf-8')
    LDA.timeLineAnalysis(realTopics, thredhold)
    LDA.plotTimeLine(2008)

#####################################################################################
# 利用LDA模型对所有文本进行主题抽取，可以抽取出两种主题，并且画在同一个图中
#
# Parameter:
#   topicCount: 设定LDA主题抽取个数（根据需要主题的存在与否来确定选择主题的个数，保证需要的主题能在抽取目标中）
#   realTopicsOne, realTopicsTwo: 根据需要选择的文章所包含的主题来确定，
#                   需要用同一组参数对模型训练两次。
#                       第一次训练根据打印出的每个主题所包含的关键词来确定需要的关键词，
#                       第二次训练利用选择出来的主题对模型重新训练然后抽取出所需要的文章
#                   TODO：只训练一次，保存每一个模型所包含每一个主题的概率，然后直接根据概率来抽取
####################################################################################
def analysisLDABoth(debug=False, version='V1', sourceTextPath = 'segTextIdx', topicListPath = 'topic.txt',
                fileTopicPath = 'fileTopic.csv', fileTopTopicPath = 'fileTopTopic.csv', timeLinePath = 'timeLine',
                topicCount = 10, realTopicsOne = [0], realTopicsTwo = [0], thredhold = 0.01, maxIter = 30):
    sourceTextWholePath = 'data/' + version + '/' + sourceTextPath
    topicListWholePath = 'data/' + version + '/' + topicListPath
    fileTopicWholePath = 'data/' + version + '/' + fileTopicPath
    fileTopTopicWholePath = 'data/' + version + '/' + fileTopTopicPath
    timeLineWholePath = 'data/' + version + '/' + timeLinePath
    if not os.path.exists(timeLineWholePath):
        os.mkdir(timeLineWholePath)
    minDf = 0.002
    maxDf = 0.6
    LDA = Analysis(sourceTextWholePath, topicListWholePath, fileTopicWholePath, timeLineWholePath, maxDf, minDf, debug, maxIter = maxIter,
                   topic = topicCount)
    LDA.LatentDirichletAllocation()
    LDA.saveTopicWords(topN=20)
    #topics = LDA.extractTopic(3)
    topics = LDA.extractTopicByProb(thredhold)
    topics.to_csv(fileTopTopicWholePath, encoding='utf-8')
    timeLineOne, timeLineTwo, fileListOne, fileListTwo = LDA.timeLineAnalysisBoth(realTopicsOne, realTopicsTwo, thredhold)
    #LDA.plotTimeLine(2008)
    LDA.plotTimeLineBoth(timeLineOne, timeLineTwo, fileListOne, fileListTwo, 2008, 2018)

####################################################################################
# 分词，为文章添加序号
####################################################################################
def segmentation(version = 'V1', sourcePath = 'text', targetPath = 'segText', targetIdxPath = 'segTextIdx', left = 2008):
    if not os.path.exists('data/' + version):
        os.mkdir('data/' + version)
    stopWordListWholePath = 'data/stopWord.txt'
    sourceWholePath = 'data/' + sourcePath + '/'
    targetWholePath = 'data/' + version + '/' + targetPath
    if not os.path.exists(targetWholePath):
        os.mkdir(targetWholePath)
    seg = Segmentation(stopWordListPath=stopWordListWholePath, sourcePath=sourceWholePath, targetPath=targetWholePath)
    seg.segmentFiles()
    targetIdxListWholePath = 'data/' + version + '/' + targetIdxPath
    if not os.path.exists(targetIdxListWholePath):
        os.mkdir(targetIdxListWholePath)
    fileList = os.listdir(targetWholePath)
    realIdx = 0
    for idx, file in enumerate(fileList):
        with open(os.path.join(targetWholePath, file), 'r', encoding='utf-8') as f:
            if int(file.split('年')[0]) < left:
                continue
            r = ' '. join(f.readlines())
            idxStr = str(realIdx).zfill(4)
            realIdx += 1
            if idx % 1000 == 0:
                print(idxStr)
            with open(os.path.join(targetIdxListWholePath, idxStr + u'.' + file.replace(' ', '')), 'w', encoding='utf-8') as w:
                w.write(r)
                w.close()
            f.close()

####################################################################################
# 直接根据统计结果进行绘图
####################################################################################
def plotPictures(figPath = 'data/V2/timeLine/time.png', timeLineCountFullPath = 'data/V2/timeLine/totalCount.txt',
                 timeLineTopicCountFullPath='data/V2/timeLine/totalTopicCount.txt', left = 2008, right = 2018):
    Analysis.loadDataPlotTimeLineTwoRange(left, right, figPath, timeLineCountFullPath, timeLineTopicCountFullPath)


if __name__ == '__main__':
    #crawl()
    segmentation(version='V3')
    analysisLDA(False, 'V3', topicCount=50, realTopics=[0], thredhold=0.01)

    # analysisKmeans()

    # segmentation('V2Whole', 'text_whole', 'segTextWhole')
    # analysisLDABoth(False, 'V2Whole_50_0.01', 'segTextIdxWhole', 'topicWhole.txt', 'fileTopicWhole.csv', 'fileTopTopicWhole.csv',
    #             'timeLineWhole', 50, [6], [6, 12], 0.01, 30)
    #plotPictures('data/V2Whole_50_0.005/time.png', 'data/V2Whole_50_0.005/label/inTopicYearOneSplit', 'data/V2Whole_50_0.005/label/inTopicYearSplit')