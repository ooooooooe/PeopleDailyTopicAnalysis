import os

def relationAnalysis(sourcePath='', baseWordList = [], futherWordList = []):
    fileList = os.listdir(sourcePath)
    baseWordContaining = 0
    futherWordContaining = 0
    for file in fileList:
        filePath = os.path.join(sourcePath, file)
        with open(filePath, 'r') as r:
            content = ''.join(r.readlines())
            flag = False
            for baseWord in baseWordList:
                if baseWord in content:
                    flag = True
                    break
            if flag:
                baseWordContaining += 1
                flagTwo = False
                for futherWord in futherWordList:
                    if futherWord in content:
                        flagTwo = True
                        break
                if flagTwo:
                    futherWordContaining += 1
                    print("Type A: %s" % file)
                else:
                    print("Type B: %s" % file)
    print("Number of articles containing basic word: %d, number of articles containing futher word: %d." %(baseWordContaining, futherWordContaining))

def moveArticlesInTopic(nameFile='', sourcePath='', targetPath='',):
    if not os.path.exists(sourcePath):
        print("Source Path Not Exists, Break")
    if not os.path.exists(targetPath):
        os.mkdir(targetPath)
    with open(nameFile, 'r') as r:
        nameList = r.readlines()
        for name in nameList:
            realName = ''.join(name.split('.')[1:]).strip()
            with open(os.path.join(targetPath, realName), 'w', encoding='utf-8') as w:
                try:
                    with open(os.path.join(sourcePath, realName), 'r') as r:
                        content = r.readlines()
                        for line in content:
                            w.write(line)
                        r.close()
                except:
                    print(realName)
                w.close()


if __name__ == '__main__':
   # baseWordList = ['1.5度', '2度']
   # futherWordList = ['气候变化', '全球变暖', '温室', '极端天气', '全球环境变化',
   #      '低碳', '可再生能源', '碳排放', '二氧化碳排放', '气候污染',
   #      '气候', '全球升温', '再生能源', 'CO2排放']
   # relationAnalysis('data/text_whole', baseWordList, futherWordList)
   moveArticlesInTopic('data/V2Whole_50_0.005/label/inTopicSplit', 'data/text_whole', 'data/final_in_topic')