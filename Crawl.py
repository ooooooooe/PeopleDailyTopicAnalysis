from Parser import Parser
import requests
import json
import time
import os
import sys
import hashlib
import math

#BaseUrl = "http://data.people.com.cn/sc/ss?qs=" #need to be change
BaseUrl = r"http://data.people.com.cn/rmrb/s?type=1&qs="
SpecificBaseUrl = r"http://data.people.com.cn/rmrb/pd.html?qs="
GlobalJsonPart = {"cds":
                      [{"cdr":"AND",
                        "cds":
                            [{"fld":"title","cdr":"OR","hlt":"true","vlr":"AND","qtp":"DEF","val":""},
                             {"fld":"subTitle","cdr":"OR","hlt":"false","vlr":"AND","qtp":"DEF","val":""},
                             {"fld":"introTitle","cdr":"OR","hlt":"false","vlr":"AND","qtp":"DEF","val":""},
                             {"fld":"contentText","cdr":"OR","hlt":"true","vlr":"AND","qtp":"DEF","val":""}]}],
                  "obs":[{"fld":"dataTime","drt":"DESC"}]}
DataPath = os.getcwd() + r'/data'
TextPath = DataPath + r'/text_whole'
UrlPath = DataPath + '/url_whole.log'
ListPath = DataPath + '/list_whole.log'

NewsPerPage = 20
class Crawl:
    keyWords = []
    keyWordListSize = 0
    jsonPart = {}
    keyWordsSize = []
    searchPageList = []

    def __init__(self, keyWords):
        self.keyWords = keyWords
        self.keyWordListSize = len(keyWords)
        self.keyWordsSize = [len(keyWords[index]) for index in range(0, self.keyWordListSize)]
        self.searchPageList = []
        self.hm = HMap()
        self.urlFile = open(UrlPath, 'w+', encoding='utf-8')
        self.fileList = [''.join(item.split('-')[1:]) for item in os.listdir(TextPath)]

    def __del__(self):
        self.urlFile.close()

    def crawlComplexSearchMethod(self, startX = 0, startY = 0, abstract=True):
        if self.keyWordListSize == 2:
            print("Size of Key Word List is TWO!")
            for indexX, keyWordFirst in enumerate(self.keyWords[0]):
                indexY = 0
                if indexX < startX:
                    continue
                for indexY, keyWordSecond in enumerate(self.keyWords[1]):
                    #print(startX, startY)
                    #print(indexX, indexY)
                    if indexX <= startX and indexY < startY:
                        continue
                    self.complexSearchWord = keyWordFirst + " " + keyWordSecond
                    searchPage, jsonPart = self.crawlSearchMethod(self.complexSearchWord)
                    searchResult = Parser.parsingSearchMethod(searchPage, jsonPart, self.complexSearchWord)
                    self.searchPageList.append(searchResult)
                    time.sleep(0.1)
                    if abstract == True:
                        print(str(indexX) + " " +str(indexY) + self.complexSearchWord)
                        self.crawlContextAbstractMethod(searchResult)
                    else:
                        self.crawlUrlMethod(searchResult)

        elif self.keyWordListSize == 1:
            print("Size of Key Word List is ONE!")
            for indexX, keyWord in enumerate(self.keyWords[0]):
                if indexX < startX:
                    continue
                self.complexSearchWord = keyWord
                searchPage, jsonPart = self.crawlSearchMethod(self.complexSearchWord)
                searchResult = Parser.parsingSearchMethod(searchPage, jsonPart, keyWord)
                self.searchPageList.append(searchResult)
                time.sleep(0.1)
                if abstract == True:
                    print(str(indexX) + " " + self.complexSearchWord)
                    self.crawlContextAbstractMethod(searchResult)
                else:
                    self.crawlUrlMethod(searchResult)
        else:
            print("Too many key words lists.")
            return

    def getCrawlList(self, searchResult):
        newsIndex = []
        try:
            listUrl = BaseUrl + searchResult.jsonPart + r'&pageNo=1&pageSize=' + str(searchResult.totalCount)
            pageRequest = requests.get(listUrl)
            #print(pageRequest)
            result = Parser.parsingSearchMethod(pageRequest, searchResult.jsonPart, keyWords="")
            #for index in range(0, int(searchResult.totalCount) - 1):
            for index, item in enumerate(result.page.find_all('a', {'class': 'open_detail_link'})):
                title = item.text
                if(title not in self.fileList):
                    newsIndex.append(index)
        except:

            return None
        print("Length of New Pages' Index is %d" %(len(newsIndex)))
        return newsIndex

    def crawlContextAbstractMethod(self, searchResult):
        #totalCount = searchResult.totalCount
        newsIndex = self.getCrawlList(searchResult)
        if newsIndex == None:
            print("Get New Pages' Index Fail!")
            newsIndex = range(0, int(searchResult.totalCount) - 1)

        for index in newsIndex:
            print("Crawling No.%d News..."%(index))
            abstractUrl = SpecificBaseUrl + searchResult.jsonPart + r'&tr=A&pageNo=' + str(int(int(index) / 100) + 1) + '&pageSize=100&position=' + str(index % 100)
            #abstractUrl = SpecificBaseUrl + searchResult.jsonPart + r'&tr=A&pageNo=1&pageSize=' + str(searchResult.totalCount) +r'&position=' + str(index)
            print(abstractUrl)
            while True:
                try:
                    pageRequest = requests.get(abstractUrl.encode('utf-8').decode('unicode-escape'))
                    break
                except:
                    print('Short Sleeping...')
                    time.sleep(60)
                    continue
            abstractPage = Parser.parsingSpecificMethod(pageRequest)
            while abstractPage == None:
                print('Sleeping...')
                time.sleep(90)
                pageRequest = requests.get(abstractUrl.encode('utf-8').decode('unicode-escape'))
                abstractPage = Parser.parsingSpecificMethod(pageRequest)
            if abstractPage != None:
                if int(abstractPage.date.split('年')[0]) < 2008:
                    print("Date of Left Pages is Less than 2008...")
                    break
                filePath = TextPath + '/' + abstractPage.date + '-' + abstractPage.title
                if not os.path.exists(filePath):
                    #abstractPage
                    self.fileList.append(abstractPage.title)
                    with open(ListPath, 'a+', encoding='utf-8') as listFile:
                        listFile.write(self.complexSearchWord + '\n')
                        listFile.write(abstractPage.date + '-' + abstractPage.title + '\n')
                        listFile.close()
                    try:
                        file = open(filePath, 'w+', encoding='utf-8')
                    except:
                        print("Open File %s Failed!" % filePath)
                        continue
                    print("Saving... " + abstractPage.date + '-' + abstractPage.title)
                    file.write(abstractPage.context)
                    file.close()
            time.sleep(0.2)

    def crawlUrlMethod(self, searchResult):
        totalCount = searchResult.totalCount
        totalPage = math.ceil(float(totalCount) / float(NewsPerPage))
        for index in range(1, int(totalPage) + 1):
            listUrl = BaseUrl + searchResult.jsonPart + r'&pageNo=' + str(index) + r'&pageSize=' + str(NewsPerPage)
            pageRequest = requests.get(listUrl)
            #print(pageRequest)
            result = Parser.parsingSearchMethod(pageRequest, searchResult.jsonPart, keyWords="")
            for item in result.page.div.div.div.next_sibling.next_sibling.next_sibling.next_sibling.div.next_sibling.next_sibling.div.next_sibling.next_sibling.next_sibling.next_sibling.find_all(
                    'a', {'class': 'open_detail_link'}):
                itemUrl = item['href']
                hashValue = hash(itemUrl)
                if hashValue in self.hm.hashMap.keys():
                    continue
                self.hm.hashMap[hashValue] = 1
                self.urlFile.write(itemUrl + '\n')

    def crawlSearchMethod(self, complexSearchWord):
        jsonPart = self.buildJson(complexSearchWord)
        print(BaseUrl + jsonPart)
        return requests.get(BaseUrl + jsonPart), jsonPart

    def buildJson(self, word):
        curJsonPart = GlobalJsonPart
        for cdsElement in curJsonPart["cds"]:
            for one in cdsElement["cds"]:
                one["val"] = word
        return json.dumps(curJsonPart)

    def browseSearchResults(self ,save = True):
        if save:
            if not os.path.exists(DataPath):
                os.mkdir(DataPath)
            if not os.path.exists(TextPath):
                os.mkdir(TextPath)
            f = open(DataPath + '/index.txt', 'w', encoding='utf-8')
        for searchPage in self.searchPageList:
            if save:
                #f.write(searchPage.jsonPart['cds'][0]['cds'][0]['val'].encode('utf-8'))
                #f.write(searchPage.keyWords.encode('utf-8').decode('unicode-escape') + '\n')
                print(searchPage.keyWords)
                f.write(searchPage.keyWords +'\n')
                #f.write('\n')
                f.write(searchPage.jsonPart + '\n')
                f.write(str(searchPage.totalCount) + '\n')
            print(searchPage.jsonPart)
            print(searchPage.totalCount)
        if save:
            f.close()

class HMap:
    def __init__(self):
        self.hashMap = {}
