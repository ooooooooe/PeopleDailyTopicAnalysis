from bs4 import BeautifulSoup
class Parser:
    @staticmethod
    def parsingSearchMethod(searchContext, jsonPart, keyWords):
        #print(searchContext.text, jsonPart)
        searchPage = BeautifulSoup(searchContext.text, "lxml")
        return SearchPage(searchPage, jsonPart, keyWords)

    @staticmethod
    def parsingSpecificMethod(specificContext):
        specificPage = BeautifulSoup(specificContext.text, "lxml")
        try:
            title = specificPage.find_all('div', {'class': 'title'})[0].text.replace('/', '-')
            contextList = [str(item.text) for item in specificPage.find_all('div', {'class': 'detail_con'})[0].find_all('p')]
            context = ''.join(contextList)
            date = specificPage.find_all('div', {'class': 'sha_left'})[0].span.text
            return SpecificPage(specificPage, title, context, date)
        except:
            return None
class BasePage:
    def __init__(self, jsonPart, keyWords):
        self.jsonPart = jsonPart
        self.keyWords = keyWords

class SearchPage(BasePage):
    def __init__(self, page, jsonPart, keyWords):
        super().__init__(jsonPart, keyWords)
        self.page = page
        self.totalCount = int(page.div.div.div.next_sibling.next_sibling.next_sibling.next_sibling.div.next_sibling.next_sibling.div.next_sibling.next_sibling.ul.a.label.text)

class SpecificPage(BasePage):
    def __init__(self, page, title, context, date):
        #super().__init__(jsonPart, keyWords)
        self.title = title
        self.context = context
        self.date = date

