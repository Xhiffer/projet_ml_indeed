from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pymongo import MongoClient
from selenium import webdriver
# pprint library is used to make the output look more pretty
from pprint import pprint
# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient('localhost', 27017)
try :
    db = client['indeed']
except :
    db = client.indeed
indeed = db['indeed']





all_query_poste = ['data scientist'] #['développeur', 'data scientist', 'data analyst', 'business intelligence']
all_query_ville = ['Île-de-France'] #['Île-de-France', 'Lyon', 'Toulouse', 'Nantes','Bordeaux']


for query_poste in all_query_poste :
    for query_ville in all_query_ville :
        lien = 'https://www.indeed.fr/emplois?q=' + poste + '&l=' + ville
        browser = webdriver.Firefox(executable_path='geckodriver.exe')
        browser.get(lien)
        browser.maximize_window()
        time = 0
        while True :
            row = {}
            if time == 1 :
                print('test')
                try :
                    t = WebDriverWait(browser, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR , '#popover-close-link')))
                except :
                    pass
                t.click()
            time += 1
            post_info =  WebDriverWait(browser, 5).until(EC.presence_of_all_elements_located((By.CLASS_NAME , 'title')))
            
            ID = browser.find_elements_by_class_name('jobsearch-SerpJobCard')
            for i,post in enumerate(post_info) : 
                post.location_once_scrolled_into_view
                post.click()
                row['query_poste'] = query_poste
                row['query_ville'] = query_ville
                row['_id'] = ID[i].get_attribute("id")
                row['titre'] = WebDriverWait(browser, 5).until(EC.presence_of_element_located((By.ID , 'vjs-jobtitle'))).text
                row['nom_de_la_boite'] =  browser.find_element_by_id('vjs-cn').text
                row['adresse'] =  browser.find_element_by_id('vjs-loc').text
                row['texte'] =  browser.find_element_by_css_selector('#vjs-content').text

                try :
                    iswhat = browser.find_element_by_css_selector('div.jobMetadataHeader-itemWithIcon:nth-child(2) > span:nth-child(2)').text
                    if '$' or '€' in iswhat :
                        row['salaire'] =  iswhat
                    else :
                        row['contrat'] =  iswhat
                        try :
                                row['salaire'] =  browser.find_element_by_css_selector('div.jobMetadataHeader-itemWithIcon:nth-child(3) > span:nth-child(2)').text  #vjs-jobinfo > div:nth-child(3)
                        except :
                            pass
                except :
                    pass

                #substract with our today date
                row['date'] = browser.find_element_by_class_name('date').text
                try : 
                    row['lien_plus_info'] = browser.find_element_by_css_selector('.ws_label').get_attribute("href")
                except :
                    pass


                try:
                    indeed.insert_one(row)
                except :
                    pass
            Next = browser.find_elements_by_css_selector('.np')
            page = browser.find_element_by_css_selector('.pagination > a:nth-child(2) > span:nth-child(1)')
            try :
                Next[0].click() if (len(Next) == 1 and int(page.text) in [1,2])  else Next[1].click()
            except :
                browser.quit() 
                break