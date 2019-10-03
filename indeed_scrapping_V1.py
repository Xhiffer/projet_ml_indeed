from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pymongo import MongoClient
from selenium import webdriver
from datetime import date
from datetime import date,datetime,timedelta
import time as t
import re
import sys
# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient('localhost', 27017)
try :
    db = client['indeed2']
except :
    db = client.indeed2
indeed2 = db['indeed2']





all_query_poste = ['développeur'] #['développeur','business intelligence'] #['data scientist','data analyst'] 
all_query_ville = ['Lyon', 'Toulouse', 'Nantes','Bordeaux','Île-de-France']

try :
    for query_poste in all_query_poste :
        for query_ville in all_query_ville :
            lien = 'https://www.indeed.fr/emplois?q=' + query_poste + '&l=' + query_ville
            browser = webdriver.Firefox(executable_path='geckodriver.exe')
            browser.get(lien)
            browser.maximize_window()
            time = 0
            while True :
                row = {}
                if time == 1 :
                    print('test')
                    try :
                        t.sleep(0.5)
                        CLICK = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR , '#popover-close-link')))
                        CLICK.click()
                        t.sleep(0.5)
                    except :
                        pass
                time += 1
                post_info =  WebDriverWait(browser, 5).until(EC.presence_of_all_elements_located((By.CLASS_NAME , 'title')))

                ID = browser.find_elements_by_class_name('jobsearch-SerpJobCard')
                for i,post in enumerate(post_info) : 
                    post.location_once_scrolled_into_view
                    post.click()
                    row['query_poste'] = query_poste
                    row['query_ville'] = query_ville
                    row['_id'] = ID[i].get_attribute("id")
                    row['titre'] = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.ID , 'vjs-jobtitle'))).text
                    row['nom_de_la_boite'] =  browser.find_element_by_id('vjs-cn').text
                    row['adresse'] =  browser.find_element_by_id('vjs-loc').text
                    row['texte'] =  browser.find_element_by_css_selector('#vjs-content').text

                    
                    
                    try :
                        iswhat = browser.find_element_by_css_selector('div.jobMetadataHeader-itemWithIcon:nth-child(2) > span:nth-child(2)').text
                        if any(char.isdigit() for char in iswhat) == True :
                            row['salaire'] =  iswhat
                        else :
                            row['contrat'] =  iswhat
                            try :
                                    row['salaire'] =  browser.find_element_by_css_selector('div.jobMetadataHeader-itemWithIcon:nth-child(3) > span:nth-child(2)').text  #vjs-jobinfo > div:nth-child(3)
                            except :
                                pass
                    except :
                        pass
                    try :
                        date_text = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR , '#vjs-footer'))).text
                        date_text = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR , '#vjs-footer'))).text
                        date_value = re.findall(r'\d+|jour|moi|heure', date_text)
                        if date_value[1] in 'mois' :
                            row['date'] = (datetime.today() - timedelta(days= (int(date_value[0])*30.5))).strftime("%d/%m/%Y, %H:%M:%S")
                        elif date_value[1] in 'minutes' :
                            row['date'] = (datetime.today() - timedelta(minutes=int(date_value[0]))).strftime("%d/%m/%Y, %H:%M:%S")
                        elif date_value[1] in 'jours' :
                            row['date'] = (datetime.today() - timedelta(days=int(date_value[0]))).strftime("%d/%m/%Y, %H:%M:%S")
                        elif date_value[1] in 'heures' :
                            row['date'] = (datetime.today() - timedelta(hours=int(date_value[0]))).strftime("%d/%m/%Y, %H:%M:%S")
                    except :
                        pass
                    try : 
                        row['lien_plus_info'] = browser.find_element_by_css_selector('.ws_label').get_attribute("href")
                    except :
                        pass


                    try:
                        indeed2.insert_one(row)
                    except :
                        pass
                Next = browser.find_elements_by_css_selector('.np')
                page = browser.find_element_by_css_selector('.pagination > a:nth-child(2) > span:nth-child(1)')
                try :
                    Next[0].click() if (len(Next) == 1 and time == 1)  else Next[1].click()
                except :
                    browser.quit() 
                    break
except Exception as e: 
    print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
    print('query_poste ' + query_poste) 
    print('\n query_ville ' + query_ville)