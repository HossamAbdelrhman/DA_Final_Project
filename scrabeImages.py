# The place where all required packages are imported.

import time
import numpy as np
import pandas as pd
# from PIL import Image
import urllib.request 
from bs4 import BeautifulSoup

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.edge.service import Service as EdgeService



from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# Put our Listing Data into pandas DataFrame.

listings_data = pd.read_csv('Airbnb_Data/listings.csv')

def save_image(img_url, img_name):
    urllib.request.urlretrieve(img_url, f'images2/{img_name}') 




print(len(listings_data['listing_url'].values[3007:]))

numper_of_d = 0

for url in listings_data['listing_url'].values[3167:]:
    numper_of_d += 1
    print(f'working on url: ({url}) and Image Numper is: ({numper_of_d})') 
    images_No = 0  
    id = int(url.split('/')[-1]) # extract the id form url
    base_url = f'{url}?modal=PHOTO_TOUR_SCROLLABLE' # get_url_of_all_img_window(url)
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    driver.get(base_url)
    WebDriverWait(driver, 20)
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, 'html.parser')    
    dialog = soup.select('.d1esrtf4[role="dialog"]')
    # print(dialog)
    WebDriverWait(driver, 20)

    if dialog :
        cls_button = driver.find_element(By.CSS_SELECTOR, ".d1esrtf4[role='dialog'] button.l1ovpqvx")
        cls_button.click()
        WebDriverWait(driver, 20)
    
    try:
        images = driver.find_elements(By.CSS_SELECTOR, 'div.d1l1iq7v img.itu7ddv')
        images = list(map(lambda img: img.get_attribute('src'), images))
        images = np.unique(np.array(images))
        images_No = len(images)
        print('images No.: ', images_No)
        all_imgs_names = []

        for i, img_url in enumerate(images):
            all_imgs_names.append(f'{id}_{i}.jpg')
            save_image(img_url, all_imgs_names[i])
    except:
        all_imgs_names = None

    driver.quit()
    #data.loc[len(data)] = {'listing_url': url, 'images_names': all_imgs_names,'images_No': images_No}
    # time.sleep(2)

#data.to_csv('data.csv', encoding='utf-8', index=False)