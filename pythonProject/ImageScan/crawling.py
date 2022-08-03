from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

driver = webdriver.Chrome('C:\\KOPO\\유틸\\chromedriver_win32 - 복사본\\chromedriver.exe')
driver.get("https://www.google.co.kr/imghp?h1=ko&tab=wi&ogb1") #이미지검색
elem = driver.find_element_by_name("q") #검색창을 찾아서
elem.send_keys("식용버섯 사진")
elem.send_keys(Keys.RETURN) #엔터키

###################스크롤바 내림################################
SCROLL_PAUSE_TIME = 1

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight") #자바스크립트의 높이를 계산한다.
cnt1 = 1

while True: # =무한반복
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  #브라우저 끝까지

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME) # 브라우저가 로딩될때 까지 기다리고

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height: #브라우저 끝이 측정된 새로운 높이와 같으면 브레이크
        try:
            driver.find_element_by_css_selector(".mye4qd").click()
        except:
            break
    last_height = new_height

###################################################

images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd") #작은이미지 선택

cnt = 1
for image in images:
    try:
        image.click()   #클릭하면 큰이미지가 나올텐데
        time.sleep(3)
        #imageUrl = driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img").get_attribute("src") #큰 이미지 선택
        #imageUrl = driver.find_element_by_xpath("//*[@id='Sva75c']/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img").find_element_by_tag_name()
        imageUrl = driver.find_element_by_xpath("//*[@id='Sva75c']/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]").find_element_by_tag_name("img").get_attribute("src")
        #"/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img"

        print(imageUrl)
        PathUrl = "C:\\Users\\KOPO\\Desktop\\사진\\식용버섯\\"

        urllib.request.urlretrieve(imageUrl, PathUrl + str(cnt) + ".jpg")
        cnt += 1
    except:
        pass

driver.close()