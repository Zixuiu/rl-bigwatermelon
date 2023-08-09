from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.common.touch_actions import TouchActions
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotVisibleException
import time, random, cv2
import numpy as np

url = 'http://localhost/watermelon'  # "http://www.wesane.com/game/654"

def process_frame42(frame):
    # 处理帧图像，将其裁剪为160x250大小，然后缩放为80x80
    frame = frame[:250, :160]
    frame = cv2.resize(frame, (80, 80))
    cv2.imwrite("filename.png", frame)
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame

class bigwaterlemon:
    def __init__(self):
        chrome_options = Options()
        self.width = 160
        self.height = 250
        self.last_score = 0
        self.episode_num = 0
        mobile_emulation = {
            "deviceMetrics": {"width": self.width, "height": self.height, "pixelRatio": 3.0},
            "userAgent": "Mozilla/5.0 (Linux; Android 4.2.1; en-us; Nexus 5 Build/JOP40D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Mobile Safari/535.19"
        }

        chrome_options = Options()
        chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
        chrome_options.add_experimental_option('w3c', False)
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

        self.browser = webdriver.Chrome(options=chrome_options)
        self.browser.get(url)
        time.sleep(10)
        self.gamecanvas = self.browser.find_element_by_id('GameCanvas')

    def get_state(self):
        # 获取游戏画面的截图，并进行预处理
        state = self.gamecanvas.screenshot("test.png")
        state = cv2.imread("test.png")
        state = process_frame42(state)
        return state

    def act(self, x):
        x = int(x)
        if x == self.width:
            x -= 1
        actions = TouchActions(self.browser)
        actions.tap_and_hold(x, 200)
        actions.move(x, 200).perform()
        time.sleep(1)
        actions.release(x, 200).perform()

    def step(self, x):
        # 执行动作并返回状态、奖励和是否结束的标志
        self.act(x)
        time.sleep(5)

        score = self.browser.execute_script("return cc.js.getClassByName('GameManager').Instance.score;")
        reward = score - self.last_score
        self.last_score = score

        done = False
        end = self.browser.execute_script("return cc.js.getClassByName('GameFunction').Instance.endOne")
        if end == 1:
            self.episode_num += 1
            self.reset()
            done = True
            self.last_score = 0
        return self.get_state(), reward, done

    def reset(self):
        # 重置游戏
        self.browser.execute_script("cc.js.getClassByName('GameManager').Instance.RestartGame.call();")
        return self.get_state()

    def sample_action(self):
        # 随机选择一个动作
        return random.randint(0, self.width)


if __name__ == "__main__":
    env = bigwaterlemon()
    for i in range(300):
        r, s, d = env.step(random.randint(0, env.width))
