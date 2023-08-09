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
    # 使用Python的切片语法，将输入图像裁剪为160x250的大小。这里，":250"表示取图像的最后250行，":160"表示取图像的前160列
    # 这意味着我们只保留了原始图像的一个子区域，裁剪后的区域大小为160x250
    # 然后，这个裁剪后的图像被赋值给变量frame，因此后续的操作都是在这个裁剪后的图像上进行的
    # 处理帧图像，将其裁剪为160x250大小，然后缩放为80x80
    frame = frame[:250, :160] 
    # 使用OpenCV的resize函数，将裁剪后的图像的大小调整为80x80。这个函数的第一个参数是源图像，第二个参数是目标大小
    # 这个操作并不会改变图像的实际像素值，只是改变了图像的大小
    frame = cv2.resize(frame, (80, 80)) 
    # 使用OpenCV的imwrite函数，将处理后的图像保存为一个名为"filename.png"的文件。这个函数的第一个参数是源图像，第二个参数是文件名
    cv2.imwrite("filename.png", frame) 
    # 使用NumPy的mean函数，计算图像在颜色通道（即最后一个维度）上的平均值。这个函数的第一个参数是源数组，第二个参数是沿着哪个维度计算平均值。这里，我们设置keepdims=True，表示在计算平均值后保留原数组的维度
    # 这个操作将每个像素的值转换为灰度值
    frame = frame.mean(2, keepdims=True) 
    # 使用NumPy的astype函数，将数组的数据类型转换为np.float32。这个函数的第一个参数是源数组，第二个参数是目标数据类型
    # 这个操作将像素值转换为浮点数
    frame = frame.astype(np.float32) 
    # 使用NumPy的乘法运算符和常数1.0 / 255.0，将每个像素值乘以(1.0 / 255.0)，相当于将像素值归一化到[0, 1]区间内
    # 这个操作将像素值归一化到[0, 1]区间内
    frame *= (1.0 / 255.0) 
    # 使用NumPy的moveaxis函数，将数组的维度顺序改变。这个函数的第一个参数是源数组，第二个参数是要移动的轴的索引，第三个参数是移动后的目标位置
    # 这个操作将最后一个维度移到最前面，使得处理后的图像成为一个灰度图像
    frame = np.moveaxis(frame, -1, 0) 
    # 返回处理后的图像
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
