# 导入Selenium库，这是一个用于自动化Web浏览器操作的库  
from selenium import webdriver  
# 导入Selenium库中的支持等待的UI类  
from selenium.webdriver.support.ui import WebDriverWait, Select  
# 导入Selenium库中的触摸动作类  
from selenium.webdriver.common.touch_actions import TouchActions  
# 导入Selenium库中的键盘操作类  
from selenium.webdriver.common.keys import Keys  
# 导入Selenium库中的Chrome浏览器选项类  
from selenium.webdriver.chrome.options import Options  
# 导入Selenium库中的常见异常类  
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotVisibleException  
  
# 导入时间库，用于控制程序等待时间  
import time, random, cv2  
# 导入NumPy库，用于处理数组和矩阵  
import numpy as np  
  
# 设置要访问的网页URL，这里为本地的一个示例地址，实际使用时需要替换为实际的网页URL  
url = 'http://localhost/watermelon'  # "http://www.wesane.com/game/654"  
  
# 定义一个函数，用于处理帧图像，具体处理方式为裁剪、缩放和归一化等操作  
    def process_frame42(frame):  
        # 将帧图像裁剪为160x250大小，然后缩放为80x80  
        frame = frame[:250, :160]  
        frame = cv2.resize(frame, (80, 80))  
        # 将处理后的帧图像保存为PNG文件  
        cv2.imwrite("filename.png", frame)  
        # 求帧图像每个像素的平均值（RGB三个通道的和再除以3），然后保持维度不变，将结果张量变成二维数组  
        frame = frame.mean(2, keepdims=True)  
        # 将张量数据类型转换为float32      
        frame = frame.astype(np.float32)  
        # 将像素值从[0, 255]缩放到[0, 1]，以便后续的归一化操作  
        frame *= (1.0 / 255.0)  
        # 将通道维度移到最前面，以便后续的神经网络处理  
        frame = np.moveaxis(frame, -1, 0)  
        # 返回处理后的帧图像  
        return frame  

# 定义一个名为bigwaterlemon的类，用于控制和操作游戏浏览器  
class bigwaterlemon:  
    def __init__(self):  
        # 创建一个Chrome浏览器选项对象，并设置一些参数，例如模拟移动设备、设置用户代理等  
        chrome_options = Options()  
        self.width = 160  # 设置需要截图的宽度为160像素  
        self.height = 250  # 设置需要截图的长度为250像素  
        self.last_score = 0  # 初始化上次得分  
        self.episode_num = 0  # 初始化游戏回合数  
        mobile_emulation = {  # 设置模拟移动设备的参数  
            "deviceMetrics": {"width": self.width, "height": self.height, "pixelRatio": 3.0},  
            "userAgent": "Mozilla/5.0 (Linux; Android 4.2.1; en-us; Nexus 5 Build/JOP40D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Mobile Safari/535.19"  
        }  
        chrome_options = Options()  # 重新创建一个Chrome浏览器选项对象，并添加之前设置的参数  
        chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)  # 添加模拟移动设备的参数  
        chrome_options.add_experimental_option('w3c', False)  # 设置Selenium使用旧版W3C标准  
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])  # 禁用日志记录  
        self.browser = webdriver.Chrome(options=chrome_options)  # 使用Chrome浏览器驱动，并传入选项参数进行初始化  
        self.browser.get(url)  # 让浏览器打开指定的URL地址  
        time.sleep(10)
        self.gamecanvas = self.browser.find_element_by_id('GameCanvas')
    def get_state(self):  
        # 获取游戏画面的截图，并进行预处理  
        state = self.gamecanvas.screenshot("test.png")  
        # 使用Selenium的截图功能获取游戏画面的截图，并将其保存为test.png文件  
        state = cv2.imread("test.png")  
        # 使用OpenCV库读取刚才保存的截图文件，将其加载为图像数组  
        state = process_frame42(state)  
        # 对加载的图像进行预处理，具体处理方式由process_frame42函数实现  
        return state  
      
    def act(self, x):  
        # 定义一个函数，用于执行具体的动作  
        x = int(x)  
        # 将输入的x转换为整数  
        if x == self.width:  
            x -= 1  
        # 如果x等于self.width，那么x减1  
        actions = TouchActions(self.browser)  
        # 创建一个TouchActions对象，用于在浏览器上执行模拟的触摸动作  
        actions.tap_and_hold(x, 200)  
        # 在指定的位置（x，200）执行单击并按住的触摸动作  
        actions.move(x, 200).perform()  
        # 在指定的位置（x，200）执行移动触摸的动作，并执行这个动作  
        time.sleep(1)  
        # 等待1秒，可能是为了等待动作完成或者动画效果显示  
        actions.release(x, 200).perform()  
        # 在指定的位置（x，200）执行释放触摸的动作，并执行这个动作  
      
    def step(self, x):  
        # 定义一个函数，用于执行一系列的动作并返回结果  
        self.act(x)  
        # 调用act函数执行指定的动作  
        time.sleep(5)  
        # 等待5秒，可能是为了等待动作完成或者动画效果显示  
        score = self.browser.execute_script("return cc.js.getClassByName('GameManager').Instance.score;")  
        # 使用Selenium的execute_script方法获取游戏分数，可能是在游戏的一个JavaScript对象中获取的  
        reward = score - self.last_score  
        # 计算奖励，即当前分数与上一次分数之差  
        self.last_score = score  
        # 更新上一次的分数为当前分数  
        done = False  
        end = self.browser.execute_script("return cc.js.getClassByName('GameFunction').Instance.endOne")  
        # 使用Selenium的execute_script方法获取游戏的一个状态，可能是在游戏的一个JavaScript对象中获取的  
        if end == 1:  
            self.episode_num += 1  
            self.reset()  
            done = True  
            self.last_score = 0  
        # 如果这个状态等于1，那么表示游戏结束，重置游戏并更新episode数量，同时将done设为True并将上一次的分数设为0  
        return self.get_state(), reward, done  
      
    def reset(self):  
        # 重置游戏  
        self.browser.execute_script("cc.js.getClassByName('GameManager').Instance.RestartGame.call();")  
        # 使用Selenium的execute_script方法执行一段JavaScript代码，这段代码的功能可能是重置游戏  
        return self.get_state()  
    # 重置游戏后获取游戏的状态，并返回这个状态  
      
    def sample_action(self):  
        # 随机选择一个动作  
        return random.randint(0, self.width)  
    # 随机生成一个介于0到self.width之间的整数，作为选择的动作

  
# 定义了一个特殊的if语句，这个语句只有在当前模块（文件）被直接运行时才会执行，而不是作为模块被导入时执行  
if __name__ == "__main__":  
  
    # 创建了一个名为env的变量，它引用了（或者实例化了）一个名为bigwaterlemon的对象（类）。bigwaterlemon可能是一个定义在其它地方的环境类（比如一个游戏环境或者一个机器学习实验环境）  
    env = bigwaterlemon()  
  
    # 创建了一个for循环，这个循环会执行300次。变量i会在每次循环时从0到299变化  
    for i in range(300):  
  
        # 调用了env对象的step方法。这个方法可能返回一个状态，一个奖励和一个是否结束的标志，或者是其它一些由bigwaterlemon类定义的数据。random.randint(0, env.width)生成了一个在0到env.width（可能是环境的宽度）之间的随机整数，这个整数作为step方法的输入参数  
        r, s, d = env.step(random.randint(0, env.width))  
  
        # r变量可能存储了step方法的返回奖励  
        # s变量可能存储了step方法的返回状态  
        # d变量可能存储了step方法的返回是否结束的标志
