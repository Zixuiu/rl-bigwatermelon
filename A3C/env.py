from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.common.touch_actions import TouchActions
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotVisibleException
import time, random, cv2
import numpy as np

# 上面的代码是使用Selenium库来模拟游戏操作的示例代码。它定义了一个名为bigwaterlemon的类，该类包含了获取游戏状态、执行动作、重置游戏等方法。
# 在主程序中，创建了一个bigwaterlemon的实例，并执行了300次循环操作，每次循环中随机选择一个动作并调用step方法获取状态、奖励和结束标志。

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
# 定义一个名为bigwaterlemon的类  
class bigwaterlemon:  
  
    # 定义类的初始化方法  
    def __init__(self):  
  
        # 创建一个Chrome选项对象，该对象可以被webdriver.Chrome用来配置浏览器驱动  
        chrome_options = Options()  
  
        # 设置width为160，height为250，可能是用于设置模拟设备的尺寸  
        self.width = 160  
        self.height = 250  
  
        # 设置last_score为0，可能是用于记录最近的得分  
        self.last_score = 0  
  
        # 设置episode_num为0，可能是用于记录游戏集数或某一阶段次数  
        self.episode_num = 0  
  
        # 创建一个字典，定义了移动设备模拟的参数，包括设备的宽度、高度和像素比率，以及用户代理字符串  
        mobile_emulation = {  
            "deviceMetrics": {"width": self.width, "height": self.height, "pixelRatio": 3.0},  
            "userAgent": "Mozilla/5.0 (Linux; Android 4.2.1; en-us; Nexus 5 Build/JOP40D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Mobile Safari/535.19"  
        }  
  
        # 将移动设备模拟选项添加到Chrome选项对象中  
        chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)  
  
        # 将w3c选项设置为False，可能用于关闭某些w3c标准兼容性相关的设置  
        chrome_options.add_experimental_option('w3c', False)  
  
        # 将'enable-logging'这个开关排除，可能用于关闭某些日志或调试功能  
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])  
  
        # 使用Chrome选项对象配置Chrome浏览器驱动，创建浏览器对象  
        self.browser = webdriver.Chrome(options=chrome_options)  
  
        # 通过浏览器对象打开指定的url，可能是游戏页面  
        self.browser.get(url)  
  
        # 等待10秒钟，可能用于等待页面加载或游戏初始化完成  
        time.sleep(10)  
  
        # 通过浏览器对象和id找到页面中的GameCanvas元素，并将其赋值给self.gamecanvas，可能是用于后续的游戏操作  
        self.gamecanvas = self.browser.find_element_by_id('GameCanvas')
# 定义一个名为get_state的方法，它是对象（self）的成员方法  
    def get_state(self):  
        # 使用self.gamecanvas.screenshot方法获取游戏画面的截图，并将其保存为名为"test.png"的图像文件  
        # 这个方法可能执行一些预处理步骤，但在这段代码中没有显示  
        # 截图结果被存储在变量state中  
        # 获取游戏画面的截图，并进行预处理  
            state = self.gamecanvas.screenshot("test.png")  
          
        # 使用cv2库的imread方法读取名为"test.png"的图像文件，并将其存储在变量state中  
        # OpenCV（cv2）是一个开源的计算机视觉和机器学习软件库，包含了多种图像处理和计算机视觉的算法  
        # 读取图像文件  
            state = cv2.imread("test.png")  
          
        # 调用process_frame42函数对读取的图像进行处理，并将结果存储在state中。这段代码没有提供process_frame42的详细内容，我们只能从函数名推测它可能对图像帧进行某种处理  
        # 对图像帧进行处理  
            state = process_frame42(state)  
          
        # 返回处理后的图像帧（状态）  
        return state  
      
# 定义一个名为act的方法，它是对象（self）的成员方法  
    def act(self, x):  
        # 将参数x转换为整数，并存储在变量x中  
        x = int(x)  
          
        # 如果x等于self.width（可能是一个表示画面宽度的属性），那么将x减1  
        if x == self.width:  
            x -= 1  
          
        # 创建一个TouchActions对象，该对象提供模拟触摸屏幕的操作。参数是self.browser，可能是一个表示浏览器实例的变量  
        actions = TouchActions(self.browser)  
          
        # 在坐标(x, 200)执行单击并保持的动作，然后移动到同一位置  
        actions.tap_and_hold(x, 200)  
        actions.move(x, 200)  
          
        # 执行上述定义的动作序列  
        actions.perform()  
          
        # 暂停1秒，可能用于等待动作执行完成或等待用户反馈  
        time.sleep(1)  
          
        # 在坐标(x, 200)执行松开的手势，然后再次执行上述动作序列  
        actions.release(x, 200)  
        actions.perform()
        # 定义一个名为step的方法，它是对象（self）的成员方法。该方法接受一个参数x，可能表示要执行的动作。  
    def step(self, x):  
        # 调用act方法执行动作x  
        # 执行动作并返回状态、奖励和是否结束的标志  
        self.act(x)  
        time.sleep(5)  
      
        # 使用浏览器对象执行一段JavaScript代码，这段代码返回游戏得分。这里假设有一个名为'cc.js.getClassByName('GameManager').Instance'的对象，并且该对象有一个名为'score'的属性。  
        score = self.browser.execute_script("return cc.js.getClassByName('GameManager').Instance.score;")  
        # 计算奖励，即当前得分与上一个得分之差  
        reward = score - self.last_score  
        # 更新上一个得分为当前得分  
        self.last_score = score  
      
        # 初始化结束标志为False  
        done = False  
        # 执行另一段JavaScript代码，获取名为'cc.js.getClassByName('GameFunction').Instance'的对象的'endOne'属性值。这里假设该对象有一个名为'endOne'的属性。  
        end = self.browser.execute_script("return cc.js.getClassByName('GameFunction').Instance.endOne")  
        # 如果end等于1，则表示游戏结束，增加episode数，重置游戏，设置结束标志为True，并将上一个得分重置为0。  
        if end == 1:  
            self.episode_num += 1  
            self.reset()  
            done = True  
            self.last_score = 0  
        # 返回当前状态、奖励和结束标志  
        return self.get_state(), reward, done  
      
    # 定义一个名为reset的方法，它是对象（self）的成员方法。该方法用于重置游戏。  
    def reset(self):  
        # 执行一段JavaScript代码，调用名为'cc.js.getClassByName('GameManager').Instance'的对象的'RestartGame'方法。这里假设该对象有一个名为'RestartGame'的方法，用于重置游戏。  
        self.browser.execute_script("cc.js.getClassByName('GameManager').Instance.RestartGame.call();")  
        # 返回重置后的状态  
        return self.get_state()  
      
    # 定义一个名为sample_action的方法，它是对象（self）的成员方法。该方法随机选择一个动作。  
    def sample_action(self):  
        # 从0到宽度（self.width）之间随机选择一个整数，作为要执行的动作  
        return random.randint(0, self.width)  
      
    # 如果当前模块作为主程序执行，则执行以下代码块。创建一个名为bigwaterlemon的实例，然后执行300次循环操作。在每次循环中，随机选择一个动作，并调用step方法获取状态、奖励和结束标志。  
    if __name__ == "__main__":  
        env = bigwaterlemon()  
        for i in range(300):  
            r, s, d = env.step(random.randint(0, env.width))
