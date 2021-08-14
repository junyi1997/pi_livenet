import os
from pygame import mixer    
from gtts import gTTS
def speak_i():
    mixer.init()    # 初始化
    if not os.path.isfile('tmp.mp3'):    # 不重要的聲音檔產生器
        tts = gTTS(text = '不重要的語音檔', lang = 'zh-tw')
        tts.save('tmp.mp3')
        print('已產生不重要的語音檔 tmp.mp3')
    #-----------------#
def bot_speak(text, lang):    # 建立自訂函式
    try: 
        mixer.music.load('tmp.mp3')    # 讀取不重要的聲音檔
        tts = gTTS(text=text, lang=lang)    
        tts.save('speak.mp3')    
        mixer.music.load('speak.mp3')	    
        mixer.music.play()    # 播放重要的聲音檔
        while(mixer.music.get_busy()):    
            continue
    except:
        print('播放音效失敗')
    #-----------------#
def speak(text):
    speak_i()
    bot_speak(text,'zh-tw')  # 說出text
    
def mymain():
    speak('我是12345678901234567890')

if __name__=="__main__":
    mymain()