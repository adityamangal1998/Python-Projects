import time
import pyttsx3

voiceEngine = pyttsx3.init()
rate = voiceEngine.getProperty('rate')
volume = voiceEngine.getProperty('volume')
voice = voiceEngine.getProperty('voice')

present_time = time.ctime()
time = int(present_time[11:13])
# SystemExit

if 17 >= time >= 12:
    mytext = "Good Afternoon Vicky"
elif 20 >= time > 17:
    mytext = "Good Evening Vicky"
elif 23 >= time > 17 or 4 > time >= 0:
    mytext = "Good Night Vicky"
elif 12 > time >= 4:
    mytext = "Good Morning Vicky"

newVoiceRate = 120
newVolume = 0.8
voiceEngine.setProperty('rate', newVoiceRate)
voiceEngine.setProperty('volume', newVolume)
voiceEngine.say(mytext)
voiceEngine.runAndWait()




