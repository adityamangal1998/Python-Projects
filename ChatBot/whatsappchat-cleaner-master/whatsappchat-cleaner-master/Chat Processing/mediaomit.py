import re

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

with open("data_chatoutout1.txt","r+",encoding='utf-8') as f:
    new_f = f.readlines()
    f.seek(0)
    for line in new_f:
        f.write(emoji_pattern.sub(r'', line))
    f.truncate()
