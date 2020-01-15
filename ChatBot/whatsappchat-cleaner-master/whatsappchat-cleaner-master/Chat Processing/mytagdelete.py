import re
filenames = ["data_chatout"]

for names in filenames:
	file_handle = open(names + ".txt",encoding='utf-8')
	print(names + ".txt has been opened")
	f = open(names + "out1.txt", 'w',encoding='utf-8')
	for line in file_handle:
		if(line.startswith("|")):
			f.write(line)
		else:
			temp = re.sub(r'.*:', ':', line)
			line = temp[2:]
			f.write(line)

file_handle.close()