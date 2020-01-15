import re
filenames = ["data_chat"]
 
for names in filenames:
	file_handle = open(names + ".txt",encoding='utf-8')
	print(names + ".txt has been opened")
	f = open(names + "out.txt", 'w',encoding='utf-8')
	for line in file_handle:
		# print(line)
		temp = re.sub(r'.*-', '-', line)
		line = temp[2:]
		f.write(line)
		print(line)
file_handle.close()
