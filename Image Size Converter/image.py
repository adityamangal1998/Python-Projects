import cv2
img = cv2.imread('Visiting Card (Back).jpg')
print(img.shape)
width = img.shape[1] * 0.264583333
height = img.shape[0] * 0.264583333
print(width)
print(height)
desired_width = int((width-208.92)/0.264583333)
desired_height = int((height-127.85)/0.264583333)
print(desired_width)
print(desired_height)
# scale_percent = 30  # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
dim = (desired_width, desired_height)
# # resize image
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imwrite('new.jpg',img)
cv2.imshow('open',img)
cv2.waitKey(0)
cv2.destroyAllWindows