############################################
#  GUI cropper image using PIL and Tkinter #
############################################

# To use, go python pythonGUI.py 'filename'

from Tkinter import *
from PIL import ImageTk, Image
import sys, os

box=[0,0,0,0]
photo=Image.new("RGB",(1000,1000),"white")
imgName= None

def motion(event):
	box[0],box[1]=event.x, event.y
	print("Top left: (%s %s)" % (event.x, event.y))
	return

def mouseRelease(event):
	box[2],box[3]=event.x,event.y
	print("Top right: (%s %s)" % (event.x,event.y))
	return
	
def keyDown(event):
	print event.char
	rect = (box[0],box[1],box[2],box[3])
	print rect
	try:
		cropImg = photo.crop(rect)
		if cropImg.size[0]==0 and cropImg.size[1]==0:
			raise SystemError("Nothing to crop")
		cropImg.save(imgName+".jpg")
		print("donezo")
	except SystemError as e:
		print("something wong")
		pass

def binding(filename):
	global photo
	global imgName
	master = Tk()
	#master.minsize(width=500,height=500)
	photo = Image.open(filename)
	filename=filename.split(".")[0]
	imgName=filename+"_cropped"
	# print photo.size
	img = ImageTk.PhotoImage(photo)
	panel = Label(master,image=img)
	master.bind('<Button-1>',motion)
	master.bind('<ButtonRelease-1>',mouseRelease)
	master.bind('<Key>',keyDown)
	panel.pack(expand="yes")
	mainloop()

if os.path.isfile(sys.argv[1]):
	binding(sys.argv[1])

