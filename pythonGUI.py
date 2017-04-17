############################################
#  GUI cropper image using PIL and Tkinter #
############################################

# To use, go python pythonGUI.py 'filename'

from Tkinter import *
from PIL import ImageTk, Image
import sys, os

box=[0,0,0,0]
boxes=[]
master = Tk()
startx,starty=0,0
rec=0
recs=[]
mainCanvas = Canvas(master,width=600,height=600)
photo=Image.new("RGB",(1000,1000),"white")
album=[]
counter=0

def motion(event):
	global box,startx, starty
	box[0],box[1]=event.x, event.y
	startx,starty = event.x,event.y
	print("Top left: (%s %s)" % (event.x, event.y))
	return

def mouseMotion(event):
	global box, boxes, rec
	mainCanvas.delete(rec)
	rec=mainCanvas.create_rectangle(startx,starty,event.x,event.y)
	rec_coor = [startx,starty,event.x,event.y]
	recs.append(rec_coor)
	return

def mouseRelease(event):
	global box, boxes, rec,recs, counter
	box[2],box[3]=event.x,event.y
	boxes.append(box)
	mainCanvas.create_rectangle(boxes[counter][0],boxes[counter][1],
		boxes[counter][2],boxes[counter][3])
	box=[0,0,0,0]
	counter+=1
	print counter
	print("Top right: (%s %s)" % (event.x,event.y))
	print boxes
	return

def keyDown(event):
	global album
	print event.char
	for b in boxes:
		rect = (b[0],b[1],b[2],b[3])
		print rect
		try:
			cropImg = photo.crop(rect)
			if cropImg.size[0]==0 and cropImg.size[1]==0:
				raise SystemError("Nothing to crop")
			cropImg.save(str(boxes.index(b))+".jpg")
			album.append(cropImg)
			print("done")
		except SystemError as e:
			print("something wong")
			pass
	print album

def binding(filename):
	global photo
	photo = Image.open(filename)
	img = ImageTk.PhotoImage(photo)
	mainCanvas.config(width=photo.size[0],height=photo.size[1])
	mainCanvas.create_image(photo.size[0]/2,photo.size[1]/2,image=img)
	master.bind('<Button-1>',motion)
	master.bind('<Button1-Motion>',mouseMotion)
	master.bind('<ButtonRelease-1>',mouseRelease)
	master.bind('<Key>',keyDown)
	mainCanvas.pack(expand="yes")
	mainloop()

if os.path.isfile(sys.argv[1]):
	binding(sys.argv[1])

