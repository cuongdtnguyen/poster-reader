############################################
#  GUI cropper image using PIL and Tkinter #
############################################

# To use, go python pythonGUI.py 'filename'

from Tkinter import *
from PIL import ImageTk, Image
import sys,os  
import numpy
from word import recognize
#import word

box=[0,0,0,0]
boxes=[]
master = Tk()
startx,starty=0,0
rec=0
mainCanvas = Canvas(master,width=600,height=600)
album=[]
counter=0
with open('lexicon.txt') as lex_file:
	lexicon = [line.rstrip('\n') for line in lex_file]

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
	return

def mouseRelease(event):
	global box, boxes, counter
	box[2],box[3]=event.x,event.y
	boxes.append(box)
	mainCanvas.create_rectangle(boxes[counter][0],boxes[counter][1],
		boxes[counter][2],boxes[counter][3])
	box=[0,0,0,0]
	counter+=1
	#print counter
	print("Top right: (%s %s)" % (event.x,event.y))
	#print boxes
	return

def keyDown(event):
	global album
	#print event.char
	for b in boxes:
		rect = (b[0],b[1],b[2],b[3])
		#print rect
		try:
			cropImg = photo.crop(rect)
			if cropImg.size[0]==0 and cropImg.size[1]==0:
				raise SystemError("Nothing to calculate")
			# cropImg.save(str(boxes.index(b))+".jpg")

			val = numpy.asarray(cropImg.convert('L'))
			album.append(val)
			
		except SystemError as e:
			print("Error:" , e)
			pass
	result1 = recognize(album, lexicon,show_graph_first_one=False, verbose=True)
	print(result1)

def binding(filename):
	global photo
	photo = Image.open(filename)
	img = ImageTk.PhotoImage(photo)
	
	mainCanvas.config(width=photo.size[0],height=photo.size[1])
	mainCanvas.create_image(0,0,image=img,anchor='nw')
	master.bind('<Button-1>',motion)
	master.bind('<Button1-Motion>',mouseMotion)
	master.bind('<ButtonRelease-1>',mouseRelease)
	master.bind('<Key>',keyDown)
	master.image = img

	mainCanvas.pack(expand="yes")
	
	#print numpy.asarray(photo.convert('L'))
	mainloop()
	
if os.path.isfile(sys.argv[1]):
	binding(sys.argv[1])
	print boxes
