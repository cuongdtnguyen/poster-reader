from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
############################################
#  GUI cropper image using PIL and Tkinter #
############################################

# To use, go python pythonGUI.py 'filename'

from Tkinter import *
from PIL import ImageTk, Image
import sys
import os
import numpy
from word import recognize, WINDOW_SIZE_RATIO

FRONT_COLOR = 'white'
BACK_COLOR = 'red'
BOX_LIMIT = 1
WINDOW_WIDTH = 600
REC_OPTIONS = {'outline': BACK_COLOR, 'width': 3}


master = Tk()
canvas = Canvas(master, cursor='cross')

box = [0, 0, 0, 0]
boxes=[]

startx, starty = 0, 0
preview_recs = []
drawn_items = []
restart = False

album=[]

with open('lexicon.txt') as lex_file:
 	lexicon = [line.rstrip('\n').lower() for line in lex_file]

def clear_items(item_list):
	for item in item_list:
		canvas.delete(item)

def mousePressed(event):
	global box, startx, starty, restart, drawn_items
	if restart:
		clear_items(drawn_items)
		drawn_items = []
		restart = False

	box[0], box[1] = event.x, event.y
	startx,starty = event.x,event.y
	print("Mouse pressed: (%s %s)" % (event.x, event.y))


def mouseMoved(event):
	global box, boxes, preview_recs
	clear_items(preview_recs)
	preview1 = canvas.create_rectangle(startx, starty, event.x, event.y,
																		 **REC_OPTIONS)
	left_x = min(startx, event.x)
	height = abs(starty - event.y)
	left_x_off = left_x + int(height * WINDOW_SIZE_RATIO)
	preview2 = canvas.create_rectangle(left_x, starty, left_x_off, event.y)
	preview_recs.append(preview1)
	preview_recs.append(preview2)


def mouseReleased(event):
	global box, boxes, preview_recs, drawn_items
	clear_items(preview_recs)
	box[2], box[3] = event.x, event.y
	if abs(box[2] - box[0]) > BOX_LIMIT and abs(box[3] - box[1]) > BOX_LIMIT:
		boxes.append(box)
		drawn = canvas.create_rectangle(*box, **REC_OPTIONS)
		drawn_items.append(drawn)
		box = [0,0,0,0]

		print("Mouse released: (%s %s)" % (event.x, event.y))


def keyDown(event):
	global album, drawn_items, canvas, boxes, restart
	if restart or len(boxes) == 0:
		return
	for b in boxes:
		try:
			cropImg = photo.crop((min(b[0],b[2]),min(b[1],b[3]),
														max(b[0],b[2]),max(b[1],b[3])))
			if cropImg.size[0] == 0 or cropImg.size[1] == 0:
				raise SystemError("Nothing to crop")

			val = numpy.asarray(cropImg.convert('L'))
			album.append(val)

		except SystemError as e:
			print("Error:" , e)

	print('Recognizing words...')
	result = recognize(album, lexicon, show_graph_first_one=False, verbose=False)
	for res in result:
		print('%s (%.4f)'%(res[0], res[1]))

	for i, b in enumerate(boxes):
		lower_x = min(b[0], b[2])
		lower_y = max(b[1], b[3])
		txt = canvas.create_text((lower_x + 5, lower_y),
																 text=result[i][0],
																 anchor='nw',
																 fill=FRONT_COLOR,
																 font=('arial', 25))
		bbox = canvas.bbox(txt)
		bbox_rec = canvas.create_rectangle(lower_x, lower_y, bbox[2] + 5, bbox[3] + 5,
																			 fill=BACK_COLOR, **REC_OPTIONS)
		canvas.tag_raise(txt)
		drawn_items.append(txt)
		drawn_items.append(bbox_rec)

	restart = True
	boxes = []
	album = []


def binding(filename):
	global photo
	photo = Image.open(filename)
	#resizing
	ratio = photo.size[1] / photo.size[0]
	photo = photo.resize((WINDOW_WIDTH, int(WINDOW_WIDTH * ratio)),Image.ANTIALIAS)

	img = ImageTk.PhotoImage(photo)

	canvas.config(width=img.width(), height=img.height())
	canvas.create_image(0, 0, image=img, anchor='nw')
	canvas.bind('<Button-1>', mousePressed)
	canvas.bind('<Button1-Motion>', mouseMoved)
	canvas.bind('<ButtonRelease-1>', mouseReleased)
	master.bind('<Key>',keyDown)
	canvas.image = img
	canvas.pack()


if __name__=='__main__':
	if os.path.isfile(sys.argv[1]):
		binding(sys.argv[1])
		mainloop()
