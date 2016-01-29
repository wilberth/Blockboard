#!/usr/bin/python

from __future__ import print_function
import numpy as np, serial, time, logging
from psychopy import core, visual, event, sound#, data
import time, copy, os

## definition section
# config: array of 16 bytes describing 4 colors
# board: array of 32 bytes, either 0 for absent, 1 for present or -1 for error
# constants to tune the experiment
debug = False # is this a debug run?
dry = False # dry runs are without the board connected
port = '/dev/ttyUSB0' # change to 'com?' for windows
referenceDuration = 0.0 # overides reference duration, use measured value when set to 0.0
if debug:
	logging.getLogger().setLevel(logging.DEBUG)
else:
	logging.getLogger().setLevel(logging.INFO)
	
#####################################  
# nTrain = number of training trails
# nTrails = number of trails per block
# nBlock = number of Blocks	ff
# iti = intertrial interval in seconds
# feedback = should feedback about trial performance be given?

nTrain = 2 #amount of training trials (normally 6)

nBlock = 2

nTrials = 8 #amount of experimental trials (normally 8)

iti = 5

feedback = False # do you get feedback about your time after every trial? True / False

ID = "Npilot21" #participant number 

#p_r = 0.10 + 0.05 * dt/(0.5s)
pNoiseFactor = 1  # if 1 Noise is on, if 0 Noise is off   ##### !!!!!! select condition here!!!! ########
pNoiseMin = 0.10 # minimum chance of noise
pNoiseMax = 0.40 # maximum chance of noise
dtNoise   = 5.0 # (s) time delay that will lead to an increase of p with 100%
swapOrder = False # Flase: start with noise blok
condition = "No condition" # set based on swapOrder and pNoiseFactor

# Setting the ID of the participant
# Be sure that the ID is unique! Otherwise data will be stored in the same data file as the data if the other participant (no data will be lost)

#labels for the conditions in order to write them in the output



########################################
# creating the data file and saving it
# data is stored in long format
# data is stored in a .txt file using tab-delimiter ("\t")
if ID+'_data.txt' not in os.listdir(os.getcwd()+"/data"):
	data = open("data/"+ID+'_data.txt',"a")
	saveList = ["ID", "Block", "blockTime","configNumber", "movementNumber", "configTime", "Color", "positionPick","positionPlace","Correct", "thinkTime", "movementTime", "noise", "Condition","\n"]
	saves = '\t'.join(saveList)
	data.write(saves)
	data.close()

if ID+'_efData.txt' not in os.listdir(os.getcwd()+"/data"):
	efData = open("data/"+ID+"_efData.txt", "a")
	saveList2 = ["ID", "Block","fatigue","engagement","pressure0", "pressure1","displeasure" "minduration", "meanduration","condition"]
	saves2 = '\t'.join(saveList2)
	efData.write(saves2+"\n")
	efData.close()

# function that draws the rectangles
def drawConfig(rectangles):
	for rect in rectangles:
		rect.setAutoDraw(True)
 
# function that stops the draw of the rectangles
def stopConfig(rectangles):
	for rect in rectangles:
		rect.setAutoDraw(False)
 


# constants that never change
initialConfig = np.array((
	1,2,3,4,
	1,2,3,4,
	1,2,3,4,
	1,2,3,4))
colorNames = ('empty','blue','yellow','red', 'darkgreen')
directionNames = ('up', 'down')
sideNames = ('left', 'right')
# (@wilbert, we changed the positionNames!)
positionNames = (
	'A4','B4','C4','D4',
	'A3','B3','C3','D3',
	'A2','B2','C2','D2',
	'A1','B1','C1','D1')

# visual stimuli
if debug:
	win = visual.Window(color='black', screen=1, fullscr=True)
else:
	win = visual.Window(color='black', screen=0, fullscr=True)
message = visual.TextStim(win, text='', wrapWidth=2)
message.autoDraw = True
rectangles = []; i = 0
for y in (-0.75, -0.25, 0.25, 0.75):
	for x in (-0.75, -0.25, 0.25, 0.75):
		rectangles.append(visual.Rect(win, width=0.2, height=0.2,pos=(x, y), 
			fillColor=colorNames[initialConfig[i]], autoDraw=True))
		i += 1

# audio stimuli
duration = 0.3 # (s) #300 ms is ok :)
attenuation = 10.0 # dB (attenuation of correctSound compared to noiseSound)
noise = np.random.uniform(low=-1.0, high=1.0, size=(int(44100*duration), 2))
noiseSound   = sound.Sound(value=noise, secs=0.3, sampleRate=44100)
noiseSound.setVolume(1.0) # maximum volume
#noiseSound.play()
print(noiseSound.getVolume())
correctSound = sound.Sound(value='C', secs=duration, octave=3)
correctSound.setVolume(0.01) # attenuation less than maximum
#correctSound.play()
print(correctSound.getVolume())
wrongSound   = sound.Sound(value='C', secs=duration, octave=5)
wrongSound.setVolume(0.01) # legacy 
#wrongSound.play()

#functions
# Function to draw .jpgs and wait for a response
def instrDraw(path):
	jpgList = os.listdir(path)
	for JPG in jpgList:
		ins = visual.ImageStim(win, image = path+JPG)
		ins.draw()
		win.flip()
		event.waitKeys()

def efQ(ID, block, condition, durations, qType):
	questions = [
		["fatigue", "Hoe vermoeid voel je je op dit moment?"],
		["engagement", "Hoezeer ging je in de taak op ('engagement')?"],
		["pressure0", "Hoe gespannen was je tijdens het uitvoeren van de taak?"],
		["pressure1", "Hoe gespannen voel je je op dit moment?"],
		["displeasure", "Hoe vervelend vond je het geluid dat je zojuist hoorde?"],
		]
	qGroup = [[0], [0, 1, 2], [4, 3]]
	answers = ["" for x in questions] # 5 empty strings
	
	stopConfig(rectangles)
	win.flip()
	time.sleep(.5)
	event.Mouse(visible = True)

	print("group {}: {}".format(qType, qGroup[qType]))
	for q in qGroup[qType]:
		question = visual.RatingScale(win, scale = questions[q][1], labels = ("helemaal niet", "heel erg"),low = 0, high = 100, tickHeight = .0, mouseOnly = True, showValue = False, acceptText = "Accept")
		while question.noResponse:
			question.draw()
			win.flip()
		answers[q] = question.getRating()
	
	saveList2 = [ID, str(block)] + [str(a) for a in answers] +  [str(min(durations)), str(sum(durations)/len(durations)), condition]
	saves2 = '\t'.join(saveList2)
	with open("data/"+ID+"_efData.txt", "a") as efData:
		efData.write(saves2+"\n")
	event.Mouse(visible = False)
	
	
def invalidConfig(c):
	'''
	return True if the configuration is violating any of the given rules (T0 to T18)
	'''
	# horizontal
	T0  =  c[0]== c[1]== c[2]== c[3]
	T1  =  c[4]== c[5]== c[6]== c[7]
	T2  =  c[8]== c[9]==c[10]==c[11]
	T3  = c[12]==c[13]==c[14]==c[15]
	#vertical
	T4  =  c[0]== c[4]== c[8]==c[12]
	T5  =  c[1]== c[5]== c[9]==c[13]
	T6  =  c[2]== c[6]==c[10]==c[14]
	T7  =  c[3]== c[7]==c[11]==c[15]
	#diagonal
	T8  =  c[0]== c[5]==c[10]==c[15]
	T9  =  c[3]== c[6]== c[9]==c[12]
	#square
	T10 =  c[0]== c[1]== c[4]== c[5]
	T11 =  c[1]== c[2]== c[5]== c[6]
	T12 =  c[2]== c[3]== c[6]== c[7]
	T13 =  c[4]== c[5]== c[8]== c[9]
	T14 =  c[5]== c[6]== c[9]==c[10]
	T15 =  c[6]== c[7]==c[10]==c[11]
	T16 =  c[8]== c[9]==c[12]==c[13]
	T17 =  c[9]==c[10]==c[13]==c[14]
	T18 = c[10]==c[11]==c[14]==c[15]
	return T0 or T1 or T2 or T3 or T4 or T5 or T6 or T7 or T8 or T9 or T10 or T10 or T11 or T12 or T13 or T14 or T15 or T16 or T17 or T18

def createValidConfig():
	config = initialConfig.copy()
	if dry:
		return config
	while invalidConfig(config):
		np.random.shuffle(config)
	return config
		
def printConfig(c):
	for y in range(3,-1, -1):
		for x in range(4):
			print('{:10s}'.format(colorNames[c[4*y+x]]), end='')
		print('')
	
def showMessage(t):
	message.text=t
	win.flip()
	
def showConfig(c):
	for i in range(16):
		rectangles[i].fillColor = colorNames[c[i]]
	win.flip()
	

def feedbackFUN(ct, nError):
	ft = visual.TextStim(win, text='Time: {:.2f} seconds'.format(ct), pos=(0,.15))
	fe = visual.TextStim(win, text='# of errors: {:d}'.format(nError), pos=(0,-.15))
	ft.draw()
	fe.draw()
	win.flip(0)
	time.sleep(iti)

	



class Board:
	''' 
	The Board class handles communication and state (occupied slots) of the blockboard.
	It does not do anything with the Psychopy windows or with board config (colors of slots)
	
	'''
	def __init__(self, debug=False):
		self.debug = debug
		if self.debug:
			self.debugWin = visual.Window(color='black', screen= 0, size=(340, 90))
			self.debugRectangles = []; i = 0
			for side in (0, 1):
				for y in (-0.75, -0.25, 0.25, 0.75):
					for x in (-0.75, -0.25, 0.25, 0.75):
						self.debugRectangles.append(visual.Rect(self.debugWin, width=0.15, height=0.2, pos=(side-0.5+0.4*x, y), autoDraw=True))

		# initialize serial
		if dry:
			self.state = np.hstack((np.ones(16, dtype=np.int), np.zeros(16, dtype=np.int)))
			self.sourceSide = 0 # left
			self.waitMove = self.waitMove2
		else:
			self.ser = serial.Serial(port, 115200)
			# The following three lines cause a reset of the Arduino. 
			# See 'Automatic (Software) Reset' in the documentation.
			# Make sure not to send data for the first second after reset.
			self.ser.setDTR(False)
			time.sleep(0.1)
			self.ser.setDTR(True)

			time.sleep(1.0)
			self.ser.flushInput()
			# initialize board
			self.getState()
			self.sourceSide = self.checkState() # left or right
		


	def showState(self):
		if self.debug:
			for i in range(32):
				self.debugRectangles[i].fillColor = ('black', 'white')[self.state[i]]
			self.debugWin.flip()

	def getState(self):
		'''
		request state from blockboard 
		ord(c):
		bit 0-1 column
		bit 2-3 row
		bit 4   side (0 = left/usb, 1 = right)
		bit 5   state (1 = present , 0 = absent)
		return value:
		32 bytes (according to first 5 bits of c) containing 0 for absent or 1 for present
		'''
		self.state = np.ones([32], dtype='uint8') * -1 # unknown
		if not dry:
			self.ser.flushInput()
			self.ser.write('A') # any byte
			s = self.ser.read(32)
		else:
			s = "".join(("{:c}".format(i) for i in range(32)))
		#print('retval: {}, {}'.format(len(s), ''.join(format(ord(x), '02x') for x in s)))
		for c in s:
			c = ord(c) # one character string to unsigned integer (cast)
			self.state[c&0x1f] = c >> 5
		self.showState()


	def checkState(self, silent=True):
		'''
		return -1 for error
		return 0 for left board contains 16 blocks
		return 1 for right board contains 16 blocks
		'''
		if min(self.state) < 0:
			if not silent:
				logging.error('Not all 32 slots reported')
			return -1
		if sum(self.state) != 16:
			if not silent:
				logging.error('Not all 16 colored blocks present')
			return -1
		if sum(self.state[0:16]) != 16 and sum(self.state[16:32]) != 16:
			if not silent:
				logging.error('Not all 16 colored blocks on one of the two boards')
			return -1
		self.sourceSide = sum(self.state[16:32]) / 16
		return self.sourceSide
		
	def waitEvent(self):
		'''
		wait for pickup or putdown
		todo: store state change
		'''
		c = ord(self.ser.read())
		direction, board, position = (c&0x20) >> 5, (c&0x10) >> 4, c&0x0f
		logging.debug('direction: {}, board: {}, position: {}'.
			format(directionNames[direction], sideNames[board], positionNames[position]))
		self.state[c&0x1f] = c >> 5
		self.showState()

		return c
		
		
	def waitMove(self, sourceSide, tTime):
		'''
		wait for an entire move (pickup till putdown) to finish. This move may be invalid.
		This function will call waitEvent at least twice.
		return: (tuple of from-board, from-index and to-index, fromTime, thinkTime) or None
		moveTime: time between first pickup and final putdown
		thinkTime: time between reference tTime and last pickup
		'''
		ups = []
		uptime = [] 
		downs = []
		while len(ups)!=1 or len(downs)!=1:
			c = self.waitEvent()
			
			direction, board, position = (c&0x20) >> 5, (c&0x10) >> 4, c&0x0f
			c&=0x1f  # remove direction information
			if not direction:
				# picking up
				thinkTime = time.time()-tTime
				ups.append(c)
				uptime.append(time.time())
				if board != sourceSide:
					logging.warning('  starting move: {} on wrong board: {}'.
						format(positionNames[position], sideNames[board]))
				if len(ups) > 1:
					logging.warning('  multiple ({}) moves started by picking up {} on board: {}'.
						format(len(ups), positionNames[position], sideNames[board]))
			else:
				# putting down
				if c in ups:
					logging.warning('  aborting move from {} on board: {}'.
						format(positionNames[position], sideNames[board]))
					i = ups.index(c)
					ups.remove(c)
					del uptime[i]
				else:
					downs.append(c)
					downtime = time.time()
					if len(ups) > 1 and len(downs) > 1:
						logging.critical('  Multiple blocks were picked up and multiple were put down in a different place. I lost track. Quitting')
						exit()
					if len(ups) > 1:
						logging.warning('  thanks for putting down {} on board: {} but {} are still in the air: {}'.
						format(positionNames[position], sideNames[board], len(ups)-len(downs), ups))
		movetime = downtime - uptime[0]
		return ups[0]>>4, ups[0]&0xf, downs[0]&0xf, movetime, thinkTime
	
	iMove = -1
	def waitMove2(self, sourceSide, tTime):
		time.sleep(.2)
		self.iMove = (self.iMove+1)%16
		return (sourceSide, self.iMove, self.iMove, 1., 1.)
		#sBoard, s, t, movetime, thinkTime
		

def waitValidMove(board, sourceSide, sourceConfig, targetConfig, block, blockTimeref, configNum, configTimeref, mN, nError, referenceDuration=None):
	'''
	wrapper for waitMove that gives feedback in case of invalid moves
	'''   
	state = board.state.copy()
	tTime =time.time()
	sBoard, s, t, movetime, thinkTime = board.waitMove(sourceSide, tTime)
	
	# save data for any (invalid) move
	if sBoard != sourceSide or sourceConfig[s] != targetConfig[t]:
		wrongSound.play()
		logging.info('WRONG MOVE put it back: {} {} {} -> {} {}'.
			format(colorNames[sourceConfig[s]], sideNames[not sBoard], 
				positionNames[t], sideNames[sBoard], positionNames[s]))
		if debug:
			showMessage('put it back: {} {} {} -> {} {}'.
				format(colorNames[sourceConfig[s]], sideNames[not sBoard], 
					positionNames[t], sideNames[sBoard], positionNames[s]))
		else:
			message.color='red'
			showMessage('X')
		# wait for correcting error
		e1 = time.time()
		while np.any(state != board.state):
			board.waitEvent()
		movetime = time.time() - e1
		message.color='white'
		showMessage('')
		logging.info('Wrong move corrected')
		correctSound.play() # this just indicates a correct correction, not a correct move
		mN = mN + 1
		
		saveList = [ID, str(block), str(time.time()-blockTimeref),str(configNum), str(mN), str(time.time()-configTimeref),colorNames[sourceConfig[s]], positionNames[s], positionNames[t],"FALSE",str(thinkTime), str(movetime), condition]
		saves = "\t".join(saveList)
		
		data = open("data/"+ID+'_data.txt',"a")
		data.write(saves+"\n")
		data.close()
		nError += 1
		return waitValidMove(board, sourceSide, sourceConfig, targetConfig, block, blockTimeref, configNum, configTimeref, mN, nError, referenceDuration=referenceDuration)
	else:
		mN = mN + 1
		configTime = time.time()-configTimeref
		if referenceDuration is None:
			bNoise = False
			dt = 0
		else:
			dt = configTime - referenceDuration*mN/16
			pNoise = 0
			if condition == "Noise":
				pNoise = np.clip(pNoiseMin + dt/dtNoise, pNoiseMin, pNoiseMax) # chance of noise
			bNoise = pNoise>np.random.uniform() # pNoise chance of True, 1-pNoise chance of False
			logging.info('  dt={:.3f}, pNoise={:.3f}, noise={}, condition: {}'.format(dt, pNoise, bNoise, condition))
		logging.info('CORRECT MOVE: {} {} ({})-> {} {}'.
			format(sideNames[sBoard], positionNames[s], colorNames[sourceConfig[s]],
				sideNames[not sBoard], positionNames[t]))
		sound = (correctSound, noiseSound)[bNoise] # choose sound
		

		time.time()-blockTimeref
		data = open("data/"+ID+'_data.txt',"a")		 
		saveList = [ID, str(block), str(time.time()-blockTimeref),str(configNum), str(mN), str(configTime),colorNames[sourceConfig[s]], positionNames[s], positionNames[t],"TRUE",str(thinkTime), str(movetime), ("FALSE", "TRUE")[bNoise],condition]
		saves = "\t".join(saveList)
		data.write(saves+"\n")
		data.close()
		sound.play()
	#return sBoard, s, t, mN, str(round(time.time()-configTimeref, 2)), nError #insane
	return sBoard, s, t, mN, configTime, nError

def experiment():
	global referenceDuration, condition
	if not debug:
		stopConfig(rectangles)
		# drawing instructions
		win.flip()
		
		instrDraw('start/')
		
	efQ(ID, "Instruction", "No condition", [0], 0) # questionaire A (no durations yet)

		
	drawConfig(rectangles)
	# starting the training 
	block = "Training"
	board = Board(debug=debug)
	
	# check starting config
	print (board.state)
	while board.checkState() == -1:
		showConfig(initialConfig)
		showMessage('initial board like this please')
		board.waitEvent() # request state from board, never do this during experiment !
	showMessage('')

	logging.debug('Initial board is {}'.format(sideNames[board.sourceSide]))
	sourceConfig = initialConfig #assume this is true
	board.showState()

	# trials
	durations = []
	if nTrain == 0:
		durations = [referenceDuration]
	blockTimeref = time.time()
	for i in range(nTrain):
		nError = 0
		mN = 0
		targetConfig = createValidConfig()
		showConfig(targetConfig)
		drawConfig(rectangles)
		win.flip()
		if debug:
			showMessage(('->', '<-')[board.sourceSide]+'like this please')
			printConfig(sourceConfig)
			print('\ntarget')
			printConfig(targetConfig)
		configTimeref = time.time()
		configNum = i+1
		for iMove in range(16):
			sBoard, s, t, mN, ct, nError = waitValidMove(board, board.sourceSide, sourceConfig, targetConfig, block, blockTimeref, configNum, configTimeref, mN, nError)
		# assume all went correct
		sourceConfig = targetConfig
		board.sourceSide = int(not board.sourceSide)
		stopConfig(rectangles)
		win.flip()
		durations.append(ct)
		if feedback:
			feedbackFUN(ct, nError)
		else:
			time.sleep(iti)
		 
	efQ(ID, block, "No condition", durations, 1) # questionaire B
	
	# show fastest training
	if referenceDuration == 0:
		referenceDuration = min(durations)
	feedbackText = visual.TextStim(win, text=" Snelste tijd: {:.3f} s, \n Roep nu de experimentleider".format(referenceDuration))
	feedbackText.draw()
	win.flip()
	event.waitKeys()
	
	#showing post training instructions
	instrDraw('preEx/') #instructions prior to experiment
	win.flip()
	# this instruction should show a "press key to hear noise" instruction
	
	# example noise after keypress
	noiseSound.play()
	event.waitKeys()
	efQ(ID, block, "No condition", durations, 2) # questionaire C

	core.wait(1)
	
	# blockBoard
	for h in range(nBlock):
		if pNoiseFactor == 0:
			condition = "Control"
		else:
			condition = ("Noise", "Control")[(h+swapOrder)%2]
		durations = []
		
		block = h + 1
		
		blockTimeref = time.time()
		for k in range(nTrials):
			nError = 0
			mN = 0
			configTimeref =time.time()
			configNum = k+1
			targetConfig = createValidConfig()
			showConfig(targetConfig)
			drawConfig(rectangles)
			win.flip()
			if debug:
				showMessage(('->', '<-')[board.sourceSide]+'like this please')
				printConfig(sourceConfig)
				print('\ntarget')
				printConfig(targetConfig)
			for iMove in range(16):
				tTime =time.time()
				sBoard, s, t, mN, ct,nError = waitValidMove(board, board.sourceSide, sourceConfig, targetConfig, block,blockTimeref,configNum,configTimeref,mN, nError, referenceDuration=referenceDuration)
			# assume all went correct
			sourceConfig = targetConfig 
			board.sourceSide = not board.sourceSide
			stopConfig(rectangles)
			win.flip()
			durations.append(float(ct))
			if feedback:
				feedbackFUN(ct, nError)
			else:
				time.sleep(iti)

		efQ(ID, block, condition, durations, 1) # questionaire B
		# after block, show average time
		feedbackText = visual.TextStim(win, text=" Gemiddelde trial tijd: {:.3f} s, \n Roep nu de experimentleider".format(sum(durations)/len(durations)))
		feedbackText.draw()
		win.flip()
		event.waitKeys()
		if not h%2:
			instrDraw('postBlock/')
		
	stopConfig(rectangles)
	instrDraw('postEx/')
def test():
	board = Board(debug=debug)
	while board.checkState() == -1:
		showConfig(initialConfig)
		showMessage('initial board like this please')
		board.waitEvent() # request state from board, never do this during experiment !
	showMessage('')

	sourceConfig = initialConfig #assume this is true
	board.showState()

## experiment section
if __name__ == '__main__':
	experiment()
  
