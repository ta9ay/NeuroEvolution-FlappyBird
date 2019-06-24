import numpy as np
import pygame, sys, math
from random import randrange as randH
from random import randint
import random
from pygame.locals import *
# import neuralnet as nn1
import NeuralNetwork as nn2

# Start pygame
pygame.init()


# Set up resolution
windowObj = pygame.display.set_mode( ( 640, 480) )
fpsTimer = pygame.time.Clock()
maxFPS = 30

# Ground Elevation (pixels)
groundLevel = 400
skyLevel = 0
highScore = 0
# Global colors
birdColor = pygame.Color(32,32,32,100)
backgroundColor = pygame.Color('#abcdef')
groundColor = pygame.Color('#993333')
groundColor1 = pygame.Color('#993333')
groundColor2 = pygame.Color('#349999')
fontColor = pygame.Color('#FFFFFF')

fontObj = pygame.font.SysFont(None, 16)

# Population
POPULATION = 100
GENERATION = 0

# NN variables
INPUT_NUM = 4
HIDDEN_NUM = 4


""" Class Definitions """
# Class for pipe obstacles
class Pipes:

    height = 0
    width = 60
    # Gap 
    gap = 150
    pos = 600
    replaced = False
    scored = False

    # Randomize pipe location
    def __init__(self):
        self.height = randH(210, groundLevel - 10)

    # Moves the pipes along the ground, checks if they're off the screen
    def move(self, movement):
        self.pos += movement
        if( self.pos + self.width < 0 ):
            return False #Return false if we moved off the screen 
        return True

    # Handles drawing the pipes to the screen
    def draw(self, surface):
        pygame.draw.rect( surface, groundColor, (self.pos, self.height, self.width, groundLevel - self.height))
        pygame.draw.rect( surface, groundColor, (self.pos, 0, self.width, self.height - self.gap))

# Class for the player 
class Bird:
    
    pos = (0,0)
    radius = 20
    
    def __init__(self, newPos, brain=None):
        """
        If a brain value is passed, creates a new bird with that brain.
        If not, creates a completely random bird with a random brain. 
        """
        self.pos = newPos
        self.velocity = 0
        self.score = 1
        self.pipescore = 0
        self.fitness = 0

        # NN with n inputs 1 output
        if brain==None:
            self.brain2 = nn2.NeuralNetwork(INPUT_NUM,HIDDEN_NUM) #needs to be changed for hnum
        else:
            self.brain2 = nn2.NeuralNetwork(INPUT_NUM,HIDDEN_NUM,brain)
    
    # Handles drawing the bird to the screen
    def draw(self, surface):
        intPos = ( int(math.floor(self.pos[0])), int(math.floor(self.pos[1])) )
        screen = pygame.Surface((640,480), pygame.SRCALPHA)
        pygame.draw.circle(screen, birdColor, intPos, self.radius)
        surface.blit(screen, (0,0))

    # Attempt to move the bird, make sure we aren't hitting the ground
    def move(self, movement):
        posX, posY = self.pos
        movX, movY = movement
        if posY - self.radius < 0:
            return False
        if( (posY + movY + self.radius) < groundLevel ):
            self.pos = (posX + movX, posY + movY)
            return True #Return if we successfuly moved
        
        self.pos = (posX, groundLevel - self.radius)
        return False

    # Test for collision with the given pipe
    def collision(self, pipe):
        posX, posY = self.pos
        collideWidth = ( pipe.pos < posX + self.radius and posX - self.radius < pipe.pos + pipe.width)
        collideTop = ( pipe.height - pipe.gap > posY - self.radius )
        collideBottom = ( posY + self.radius > pipe.height )
        if (  collideWidth and ( collideTop or collideBottom)):
            return True
        return False
    
    def jump(self):
        self.score+=1
        self.velocity = -15
    
    def think(self,pipes):
        """
        Trying a couple of different approaches.
        1) two inputs. horizontal distance from the closest pipe and bertical distance to the center of gap of pipe
        2) Three inputs. All of the above plus velocity.
        3) four inputs. All the above, except no vertical distance to center of gap, instead location of bottom and top of gap.

        """
        
        # Find closest pipe
        closest = pipes[0]
        for pipe in pipes:
            if (closest.pos + closest.width) - self.pos[0] < 0 or (closest.pos + closest.width) - self.pos[0] > (pipe.pos + pipe.width) - self.pos[0]:
                closest = pipe 
        
        # Horizontal distance
        hdist = ((closest.pos + closest.width) - self.pos[0]) / 640

        # Vertival distance
        # vdist = (bird.pos[1] - (closest.height - (closest.gap/2))) / 480
        vdistBottom = (bird.pos[1] - closest.height) / 480
        vdistTop = (bird.pos[1] - (closest.height - closest.gap)) / 480
        
        # Velocity of the bird
        y_vel = self.velocity/100
        inputs = np.array([
            [y_vel,hdist,vdistTop,vdistBottom]
        ])
        output = self.brain2.feedforward(inputs,HIDDEN_NUM)
        if output[0][0]>0:
            self.jump()



""" Game Area """
birds = []
saved = []
previousFitness = 0


def calculateFitness():
    """ 
    Calculate Fitness of each bird based on its normalized score 
    """
    global saved
    global previousFitness
    sumScore = 0
    for bird in saved:
        # print(bird.score)
        sumScore += bird.score
    for bird in saved:
        bird.fitness = (bird.score / sumScore)
        

def pickOne():
    """
    Return the one bird from the saved birds list.
    This bird is used to perform mutation and then create a new generation of birds.
    Initial approach was to select the bird with highest fitness value. However that didn't perform well.
    """ 
    # global previousFitness
    # temp = sorted(saved, key=lambda obj: obj.fitness,reverse=True)
    # print("Fitness of generation: ",temp[0].fitness)
    # # previousFitness=temp[0].fitness
    # return temp[0]
    """
    A different method based on improved pool selection.
    """
    index = 0
    r = random.random()
    temp = sorted(saved, key=lambda obj: obj.fitness,reverse=True)
    while r>0:
        r=r-temp[index].fitness
        index+=1
    index-=1
    return temp[index]

def mutate(numbers,mutationRate):
    """
    Performs Mutation on the matrices. Mutation Rate is the probability of mutation of each element of the martix.
    """
    for val in np.nditer(numbers['Wx'],op_flags=['readwrite']):
        val[...] = mutateValue(val,mutationRate)
    for val in np.nditer(numbers['Wh'],op_flags=['readwrite']):
        val[...] = mutateValue(val,mutationRate)
    for val in np.nditer(numbers['hiddenbias'],op_flags=['readwrite']):
        val[...] = mutateValue(val,mutationRate)
    numbers['outputbias'] = mutateValue(numbers['outputbias'],mutationRate)
    
    return numbers
    

def mutateValue(value,mutationRate):
    if random.random() < mutationRate:
        return np.random.normal()
    else:
        return value



def newGeneration(firstgen=True):
    """ 
    If this is the first generation, each bird gets a new NN for its brain. 
    If not, the bird gets a brain from the previous generation.
    """
    global saved
    global GENERATION
    GENERATION += 1
    calculateFitness()
    if firstgen==True:
        for i in range(POPULATION):
            birds.append(Bird((windowObj.get_width() / 4 , windowObj.get_height() / 2)))
    
    else:
        #pickOne() returns one bird
        chosenBird = pickOne()
        # take the numbers of that bird and pass it to the Bird constructor
        for i in range(POPULATION):
            newBrain = mutate(chosenBird.brain2.numbers,0.1)
            birds.append(Bird((windowObj.get_width() / 4 , windowObj.get_height() / 2),newBrain))
    saved = []



pipes = [ Pipes()]
gravity = 2

newGeneration()

# Called to reset the game when you lose
def resetGame():
    global highScore

    if ( bird.pipescore > highScore ):
        highScore = bird.pipescore
    
    global pipes
    del pipes[:]
    pipes = [ Pipes() ]

    windowObj.fill(pygame.Color('#230056'))
    newGeneration(firstgen=False)



def pause():
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if ( event.key == K_ESCAPE):
                    return
    
    
# Main game loop
while True:
    """
        When collision is detected, save the bird in a different list called "saved" and then discard the bird.
        Resets the game when no birds are left.
    """    
    windowObj.fill(backgroundColor)

    # Check for events
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if ( event.key == K_ESCAPE):
                pause()
            # If the player hits a key, set velocity upward
            
            if event.key == K_SPACE:
                
                bird.velocity = -20
   
    for pipe in pipes:
        if not pipe.replaced and pipe.pos < windowObj.get_width() / 2  :
            pipes[len(pipes):] = [Pipes()]
            pipe.replaced = True
        pipe.draw(windowObj)
        if( not pipe.move(-10)):
            del pipes[0]
    
    # Use the brain to think
    for bird in birds:
        bird.think(pipes)
        
        # Add acceleration from gravity
        bird.velocity += gravity

        if (not bird.move((0, bird.velocity))):
            saved.append(bird)
            birds.remove(bird)
            bird.velocity = 0
        
        for pipe in pipes:
        
            if (bird.collision(pipe)):
                saved.append(bird)
                try:
                    birds.remove(bird) # Throws a "x not in list" error sometimes. Why?
                except:
                    pass
            if (pipe.pos + pipe.width == bird.pos[0] ):
                bird.score += 10
                bird.pipescore +=1
                
                pipe.scored = True
            
            
        if len(birds) == 0:
            resetGame()

        # Draw stuff
        scoreSurface = fontObj.render( 'Score: ' + str(highScore) + ' Gen: ' + str(GENERATION), False, fontColor)
        scoreRect = scoreSurface.get_rect()
        scoreRect.topleft = (windowObj.get_height() / 2 , 10)
        windowObj.blit(scoreSurface, scoreRect)
        pygame.draw.rect(windowObj, groundColor, (0, groundLevel, windowObj.get_width(), windowObj.get_height()) )

        bird.draw(windowObj)
        
    pygame.display.update()
    fpsTimer.tick(maxFPS)
