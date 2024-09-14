import pygame
import numpy as np
from net import network

# from keras.api.datasets import mnist

pygame.init

# (train_X, train_y), (test_X, test_y) = mnist.load_data()

screen_height = 500
screen_width = 500
screen_size = (screen_width, screen_height)
pygame.display.set_caption("ඞ")
screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE)

grid_width = 28
grid_height = 28
grid_size = (grid_width, grid_height)

grid = [[0 for col in range(grid_width)] for row in range(grid_height)]

wbVec = np.load("Neural Net\mnist_16x16_model.npy")
net = network(28 * 28, 2, 16, 10)
net.initWeightsBiases(wbVec)


# mouse input
def getX():
    return pygame.mouse.get_pos()[0]


def getY():
    return pygame.mouse.get_pos()[1]


def getMouseLeft():
    return pygame.mouse.get_pressed(3)[0]


def getMouseRight():
    return pygame.mouse.get_pressed(3)[2]


# display
def drawGrid():
    for row in range(0, grid_height):
        for col in range(0, grid_height):
            pygame.draw.rect(
                screen,
                (grid[row][col] * 255, grid[row][col] * 255, grid[row][col] * 255),
                pygame.Rect(
                    screen_width * col / grid_width,
                    screen_height * row / grid_height,
                    screen_width / grid_width + 1,
                    screen_height / grid_height + 1,
                ),
            )


def resetGrid():
    for row in range(0, grid_height):
        for col in range(0, grid_width):
            grid[row][col] = 0


def updateGrid():
    if getMouseRight():
        resetGrid()

    elif getMouseLeft():
        gridx = (int)(np.floor(getX() / (screen_width / grid_width)))
        gridy = (int)(np.floor(getY() / (screen_height / grid_height)))

        colorPixel(gridx, gridy, 0.5)
        colorPixel(gridx + 1, gridy, 0.1)
        colorPixel(gridx - 1, gridy, 0.1)
        colorPixel(gridx, gridy + 1, 0.1)
        colorPixel(gridx, gridy - 1, 0.1)


def colorPixel(x, y, value):
    if (x < grid_width) & (y < grid_height) & (x >= 0) & (y >= 0):
        if grid[y][x] + value < 1:
            grid[y][x] += value
        else:
            grid[y][x] = 1


# right = 0
# for i in range(0, len(test_X)):

#     output, z = net.forwardProp(np.array(test_X[i]).flatten())

#     max = -1
#     maxNum = 0
#     for j in range(0, 10):
#         if output[-1][j] > max:
#             max = output[-1][j]
#             maxNum = j

#     if maxNum == test_y[i]:
#         right += 1

#     print(right / (i + 1))

running = True
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        updateGrid()

    output, z = net.forwardProp(np.array(grid).flatten())

    max = -1
    maxNum = 0
    for i in range(0, 10):
        if output[-1][i] > max:
            max = output[-1][i]
            maxNum = i

    print(maxNum)

    drawGrid()
    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()
