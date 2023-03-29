import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
CORLOR1 = (255, 0, 255)
CORLOR2 = (255,100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.direction2 = Direction.RIGHT
        #320,240
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.head2 = Point(self.w/2-BLOCK_SIZE, self.h/2-BLOCK_SIZE)
        self.snake2 = [self.head2,
                      Point(self.head2.x-BLOCK_SIZE, self.head2.y),
                      Point(self.head2.x-(2*BLOCK_SIZE), self.head2.y)]
        self.score = 0
        self.score2 = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake and self.food in self.snake2:
            self._place_food()
        


    def play_step(self, action,action2):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action,action2) # update the head
        self.snake.insert(0, self.head)
        self.snake2.insert(0, self.head2)

        # 3. check if game over
        reward = 0
        reward2 = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake)or self.frame_iteration > 100*len(self.snake2):
            game_over = True
            reward = -10
            return reward, game_over, self.score,reward, game_over, self.score2

        # 4. place new food or just move
        if self.head == self.food :
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        if self.head2 == self.food :
            self.score2 += 1
            reward2 = 10
            self._place_food()
        else:
            self.snake2.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score,reward2, game_over, self.score2


    def is_collision(self, pt=None, pt2=None):
        if pt is None:
            pt = self.head
        if pt2 is None:
            pt2 = self.head
        # hits boundary
        if (pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0) or ( pt2.x > self.w - BLOCK_SIZE or pt2.x < 0 or pt2.y > self.h - BLOCK_SIZE or pt2.y < 0):
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        if pt2 in self.snake2[1:]:
            return True
        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        for pt in self.snake2:
            pygame.draw.rect(self.display, CORLOR1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, CORLOR2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))


        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        text = font.render("Score2: " + str(self.score2), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action,action2):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        idx2 = clock_wise.index(self.direction2)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        if np.array_equal(action2, [1, 0, 0]):
            new_dir2 = clock_wise[idx2] # no change
        elif np.array_equal(action2,[0, 1, 0]):
            next_idx2 = (idx2 + 1) % 4
            new_dir2 = clock_wise[next_idx2] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx2 = (idx2 - 1) % 4
            new_dir2 = clock_wise[next_idx2] # left turn r -> u -> l -> d

        self.direction = new_dir
        self.direction2 = new_dir2

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

        x = self.head2.x
        y = self.head2.y
        if self.direction2 == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction2 == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction2 == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction2 == Direction.UP:
            y -= BLOCK_SIZE

        self.head2 = Point(x, y)