import pygame
import sys
import random
import numpy as np
import math as math
from helper import plot

pygame.init()

# Define screen dimensions and colors
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Create the display surface
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Snake Game')


def draw_snake(snake):
    for segment in snake:
        pygame.draw.rect(screen, GREEN, pygame.Rect(segment[0], segment[1], 20, 20))


def draw_food(food_position):
    pygame.draw.rect(screen, RED, pygame.Rect(food_position[0], food_position[1], 20, 20))

def display_score(score):
    font = pygame.font.Font(None, 36)
    text = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(text, (SCREEN_WIDTH - 150, 10))

def game_over_screen(score):
    font = pygame.font.Font(None, 48)
    text = font.render(f"Game Over! Score: {score}", True, (0, 0, 0))
    screen.blit(text, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 30))
    pygame.display.flip()
    pygame.time.wait(2000)




def move_snake(snake, direction):
    x, y = direction
    head = snake[-1].copy()
    head[0] += x * 20
    head[1] += y * 20
    snake.append(head)
    snake.pop(0)


def check_collision(snake):
    head = snake[-1]
    return head in snake[:-1] or head[0] < 0 or head[0] >= SCREEN_WIDTH or head[1] < 0 or head[1] >= SCREEN_HEIGHT

def step(snake, action,food_position):
    snake2=snake
    move_snake(snake2,  action)
    state = snake2[-1][1]* 2+snake2[-1][0]//20
    if snake2[-1] == food_position:
        done =True
    else:
        done =False
    reward = 1 if done else 0.
    return state, reward, done

def gameLoop():
    plot_scores = []
    plot_mean_scores = []
    total_score=0
    num_game=0
    # Initialize the snake and food
    s=1200
    a=4
    Pi=np.zeros([s,a])
    q = np.zeros([s,a])
    max_episode=100000000
    gamma = 0.9
    step_size = 0.1
    epsilon = 0.1
#Initialize S
    score = 0
    snake = [[100, 100], [120, 100], [140, 100]]
    food_position = [random.randrange(0, SCREEN_WIDTH, 20), random.randrange(0, SCREEN_HEIGHT, 20)]
    clock = pygame.time.Clock()
    direction = [1, 0]
    current_state=107
    for i in range (max_episode):
        while True:
            if(np.random.rand()<=epsilon):
                A=np.random.randint(4)
            else:
                A=np.random.choice(np.flatnonzero(q[current_state]==np.max(q[current_state])))           
            #take A
            for j in range(4):
                Pi[current_state][j]=0
                Pi[current_state][A]=1
                
            state, reward, done= step(snake, direction,food_position)
            #update Q(S,A)
            q[current_state,A]=q[current_state,A]+step_size*( reward+ gamma*np.max(q[state])-q[current_state,A])
            current_state=state

            if A==0:
                direction = [0, 1]
            elif  A==1 :
                direction = [0, -1]
            elif  A==2 :
                direction = [1, 0]
            else :
                direction = [-1, 0]

            
            move_snake(snake,  direction)
            # Check collisions
            if check_collision(snake):
                num_game+=1
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / num_game
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
                # Reset the game
                snake = [[100, 100], [120, 100], [140, 100]]
                food_position = [random.randrange(0, SCREEN_WIDTH, 20), random.randrange(0, SCREEN_HEIGHT, 20)]
                direction = [1, 0]
                score = 0

            if snake[-1] == food_position:
                snake.insert(0, snake[0].copy())
                food_position = [random.randrange(0, SCREEN_WIDTH, 20), random.randrange(0, SCREEN_HEIGHT, 20)]
                score += 1

            # Draw everything
            screen.fill(WHITE)
            draw_snake(snake)
            draw_food(food_position)
            display_score(score)  # Display the score
            pygame.display.flip()

            if(done):
                break
            # Cap the frame rate
            clock.tick(10)


gameLoop()



def reset():
    game_over_screen(score)  # Show the game over screen
    # Reset the game
    snake = [[100, 100], [120, 100], [140, 100]]
    food_position = [random.randrange(0, SCREEN_WIDTH, 20), random.randrange(0, SCREEN_HEIGHT, 20)]
    direction = [1, 0]
    score = 0
    state=207
    return snake,state,food_position,direction






