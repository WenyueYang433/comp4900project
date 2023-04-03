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

direction_dict = {
    (0, 1): 0,  # up
    (0, -1): 1,  # down
    (1, 0): 2,  # right
    (-1, 0): 3  # left
}

fruit_relative_position_dict = {
    (-1, -1): 0,
    (-1, 0): 1,
    (-1, 1): 2,
    (0, -1): 3,
    (0, 0): 4,
    (0, 1): 5,
    (1, -1): 6,
    (1, 0): 7,
    (1, 1): 8,
}


def get_obstacles(head, direction, snake):
    dx, dy = direction
    straight = ((head[0] + dx * 20, head[1] + dy * 20) in snake[:-1]) or \
               (head[0] + dx * 20 < 0) or (head[0] + dx * 20 >= SCREEN_WIDTH) or \
               (head[1] + dy * 20 < 0) or (head[1] + dy * 20 >= SCREEN_HEIGHT)

    left = ((head[0] - dy * 20, head[1] + dx * 20) in snake[:-1]) or \
           (head[0] - dy * 20 < 0) or (head[0] - dy * 20 >= SCREEN_WIDTH) or \
           (head[1] + dx * 20 < 0) or (head[1] + dx * 20 >= SCREEN_HEIGHT)

    right = ((head[0] + dy * 20, head[1] - dx * 20) in snake[:-1]) or \
            (head[0] + dy * 20 < 0) or (head[0] + dy * 20 >= SCREEN_WIDTH) or \
            (head[1] - dx * 20 < 0) or (head[1] - dx * 20 >= SCREEN_HEIGHT)

    return [int(straight), int(left), int(right)]


def get_state(snake, direction, food_position):
    head = snake[-1]

    # Calculate the relative position of the fruit
    fruit_relative_position = (
        int(np.sign(food_position[0] - head[0])),
        int(np.sign(food_position[1] - head[1]))
    )

    # Calculate the absolute distance
    absolute_distance = abs(food_position[0] - head[0]) + abs(food_position[1] - head[1])

    # Discretize the distance into categories
    distance_category = 0
    if absolute_distance <= SCREEN_WIDTH / 128:
        distance_category = 0
    elif absolute_distance <= SCREEN_WIDTH / 64:
        distance_category = 1
    elif absolute_distance <= SCREEN_WIDTH / 32:
        distance_category = 2
    elif absolute_distance <= SCREEN_WIDTH / 16:
        distance_category = 3   
    elif absolute_distance <= SCREEN_WIDTH / 8:
        distance_category = 4
    elif absolute_distance <= SCREEN_WIDTH / 4:
        distance_category = 5
    elif absolute_distance <= SCREEN_WIDTH / 2:
        distance_category = 6   
    else:
        distance_category = 7

    # Obstacles
    obstacles = [
        int(head[1] <= 0 or [head[0], head[1] - 20] in snake[:-1]),  # up
        int(head[1] >= SCREEN_HEIGHT - 20 or [head[0], head[1] + 20] in snake[:-1]),  # down
        int(head[0] <= 0 or [head[0] - 20, head[1]] in snake[:-1]),  # left
        int(head[0] >= SCREEN_WIDTH - 20 or [head[0] + 20, head[1]] in snake[:-1])  # right
    ]

    obstacles_int = sum([2 ** i * obstacles[i] for i in range(4)])

    state = (direction, fruit_relative_position, obstacles_int, distance_category)
    return state




def encode_state(state):
    direction, fruit_relative_position, obstacles_int, distance_category = state

    # Encode the direction and fruit relative position as before
    direction_encoded = direction_dict[tuple(direction)]
    fruit_relative_position_encoded = fruit_relative_position_dict[tuple(fruit_relative_position)]

    # Calculate the total number of states
    total_states = 4 * 9 * 16 * 3

    # Calculate the state index
    state_index = (
        direction_encoded * 9 * 16 * 8
        + fruit_relative_position_encoded * 16 * 8
        + obstacles_int * 8
        + distance_category
    )

    return state_index







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
    return snake


def check_collision(snake):
    head = snake[-1]
    return head in snake[0:-1] or head[0] < 0 or head[0] > SCREEN_WIDTH or head[1] < 0 or head[1] > SCREEN_HEIGHT

def step(snake, action, food_position, score,current_direction):
    # Update direction based on the action
    if action == 0:
        new_direction = (0, 1)
    elif action == 1:
        new_direction = (0,-1)
    elif action == 2:
        new_direction = (1,0)
    elif action == 3:
        new_direction = (-1,0)

 # Check if the new direction is opposite to the current direction
    if (new_direction[0] == -current_direction[0] and new_direction[1] == -current_direction[1]):
        new_direction = current_direction



    # Move the snake
    next_snake = snake.copy()
    next_snake = move_snake(next_snake, new_direction)

    # Check for food consumption
    if next_snake[-1] == food_position:
        next_snake.insert(0, next_snake[0].copy())
        next_food_position = [random.randrange(0, SCREEN_WIDTH, 20), random.randrange(0, SCREEN_HEIGHT, 20)]
        score +=1
    else:
        next_food_position = food_position

    # Get the next state
    next_state = get_state(next_snake, new_direction, next_food_position)

    return next_snake, next_state, next_food_position, new_direction, score


def reset():
    # Reset the game
    snake = [[100, 100], [120, 100], [140, 100]]
    food_position = [random.randrange(0, SCREEN_WIDTH, 20), random.randrange(0, SCREEN_HEIGHT, 20)]
    direction = (1, 0)
    
    # Reset the state
    state = get_state(snake, direction, food_position)
    
    return snake, state, food_position, direction



def gameLoop():
    plot_scores = []
    plot_mean_scores = []
    total_score=0
    num_game=0
    # Initialize the snake and food
    state_space  =  4 * 9 * 16 * 8
  # all state combinations
    action_space = 4
    q_table = np.zeros((state_space, action_space))
    max_episodes = 1000000
    gamma = 1
    step_size = 0.05
    epsilon = 0.03
    #Initialize S
    snake = [[100, 100], [120, 100], [140, 100]]
    food_position = [random.randrange(0, SCREEN_WIDTH, 20), random.randrange(0, SCREEN_HEIGHT, 20)]
    clock = pygame.time.Clock()
    currnet_direction = (1, 0)
    for episode in range(max_episodes):
        score = 0
        snake, state, food_position, currnet_direction = reset()
        state_index = encode_state(state)

        while True:
            if np.random.rand() <= epsilon:
                action = np.random.randint(action_space)
            else:
                action = np.argmax(q_table[state_index])

            # Take action and observe new state and reward
            next_snake, next_state, next_food_position, next_direction, score = step(snake, action, food_position,score,currnet_direction)
            next_state_index = encode_state(next_state)
            
            reward = 100 if next_snake[-1] == next_food_position else -1
            done = check_collision(next_snake)
            if done:
                reward = -100
            else: 
                reward = -1

            # Update Q-table
            q_table[state_index, action] = q_table[state_index, action] + step_size * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index, action])

            # Update variables for the next iteration
            snake, state, food_position, currnet_direction = next_snake, next_state, next_food_position, next_direction
            state_index = next_state_index

            

            # Draw everything
            screen.fill(WHITE)
            draw_snake(snake)
            draw_food(food_position)
            display_score(score)  # Display the score
            pygame.display.flip()

            if(done):
                break
            # Cap the frame rate
            if episode <1000:
                clock.tick(1000)
            else:
                clock.tick(10)
        total_score += score
        num_game+=1
        avg = total_score/num_game
        plot_scores.append(score)
        plot_mean_scores.append(avg)
        plot(plot_scores,plot_mean_scores)



gameLoop()




