import pygame
import sys
import random


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





def gameLoop():
    # Initialize the snake and food
    score = 0
    snake = [[100, 100], [120, 100], [140, 100]]
    food_position = [random.randrange(0, SCREEN_WIDTH, 20), random.randrange(0, SCREEN_HEIGHT, 20)]
    clock = pygame.time.Clock()
    direction = [1, 0]
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != [0, 1]:
                    direction = [0, -1]
                if event.key == pygame.K_DOWN and direction != [0, -1]:
                    direction = [0, 1]
                if event.key == pygame.K_LEFT and direction != [1, 0]:
                    direction = [-1, 0]
                if event.key == pygame.K_RIGHT and direction != [-1, 0]:
                    direction = [1, 0]

        # Move the snake
        move_snake(snake, direction)

        # Check collisions
        if check_collision(snake):
            game_over_screen(score)  # Show the game over screen
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

        # Cap the frame rate
        clock.tick(10)


gameLoop()