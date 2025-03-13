import pygame
import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Hand Gesture Coding Blocks")
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

# Load MediaPipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Webcam Setup
cap = cv2.VideoCapture(0)

# Predefined Java Questions
java_questions = [
    ["public class Main {", "public static void main(String[] args) {", "System.out.println(\"Hello, World!\");", "}", "}"],
    ["class Sum {", "public static void main(String[] args) {", "int a = 5, b = 10;", "System.out.println(a + b);", "}"],
    ["class Loop {", "public static void main(String[] args) {", "for(int i = 0; i < 5; i++)", "System.out.println(i);", "}"],
]

# Define Code Blocks
class CodeBlock:
    def __init__(self, text, x, y):  # Fixed __init__ method
        self.text = text
        self.rect = pygame.Rect(x, y, 300, 50)

    def draw(self):
        pygame.draw.rect(screen, (50, 150, 250), self.rect, border_radius=10)
        text_surface = font.render(self.text, True, (255, 255, 255))
        screen.blit(text_surface, (self.rect.x + 10, self.rect.y + 10))

# Load initial question
def load_new_question():
    global code_blocks, correct_order, pointer_index
    correct_order = random.choice(java_questions)  # Pick a correct order
    shuffled_lines = correct_order[:]  # Copy list to avoid modifying original
    random.shuffle(shuffled_lines)  # Shuffle for challenge
    code_blocks = [CodeBlock(line, 250, 100 + i * 100) for i, line in enumerate(shuffled_lines)]
    pointer_index = 0  # Reset pointer

# Score
score = 0
load_new_question()

pointer_index = 0  # Pointer position
last_fingers_up = 0  # To track gesture changes

def detect_gesture():
    success, frame = cap.read()
    if not success:
        return 0, np.zeros((150, 200, 3), dtype=np.uint8)  # Return black frame if no webcam

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    fingers_up = 0
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            fingers = [
                hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,  # Index finger
                hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,  # Middle finger
                hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,  # Ring finger
                hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y   # Pinky finger
            ]
            fingers_up = sum(fingers)
            break  # Process only one hand
    return fingers_up, frame

def check_correctness():
    return [block.text for block in code_blocks] == correct_order

running = True
while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    fingers_up, frame = detect_gesture()

    if fingers_up != last_fingers_up:
        if fingers_up == 1 and pointer_index > 0:
            pointer_index -= 1  # Move up
        elif fingers_up == 2 and pointer_index < len(code_blocks) - 1:
            pointer_index += 1  # Move down
        elif fingers_up == 3 and pointer_index > 0:
            # Swap current block with previous one
            code_blocks[pointer_index].text, code_blocks[pointer_index - 1].text = (
                code_blocks[pointer_index - 1].text,
                code_blocks[pointer_index].text,
            )
        elif fingers_up == 4 and pointer_index < len(code_blocks) - 1:
            # Swap current block with next one
            code_blocks[pointer_index].text, code_blocks[pointer_index + 1].text = (
                code_blocks[pointer_index + 1].text,
                code_blocks[pointer_index].text,
            )

    last_fingers_up = fingers_up

    # Draw Code Blocks
    for i, block in enumerate(code_blocks):
        block.draw()
        if i == pointer_index:
            pygame.draw.rect(screen, (255, 0, 0), block.rect, 3)  # Highlight current selection

    # Show webcam feed
    frame = cv2.resize(frame, (200, 150))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
    screen.blit(frame_surface, (600, 20))

    # Check correctness
    if check_correctness():
        score += 10  # Increase score
        pygame.display.flip()
        pygame.time.delay(2000)  # Show success message for 2 seconds
        load_new_question()  # Load new question

    # Display score
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (50, 50))

    pygame.display.flip()
    clock.tick(30)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
