import cv2
import imutils
import numpy as np
import sys
import math
import time
from typing import List, Tuple
from tensorflow.python.keras.models import load_model

# Global variables
classes = np.arange(0, 10)
input_size = 48

# Load the OCR model
model = load_model('model-OCR.h5')

def get_perspective(img, location, height=900, width=900):
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(np.float32([location[0], location[3], location[1], location[2]]),
                                         np.float32([[0, 0], [width, 0], [0, height], [width, height]]))
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result


def predict_numbers(boxes, model):
    numbers = [np.argmax(model.predict(np.expand_dims(box, axis=0))[0]) for box in boxes]
    return numbers

def find_empty_cells(numbers):
    return [i for i, num in enumerate(numbers) if num == 0]

def fill_empty_cells(numbers, empty_cells):
    return [num if i not in empty_cells else 0 for i, num in enumerate(numbers)]

def convert_to_2d_array(numbers):
    return [numbers[i:i + 9] for i in range(0, len(numbers), 9)]

def find_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (5, 5), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)

    keypoints = cv2.findContours(img_threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)

    # new_img = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break

    result = get_perspective(img, location)
    return result, location


def split_boxes(board):
    rows = np.vsplit(board, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            # Convert the box to grayscale before resizing
            box_gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
            box_resized = cv2.resize(box_gray, (input_size, input_size))/255.0
            # cv2.imshow("Splitted block", box_resized)
            # cv2.waitKey(50)
            boxes.append(np.expand_dims(box_resized, axis=-1))  
    cv2.destroyAllWindows()
    return boxes

def detect_numbers(image_path):
    original_img = cv2.imread(image_path)
    processed_img, roi_location = find_board(original_img)
    transformed_img = get_perspective(original_img, roi_location)
    boxes = split_boxes(transformed_img)
    numbers = predict_numbers(boxes, model)
    empty_cells = find_empty_cells(numbers)
    filled_numbers = fill_empty_cells(numbers, empty_cells)
    two_d_array = convert_to_2d_array(filled_numbers)

    return two_d_array



def is_safe(grid: List[List[int]], row: int, col: int, num: int, size: int) -> bool:
    for x in range(size):
        if grid[row][x] == num or grid[x][col] == num:
            return False

    sqrt_size = int(math.sqrt(size))
    start_row, start_col = row - row % sqrt_size, col - col % sqrt_size
    for i in range(sqrt_size):
        for j in range(sqrt_size):
            if grid[i + start_row][j + start_col] == num:
                return False
    return True

def count_possibilities(grid: List[List[int]], row: int, col: int, size: int) -> int:
    used = [False] * (size + 1)

    for i in range(size):
        if grid[row][i] != 0:
            used[grid[row][i]] = True

    for i in range(size):
        if grid[i][col] != 0:
            used[grid[i][col]] = True

    sqrt_size = int(math.sqrt(size))
    start_row, start_col = row - row % sqrt_size, col - col % sqrt_size
    for i in range(sqrt_size):
        for j in range(sqrt_size):
            if grid[i + start_row][j + start_col] != 0:
                used[grid[i + start_row][j + start_col]] = True

    count = 0
    for i in range(1, size + 1):
        if not used[i]:
            count += 1

    return count

def find_empty_cell_with_fewest_possibilities(grid: List[List[int]], size: int) -> Tuple[int, int]:
    min_possibilities = size + 1
    cell_with_min_possibilities = (-1, -1)

    for i in range(size):
        for j in range(size):
            if grid[i][j] == 0:
                possibilities = count_possibilities(grid, i, j, size)
                if possibilities < min_possibilities:
                    min_possibilities = possibilities
                    cell_with_min_possibilities = (i, j)

    return cell_with_min_possibilities

def solve_sudoku(grid: List[List[int]], size: int) -> bool:
    row, col = find_empty_cell_with_fewest_possibilities(grid, size)

    if row == -1 and col == -1:
        return True

    for num in range(1, size + 1):
        if is_safe(grid, row, col, num, size):
            grid[row][col] = num
            if solve_sudoku(grid, size):
                return True
            grid[row][col] = 0

    return False



def main():
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <image_file>")
        sys.exit(1)

    image_path = sys.argv[1]
    detected_numbers = detect_numbers(image_path)

    puzzle = []

    for row in detected_numbers:
        print (row,"\n")
        puzzle.append(row)

    

    start_time = time.time()

    if solve_sudoku(puzzle, len(puzzle)):
        end_time = time.time()
        duration = end_time - start_time

        print(f"Sudoku solved in {duration:.3f} seconds.")

        with open(sys.argv[1] + "_output.txt", 'w') as outFile:
            for row in puzzle:
                outFile.write(" ".join(map(str, row)) + "\n")

    else:
        with open(sys.argv[1] + "_output.txt", 'w') as outFile:
            outFile.write("No Solution")

if __name__ == "__main__":
    main()