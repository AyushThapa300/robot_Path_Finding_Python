import numpy as np
import matplotlib.pyplot as plt
import heapq
import random

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    
    came_from = {}
    cost_so_far = {start: 0}
    
    while open_set:
        _, current_cost, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in [(0,1),(1,0),(-1,0),(0,-1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 1:  # obstacle
                    continue
                new_cost = current_cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

    return None  # No path found

def plot_grid(grid, path=None, start=None, goal=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='Greys', origin='upper')
    
    if path:
        px, py = zip(*path)
        plt.plot(py, px, color='blue', linewidth=2, label="Path")
    
    if start:
        plt.scatter(start[1], start[0], c='green', s=100, label="Start")
    if goal:
        plt.scatter(goal[1], goal[0], c='red', s=100, label="Goal")

    plt.title("Robot Path Planning (A*) with Random Obstacles")
    plt.grid(True)
    plt.legend()
    plt.xticks(np.arange(grid.shape[1]))
    plt.yticks(np.arange(grid.shape[0]))
    plt.show()

def generate_random_grid(rows, cols, obstacle_prob=0.2, start=None, goal=None):
    grid = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if random.random() < obstacle_prob:
                grid[i][j] = 1
    # Ensure start and goal are not blocked
    if start:
        grid[start] = 0
    if goal:
        grid[goal] = 0
    return grid

def main():
    print("--- Robot Path Planning using A* with Random Obstacles ---")
    
    rows = 15
    cols = 15
    start = (0, 0)
    goal = (14, 14)

    # Generate grid with random obstacles
    grid = generate_random_grid(rows, cols, obstacle_prob=0.25, start=start, goal=goal)

    # Find and plot path
    path = a_star(grid, start, goal)
    if path:
        print("\nPath found!")
        plot_grid(grid, path, start, goal)
    else:
        print("\nNo path found. Try re-running for a new random grid.")
        plot_grid(grid, None, start, goal)

if __name__ == "__main__":
    main()
