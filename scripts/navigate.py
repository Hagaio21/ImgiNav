#!/usr/bin/env python3
"""Navigation algorithms (A* and RRT*) for pathfinding from camera to objects."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import json
import random
import math
from heapq import heappush, heappop

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.taxonomy import Taxonomy


def is_collision_free(grid, p1, p2, step_size=0.5):
    """Check if line segment from p1 to p2 is collision-free."""
    h, w = grid.shape
    x1, y1 = p1
    x2, y2 = p2
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if dist == 0:
        return True
    
    num_steps = int(dist / step_size) + 1
    for i in range(num_steps + 1):
        t = i / num_steps if num_steps > 0 else 0
        x, y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        gx, gy = int(x), int(y)
        if not (0 <= gx < w and 0 <= gy < h) or grid[gy, gx] == -1:
            return False
    return True


def find_goal_near_bbox(grid, bbox):
    """Find walkable point near object bbox."""
    h, w = grid.shape
    for offset in range(1, 5):
        for x in range(bbox['min_x'], bbox['max_x'] + 1):
            for y in [bbox['min_y'] - offset, bbox['max_y'] + offset]:
                if 0 <= y < h and 0 <= x < w and grid[y, x] != -1:
                    return (x, y)
        for y in range(bbox['min_y'], bbox['max_y'] + 1):
            for x in [bbox['min_x'] - offset, bbox['max_x'] + offset]:
                if 0 <= y < h and 0 <= x < w and grid[y, x] != -1:
                    return (x, y)
    return None


def astar_path(grid, start, goal_region):
    """A* pathfinding algorithm."""
    h, w = grid.shape
    start_x, start_y = int(start[0]), int(start[1])
    
    if not (0 <= start_x < w and 0 <= start_y < h):
        return None
    
    # Find walkable start if needed
    if grid[start_y, start_x] == -1:
        for radius in range(1, 10):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = start_x + dx, start_y + dy
                    if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] != -1:
                        start_x, start_y = nx, ny
                        break
                if grid[start_y, start_x] != -1:
                    break
            if grid[start_y, start_x] != -1:
                break
        if grid[start_y, start_x] == -1:
            return None
    
    goal = find_goal_near_bbox(grid, goal_region['bbox'])
    if goal is None:
        return None
    goal_x, goal_y = goal
    
    open_set = [(0, start_x, start_y)]
    came_from = {}
    g_score = np.full((h, w), np.inf, dtype=np.float32)
    g_score[start_y, start_x] = 0
    visited = np.zeros((h, w), dtype=bool)
    
    while open_set:
        _, cx, cy = heappop(open_set)
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        
        if cx == goal_x and cy == goal_y:
            path = []
            current = (cx, cy)
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(current)
            return path[::-1]
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] != -1 and not visited[ny, nx]:
                tentative_g = g_score[cy, cx] + 1
                if tentative_g < g_score[ny, nx]:
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[ny, nx] = tentative_g
                    f = tentative_g + abs(nx - goal_x) + abs(ny - goal_y)
                    heappush(open_set, (f, nx, ny))
    
    return None


def rrt_star_path(grid, start, goal_region, max_iterations=5000, step_size=2.0, goal_radius=2.0):
    """RRT* pathfinding algorithm."""
    h, w = grid.shape
    start_x, start_y = float(start[0]), float(start[1])
    
    bbox = goal_region['bbox']
    goal_candidates = []
    for x in range(bbox['min_x'], bbox['max_x'] + 1):
        for y, off in [(bbox['min_y'] - 1, -1), (bbox['max_y'] + 1, 1)]:
            if 0 <= y < h and 0 <= x < w and grid[y, x] != -1:
                goal_candidates.append((float(x), float(y)))
    for y in range(bbox['min_y'], bbox['max_y'] + 1):
        for x, off in [(bbox['min_x'] - 1, -1), (bbox['max_x'] + 1, 1)]:
            if 0 <= y < h and 0 <= x < w and grid[y, x] != -1:
                goal_candidates.append((float(x), float(y)))
    
    if not goal_candidates:
        return None
    
    best_goal = min(goal_candidates, key=lambda g: math.sqrt((g[0]-start_x)**2 + (g[1]-start_y)**2))
    goal_x, goal_y = best_goal
    
    tree = {(start_x, start_y): (None, 0.0)}
    
    for _ in range(max_iterations):
        if random.random() < 0.1:
            rand_x = goal_x + random.uniform(-goal_radius, goal_radius)
            rand_y = goal_y + random.uniform(-goal_radius, goal_radius)
        else:
            rand_x, rand_y = random.uniform(0, w-1), random.uniform(0, h-1)
        
        nearest = min(tree.keys(), key=lambda n: (n[0]-rand_x)**2 + (n[1]-rand_y)**2)
        
        nx, ny = nearest
        dist = math.sqrt((rand_x - nx)**2 + (rand_y - ny)**2)
        if dist > step_size:
            angle = math.atan2(rand_y - ny, rand_x - nx)
            new_x = nx + step_size * math.cos(angle)
            new_y = ny + step_size * math.sin(angle)
        else:
            new_x, new_y = rand_x, rand_y
        
        if not (0 <= new_x < w and 0 <= new_y < h):
            continue
        if not is_collision_free(grid, nearest, (new_x, new_y)):
            continue
        
        new_node = (new_x, new_y)
        search_radius = step_size * 2.0
        nearby = [(n, math.sqrt((n[0]-new_x)**2 + (n[1]-new_y)**2)) for n in tree if math.sqrt((n[0]-new_x)**2 + (n[1]-new_y)**2) < search_radius]
        
        best_parent = nearest
        best_cost = tree[nearest][1] + math.sqrt((new_x-nearest[0])**2 + (new_y-nearest[1])**2)
        
        for node, dist in nearby:
            if is_collision_free(grid, node, new_node):
                cost = tree[node][1] + dist
                if cost < best_cost:
                    best_cost = cost
                    best_parent = node
        
        tree[new_node] = (best_parent, best_cost)
        
        for node, dist in nearby:
            if node != best_parent:
                new_cost = best_cost + dist
                if new_cost < tree[node][1] and is_collision_free(grid, new_node, node):
                    tree[node] = (new_node, new_cost)
        
        if math.sqrt((new_x - goal_x)**2 + (new_y - goal_y)**2) < goal_radius:
            path = []
            current = new_node
            while current is not None:
                path.append((int(current[0]), int(current[1])))
                current = tree[current][0]
            return path[::-1]
    
    return None


def draw_dashed_line(draw, p1, p2, color, dash=8, gap=4, width=2):
    """Draw dashed line from p1 to p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist == 0:
        return
    dx, dy = dx / dist, dy / dist
    pos, drawing = 0, True
    while pos < dist:
        if drawing:
            end = min(pos + dash, dist)
            draw.line([(p1[0]+dx*pos, p1[1]+dy*pos), (p1[0]+dx*end, p1[1]+dy*end)], fill=color, width=width)
            pos += dash
        else:
            pos += gap
        drawing = not drawing


def main():
    parser = argparse.ArgumentParser(description="Navigation with A* and RRT*")
    parser.add_argument("occupancy_grid_path", type=Path)
    parser.add_argument("objects_path", type=Path)
    parser.add_argument("cleaned_layout_path", type=Path)
    parser.add_argument("--taxonomy", type=Path, default=Path("../data_preparation_v2/taxonomy.json"))
    parser.add_argument("--algorithm", choices=["astar", "rrt_star"], default="astar")
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--step-size", type=float, default=2.0)
    args = parser.parse_args()

    grid = np.load(args.occupancy_grid_path)
    with open(args.objects_path) as f:
        data = json.load(f)
    
    camera_pos = tuple(data['camera_pos'])
    scale_x, scale_y = data['scale']
    
    taxonomy = Taxonomy(args.taxonomy)
    categories = taxonomy.data.get("categories", [])
    id2cat = {idx + 1: cat for idx, cat in enumerate(categories)}
    id2cat[0] = "Unknown"
    cat_colors = taxonomy.data.get("category_to_color", {})

    layout = Image.open(args.cleaned_layout_path).convert("RGB")
    draw = ImageDraw.Draw(layout)
    
    cam_x, cam_y = int(camera_pos[0] * scale_x), int(camera_pos[1] * scale_y)
    draw.ellipse([cam_x-5, cam_y-5, cam_x+5, cam_y+5], fill=(255, 0, 0))

    paths = []
    for obj in data['objects']:
        bbox = obj.get('bbox_grid', obj['bbox'])
        
        if args.algorithm == "astar":
            path = astar_path(grid, camera_pos, {'bbox': bbox})
        else:
            path = rrt_star_path(grid, camera_pos, {'bbox': bbox}, args.max_iterations, args.step_size)
        
        if not path:
            continue
        
        cat = id2cat.get(obj['class_id'], "Unknown")
        color = tuple(cat_colors.get(cat, [127, 127, 127]))
        
        cx, cy = obj.get('centroid_grid', obj['centroid'])
        cx, cy = int(cx * scale_x), int(cy * scale_y)
        draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=color, outline=(0,0,0))
        
        for i in range(len(path) - 1):
            p1 = (int(path[i][0] * scale_x), int(path[i][1] * scale_y))
            p2 = (int(path[i+1][0] * scale_x), int(path[i+1][1] * scale_y))
            draw_dashed_line(draw, p1, p2, color)
        
        paths.append({'class_id': obj['class_id'], 'centroid': obj['centroid'], 'bbox': bbox, 'path': path})

    base = args.cleaned_layout_path.stem.replace("_cleaned", "")
    out_dir = args.cleaned_layout_path.parent
    layout.save(out_dir / f"{base}_paths_{args.algorithm}.png")
    with open(out_dir / f"{base}_paths_{args.algorithm}.json", 'w') as f:
        json.dump({'camera_pos': camera_pos, 'paths': paths}, f, indent=2)
    
    print(f"Found {len(paths)} paths using {args.algorithm}")


if __name__ == "__main__":
    main()