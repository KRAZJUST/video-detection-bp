DIRECTIONS = {
        'N': 'north',
        'S': 'south',
        'E': 'east',
        'W': 'west',
        'NE': 'north-east',
        'NW': 'north-west',
        'SE': 'south-east',
        'SW': 'south-west'
    }
COLORS = ['red', 'blue', 'green', 'yellow', 'white', 'orange', 'purple', 'brown', 'black', 'pink', 'beige', 'gray']
OBJECTS = ['person', 'car', 'truck', 'bus']
QUADRANTS = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
INTERACTIONS = ['near', 'touching', 'overlapping', 'far']

# Color map for drawing bounding boxes in BGR format as the OpenCV library uses this format
# Format: (B, G, R)
#TODO: Maybe adjust the brown and beige colors
COLOR_MAP = {
    'red': (30, 30, 180),     
    'blue': (180, 30, 30),    
    'green': (30, 150, 30),   
    'yellow': (50, 180, 255), 
    'white': (220, 220, 220), 
    'orange': (30, 90, 180),  
    'purple': (120, 30, 120), 
    'brown': (20, 50, 100),   
    'black': (10, 10, 10),
    'pink': (200, 150, 200), 
    'beige': (190, 170, 130), 
    'gray': (100, 100, 100)   
}