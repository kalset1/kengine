import pygame
import numpy as np
import argparse
from parser import parser

"""
Point class for convient conversion of points to homogenous 
and heterogenous points (represented as numpy vectors)
"""
class Point:
    def __init__(self, x : float, y : float, z : float):
        self.x = x
        self.y = y
        self.z = z
        self.he = np.array([self.x, self.y, self.z])
        self.ho = np.array([self.x, self.y, self.z, 1])


"""
Shape class: defines the shape to be represented. This can be done through a list of vertices
and faces. Additionally, the face_surface's are stored as a list of points with their associated
score, which is then used to determine the amount of shading necessary. 
"""
class Shape:
    def __init__(self, num_vertices : int, num_faces : int, vertices : list, faces : list):
        self.num_vertices = num_vertices
        self.num_faces = num_faces
        self.vertices = vertices
        self.faces = faces

        self.vertex_points = self.gen_points()

        self.face_surface = None
        self.pts_2d = None
        self.rotation = np.diag((1, 1, 1, 1))
        self.initial_scale = None 

    def gen_points(self) -> list:
        pts = []
        flattened_faces = np.array(self.faces).flatten()

        for vertex in self.vertices:
            # do not include vertex in points if it is never referenced in faces
            if vertex[0] - 1 not in flattened_faces: # to account for 1 indexing vs. 0 indexing
                continue

            x, y, z = vertex[1:]
            pt = Point(x, y, z)
            pts.append(pt)

        return pts
    
    def lines(self) -> list:
        lines = []
        for face in self.faces:
            line_1 = (self.pts_2d[face[0]], self.pts_2d[face[1]])
            line_2 = (self.pts_2d[face[1]], self.pts_2d[face[2]])
            line_3 = (self.pts_2d[face[2]], self.pts_2d[face[0]])
            lines.extend([line_1, line_2, line_3])

        return lines
    
    """
    Convert 3D points to 2D points. Uses the convert_point_to_2d 
    function from the Camera class to convert all vertex points to their 2D representation,
    and then appropriately scale them to the correct size for display on the window.
    It utilizes an initial scaling factor to then determine future point coordinates,
    to prevent warping of the shape.
    """
    def convert_pts(self, cam) -> None:
        pts = []

        for pt in self.vertex_points:
            result = cam.convert_point_to_2d(pt)
            if result is None:
                pts.append(None)
                continue
            x_2d, y_2d = result
            pts.append([x_2d, y_2d])

        valid_pts = [pt for pt in pts if pt is not None]
        
        if not valid_pts:
            self.pts_2d = pts
            return

        # bounding box
        pts_array = np.array(valid_pts)
        min_x = np.min(pts_array[:, 0])
        max_x = np.max(pts_array[:, 0])
        min_y = np.min(pts_array[:, 1])
        max_y = np.max(pts_array[:, 1])

        width = max_x - min_x
        height = max_y - min_y
        if self.initial_scale is None:
            self.initial_scale = min(cam.width / width, cam.height / height) * 0.5
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        for i, pt in enumerate(pts):
            if pt is not None:
                x = (pt[0] - center_x) * self.initial_scale + cam.width / 2
                y = (pt[1] - center_y) * self.initial_scale + cam.height / 2
                pts[i] = [x, y]

        self.pts_2d = pts

    """
    Incrementally update the coordinates of the vertex points,
    by rotating them by the specified rotation matrix
    """
    def rotate(self, rotation_matrix : np.array) -> None:
        self.rotation = self.rotation @ rotation_matrix # update rotation matrix

        for pt in self.vertex_points:
            pt.ho = rotation_matrix @ pt.ho # update vertex points


    """
    Calculate the associated shading factor with each face. This is done through
    computing the vector normal to each face, and then taking the dot product
    of this vector with the light direction towards the shape. This is a mathematical
    representation of how light hits an object: an object has a face, which depending
    on its orientation towards light (in this case, the z-axis of the camera), will
    appear to be "lighter" or "darker". It becomes lighter the more direct the light
    hits the face, which can be represented as a dot product between the vector normal
    of the face and light's direction. This function does exactly that, calculating
    the vector normal for each face, and then taking the dot product of this vector
    with the z-axis. The culling test is also used to "cull" faces which have a 
    0 or negative dot product. This physically represents a face which is not in
    view of the lighting direction, and thus should not be shown.
    """
    def calculate_face_shade_scores(self) -> None:
        surfaces_and_score = []

        camera_pos = np.array([0, 0, 1e12])  # camera is at [0,0,1e12]

        for face in self.faces:
            points = [self.pts_2d[face[0]], self.pts_2d[face[1]], self.pts_2d[face[2]]]

            p0 = self.vertex_points[face[0]].ho[:3]
            p1 = self.vertex_points[face[1]].ho[:3]
            p2 = self.vertex_points[face[2]].ho[:3]

            vec1 = p1 - p0
            vec2 = p2 - p0
            normal = np.cross(vec1, vec2)

            # skip degenerates
            normal_length = np.linalg.norm(normal)
            if normal_length < 1e-6:
                continue
                
            normal = normal / normal_length

            face_center = (p0 + p1 + p2) / 3.0

            view_dir = camera_pos - face_center
            view_dir = view_dir / np.linalg.norm(view_dir)

            # culling test
            dot_product = np.dot(normal, view_dir)
            
            if dot_product <= 0: 
                surfaces_and_score.append([points, None])
            else:
                score = dot_product
                surfaces_and_score.append([points, score])

        self.face_surface = surfaces_and_score



"""
The camera class defines a camera object which features a coordinate
frame transfrom from the camera to the world and from the world to 
the camera. This is then used in the function convert_point_to_2d,
which takes in a 3D point from the world and converts it into a 2D point
in the camera's frame.
"""
class Camera:
    def __init__(self, canvas_size : list):
        self.width, self.height = canvas_size

        # camera faces -z direction (into screen), position at [0,0,1e12]
        self.camera_to_world = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, -1, 1e12],  # change z direction and position
                                        [0, 0, 0, 1]])
        self.world_to_camera = np.linalg.inv(self.camera_to_world)

    def calculate_world_to_camera(self) -> None:
        self.world_to_camera = np.linalg.inv(self.camera_to_world)

    """
    Converts 3D point in the world frame to a point in the camera's 3D coordinate frame. 
    It then converts to 2D, utilizing the property of similar triangles to compute its 
    x and y location in the 2D image plane
    """
    def convert_point_to_2d(self, pt: Point):
        pt_in_camera_frame = (self.world_to_camera @ pt.ho.T)[:-1]  # homogeneous point
        x_3d, y_3d, z_3d = pt_in_camera_frame

        if z_3d == 0:
            return None  # avoid division by zero

        x_2d = x_3d / -z_3d
        y_2d = y_3d / -z_3d

        if abs(x_2d) > self.width / 2 or abs(y_2d) > self.height / 2:
            return None  # return None (error)

        return x_2d, y_2d


"""
The kengine class houses the driver code for pygame and drawing the shapes.
It creates Camera and Shape objects for itself, and then runs methods 
from these classes to update points dynamically based on user input. 
It then draws the Shape based on these updated points and (optionally) shades them.
"""
class kEngine:
    def __init__(self, filepath, canvas_size, mode):
        self.canvas_size = canvas_size
        self.height, self.width = canvas_size

        obj_info = parser(filepath)
        self.shape = Shape(obj_info['num_vertices'], 
                          obj_info['num_faces'], 
                          obj_info['vertices'], 
                          obj_info['faces'])
        
        self.cam = Camera(canvas_size)

        self.disp = None

        self.mouse_pressed = None
        self.prev_mouse_pos = None

        self.mode = mode

    """
    Computes the change in the mouse's position based upon
    a previous value and whether or not the mouse was pressed on the
    previous timestep. 
    """
    def get_delta_mouse(self):
        currently_pressed = pygame.mouse.get_pressed()[0]
        curr_x, curr_y = pygame.mouse.get_pos()
        prev_x, prev_y = self.prev_mouse_pos

        if (currently_pressed and self.mouse_pressed):

            self.prev_mouse_pos = [curr_x, curr_y]
            self.mouse_pressed = currently_pressed

            return (curr_x - prev_x, curr_y - prev_y)
        
        else:
            self.prev_mouse_pos = [curr_x, curr_y]
            self.mouse_pressed = currently_pressed
            return None
        
    """
    Computes a new rotation matrix based on the change
    in the mouse's position. It creates two rotation matrices for the x-axis
    and y-axis, and then multiplies them to get the total rotation change 
    that will end up being applied to the points.
    """
    def perform_rotation(self) -> None:
        delta = self.get_delta_mouse()

        if delta is None:  # in case there was no change in delta
            return

        delta_x, delta_y = delta

        sensitivity = 0.005
        angle_x = -delta_x * sensitivity  # Inverted sign for x rotation
        angle_y = -delta_y * sensitivity  # Inverted sign for y rotation

        rotation_about_y = np.array([[np.cos(-angle_x), 0, np.sin(-angle_x), 0],
                                     [0, 1, 0, 0],
                                     [-np.sin(-angle_x), 0, np.cos(-angle_x), 0],
                                     [0, 0, 0, 1]])

        rotation_about_x = np.array([[1, 0, 0, 0],
                                     [0, np.cos(angle_y), -np.sin(angle_y), 0],
                                     [0, np.sin(angle_y), np.cos(angle_y), 0],
                                     [0, 0, 0, 1]])

        rotation_matrix = rotation_about_x @ rotation_about_y

        self.shape.rotate(rotation_matrix)


    def draw_shape(self):
        self.shape.convert_pts(self.cam)
        lines = self.shape.lines()

        for line in lines:
            pygame.draw.line(self.disp, (0, 0, 255), line[0], line[1], width=3)

        for pt in self.shape.pts_2d:
            pygame.draw.circle(self.disp, (0, 0, 255), pt, 5)


    """
    Determines the color of the face's surface based on the score it 
    obtained from the calculate_face_shade_scores function 
    (method for score calculation described in function comment)
    """
    def shade(self):
        self.shape.calculate_face_shade_scores()
        for face in self.shape.face_surface:
            points, score = face
            
            if score == None: # triangle culling, we shouldn't show this
                continue

            pygame.draw.polygon(self.disp, (0, 0, int(95 + (255-95)*score)), points)


    def run(self):
        pygame.init()

        self.GAME_FONT = pygame.font.SysFont('Times New Roman', 30)

        self.disp = pygame.display.set_mode(self.canvas_size)
        self.mouse_pressed = False
        self.prev_mouse_pos = pygame.mouse.get_pos()

        run = True
        while run:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            # first, fill display with white color background
            self.disp.fill((255, 255, 255))

            # then draw the shape on the canvas
            self.draw_shape()

            # if mode is 2, shade in the shape
            if (self.mode == "2"):
                self.shade()

            # calculate new rotations based on mouse input
            self.perform_rotation()

            pygame.display.flip()

        pygame.quit()





if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="3D graphics engine built entirely with numpy and drawn using pygame.\n")
    argparser.add_argument("-m", "--mode", help = "select which mode to display (1 for wireframe, 2 for shading), default = 2\n")
    argparser.add_argument("-i", "--input", help = "file path to input, default = ../data/object.txt\n")
    argparser.add_argument("-s", "--size", help = "window size to display, default = 1000,1000\n")
    args = argparser.parse_args()

    input = args.input if args.input != None else '../data/object.txt'
    mode = args.mode if args.mode != None else "2"
    size = args.size if args.size != None else [1000, 1000]

    if (mode != '1' and mode != '2'):
        print("\nInvalid mode entered! (use 1 or 2)\n")
        exit()

    try:
        size = size.replace(' ', "").split(',')
        size = list(map(int, size))
    except:
        print("\nInvalid size entered! (enter two comma separated values with no spaces)\n")
        exit()

    engine = kEngine(input, size, mode)
    engine.run()