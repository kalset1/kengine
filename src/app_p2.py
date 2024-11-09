import pygame
import numpy as np
from parser import parser

class Point:
    def __init__(self, x : float, y : float, z : float):
        self.x = x
        self.y = y
        self.z = z
        self.he = np.array([self.x, self.y, self.z])
        self.ho = np.array([self.x, self.y, self.z, 1])

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

    def gen_points(self) -> list:
        pts = []
        for vertex in self.vertices:
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
    
    def convert_pts(self, cam) -> None:
        pts = []

        for pt in self.vertex_points:
            x_2d, y_2d = cam.convert_point_to_2d(pt)
            pts.append([x_2d, y_2d])

        pts = np.array(pts) / np.max(np.linalg.norm((np.array(pts))))

        for i, pt in enumerate(pts):
            pts[i][0] = (pt[0] + 1) * (cam.width / 2)
            pts[i][1] = (pt[1] + 1) * (cam.height / 2)

        self.pts_2d = pts

    def rotate(self, rotation_matrix : np.array) -> None:
        self.rotation = self.rotation @ rotation_matrix # update rotation matrix

        for pt in self.vertex_points:
            pt.ho = rotation_matrix @ pt.ho # update vertex points

    def face_points_and_normal(self) -> None:
        surfaces_and_score = []

        for face in self.faces:
            points = [self.pts_2d[face[0]], self.pts_2d[face[1]], self.pts_2d[face[2]]]

            a = np.array((self.vertex_points[face[1]].ho)[:-1]) - np.array((self.vertex_points[face[0]].ho)[:-1])
            b = np.array((self.vertex_points[face[2]].ho)[:-1]) - np.array((self.vertex_points[face[0]].ho)[:-1])

            normal = np.cross(a, b)
            normal = normal / np.linalg.norm(normal)

            z_axis = self.rotation[:3, 2] - np.array([0, 0, -10]) # position of camera in global frame
            z_axis = z_axis / np.linalg.norm(z_axis)
 
            if ((np.dot(normal, z_axis)) >= 0): # triangle culling 
                score = max(0, np.dot(normal, z_axis))
            else:
                score = None

            surfaces_and_score.append([points, score])

        self.face_surface = surfaces_and_score




        

class Camera:
    def __init__(self, canvas_size : list):
        self.width, self.height = canvas_size
        self.camera_to_world = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, -10],
                                         [0, 0, 0, 1]])
        self.world_to_camera = np.linalg.inv(self.camera_to_world)

    def calculate_world_to_camera(self) -> None:
        self.world_to_camera = np.linalg.inv(self.camera_to_world)

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



class kEngine:
    def __init__(self, filepath, canvas_size):
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
        
    
    def perform_rotation(self) -> None:
        delta = self.get_delta_mouse()

        if delta is None:  # in case there was no change in delta
            return

        delta_x, delta_y = delta

        sensitivity = 0.005
        angle_x = delta_x * sensitivity
        angle_y = delta_y * sensitivity

        rotation_about_y = np.array([[np.cos(angle_x), 0, np.sin(angle_x), 0],
                                     [0, 1, 0, 0],
                                     [-np.sin(angle_x), 0, np.cos(angle_x), 0],
                                     [0, 0, 0, 1]])

        rotation_about_x = np.array([[1, 0, 0, 0],
                                     [0, np.cos(-angle_y), -np.sin(-angle_y), 0],
                                     [0, np.sin(-angle_y), np.cos(-angle_y), 0],
                                     [0, 0, 0, 1]])

        rotation_matrix = rotation_about_x @ rotation_about_y

        self.shape.rotate(rotation_matrix)


    def draw_shape(self):
        self.shape.convert_pts(self.cam)
        lines = self.shape.lines()

        for line in lines:
            pygame.draw.line(self.disp, (0, 0, 255), line[0], line[1], width=3)

        for i, pt in enumerate(self.shape.pts_2d):
            pygame.draw.circle(self.disp, (0, 0, 255), pt, 5)
            text = self.GAME_FONT.render(f"{i + 1}", False, (0, 0, 0))
            self.disp.blit(text, (pt[0], pt[1] + 10))


    def shade(self):
        self.shape.face_points_and_normal()
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

            self.disp.fill((255, 255, 255))

            self.draw_shape()

            self.perform_rotation()

            self.shade()

            pygame.display.flip()

        pygame.quit()





if __name__ == '__main__':
    engine = kEngine("../data/object.txt", [1000, 1000])
    engine.run()