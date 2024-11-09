"""
simple method for extracting relevant information from object.txt
"""

def parser(file_path: str) -> dict:
    # initial dictionary 
    object_info = {'num_vertices': 0, 'num_faces' : 0, 'vertices' : [], 'faces' : []}

    with open(file_path, 'r') as f:
        num_vertices, num_faces = map(int, f.readline().split(","))
        object_info['num_vertices'], object_info['num_faces'] = num_vertices, num_faces

        for _ in range(num_vertices):
            vertex = list(map(float, f.readline().split(",")))
            vertex[0] = int(vertex[0])

            object_info['vertices'].append(vertex)

        for _ in range(num_faces):
            face = list(map(lambda x: int(x) - 1, f.readline().split(",")))
            object_info['faces'].append(face)

    return object_info
