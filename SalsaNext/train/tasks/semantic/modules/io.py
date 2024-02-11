import numpy as np

import DracoPy
from plyfile import PlyData, PlyElement

def write_draco(scan: np.array, path: str, qp: int):
    binary = DracoPy.encode(
        scan[:, :3],
        quantization_bits=qp, compression_level=1,
        quantization_range=-1, quantization_origin=None,
        create_metadata=False, preserve_order=False
    )

    with open(path, 'wb') as file:
        file.write(binary)

def read_draco(path: str):
    with open(path, 'rb') as file:
        data = file.read()
        mesh = DracoPy.decode(data)
    return mesh

def write_ply(scan: np.array, path: str, format: str="binary"):
    """Save a point cloud as a Ply file

    :param pointcloud: Point cloud to save
    :param path: File path where to point cloud should be saved
    """
    formats = ["ascii", "binary"]
    if format not in formats:
        raise ValueError(f"ply format {format} not supported. Must be one of {formats}")
    write_text = format == "ascii"

    elements = []
    types = []

    elements.append(scan[:, :3])
    types.extend((('x', 'single'), ('y', 'single'), ('z', 'single')))

    #elements.append(np.zeros((len(scan), 3)))
    #types.extend((('red', 'u1'), ('green', 'u1'), ('blue', 'u1')))

    elements = np.hstack(tuple(elements))

    vertices = np.array(
        [tuple(x) for x in elements.tolist()],# TODO: Slow! Should be replaced with something faster if possible
        dtype=types)
    
    PlyData([PlyElement.describe(vertices, 'vertex')], text=write_text).write(path)

def read_velodyne(path: str):
    scan = np.fromfile(path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def write_velodyne(scan: np.array, path: str):
    scan = np.array(scan, dtype=np.float32)
    scan.tofile(path)

def write_labels(labels: np.array, path: str):
    labels.tofile(path)

def read_labels(path: str):
    label = np.fromfile(path, dtype=np.int32) & 0xFFFF
    return label