from pathlib import Path

# breadboard_classifier: TensorFlow, DocAligner and pycpd made optional (see __init__.py).
# TF only loads the corner classifier, which is not used by normalize_image(); DocAligner can
# be replaced by any callable passed as Normalizer(corner_model=...); pycpd is only needed by
# the *_cpd registration methods.
try:
    import tensorflow as tf
except ImportError:
    tf = None
import numpy as np
try:
    from docaligner import DocAligner
except ImportError:
    DocAligner = None
import cv2
from PIL import Image, ImageOps
import os
from fastquadtree import QuadTree
try:
    from pycpd import RigidRegistration, AffineRegistration
except ImportError:
    RigidRegistration = AffineRegistration = None
from typing import Literal, Optional

import numpy as np

try:
    import open3d as o3d
    USE_OPEN3D = True
    print("Using Open3D for ICP")
except ImportError: 
    USE_OPEN3D = False
    print("Falling back to vendored ICP")


import random
from skimage import transform

# from https://github.com/ClayFlannigan/icp
# edited to allow arrays of different length
from .vendor.icp import icp

def draw_corners(image, corners, radius=None, thickness=None, colors=None):
    # red, green, blue, magenta
    if colors is None:
        colors=[[0,0,255], [0, 255, 0], [255, 0, 0], [255, 0, 255]]
    corners = np.array(corners)

    annotated = np.copy(image)

    if radius == None:
        radius = np.min(image.shape[:2]) // 40

    if thickness == None:
        thickness = radius * 2

    for i, corner in enumerate(corners):
        annotated = cv2.circle(annotated, corner.astype(int), radius=radius, color=colors[i % 4], thickness=thickness)
        annotated = cv2.putText(annotated, f"{i}", (int(corner[0]) - radius // 2, int(corner[1]) + radius // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
    return annotated


def is_landscape(corners):
    even_dist = np.linalg.norm(corners[0] - corners[1]) + np.linalg.norm(corners[2] - corners[3]) 
    odd_dist = np.linalg.norm(corners[1] - corners[2]) + np.linalg.norm(corners[3] - corners[0]) 

    return even_dist > odd_dist

    # center = corners.mean(axis=0)

    # dif = (corners - center)
    # a_dif = np.abs(dif).mean(axis=0)

    # return a_dif[0] > a_dif[1]

def reorder_corners(corners):

    center = corners.mean(axis=0)

    relative = corners - center

    angles = -np.atan2(relative[:, 1], relative[:, 0])

    indices = angles.argsort()

    corners = corners[indices]

    portrait_indices = np.array([1, 2, 3, 0])

    if not is_landscape(corners):
        corners = corners[portrait_indices]

    return corners


def skew_metric(corners):
    length_diff = abs(np.linalg.norm(corners[0] - corners[1]) - np.linalg.norm(corners[2] - corners[3]))
    width_diff = abs(np.linalg.norm(corners[0] - corners[3]) - np.linalg.norm(corners[1] - corners[2]))
    return length_diff + width_diff;

def aspect_metric(corners):
    target_aspect_ratio = 1024/340
    length = np.linalg.norm(corners[0] - corners[1]) + np.linalg.norm(corners[2] - corners[3])
    width =  np.linalg.norm(corners[0] - corners[3]) + np.linalg.norm(corners[1] - corners[2])
    return abs(length / width - target_aspect_ratio)

def crop_square(image, l, size):
    h, w = image.shape[:2]
    l[0] = min(l[0], w - size)
    l[1] = min(l[1], h - size)
    return image[l[1]:l[1]+size, l[0]:l[0]+size]

def resize_width(image: np.ndarray, target_width: int):
    h, w = image.shape[:2]
    factor = float(target_width) / float(w)
    return cv2.resize(image, (target_width, int(h * factor)), interpolation=cv2.INTER_NEAREST)

def resize_height(image: np.ndarray, target_height: int):
    h, w = image.shape[:2]
    factor = float(target_height) / float(h)
    return cv2.resize(image, (int(w * factor), target_height), interpolation=cv2.INTER_AREA)

def get_source_corners_from_label(filename):
    source_corners = np.loadtxt(filename, dtype=np.float32)[1:].reshape(4 ,2)
    return reorder_corners(source_corners)

def image_label_pairs(image_dir, label_dir):
    for index, file in enumerate(os.listdir(os.fsencode(image_dir))):
        filename, ext = os.path.splitext(os.fsdecode(file))
        if ext.lower().endswith(Normalizer._image_extensions):
            image_pil = Image.open(image_dir + filename + ext)
            image_pil = ImageOps.exif_transpose(image_pil)
            image = np.array(image_pil)
            label = get_source_corners_from_label(label_dir + filename + '.txt')
            yield filename, image, label


def normalize_points(points):
    normalized = np.copy(points)

    translation = -np.mean(normalized, axis=0)

    normalized += translation

    max_dist = np.max(np.abs(normalized)) 
    scale_val = 1.0 / max_dist if max_dist != 0 else 1.0
    scale = np.array([scale_val, scale_val])

    normalized *= scale

    m = np.eye(3)
    m[0, 0] = scale[0]
    m[1, 1] = scale[1]

    m[:2, 2] = translation * scale

    return normalized, m



class PinGrid:
    _size: np.ndarray
    _pad: np.ndarray
    _grid_size: np.ndarray = np.array([65.1, 21.25])

    _base_points: np.ndarray
    points: np.ndarray
    labels = None

    # grid spacing in pixels
    pitch: np.ndarray

    _quadtree: QuadTree

    def __init__(self, size: np.ndarray, padding: np.ndarray=np.array([0, 0])):
        self._size = size
        self._pad = padding
        self._base_points, self.labels = PinGrid.base_pin_holes()
        
        print("Creating pingrid with", len(self._base_points), "points and", len(self.labels), "labels")
        
        low = np.array([0, 0], dtype=np.float64)
        high = self._size * 1.01

        padded_base_points = self._base_points * (1.0 - padding * 2) + padding
        self.points = np.array(padded_base_points * self._size, dtype=np.float32)
        self._quadtree = QuadTree((low[0], low[1], high[0] + 1, high[1] + 1), capacity=16)
        self._quadtree.insert_many_np(self.points)

        self.pitch = self._size / self._grid_size
    
    
    def transform_points_3x3(points, matrix):
        if points.size == 0:
            return points
        original_shape = points.shape
        points_transformed = points.reshape(-1, 1, 2)
        points_transformed = cv2.perspectiveTransform(points_transformed, matrix)
        return points_transformed.reshape(original_shape)
    
    def nearest_neighbors(self, points):
        target_correspondences = []
        for point in points:
            _, c = self._quadtree.nearest_neighbor_np((point[0], point[1]))
            target_correspondences.append(c)
        return np.array(target_correspondences, dtype=np.float32)


    def o3d_point_cloud_from_points(p):
        p_3d = np.hstack((p, np.zeros((p.shape[0], 1))))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p_3d)
        return pcd

    def icp_open3d(self, source):
        threshold=int(np.mean(self.pitch) * 4)

        source_pc = PinGrid.o3d_point_cloud_from_points(source)
        target_pc = PinGrid.o3d_point_cloud_from_points(self.points)

        trans_init = np.eye(4)

        # https://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pc, target_pc, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
        h = reg_p2p.transformation
        source_pc.transform(h)
        
        transformed = np.asarray(source_pc.points)[:, :2]
        # h = np.linalg.inv(h)
        # drop the z axis
        indices = [0, 1, 3]
        h = h[np.ix_(indices, indices)]
        return transformed, np.asarray(h)[:3, :3]


    def fit_icp(self, source: np.ndarray):
        if USE_OPEN3D:
            transformed, h = self.icp_open3d(source)
        else:
            h, distances, iterations = icp(source, self.points, max_iterations=50)
        return h
    
    def rigid_cpd(source, target):
        # s_n, s_m = normalize_points(source)
        t_n, t_m = normalize_points(target)
        s_m = t_m
        s_n = PinGrid.transform_points_3x3(source, t_m)

        reg = RigidRegistration(X=t_n, Y=s_n, w=0.2, sigma2=0.05)
        transformed, ((scale, rotation, translation)) = reg.register()
        h = np.eye(3)
        h[:2, 2] = translation
        h[:2, :2] = scale * rotation

        t_m_inv = np.linalg.inv(t_m)

        # h maps from s_m to t_m
        # make it go from s_m to world
        h = t_m_inv @ h

        # make it go from world to world
        h = h @ s_m

        transformed = PinGrid.transform_points_3x3(transformed, t_m_inv)

        return (transformed, h)

    def rigid_cpd_rev(source, target):
        # s_n, s_m = normalize_points(source)
        t_n, t_m = normalize_points(target)
        s_m = t_m
        s_n = PinGrid.transform_points_3x3(source, t_m)

        reg = RigidRegistration(X=s_n, Y=t_n, w=0.2, sigma2=0.05)
        transformed, ((scale, rotation, translation)) = reg.register()
        h = np.eye(3)
        h[:2, 2] = translation
        h[:2, :2] = scale * rotation

        # s_m to t_m
        h = np.linalg.inv(h)

        s_m_inv = np.linalg.inv(s_m)

        # h maps from s_m to t_m
        # make it go from t_m to world
        h = s_m_inv @ h

        # make it go from world to world
        h = h @ t_m

        transformed = PinGrid.transform_points_3x3(source, h)

        return (transformed, h)

    def affine_cpd(source, target):
        # s_n, s_m = normalize_points(source)
        t_n, t_m = normalize_points(target)
        s_m = t_m
        s_n = PinGrid.transform_points_3x3(source, t_m)

        reg = AffineRegistration(X=t_n, Y=s_n, w=0.1, sigma2=0.05)
        transformed, ((affine, translation)) = reg.register()
        
        affine_3 = np.eye(3)
        affine_3[:2, :2] = affine.T # apparently you need to transpose here for handedness or something, according to chatGPT

        h = np.eye(3)
        h[:2, 2] = translation

        h = h @ affine_3

        t_m_inv = np.linalg.inv(t_m)

        # h maps from s_m to t_m
        # make it go from s_m to world
        h = t_m_inv @ h

        # make it go from world to world
        h = h @ s_m

        transformed = PinGrid.transform_points_3x3(transformed, t_m_inv)

        return (transformed, h)

    def affine_cpd_rev(source, target):

        # s_n, s_m = normalize_points(source)
        t_n, t_m = normalize_points(target)
        s_m = t_m
        s_n = PinGrid.transform_points_3x3(source, t_m)

        reg = AffineRegistration(X=s_n, Y=t_n, w=0.05, sigma2=0.05)
        _, ((affine, translation)) = reg.register()
        
        affine_3 = np.eye(3)
        affine_3[:2, :2] = affine.T # apparently you need to transpose here to account for row vs column vector conventions, according to chatGPT

        h = np.eye(3)
        h[:2, 2] = translation

        h = h @ affine_3

        h = np.linalg.inv(h)

        t_m_inv = np.linalg.inv(t_m)

        # h maps from normalized source to normalized target
        # make it go from normalized source to input coordinates
        h = t_m_inv @ h

        # make it go input coordinates to input coordinates
        h = h @ s_m

        transformed = PinGrid.transform_points_3x3(source, h)

        return (transformed, h)

    def refine_x_ransac(self, source: np.ndarray):
        """
        Guess random initial transformations, use nearest neighbors to guess at correspondences,
        and run it through cv2.findHomography() N times and keep the transform with the most inliners

        findHomography() is the only method here that handles perspective, but it needs 1-1 correspondences
        between points in the source and target arrays. Nearest neighbors alone are not a great way to do
        this because of grid aliasing, so if the source is not already mostly aligned this method tends to 
        fits one side well and lets the other side explode into perspective distortions.
        """
        
        best_matches = 0
        best_transform = np.eye(3)
        offsets = np.array([-2, -1, 0, 1, 2]) * np.mean(self.pitch)
        for i in range(len(offsets)):

            guess_transform = transform.AffineTransform(
                scale=np.array([1, 1]),
                rotation=0.0,
                translation=np.array([offsets[i], 0])
            ).params

            source_transformed = PinGrid.transform_points_3x3(source, guess_transform)

            target_correspondences = self.nearest_neighbors(source_transformed)

            if len(target_correspondences) < 4:
                continue

            # Most of these methods seem to give better results fitting the target to the source rather than vice versa
            h, mask = cv2.findHomography(target_correspondences, source_transformed, method=cv2.RANSAC, ransacReprojThreshold=np.mean(self.pitch)/4.0)
            if h is None or mask is None:
                continue
            h = np.linalg.inv(h)

            inliners = np.count_nonzero(mask)
            if inliners > best_matches:
                best_matches = inliners
                best_transform = h @ guess_transform
        return best_transform

    def refine_x_icp(self, source):
        best_score = PinGrid.single_score(self.evaluate_fit(source))
        best_transform = np.eye(3)
        best_refined = source
        offsets = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]) * np.mean(self.pitch)
        for i in range(len(offsets)):

            guess_transform = transform.AffineTransform(
                scale=np.array([1, 1]),
                rotation=0.0,
                translation=np.array([offsets[i], 0])
            ).params
            

            source_transformed = PinGrid.transform_points_3x3(source, guess_transform)

            h = self.fit_icp(source_transformed)

            source_refined = PinGrid.transform_points_3x3(source_transformed, h)

            score = PinGrid.single_score(self.evaluate_fit(source_refined))
            if score < best_score:
                best_score = score
                best_transform = h @ guess_transform
                best_refined = source_refined
                if offsets[i] != 0:
                    print("improved by shifting horizontally")
        return best_refined, best_transform


    def single_score(multi_score):
        """
        Combine inliner rmse, inliner ratio, and duplicate ratio arbitrarily into a single score.
        Definitely not the correct way to do this
        """
        rmse, inliner_ratio, duplicate_ratio = multi_score
        return (rmse / np.maximum(inliner_ratio ** 2, 0.0001))

    def fit_cpd_ransac(self, source):
        """
        Attempt to combine the strengths of each method. ICP is stable but limited to rigid transforms, affine CPD is decent 
        but needs a close starting alignment to not explode, cv2.findHomography with RANSAC is great if
        the points are already well aligned across the board but seems very sensitive to poor initial alignment.

        Run the best performing methods in decreasing order of stability and increasing order of quality, keeping
        the best performing transform to avoid degenerate edge cases.
        """
        best_score = PinGrid.single_score(self.evaluate_fit(source))
        best_h = np.eye(3)
        best_t = source

        h1 = self.fit_icp(source)
        t1 = PinGrid.transform_points_3x3(source, h1)
        c_score = PinGrid.single_score(self.evaluate_fit(t1))
        if c_score < best_score:
            # print("icp did something, improved score by", best_score - c_score)
            best_score = c_score
            best_h = h1
            best_t = t1

        t2, h2 = PinGrid.affine_cpd_rev(best_t, self.points)
        c_score = PinGrid.single_score(self.evaluate_fit(t2))
        if c_score < best_score:
            # print("affine CPD did something, improved score by", best_score - c_score)
            best_score = c_score
            best_h = h2 @ best_h
            best_t = t2

        h3 = self.refine_x_ransac(best_t)
        t3 = PinGrid.transform_points_3x3(best_t, h3)
        c_score = PinGrid.single_score(self.evaluate_fit(t3))
        if c_score < best_score:
            # print("ransac did something, improved score by", best_score - c_score)
            best_score = c_score
            best_h = h3 @ best_h
            best_t = t3

        t4, h4 = self.refine_x_icp(best_t)
        c_score = PinGrid.single_score(self.evaluate_fit(t4))
        if c_score < best_score:
            # print("refine off-by-one did something, improved score by", best_score - c_score)
            best_score = c_score
            best_h = h4 @ best_h
            best_t = t4
            
        return best_t, best_h



    def fit_icp_ransac(self, source):
        """
        Attempt to combine the strengths of each method. ICP is stable but limited to rigid transformations, cv2.findHomography with RANSAC 
        can correct for perspective if the points are already well aligned across the board but seems very sensitive to poor initial alignment.

        Run the best performing methods in decreasing order of stability and increasing order of quality, keeping
        the best performing transform.
        """
        best_score = PinGrid.single_score(self.evaluate_fit(source))
        best_h = np.eye(3)
        best_t = source

        h1 = self.fit_icp(source)
        t1 = PinGrid.transform_points_3x3(source, h1)
        c_score = PinGrid.single_score(self.evaluate_fit(t1))
        if c_score < best_score:
            # print("icp did something, improved score by", best_score - c_score)
            best_score = c_score
            best_h = h1
            best_t = t1

        h2 = self.refine_x_ransac(best_t)
        t2 = PinGrid.transform_points_3x3(best_t, h2)
        c_score = PinGrid.single_score(self.evaluate_fit(t2))
        if c_score < best_score:
            # print("ransac did something, improved score by", best_score - c_score)
            best_score = c_score
            best_h = h2 @ best_h
            best_t = t2

        # hack to try and prevent RANSAC from finding off-by-one solutions. Can probably be removed
        t3, h3 = self.refine_x_icp(best_t)
        c_score = PinGrid.single_score(self.evaluate_fit(t3))
        if c_score < best_score:
            # print("refine off-by-one did something, improved score by", best_score - c_score)
            best_score = c_score
            best_h = h3 @ best_h
            best_t = t3
            
        return best_t, best_h


    def eval_rmse(src, tgt):
        assert src.shape == tgt.shape

        if len(src) == 0:
            return 0.0

        # no idea how or why this works https://stackoverflow.com/questions/21926020/how-to-calculate-rmse-using-ipython-numpy
        return np.linalg.norm(src - tgt) / np.sqrt(len(src))

    def evaluate_fit(self, pts: np.ndarray):
        """
        Returns a tuple of (inliner RMSE, inliner ratio, duplicate ratio)
        """
        neighbors = self.nearest_neighbors(pts)

        distances = np.linalg.norm(pts - neighbors, axis=1)

        inliner_mask = distances < np.mean(self.pitch) * 0.25

        inliners = pts[inliner_mask]
        inliner_neighbors = neighbors[inliner_mask]

        rmse = PinGrid.eval_rmse(inliners, inliner_neighbors)

        if inliners.size == 0:
            rmse = np.mean(self.pitch) * 0.25
 
        # rescale so its resolution agnostic and roughly in units of pin holes
        rmse_grid = rmse / self._size[0]
        rmse_grid *= np.mean(self.pitch)

        # more than one detected pinhole maps to a target pinhole
        # Attempt to catch off-by-one scale problems that might optimize RMSE on noisy inputs
        duplicates = len(neighbors) - len(np.unique(neighbors, axis=0))

        # 1 is perfect, 0 is completely degenerate
        duplicate_score = 1.0 - duplicates/len(neighbors) + (1.0 / len(neighbors))
        
        inliner_ratio = len(inliners) / len(pts)

        return rmse_grid, inliner_ratio, duplicates/len(neighbors)

    def closest_grid_cell(self, pos: np.ndarray):
        ''' 
        Returns a (point, label) pair, or None if there was an internal error 
        '''
        res = self._quadtree.nearest_neighbor_np(tuple(pos.flatten()))
        if res is None:
            return None
        i, point = res
        return point, self.labels[i]

    def label_to_string(label):
        grid_name, x, y = label
        if grid_name == 'base_top':
            letters = ['a', 'b', 'c', 'd', 'e']
            return f"Center grid at {x+1}{letters[y]}"
        if grid_name == 'base_bot':
            letters = ['f', 'g', 'h', 'i', 'j']
            return f"Center grid at {x+1}{letters[len(letters)-y-1]}"
        if grid_name == 'rail_top':
            polarity = ['-', '+']
            return f"Top rail at {x+3+x//5}{polarity[1-y]}"
        if grid_name == 'rail_bot':
            polarity = ['-', '+']
            return f"Bottom rail at {x+3+x//5}{polarity[y]}"
        return "Unknown grid name"

    def grid_points(x, y):
        X, Y = np.meshgrid(x, y, indexing='ij')
        return np.stack([X, Y], axis=-1).reshape(-1, 2)

    def base_pin_holes():
        """
        A rough manually tuned grid. 
        Made by visually aligning the points to a stacked image of the training data with pinhole detections highlighted.

        Returns a (points, labels) pair, where each label is itself a (grid_name, x, y) pair.

        There are 4 grids for the 4 connected grids on the breadboard (The center grid halves are not connected to each other):
        - rail_top 50x2
        - rail_bot 50x2
        - base_top 63x5
        - base_bot 63x5

        This is a terrible function and in need of refactoring
        """
        labels = []
    
        bb_size = PinGrid._grid_size
        center_pos = np.array([1.525, 5.15])
        center_base = PinGrid.grid_points(np.arange(63), np.arange(5)).astype(float)
        center_labels = []
        for i, (x, y) in enumerate(center_base):
            center_labels.append(('base_top', 63 - int(x) - 1, int(y)))
        center_base += center_pos

        rail_pos = np.array([3.525, 1.35])
        rail_x = np.zeros((50, 1))
        for i in range(59):
            rail_x[i - i//6] = i
        
        rail_base = PinGrid.grid_points(rail_x, np.arange(2)) + rail_pos
        rail_labels = []
        for x, _ in enumerate(rail_x):
            for y in range(2):
                rail_labels.append(('rail_top', len(rail_x) - x - 1, y))
        
        pts = np.concatenate((rail_base, center_base), axis=0)
        labels = rail_labels + center_labels

        points_reflected = np.copy(pts)
        points_reflected[:, 1] = bb_size[1] - points_reflected[:, 1]

        labels_reflected = labels.copy()
        for i in range(len(labels_reflected)):
            name, x, y = labels_reflected[i]
            if name == 'base_top':
                labels_reflected[i] = ('base_bot', x, y)
            if name == 'rail_top':
                labels_reflected[i] = ('rail_bot', x, y)

        pts = np.concatenate((pts, points_reflected), axis=0)

        labels = labels + labels_reflected

        return pts / bb_size, labels


class Normalizer:
    """
    A class to handle extracting the breadboard from an image.
    Makes a base assumption that the breadboard exists and is the focus of the photo.
    """

    _corner_rough_model = None

    _corner_flip_model = None

    target_size: np.ndarray = np.array([1024, 340])

    # corner_flip_class_names = ['flipped', 'correct', 'obstructed', 'missed']
    corner_flip_class_names = ['corner', 'invalid']

    # not sure how else to return so many values in an ergonomic way
    last_score = None # (inliner_rmse, inliner_ratio, duplicate_ratio)
    last_pinhole_detections = None
    last_rough_corners = None
    last_homography = None
    last_grayscale = None

    pad: np.ndarray = np.array([0.00, 0.00])

    pingrid: PinGrid

    destination_corners: np.ndarray = np.array([
        [target_size[0] * pad[0], target_size[1] * pad[1]],
        [target_size[0] * (1.0 - pad[0]), target_size[1] * pad[1]],
        [target_size[0] * (1.0 - pad[0]), target_size[1] * (1.0 - pad[1])],
        [target_size[0] * pad[0], target_size[1] * (1.0 - pad[1])],
    ], dtype=np.float32)

    corner_size: int = 32
    model_pad = np.array([0.00, 0.00])
    corner_fill: float = 1.0


    RegistrationMethod = Optional[Literal["affine_cpd", "rigid_cpd", "icp", "icp_ransac", "cpd_ransac"]]
    

    def __init__(self, padding=None, output_resolution=None, raw_pingrid: PinGrid = None, corner_model=None):

        if raw_pingrid is not None:
            assert padding is None or padding == raw_pingrid._pad
            assert output_resolution is None or output_resolution == raw_pingrid._size
            self.pingrid = raw_pingrid
            padding = raw_pingrid._pad
            output_resolution = raw_pingrid._size

        if padding is not None:
            if isinstance(padding, float):
                self.pad = np.array([padding, padding])
            else:
                self.pad = np.array(padding)
        if output_resolution is not None:
            if isinstance(output_resolution, float):
                self.target_size = (self.target_size.astype(float) * output_resolution).astype(int)
            else:
                self.target_size = np.array(output_resolution).astype(int)

        if raw_pingrid is None:
            self.pingrid = PinGrid(self.target_size, self.pad)
        
        self.destination_corners: np.ndarray = np.array([
            [self.target_size[0] * self.pad[0], self.target_size[1] * self.pad[1]],
            [self.target_size[0] * (1.0 - self.pad[0]), self.target_size[1] * self.pad[1]],
            [self.target_size[0] * (1.0 - self.pad[0]), self.target_size[1] * (1.0 - self.pad[1])],
            [self.target_size[0] * self.pad[0], self.target_size[1] * (1.0 - self.pad[1])],
        ], dtype=np.float32)
                
        self._corner_rough_model = corner_model if corner_model is not None else DocAligner()
        self.destination_corners = reorder_corners(self.destination_corners)
        src_dir = Path(__file__).parent.parent
        model_path = src_dir / "weights" / "corner_orientation.keras"
        if tf is not None:
            try:
                self._corner_flip_model = tf.keras.models.load_model(model_path)
            except:
                print(f"Warning: unable to find corner flip model in {model_path}")
                pass
        return
    
    def crop_corners(self, image):
        """Returns an an array containing the 4 square corners of the image, cropped according to corner_size"""
        h, w = (image.shape[0], image.shape[1])

        o = np.array([w, h]) * self.model_pad - self.corner_size * (1.0 - self.corner_fill)
        o = np.array(o, dtype=int)
        oi = np.array([w, h]) * (1.0 - self.model_pad) - self.corner_size * self.corner_fill
        oi = np.array(oi, dtype=int)
        return np.array([
            crop_square(image, [o[0], oi[1]], self.corner_size),
            crop_square(image, [oi[0], oi[1]], self.corner_size),
            crop_square(image, [oi[0], o[1]], self.corner_size),
            crop_square(image, [o[0], o[1]], self.corner_size),
    ])

    def find_rough_corners(self, image):
        """
        Finds the corners in an image. Returned values are in pixels, and the corners are in the following order,
        with respect to the shape in the image and not the orientation of the breadboard

        3------2\n
        0------1

        Returns None if the model failed to find all 4 corners
        """
        source_corners = self._corner_rough_model(image)

        if len(source_corners) != 4:
            return None
        
        return reorder_corners(source_corners)

    def warp_image(self, image, corners):
        """
        Warps the image so the provided corners are mapped to Normalizer.destination_corners. 

        Returns a (image, transform) pair
        """

        transform = cv2.getPerspectiveTransform(corners, self.destination_corners)

        return cv2.warpPerspective(image, transform, dsize=self.target_size), transform

    def find_refinement_transform(self, rough_norm, registration: RegistrationMethod = 'icp_ransac', debug=False):
        """
        Returns the refinement transform in output space
        """
        keypoints = self.find_circles(rough_norm, debug=False)

        source = []
        for keypoint in keypoints:
            source.append(keypoint.pt)
        source = np.array(source)

        if len(source) == 0:
            return np.eye(3), None

        self.last_pinhole_detections = source
        if registration is None:
            transformed, h = source, np.eye(3)
        elif registration == "affine_cpd":
            transformed, h = PinGrid.affine_cpd_rev(source, self.pingrid.points)
        elif registration == "rigid_cpd":
            transformed, h = PinGrid.rigid_cpd_rev(source, self.pingrid.points)
        elif registration == "icp":
            h = self.pingrid.fit_icp(source)
            transformed = PinGrid.transform_points_3x3(source, h)
        elif registration == "icp_ransac":
            transformed, h = self.pingrid.fit_icp_ransac(source)
        elif registration == "cpd_ransac":
            transformed, h = self.pingrid.fit_cpd_ransac(source)
        else:
            print(f"Warning: invalid registration method \"{registration}\", falling back to ICP")
            h = self.pingrid.fit_icp(source)
            transformed = PinGrid.transform_points_3x3(source, h)

        rmse, inliner_ratio, dup_ratio  = self.pingrid.evaluate_fit(transformed)

        if np.array_equal(h, np.eye(3)):
            print("No refinement transform found")

        if debug:
            vis = rough_norm.copy()
            for kp in keypoints:
                vis = cv2.circle(vis, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 0, 255), -1)

            for kp in self.pingrid.points.astype(int):
                vis = cv2.circle(vis, kp, 1, (255, 255, 255), -1)
            
            for kp in transformed.astype(int):
                vis = cv2.circle(vis, kp, 1, (0, 255, 0), -1)

            cv2.imshow("Pinhole Detections", vis)

        return h, (rmse, inliner_ratio, dup_ratio)


    def find_normalization_transform(self, image, registration: RegistrationMethod = 'icp_ransac', debug=False):
        """
        Returns a (transform, score) tuple. If the process failed, returns (None, None)
        """
        rough_corners = self.find_rough_corners(image)
        if rough_corners is None:
            return (None, None)
        
        rough_transform = cv2.getPerspectiveTransform(rough_corners, self.destination_corners)
        rough_norm = cv2.warpPerspective(image, rough_transform, dsize=self.target_size)

        self.last_rough_corners = rough_corners

        refinement_transform, score = self.find_refinement_transform(rough_norm, registration=registration, debug=debug)

        return refinement_transform @ rough_transform, score


    def normalize_image(self, image, registration: RegistrationMethod = 'icp_ransac', debug=False):
        """
        Returns an (image, corners, score) pair where:
        - image is the normalize image of size self.target_size, with the corners of the breadboard at
          self.destination_corners
        - the positive rail is on top
        - corners is the pixel-space position of the corners in the original image
          - so corners[0] and corners[1] are always the bottom of the breadboard, with the negative rail on the bottom
        - score is a number representing the alignment quality between the detected pinholes and the target grid
          - Currently only 1, 0.75, or 0
            - 1 is good, 0.75 is close but imperfect, and 0 is a failed alignment
        """

        source_corners = self.find_rough_corners(image)
        if source_corners is None:
            return (None, None, None)

        self.last_rough_corners = source_corners

        norm_rough, h = self.warp_image(image, source_corners)

        output_refinement, h_score = self.find_refinement_transform(norm_rough, registration=registration, debug=debug)

        last_score_full = h_score

        refined_h = output_refinement @ h

        if h_score is not None:
            inliner_rmse, inliner_ratio, duplicate_ratio = h_score
            score = 1.0
            if inliner_rmse > 0.1 or inliner_ratio < 0.85 or duplicate_ratio > 0.075:
                score = 0.75
            if inliner_rmse > 0.15 or inliner_ratio < 0.8 or duplicate_ratio > 0.15:
                refined_h = h
                score = 0.0
        else:
            score = 0.0

        self.last_homography = refined_h

        norm = cv2.warpPerspective(image, refined_h, dsize=self.target_size)

        label = self.breadboard_orientation_cv(norm)

        if label == 'flipped':
            norm = np.rot90(norm, k=2)
            c = self.target_size / 2 - 0.5
            rot180 = cv2.getRotationMatrix2D(c, 180.0, 1.0)
            rot180 = np.vstack([rot180, [0, 0, 1]])
            refined_h = rot180 @ refined_h
            output_refinement = rot180 @ output_refinement
            self.last_homography=refined_h

        # this is kind of sketchy
        source_corners = PinGrid.transform_points_3x3(self.destination_corners, np.linalg.inv(refined_h))

        if score > 0.0:
            self.last_pinhole_detections = PinGrid.transform_points_3x3(self.last_pinhole_detections, output_refinement)
       #else:
        #    self.last_pinhole_detections = np.empty((0, 2), dtype=np.float32)
        
        return norm, source_corners, score

    _image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')


    def _visualize_corner_classifier(self, image, window_name):
        """
        The corner classifier is not currently functional
        """

        if self._corner_flip_model is None:
            return

        source_corners = self.find_rough_corners(image)

        image_bgr = np.flip(image, axis=-1)

        if source_corners is None:
            return image_bgr

        normalized_image, transform = self.warp_image(image, source_corners)


        norm_bgr = np.flip(normalized_image, axis=-1)

        

        corner_crops = self.crop_corners(norm_bgr)

        corner_flip_predictions = self._corner_flip_model.predict(corner_crops, verbose=0)

        # for i in range(4):
        #     index = np.argmax(corner_flip_predictions[i])
        #     label = self.corner_flip_class_names[index]
        #     corner_crops[i] = cv2.putText(corner_crops[i], label, (4, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        corner_crops_resized = np.zeros((4, self.corner_size * 4, self.corner_size * 4, 3))

        label_indices = corner_flip_predictions[:, 0] > 0.5
        label_colors = [(0, 0, 255), (0, 255, 0)]
        corner_colors = []
        for i in label_indices:
            corner_colors.append(label_colors[int(i)])

        print(corner_flip_predictions)
        print(label_indices)


        # for i in range(4):
        #     index = corner_flip_predictions[:][0] > 0.5
        #     label = self.corner_flip_class_names[index]
        #     corner_crops_resized[i] = cv2.resize(corner_crops[i], (self.corner_size * 4, self.corner_size * 4), cv2.INTER_NEAREST) / 256
        #     corner_crops_resized[i] = cv2.putText(corner_crops_resized[i], label, (4, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # corner_stack = resize_width(np.hstack(corner_crops_resized), self.target_size[0] * 4)

        annotated = draw_corners(image_bgr, source_corners, colors=corner_colors)


        annotated = resize_width(annotated, 1920)

        skew = 1.0 - (skew_metric(source_corners) / image.shape[0]) * 5
        aspect = 1.0 - aspect_metric(source_corners)
        metric = (skew * 0.5 + 0.5) * aspect

        if label_indices.astype(int).sum() < 3:
            metric *= 0.75

        # total_pred = np.sum(corner_flip_predictions[:, :2], axis=0)
        # index = np.argmax(total_pred)
        # label = self.corner_flip_class_names[index]
        box_color = (0, 0, 100)
        if metric > 0.5:
            box_color = (0, 100, 0)
        cv2.rectangle(annotated, (10, 10), (1200, 260), box_color, -1)
        cv2.putText(annotated, f"Skew: {skew:.2f}", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
        cv2.putText(annotated, f"Aspect: {aspect:.2f}", (16, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
        cv2.putText(annotated, f"Validation Metric: {metric:.2f}", (16, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
        # annotated = np.vstack([annotated, corner_stack])

        return annotated

    def find_circles(self, image_padded, debug=  False):
        # tuned for np.array([1024, 340]) with 0 padding

        base_size = np.array([1024, 340])

        h, w = image_padded.shape[:2]

        ch = int(self.pad[1] * h)
        cw = int(self.pad[0] * w)
        image = image_padded[ch:h-ch, cw:w-cw, :].copy()

        h, w = image.shape[:2]

        image_size = np.array([w, h])

        scale = w / base_size[0]
        image_resized = resize_width(image, base_size[0])


        params = cv2.SimpleBlobDetector_Params()
        
        # Thresholds for binarization
        params.minThreshold = 50
        params.maxThreshold = 150
        
        params.filterByArea = True
        params.minArea = 10
        
        params.filterByCircularity = True
        params.minCircularity = 0.4
        # params.maxCircularity = 0.95
        
        params.filterByConvexity = True
        params.minConvexity = 0.65
        
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        # params.filterByColor = True
        # params.blobColor = 0
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        
        image_float = np.copy(image_resized).astype(np.float32)
        blur = cv2.blur(image_float, (16, 16))

        image_float = image_float / np.maximum(blur, 1.0 / 512.0)
        image = np.clip(image_float * 200 - 100, 0, 255).astype(np.uint8)


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use CLAHE for better local contrast at the cost of speed
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # gray = clahe.apply(gray)

        keypoints = detector.detect(gray)

        if debug:
            vis = np.stack((gray,) * 3, axis=-1)
            for kp in keypoints:
                
                vis = cv2.circle(vis, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 255, 255), -1)

            cv2.imshow("Pinhole Detections clahe", vis)
            # cv2.imshow("Annotated image 2", edges)
            # cv2.waitKey(0)
            # cv2.imshow("Annotated image 2", line_img)
            # cv2.waitKey(0)
            # Detect blobs

        for kp in keypoints:
            kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)

            kp.pt = (
                (kp.pt[0] + cw),
                (kp.pt[1] + ch)
            )


            kp.size *= scale

        return keypoints

    def breadboard_orientation_cv(self, image_padded):
        # tuned for np.array([1024, 340]) with 0 padding
        base_size = np.array([1024, 340])

        h, w = image_padded.shape[:2]

        ch = int(self.pad[1] * h)
        cw = int(self.pad[0] * w)
        image = image_padded[ch:h-ch, cw:w-cw, :]

        h, w = image.shape[:2]

        image_size = np.array([w, h])

        scale = w / base_size[0]
        image = resize_width(image, base_size[0])


        norm_bgr = np.flip(image, axis=-1)

        avg = cv2.blur(norm_bgr, (64, 64))

        norm_float = norm_bgr.astype(np.float32)
        norm_float /= np.maximum(avg, 1.0 / 512.0)
        norm_float *= 255

        norm_float = np.clip(norm_float, 0, 255)

        length = np.linalg.norm(norm_float, axis=2, keepdims=True)
        length = np.nan_to_num(length) + 0.001
        normalized = norm_float / length
        zeros = np.zeros_like(norm_float[:, :1, 0])

        red = (np.dot(normalized, np.array([0, 0, 1])))
        red -= np.mean(red)
        red = cv2.blur(red, (3, 3))
        red = np.median(red, axis=1)[:, np.newaxis]
        red /= np.max(red)
        annotated_red = red * 255.0
        annotated_red = np.clip(annotated_red, 0, 255)
        annotated_red = np.stack((zeros, zeros, annotated_red), axis=-1)

        blue = (np.dot(normalized, np.array([1, 0, 0])))
        blue -= np.mean(blue)
        blue = cv2.blur(blue, (3, 3))
        blue = np.median(blue, axis=1)[:, np.newaxis]
        blue /= np.max(blue)
        annotated_blue = blue * 255.0
        annotated_blue = np.clip(annotated_blue, 0, 255)
        annotated_blue = np.stack((annotated_blue, zeros, zeros), axis=-1)

        red = red.flatten()
        blue =  blue.flatten()

        # ignore the center of the image
        crop_size = int(len(red) * 0.33)

        red_top = red[:crop_size]
        red_bot = red[-crop_size:]

        blue_top = blue[:crop_size]
        blue_bot = blue[-crop_size:]

        red_top_peak = np.argmax(red_top)
        blue_top_peak = np.argmax(blue_top)

        red_bot_peak = np.argmax(red_bot)
        blue_bot_peak = np.argmax(blue_bot)

        # weight candidate labels by relative redness/blueness of their detected rails
        top_vote = np.sign(blue_top_peak - red_top_peak) * red_top[red_top_peak] * blue_top[blue_top_peak]
        bot_vote = np.sign(blue_bot_peak - red_bot_peak) * red_bot[red_bot_peak] * blue_bot[blue_bot_peak]

        vote = np.sign(top_vote + bot_vote)
        confidence = np.max((np.abs(top_vote), np.abs(bot_vote)))

        label = "unknown"

        if confidence >= 0.01:
            if vote == -1:
                label = "flipped"
            elif vote == 1:
                label = "correct"
            else:
                label = "disputed"
        
        return label

    def __filter_tails(v, l: int = 15):
        """
        Try to correct for background leaking around the edges of the breadboard by
        ignoring edges of the array until something changes
        """
        b = v[0]
        t = v[-1]

        m = np.mean(v)
        for i in range(0, l):
            if v[i] > 1.2 * b or v[i] < 0.8 * t:
                break
            v[i] = m
        
        for i in reversed(range(len(v) - l, len(v))):
            if v[i] > 1.2 * t or v[i] < 0.8 * t:
                break
            v[i] = m
            
            
        return v

    def _visualize_orientation_detector(self, image, window_name):

        transform, _ = self.find_normalization_transform(image)

        if transform is None:
            return None

        source_corners = PinGrid.transform_points_3x3(self.destination_corners, np.linalg.inv(transform))

        normalized_image, _ = self.warp_image(image, source_corners)

        norm_bgr = np.flip(normalized_image, axis=-1)

        avg = cv2.blur(norm_bgr, (128, 128))

    

        norm_float = norm_bgr.astype(np.float32)
        norm_float /= avg
        norm_float *= 255 / 2

        norm_float = np.clip(norm_float, 0, 255)


        length = np.linalg.norm(norm_float, axis=2, keepdims=True)
        length = np.nan_to_num(length) + 0.01
        normalized = norm_float / length


        # hsv = cv2.cvtColor(norm_float, cv2.COLOR_BGR2HSV)
        # mask = hsv[:, :, 1] <= 0.1
        

        red = normalized[:, :, 2]
        # red[mask] = 0
        red -= np.mean(red)
        red = cv2.blur(red, (3, 3))
        red_pre_median = np.clip(np.copy(red) * 255, 0, 255).astype(np.uint8)
        red = np.median(red, axis=1)[:, np.newaxis]
        red /= np.max(red)
        annotated_red = red * 255.0
        annotated_red = np.clip(annotated_red, 0, 255)

        blue = normalized[:, :, 0]
        # blue[mask] = 0
        blue -= np.mean(blue)
        blue = cv2.blur(blue, (3, 3))
        blue_pre_median = np.clip(np.copy(blue) * 255, 0, 255).astype(np.uint8)
        blue = np.median(blue, axis=1)[:, np.newaxis]
        blue /= np.max(blue)
        annotated_blue = blue * 255.0
        annotated_blue = np.clip(annotated_blue, 0, 255)

        red = red.flatten()
        blue =  blue.flatten()

        crop_size = int(len(red) * 0.33)

        red_top = Normalizer.__filter_tails(red[:crop_size])
        red_bot = Normalizer.__filter_tails(red[-crop_size:])

        blue_top = Normalizer.__filter_tails(blue[:crop_size])
        blue_bot = Normalizer.__filter_tails(blue[-crop_size:])

        red_top_peak = np.argmax(red_top)
        blue_top_peak = np.argmax(blue_top)

        red_bot_peak = np.argmax(red_bot)
        blue_bot_peak = np.argmax(blue_bot)

        top_vote = np.sign(blue_top_peak - red_top_peak) * red_top[red_top_peak] * blue_top[blue_top_peak]
        bot_vote = np.sign(blue_bot_peak - red_bot_peak) * red_bot[red_bot_peak] * blue_bot[blue_bot_peak]

        vote = np.sign(top_vote + bot_vote)
        confidence = np.max((np.abs(top_vote), np.abs(bot_vote)))

        label = "unknown"

        if confidence >= 0.1:
            if vote == -1:
                label = "flipped"
            elif vote == 1:
                label = "correct"
            elif vote == 0:
                label = "disputed"
            else:
                label = "np.sign was not -1, 0 or 1"

        norm_bgr_scaled = norm_float.astype(np.uint8)

        norm_bgr_flipped = norm_bgr

        if label == 'flipped':
            norm_bgr_flipped = np.flipud(norm_bgr_flipped)


        annotated_red = annotated_red.astype(np.uint8)

        annotated_blue = annotated_blue.astype(np.uint8)

        annotated_red = cv2.resize(annotated_red, [norm_bgr.shape[1], norm_bgr.shape[0]], interpolation=cv2.INTER_NEAREST)

        annotated_blue = cv2.resize(annotated_blue, [norm_bgr.shape[1], norm_bgr.shape[0]], interpolation=cv2.INTER_NEAREST)

        normalized[:, :, 1] = 0
        normalized = np.flip(normalized, axis=-1)
        normalized *= 16

        pre_median_vis =  np.stack([blue_pre_median, np.zeros_like(annotated_blue), red_pre_median], axis=-1)
        pre_median_vis = np.clip(pre_median_vis * 8, 0, 255)

        annotated = np.vstack([norm_float.astype(np.uint8), pre_median_vis, np.stack((annotated_blue, np.zeros_like(annotated_blue), annotated_red), axis=-1)])

        image_resized = resize_height(image, annotated.shape[0])

        factor = image_resized.shape[0] / image.shape[0]

        image_resized = np.flip(image_resized, axis=-1) # convert to BGR
        if source_corners is not None:
            image_resized = draw_corners(image_resized, source_corners * factor)

        annotated = np.hstack([image_resized, annotated])

        annotated = cv2.putText(annotated, label, (36, 122), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 16)
        annotated = cv2.putText(annotated, label, (38, 124), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 8)

        annotated = np.vstack([annotated, resize_width(norm_bgr_flipped, annotated.shape[1])])

        return annotated


    def visualize_orientation(self, path):

        window_name = "Annotated image"

        if not os.path.exists(path):
            print(f"Failed to find path at {path}")
            return

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        if os.path.isfile(path):
            annotated = self._visualize_orientation_detector(path, window_name)
            cv2.imshow(window_name, annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(self._image_extensions):
                    image = Image.open(entry.path)
                    image = np.asarray(image)
                    annotated = self._visualize_corner_classifier(image, window_name)
                    
                    if annotated is None:
                        print(f"Failed to find corners at {entry.path}")
                        continue
                    
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
        
        cv2.destroyAllWindows()


        

    









