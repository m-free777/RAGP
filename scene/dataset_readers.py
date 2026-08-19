# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import open3d as o3d

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    K: np.array
    sky_mask: np.array
    dynamic_prob: np.array
    normal: np.array
    depth: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}

def _normalize_dynamic_prob(prob):
    if prob is None:
        return None
    prob = np.asarray(prob).astype(np.float32)
    if prob.ndim == 3:
        prob = np.squeeze(prob)
        if prob.ndim == 3:
            prob = prob[..., 0]
    if prob.size > 0 and prob.max() > 1.0:
        prob = prob / 255.0
    return np.clip(prob, 0.0, 1.0)

def _read_mask_file(path, mask_type):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        arr = np.array(Image.open(path).convert("L"))
    arr = _normalize_dynamic_prob(arr)
    if mask_type == "static":
        arr = 1.0 - arr
    return arr

def _candidate_mask_dirs(image_path, image_name, dynamic_mask_dir=None):
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    dirs = []
    if dynamic_mask_dir:
        # Supports layouts such as:
        #   mask_root/IMG_6385/dynamic_prob.npy
        #   mask_root/test/IMG_6385/dynamic_prob.npy
        #   mask_root/train/IMG_6385/dynamic_prob.npy
        #   mask_root/test/IMG_6385 passed directly as --dynamic_mask_dir
        dirs.extend([
            dynamic_mask_dir,
            os.path.join(dynamic_mask_dir, image_name),
            os.path.join(dynamic_mask_dir, image_stem),
            os.path.join(dynamic_mask_dir, "train", image_name),
            os.path.join(dynamic_mask_dir, "train", image_stem),
            os.path.join(dynamic_mask_dir, "test", image_name),
            os.path.join(dynamic_mask_dir, "test", image_stem),
        ])

    base_without_ext = os.path.splitext(image_path)[0]
    for repl in ["dynamic_masks", "dynamic_mask", "masks"]:
        dirs.append(base_without_ext.replace("images", repl))

    deduped = []
    seen = set()
    for d in dirs:
        d = os.path.normpath(d)
        if d not in seen:
            deduped.append(d)
            seen.add(d)
    return deduped

def try_load_dynamic_prob(image_path, image_name, dynamic_mask_dir=None):
    # Prefer probability, then binary dynamic mask, then inverse static mask.
    file_candidates = [
        ("dynamic_prob.npy", "dynamic"),
        ("dynamic_prob.png", "dynamic"),
        ("dynamic_mask.npy", "dynamic"),
        ("dynamic_mask.png", "dynamic"),
        ("static_mask.npy", "static"),
        ("static_mask.png", "static"),
    ]
    for mask_dir in _candidate_mask_dirs(image_path, image_name, dynamic_mask_dir):
        for filename, mask_type in file_candidates:
            cand = os.path.join(mask_dir, filename)
            if os.path.exists(cand):
                return _read_mask_file(cand, mask_type)
    return None

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, sky_seg=False, load_dynamic_mask=False, dynamic_mask_dir=None, load_normal=False, load_depth=False):
    cam_infos = []
    dynamic_loaded_count = 0
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        if sky_seg:
            sky_path = image_path.replace("images", "mask")[:-4]+".npy"
            sky_mask = np.load(sky_path).astype(np.uint8)
        else:
            sky_mask = None

        dynamic_prob = try_load_dynamic_prob(image_path, image_name, dynamic_mask_dir) if load_dynamic_mask else None
        if dynamic_prob is not None:
            dynamic_loaded_count += 1

        if load_normal:
            normal_path = image_path.replace("images", "normals")[:-4]+".npy"
            normal = np.load(normal_path).astype(np.float32)
            normal = (normal - 0.5) * 2.0
        else:
            normal = None

        if load_depth:
            depth_path = image_path.replace("images", "metricdepth")[:-4]+".npy"
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              K=intr.params, sky_mask=sky_mask, dynamic_prob=dynamic_prob, normal=normal, depth=depth)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    if load_dynamic_mask:
        print(f"[dynamic_mask] loaded {dynamic_loaded_count}/{len(cam_infos)} masks from {dynamic_mask_dir}")
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, sky_seg=False, load_dynamic_mask=False, dynamic_mask_dir=None, load_normal=False, load_depth=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        sky_seg=sky_seg,
        load_dynamic_mask=load_dynamic_mask,
        dynamic_mask_dir=dynamic_mask_dir,
        load_normal=load_normal,
        load_depth=load_depth,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        if 'waymo' in path:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != (llffhold-1)]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == (llffhold-1)]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", is_train=True, load_dynamic_mask=False, dynamic_mask_dir=None):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            sky_mask = np.ones_like(image)[:, :, 0].astype(np.uint8)
            dynamic_prob = try_load_dynamic_prob(image_path, image_name, dynamic_mask_dir) if load_dynamic_mask else None

            if is_train:
                normal_path = image_path.replace("train", "normals")[:-4]+".npy"
                normal = np.load(normal_path).astype(np.float32)
                normal = (normal - 0.5) * 2.0
            else:
                normal = np.zeros_like(image).transpose(2, 0, 1)

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            K=np.array([1, 2, 3, 4]), sky_mask=sky_mask, dynamic_prob=dynamic_prob, normal=normal, depth=None))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", load_dynamic_mask=False, dynamic_mask_dir=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, load_dynamic_mask=load_dynamic_mask, dynamic_mask_dir=dynamic_mask_dir)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, is_train=False, load_dynamic_mask=load_dynamic_mask, dynamic_mask_dir=dynamic_mask_dir)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}
