"""Geometric grasp candidates from a segmented RGB-D point cloud.

Fallback for ``plan_grasp`` when Contact-GraspNet returns no candidates (thin or small
segments). It reuses cap-x's existing geometry helpers -- the oriented bounding box from
``common.get_oriented_bounding_box_from_3d_points`` (noise + statistical outlier removal +
Open3D OBB, the same box ``get_object_pose`` / ``sample_grasp_pose_simple`` grasp from) and the
pinhole back-projection in ``capx.utils.depth_utils`` -- and derives every number from the
observed points and the gripper's dimensions. Nothing here is task specific and no geometry is
invented: if the cloud is too small, too wide for the fingers, or every candidate collides,
:class:`InsufficientGeometryError` is raised with the counts that led there.

Frame conventions (camera frame in, camera frame out, like Contact-GraspNet): a returned pose is
the pose of the tool-center point (TCP, between the fingertips) with ``z`` = approach direction
(the gripper moves along +z into the grasp), ``y`` = finger closing axis, ``x = y x z``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from capx.integrations.franka.common import get_oriented_bounding_box_from_3d_points as _get_obb
from capx.utils.depth_utils import depth_to_pointcloud


class InsufficientGeometryError(ValueError):
    """The segmented point cloud does not support a geometric grasp."""


@dataclass(frozen=True)
class GripperGeometry:
    """Parallel-jaw gripper dimensions in the TCP frame (y = closing axis, z = approach)."""

    max_opening: float = 0.08  # Franka Hand: 80 mm maximum finger separation
    finger_depth: float = 0.045  # usable finger length behind the TCP (pad length)
    finger_thickness: float = 0.02  # finger extent along x and finger width along y
    palm_half_width: float = 0.10  # the hand body spans ~0.2 m along the closing axis
    palm_half_thickness: float = 0.03  # hand body half extent along x
    palm_depth: float = 0.06  # hand body extent behind the finger bases
    min_closing_extent: float = 0.004  # thinner than this is a surface (single visible face), not a solid to pinch
    approach_clearance: float = 0.10  # straight-line approach corridor checked behind the palm


FRANKA_HAND = GripperGeometry()


def object_points_from_mask(depth: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Camera-frame points of the masked pixels that carry a finite, positive depth."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    pts = depth_to_pointcloud(depth, np.asarray(intrinsics, dtype=np.float64), filter_invalid=False)
    sel = (mask > 0).reshape(-1) & np.isfinite(depth).reshape(-1) & (depth.reshape(-1) > 0)
    return pts[sel]


def scene_points_from_depth(depth: np.ndarray, intrinsics: np.ndarray, stride: int = 4) -> np.ndarray:
    """Sub-sampled camera-frame points of the whole depth image (table, other objects, robot)."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth_to_pointcloud(depth, np.asarray(intrinsics, dtype=np.float64), subsample_factor=max(1, stride))


def foreground_depth_band(points: np.ndarray, *, min_points: int = 24, depth_gap: float = 0.02) -> np.ndarray:
    """Points of the nearest contiguous depth band (camera z) that holds at least ``min_points``.

    Sorted by depth, the first jump larger than ``depth_gap`` after ``min_points`` points closes the
    band; everything behind it is background seen through or around the object.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= min_points:
        return pts
    order = np.argsort(pts[:, 2])
    z = pts[order, 2]
    gaps = np.flatnonzero(np.diff(z) > depth_gap)
    gaps = gaps[gaps + 1 >= min_points]
    if len(gaps) == 0:
        return pts
    return pts[order[: gaps[0] + 1]]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _occluded(samples: np.ndarray, depth: np.ndarray | None, intrinsics: np.ndarray | None, margin: float = 0.01) -> np.ndarray:
    """True for camera-frame sample points that lie behind the observed surface along their own
    camera ray (single-view shadow: unobserved space that may be solid)."""
    if depth is None or intrinsics is None or len(samples) == 0:
        return np.zeros(len(samples), dtype=bool)
    z = samples[:, 2]
    ok = z > 1e-6
    u = np.full(len(samples), -1.0)
    v = np.full(len(samples), -1.0)
    u[ok] = samples[ok, 0] * intrinsics[0, 0] / z[ok] + intrinsics[0, 2]
    v[ok] = samples[ok, 1] * intrinsics[1, 1] / z[ok] + intrinsics[1, 2]
    h, w = depth.shape[:2]
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    inside = ok & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.zeros(len(samples), dtype=bool)
    d = depth[vi[inside], ui[inside]]
    out[inside] = np.isfinite(d) & (d > 0) & (d + margin < z[inside])
    return out


def _box_samples(rot: np.ndarray, tcp: np.ndarray, xr, yr, zr, n=(3, 5, 4)) -> np.ndarray:
    """Camera-frame grid samples of a box given in the grasp frame (ranges along x, y, z)."""
    gx, gy, gz = np.meshgrid(np.linspace(*xr, n[0]), np.linspace(*yr, n[1]), np.linspace(*zr, n[2]), indexing="ij")
    local = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    return local @ rot.T + tcp


def geometric_grasp_candidates(
    object_points: np.ndarray,
    scene_points: np.ndarray | None = None,
    gripper: GripperGeometry = FRANKA_HAND,
    *,
    depth: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    max_candidates: int = 8,
    min_points: int = 24,
    depth_gap: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pinch-grasp candidates for one segmented object.

    Algorithm (all quantities measured on ``object_points``, camera frame):

    0. Only the nearest contiguous depth band of the mask is used: depth seen through holes in
       the object or past its silhouette belongs to the background.
    1. Oriented bounding box of the object (cap-x helper) for the global shape and axes.
    2. Grasp anchors: the object points nearest to the box centre and to the centres of its six
       faces. Snapping to real points keeps the TCP on observed geometry (a hollow frame has no
       material at its centre).
    3. Closing axes per anchor: the box's thinnest axis and, when ``depth``/``intrinsics`` are
       given, the principal axes of the points within a finger length of the anchor (a bar's
       cross-section inside a wide frame); only axes along which that extent fits the gripper
       opening are kept.
    4. Approach directions: the camera line of sight and the box / local axes, projected
       perpendicular to the closing axis. Without a depth image, directions pointing back
       towards the camera are dropped (the line of sight is the only free space we know
       about); with one, the shadow test in step 5 decides.
    5. For each (anchor, closing axis, approach, closing sign) the TCP is placed so the fingers
       cover the object's local extent along the approach (never deeper than the finger length,
       so the palm stays clear), and the candidate is kept only if object points lie between the
       fingers and the finger volumes, the hand body and the straight-line approach corridor are
       free: no scene point inside them and, when ``depth``/``intrinsics`` are given, none of
       them in the camera's shadow (space behind the observed surface is unknown and treated as
       solid -- this is what rules out closing on a single visible face or sweeping the hand
       through the hidden side of a box).
    6. Candidates are scored by alignment with the line of sight (1 = straight along it).

    Returns:
        poses: (K, 4, 4) TCP poses in the camera frame, best first.
        scores: (K,) alignment scores in (0, 1].
        report: counts and the box / per-candidate numbers behind the result.

    Raises:
        InsufficientGeometryError: too few points, nothing fits the gripper, or no
            collision-free candidate; the message carries the counts.
    """
    pts = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    n_mask_points = len(pts)
    report: dict[str, Any] = {"mask_points": n_mask_points, "gripper": asdict(gripper)}
    if n_mask_points < min_points:
        raise InsufficientGeometryError(
            f"only {n_mask_points} valid 3D points in the segment (need at least {min_points}) -- too little "
            "geometry for an oriented bounding box; check the segmentation mask and upstream depth filters"
        )
    pts = foreground_depth_band(pts, min_points=min_points, depth_gap=depth_gap)
    n_points = len(pts)
    report["object_points"] = n_points
    report["background_points_dropped"] = n_mask_points - n_points
    scene = pts if scene_points is None else np.asarray(scene_points, dtype=np.float64).reshape(-1, 3)
    scene = scene[np.isfinite(scene).all(axis=1)]
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float64)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        intrinsics = np.asarray(intrinsics, dtype=np.float64)

    try:
        obb = _get_obb(pts)
    except RuntimeError as exc:  # qhull on degenerate clouds
        raise InsufficientGeometryError(
            f"could not fit an oriented bounding box to the {n_points} segment points: {exc}"
        ) from exc
    center = np.asarray(obb["center"], dtype=np.float64).reshape(3)
    extent = np.asarray(obb["extent"], dtype=np.float64).reshape(3)
    axes = np.asarray(obb["R"], dtype=np.float64).reshape(3, 3).copy()  # columns = box axes
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
    order = np.argsort(extent)
    thin = int(order[0])
    report["obb"] = {"center": center.round(4).tolist(), "extent": extent.round(4).tolist(),
                     "axes": axes.round(4).tolist(), "closing_axis": thin,
                     "closing_extent_m": round(float(extent[thin]), 4)}
    view = _unit(center)  # camera at the origin looks along the ray through the object centre
    fits = gripper.max_opening - 0.004

    # --- anchors: real object points nearest to the box centre and its six face centres ---
    targets = [center]
    for a in range(3):
        targets.append(center + axes[:, a] * extent[a] / 2.0)
        targets.append(center - axes[:, a] * extent[a] / 2.0)
    anchors: list[np.ndarray] = []
    for t in targets:
        p = pts[np.argmin(np.linalg.norm(pts - t, axis=1))]
        if all(np.linalg.norm(p - q) > 0.01 for q in anchors):
            anchors.append(p)

    half_open = gripper.max_opening / 2.0
    half_ft = gripper.finger_thickness / 2.0
    rejections = {"behind_camera": 0, "nothing_between_fingers": 0, "finger_collision": 0,
                  "palm_or_corridor_collision": 0, "occluded_space": 0}
    candidates: list[dict[str, Any]] = []
    n_closing_axes = 0
    for anchor in anchors:
        # closing axes: global thinnest axis + local principal axes whose extent fits the opening
        closing_options: list[tuple[np.ndarray, float, str]] = []
        if gripper.min_closing_extent <= extent[thin] <= fits:
            closing_options.append((_unit(axes[:, thin]), float(extent[thin]), "box"))
        local = pts[np.linalg.norm(pts - anchor, axis=1) <= gripper.finger_depth]
        local_axes = None
        # Local cross-sections are only trusted when the camera-shadow test can run: a single
        # visible face also looks "thin" along its normal, and only the shadow test can tell
        # that the far finger would end up inside the object.
        if len(local) >= 10 and depth is not None and intrinsics is not None:
            cen = local.mean(axis=0)
            _, _, vt = np.linalg.svd(local - cen, full_matrices=False)
            local_axes = vt  # rows = principal directions
            for d in local_axes:
                ext = float(np.ptp((local - cen) @ d))
                if gripper.min_closing_extent <= ext <= fits and all(abs(float(np.dot(d, c))) < np.cos(np.deg2rad(15.0)) for c, _, _ in closing_options):
                    closing_options.append((_unit(d), ext, "local"))
        n_closing_axes += len(closing_options)
        for closing, closing_extent, closing_kind in closing_options:
            # approach directions perpendicular to the closing axis, from the camera's side
            raw_dirs = [view] + [sgn * axes[:, a] for a in range(3) for sgn in (1.0, -1.0)]
            if local_axes is not None:
                raw_dirs += [sgn * d for d in local_axes for sgn in (1.0, -1.0)]
            kept_dirs: list[np.ndarray] = []
            for d in raw_dirs:
                d = d - np.dot(d, closing) * closing
                if np.linalg.norm(d) < 0.3:
                    continue
                d = _unit(d)
                # Without a depth image the only free space we can vouch for is the line of
                # sight, so approaches pointing back at the camera are dropped; with one, the
                # shadow test below decides on evidence instead.
                if depth is None and float(np.dot(d, view)) < -0.15:
                    rejections["behind_camera"] += 1
                    continue
                if all(float(np.dot(d, k)) < np.cos(np.deg2rad(10.0)) for k in kept_dirs):
                    kept_dirs.append(d)
            for approach in kept_dirs:
                rel = pts - anchor
                lateral = rel - np.outer(rel @ approach, approach)
                near_line = np.linalg.norm(lateral, axis=1) <= half_open + half_ft
                s = np.sort(rel[near_line] @ approach)
                if len(s) == 0:
                    continue
                # thickness of the nearest contiguous run of material along the approach (a hollow
                # frame's far bar must not count), then centre that run on the finger pads while
                # keeping the palm at least 1 cm clear of the near surface
                gaps = np.flatnonzero(np.diff(s) > 0.01)
                s_near_end = float(s[gaps[0]] if len(gaps) else s[-1])
                thick_near = min(s_near_end - float(s[0]), gripper.finger_depth)
                s_tcp = float(s[0]) + min(thick_near / 2.0 + gripper.finger_depth / 2.0, gripper.finger_depth - 0.01)
                tcp = anchor + approach * s_tcp
                for sign in (1.0, -1.0):
                    y = sign * closing
                    z = approach
                    x = _unit(np.cross(y, z))
                    rot = np.column_stack([x, y, z])
                    qo = (pts - tcp) @ rot
                    between = (np.abs(qo[:, 1]) <= half_open) & (np.abs(qo[:, 0]) <= half_ft + 0.005) \
                        & (qo[:, 2] >= -gripper.finger_depth) & (qo[:, 2] <= 0.005)
                    n_between = int(between.sum())
                    if n_between < 3:
                        rejections["nothing_between_fingers"] += 1
                        continue
                    qs = (scene - tcp) @ rot
                    in_fingers = (np.abs(qs[:, 1]) >= half_open + 0.002) \
                        & (np.abs(qs[:, 1]) <= half_open + gripper.finger_thickness) \
                        & (np.abs(qs[:, 0]) <= half_ft) & (qs[:, 2] >= -gripper.finger_depth + 0.002) & (qs[:, 2] <= 0.0)
                    if in_fingers.any():
                        rejections["finger_collision"] += 1
                        continue
                    z_lo = -(gripper.finger_depth + gripper.palm_depth + gripper.approach_clearance)
                    in_palm = (np.abs(qs[:, 1]) <= gripper.palm_half_width) & (np.abs(qs[:, 0]) <= gripper.palm_half_thickness) \
                        & (qs[:, 2] >= z_lo) & (qs[:, 2] <= -gripper.finger_depth - 0.002)
                    if in_palm.any():
                        rejections["palm_or_corridor_collision"] += 1
                        continue
                    # single-view shadow test on the same volumes
                    samples = np.vstack([
                        _box_samples(rot, tcp, (-half_ft, half_ft), (half_open + 0.002, half_open + gripper.finger_thickness),
                                     (-gripper.finger_depth + 0.002, 0.0)),
                        _box_samples(rot, tcp, (-half_ft, half_ft), (-half_open - gripper.finger_thickness, -half_open - 0.002),
                                     (-gripper.finger_depth + 0.002, 0.0)),
                        _box_samples(rot, tcp, (-gripper.palm_half_thickness, gripper.palm_half_thickness),
                                     (-gripper.palm_half_width, gripper.palm_half_width),
                                     (z_lo, -gripper.finger_depth - 0.002), n=(3, 7, 6)),
                    ])
                    n_occ = int(_occluded(samples, depth, intrinsics).sum())
                    if n_occ:
                        rejections["occluded_space"] += 1
                        continue
                    pose = np.eye(4)
                    pose[:3, :3] = rot
                    pose[:3, 3] = tcp
                    candidates.append({
                        "pose": pose,
                        "score": float(0.5 + 0.5 * np.dot(approach, view)),
                        "tcp": tcp.round(4).tolist(),
                        "approach": approach.round(3).tolist(),
                        "closing": y.round(3).tolist(),
                        "closing_extent_m": round(closing_extent, 4),
                        "closing_axis_from": closing_kind,
                        "points_between_fingers": n_between,
                        "near_thickness_m": round(thick_near, 4),
                    })
    report["anchors"] = len(anchors)
    report["closing_axes"] = n_closing_axes
    report["rejections"] = rejections
    if n_closing_axes == 0:
        raise InsufficientGeometryError(
            f"nothing to pinch: the object's thinnest box extent is {extent[thin]:.4f} m and no local cross-section "
            f"fits the {gripper.max_opening:.3f} m gripper opening (a graspable extent is "
            f"{gripper.min_closing_extent:.3f}-{fits:.3f} m; box extents {extent.round(3).tolist()} m over "
            f"{n_points} points)"
        )
    if not candidates:
        raise InsufficientGeometryError(
            f"no collision-free pinch grasp on the segment ({n_points} points, box extents "
            f"{extent.round(3).tolist()} m): {len(anchors)} anchors x {n_closing_axes} closing axes tried; "
            f"rejected {rejections}"
        )
    # drop near-duplicates (same TCP and axes), best first
    candidates.sort(key=lambda c: (-c["score"], -c["points_between_fingers"]))
    unique: list[dict[str, Any]] = []
    for c in candidates:
        if not any(np.linalg.norm(c["pose"][:3, 3] - u["pose"][:3, 3]) < 0.005
                   and np.allclose(c["pose"][:3, :3], u["pose"][:3, :3], atol=1e-3) for u in unique):
            unique.append(c)
    candidates = unique[:max_candidates]
    poses = np.stack([c["pose"] for c in candidates])
    scores = np.array([c["score"] for c in candidates], dtype=np.float64)
    report["candidates"] = [{k: v for k, v in c.items() if k != "pose"} for c in candidates]
    return poses, scores, report


__all__ = [
    "FRANKA_HAND",
    "GripperGeometry",
    "InsufficientGeometryError",
    "foreground_depth_band",
    "geometric_grasp_candidates",
    "object_points_from_mask",
    "scene_points_from_depth",
]
