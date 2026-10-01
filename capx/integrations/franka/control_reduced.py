import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
from capx.integrations.base_api import ApiBase
from capx.integrations.franka.common import (
    apply_tcp_offset,
    close_gripper as _close_gripper,
    close_gripper_arm1 as _close_gripper_arm1,
    extract_arm_joints,
    get_oriented_bounding_box_from_3d_points as _get_obb,
    open_gripper as _open_gripper,
    open_gripper_arm1 as _open_gripper_arm1,
    quat_wxyz_to_xyzw,
    solve_ik_with_convergence,
    transform_pose_arm0_to_arm1,
)
from capx.integrations.vision.graspnet import init_contact_graspnet
from capx.integrations.vision.molmo import init_molmo, molmo_service_available
from capx.integrations.motion.pyroki import init_pyroki, init_pyroki_ik_check, init_pyroki_trajopt
from capx.integrations.franka.geometric_grasp import (
    InsufficientGeometryError,
    geometric_grasp_candidates,
    object_points_from_mask,
    scene_points_from_depth,
)

from capx.integrations.vision.owlvit import init_owlvit
from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore
from capx.integrations.vision.sam2 import init_sam2
from capx.integrations.vision.sam3 import init_sam3, init_sam3_point_prompt
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud, depth_to_pointcloud, depth_to_rgb
from capx.utils.visualization_utils import (
    draw_molmo_point,
    draw_oriented_bounding_box,
    overlay_segmentation_masks,
    render_cylinder_axis,
)


class NoGraspCandidatesError(RuntimeError):
    """``plan_grasp`` found no grasp for the segmented object.

    Raised when Contact-GraspNet returns no candidates *and* the geometric fallback cannot
    produce a collision-free, reachable pinch grasp from the segment's 3D points. The message
    starts with ``no_grasp_candidates:`` and states the mask size, depth range, planner depth
    window and why the fallback gave up; ``details`` holds the same as a dict.
    """

    code = "no_grasp_candidates"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"{self.code}: {message}")
        self.details = details or {}


# ------------------------------- Control API ------------------------------
class FrankaControlApiReduced(ApiBase):
    """
    Robot control helpers for Franka.
    """

    def __init__(
        self,
        env: BaseEnv,
        tcp_offset: list[float] | None = [0.0, 0.0, -0.107],
        is_spill_wipe: bool = False,
        is_peg_assembly: bool = False,
        is_handover: bool = False,
        bimanual: bool = False,
        real: bool = False,
        use_sam3: bool = True,
    ) -> None:
        super().__init__(env)
        self._TCP_OFFSET = np.array(tcp_offset, dtype=np.float64)
        self.use_sam3 = use_sam3
        print("init franka control api")
        self.grasp_net_plan_fn = (
            init_contact_graspnet()
        )  # TODO: refactor this and use registered api instead
        print("init grasp net plan fn")
        if self.use_sam3:
            self.sam3_seg_fn = init_sam3()
            self.sam3_point_prompt_fn = init_sam3_point_prompt()
            print("init sam3 seg fn")
        else:
            self.owl_vit_det_fn = init_owlvit(device="cuda")
            print("init owlvit det fn")
            self.sam2_seg_fn = init_sam2()
            print("init sam2 seg fn")
        self.molmo_point_fn = init_molmo()
        # No task YAML launches a Molmo server; when nothing listens on its port the tool
        # is withheld from the generated code's API (see functions()) rather than handed
        # to the model as an option that can only fail.
        self.molmo_available = molmo_service_available()
        print("init molmo point fn" + ("" if self.molmo_available
                                       else " (service unreachable: point_prompt_molmo disabled)"))

        self.ik_solve_fn = init_pyroki()
        self.ik_check_fn = init_pyroki_ik_check()  # IK + achieved pose, for reachability checks
        self.trajopt_plan_fn = init_pyroki_trajopt()
        self.cfg = None
        self._last_observation: dict[str, Any] | None = None
        # Structured account of the last plan_grasp call (grasp source, candidate counts,
        # fallback geometry, IK checks); recorders may read and clear it.
        self.last_call_report: dict[str, Any] | None = None
        self.is_spill_wipe = is_spill_wipe
        self.is_peg_assembly = is_peg_assembly
        self.is_handover = is_handover
        self.bimanual = bimanual
        self.real = real
    def functions(self) -> dict[str, Any]:
        fns = {"get_observation": self.get_observation}
        if self.molmo_available:
            fns["point_prompt_molmo"] = self.point_prompt_molmo
        if self.use_sam3:
            fns["segment_sam3_text_prompt"] = self.segment_sam3_text_prompt
            fns["segment_sam3_point_prompt"] = self.segment_sam3_point_prompt
        else:
            fns["detect_object_owlvit"] = self.detect_object_owlvit
            fns["segment_sam2"] = self.segment_sam2
        # if not self.is_spill_wipe:
        #     if not self.is_peg_assembly:
        fns["plan_grasp"] = self.plan_grasp
        fns["get_oriented_bounding_box_from_3d_points"] = (
            self.get_oriented_bounding_box_from_3d_points
        )
        # if not self.bimanual:
        #     fns["open_gripper"] = self.open_gripper
        #     fns["close_gripper"] = self.close_gripper
        if self.bimanual:
            fns["solve_ik_arm0"] = self.solve_ik_arm0
            fns["solve_ik_arm1"] = self.solve_ik_arm1
            fns["move_to_joints_both"] = self.move_to_joints_both
            fns["move_to_joints_arm0"] = self.move_to_joints_arm0
            fns["move_to_joints_arm1"] = self.move_to_joints_arm1
            fns["open_gripper_arm0"] = self.open_gripper_arm0
            fns["close_gripper_arm0"] = self.close_gripper_arm0
            fns["open_gripper_arm1"] = self.open_gripper_arm1
            fns["close_gripper_arm1"] = self.close_gripper_arm1
        else:
            fns["solve_ik"] = self.solve_ik
            # fns["traj_plan"] = self.traj_plan
            # fns["move_along_trajectory"] = self.move_along_trajectory
            fns["move_to_joints"] = self.move_to_joints
            fns["open_gripper"] = self.open_gripper
            fns["close_gripper"] = self.close_gripper

        return fns

    def get_observation(self) -> dict[str, Any]:
        """Get the observation of the environment.
        Returns:
            observation:
                A dictionary containing the observation of the environment.
                The dictionary contains the following keys:
                - ["robot0_robotview"]["images"]["rgb"]: Current color camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["robot0_robotview"]["images"]["depth"]: Current depth camera image as a numpy array of shape (H, W), dtype float32.
                - ["robot0_robotview"]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["robot0_robotview"]["pose_mat"]: Camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                  It is the camera-to-robot-base transform: ``pose_mat @ [x, y, z, 1]`` maps a
                  camera-frame point into the robot base frame the pose functions use.

                Depth contract (same for every camera key in the observation): ``depth`` is the
                metric distance in meters along the camera's optical axis, one value per pixel.
                Every pixel is valid -- there are no 0 / NaN / inf "holes" to filter out and the far
                plane lies well beyond the workspace. The cameras are fixed scene cameras that can
                be several meters from the table, so do NOT discard pixels with a hard-coded depth
                range (e.g. ``depth < 2.0``): such a cut-off can remove the whole object and leave
                an empty point cloud. Select geometry with the segmentation mask, and express any
                spatial bounds in the robot base frame after applying ``pose_mat``.
        """
        self._log_step("get_observation", "Capturing camera observation …")
        obs = self._env.get_observation()
        obs["robot0_robotview"]["images"]["depth"] = obs["robot0_robotview"]["images"]["depth"].squeeze(-1)
        self._last_observation = obs
        self._log_step_update(images=obs["robot0_robotview"]["images"]["rgb"])
        return obs

    # - ["robot_joint_pos"]: Current joint positions of the robot (including gripper as the last element) as a numpy array of shape (8,), dtype float64.
    # - ["robot_cartesian_pose_wxyz_xyz"]: Current Cartesian pose (quaternion wxyz, then position xyz) of the robot (including gripper as the last element) as a numpy array of shape (8,), dtype float64.

    # --------------------------------------------------------------------- #
    # Vision models: OWL-ViT detection + SAM2 segmentation (use_sam3=False)
    # --------------------------------------------------------------------- #

    def detect_object_owlvit(
        self,
        rgb: np.ndarray,
        text: str,
    ) -> list[dict[str, Any]]:
        """Run OWL-ViT open-vocabulary detection on a single RGB image.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            text:
                Natural language text query for OWL-ViT.

        Returns:
            detections:
                A list of dictionaries, one per detected box. Each dict contains:

                  - "box":   [x1, y1, x2, y2] in pixel coordinates (float)
                  - "label": str, the text label that matched best
                  - "score": float, confidence score in [0, 1]

        Example:
            >>> rgb = obs["robot0_robotview"]["images"]["rgb"]
            >>> dets = detect_object_owlvit(rgb, text="red mug")
            >>> if dets:
            ...     best = max(dets, key=lambda d: d["score"])
            ...     print(best["box"], best["label"], best["score"])
        """
        self._log_step("OWL-ViT Detection", f"Running OWL-ViT for '{text}' …", images=rgb)
        results = self.owl_vit_det_fn(rgb, texts=[[text]])
        if results:
            best_score = max(d["score"] for d in results)
            self._log_step_update(text=f"{len(results)} detection(s), best score: {best_score:.3f}")
        else:
            self._log_step_update(text="No detections.")
        return results

    def segment_sam2(
        self,
        rgb: np.ndarray,
        box: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Run SAM2 segmentation on an RGB image, optionally conditioned on a box.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            box:
                Optional bounding box [x1, y1, x2, y2] in pixel coordinates, float.
                If provided, SAM2 will segment primarily within this region.
                If None, SAM2 runs in global mode over the whole image.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool or uint8,
                              where True/1 means the pixel belongs to the instance.
                  - "score": float confidence score (if provided by SAM2).

        Example:
            >>> rgb = obs["robot0_robotview"]["images"]["rgb"]
            >>> dets = detect_object_owlvit(rgb, text="red mug")
            >>> best = max(dets, key=lambda d: d["score"])
            >>> masks = segment_sam2(rgb, box=best["box"])
        """
        box_str = f" with box {box}" if box is not None else ""
        self._log_step("SAM2 Segmentation", f"Running SAM2{box_str} …", images=rgb)
        results = self.sam2_seg_fn(rgb, box=box)
        masks = [r["mask"] for r in results if r.get("score", 0) > 0.05]
        if masks:
            vis = overlay_segmentation_masks(rgb, masks)
            self._log_step_update(text=f"Returned {len(results)} mask(s)", images=vis)
        else:
            self._log_step_update(text="No masks returned.")
        return results

    # --------------------------------------------------------------------- #
    # Vision models: SAM3 segmentation (use_sam3=True)
    # --------------------------------------------------------------------- #

    def segment_sam3_point_prompt(
        self,
        rgb: np.ndarray,
        point_coords: tuple[float, float],
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image, optionally conditioned on an image coordinate point prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            point_coords:
                (x, y) pixel coordinates of the point prompt.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool,
                              where True means the pixel belongs to the instance.
                  - "score": float confidence score.

        Example:
            >>> rgb = obs["robot0_robotview"]["images"]["rgb"]
            >>> masks = segment_sam3_point_prompt(rgb, (100, 100))
        """
        self._log_step("SAM3 Point Segmentation", f"Running SAM3 point-prompt at ({point_coords[0]}, {point_coords[1]}) …", images=rgb)
        results = self.sam3_point_prompt_fn(Image.fromarray(rgb), point_coords)
        masks = [r["mask"] for r in results if r.get("score", 0) > 0.05]
        if masks:
            vis = overlay_segmentation_masks(rgb, masks)
            if hasattr(self._env, "set_sam3_mask"):
                self._env.set_sam3_mask(vis)
            self._log_step_update(text=f"Returned {len(results)} mask(s)", images=vis)
        else:
            self._log_step_update(text="No mask beyond threshold.")
        return results

    def segment_sam3_text_prompt(
        self,
        rgb: np.ndarray,
        text_prompt: str,
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image conditioned on a text prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            text_prompt:
                Text prompt for SAM3 segmentation.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool,
                              where True means the pixel belongs to the instance.
                  - "box": list [x1, y1, x2, y2] in pixel coordinates.
                  - "score": float confidence score.

        Example:
            >>> rgb = obs["robot0_robotview"]["images"]["rgb"]
            >>> masks = segment_sam3(rgb, text_prompt="red mug")
        """
        self._log_step("SAM3 Text Segmentation", f"Running SAM3 text-prompt: '{text_prompt}' …", images=rgb)
        results = self.sam3_seg_fn(rgb, text_prompt=text_prompt)
        masks = [r["mask"] for r in results if r.get("score", 0) > 0.05]
        if masks:
            best_score = max(r.get("score", 0) for r in results)
            vis = overlay_segmentation_masks(rgb, masks)
            if hasattr(self._env, "set_sam3_mask"):
                self._env.set_sam3_mask(vis)
            self._log_step_update(text=f"Returned {len(results)} mask(s), best score: {best_score:.3f}", images=vis)
        else:
            self._log_step_update(text="No masks returned.")
        return results

    # --------------------------------------------------------------------- #
    # Molmo point prompt
    # --------------------------------------------------------------------- #
    def point_prompt_molmo(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> dict[str, tuple[int | None, int | None]]:
        """Use Molmo to point to a coordinate in the image based on a text prompt.

        Args:
            image: np.ndarray: The RGB image to process. Shape: (H, W, 3), dtype uint8.
            text_prompt: str: The text prompt to point to.

        Returns:
            dict[str, tuple[int | None, int | None]]: Pixel coordinates for each
            object query; (None, None) if parsing failed.
        """
        if not self.molmo_available:
            raise RuntimeError("point_prompt_molmo is unavailable: no Molmo service is reachable "
                               "(set CAPX_MOLMO_URL or start one on 127.0.0.1:8122); use "
                               "segment_sam3_text_prompt instead")
        self._log_step("Molmo Point Prompt", f"Querying Molmo for '{text_prompt}' …", images=image)
        result = self.molmo_point_fn(Image.fromarray(image), objects=[text_prompt])
        if None not in result.values():
            molmo_image = draw_molmo_point(image, result)
            if hasattr(self._env, "set_molmo_image"):
                self._env.set_molmo_image(molmo_image)
            self._log_step_update(text=f"Result: {result}", images=molmo_image)
        else:
            self._log_step_update(text="No point found.")
        return result

    def get_oriented_bounding_box_from_3d_points(self, points: np.ndarray) -> dict[str, Any]:
        """Get the oriented bounding box from 3D points.

        Args:
            points: np.ndarray: The 3D points to get the oriented bounding box from.
                Shape: (N, 3), dtype float64.

        Returns:
            dict[str, Any]: The oriented bounding box. The dictionary contains the following keys:
                - "center": np.ndarray: The center of the oriented bounding box in point cloud frame.
                - "extent": np.ndarray: The extent of the oriented bounding box.
                - "R": np.ndarray: The rotation matrix of the oriented bounding box in point cloud frame.

        Example:
            >>> points = np.random.randn((100, 3))
            >>> obb = get_oriented_bounding_box_from_3d_points(points)

        Raises:
            ValueError: if ``points`` is not (N, 3) or holds fewer than 4 points.
            RuntimeError: if no box can be fitted (too few distinct points after the
                helper's statistical outlier removal).
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"get_oriented_bounding_box_from_3d_points: expected points of shape (N, 3), got {points.shape}"
            )
        if len(points) < 4:
            raise ValueError(
                f"get_oriented_bounding_box_from_3d_points: received {len(points)} point(s); an oriented "
                "bounding box needs at least 4 non-coplanar points. If the points came from a masked depth "
                "image, check the segmentation mask and any depth / workspace filters applied upstream "
                "(see the depth contract in get_observation)."
            )
        try:
            return _get_obb(points)
        except RuntimeError as exc:  # qhull errors from Open3D on degenerate or too few points
            raise RuntimeError(
                f"get_oriented_bounding_box_from_3d_points: could not fit an oriented bounding box to "
                f"{len(points)} points (too few distinct points after the helper's statistical outlier "
                f"removal); check the segmentation mask and upstream depth / workspace filters: {exc}"
            ) from exc

    # --------------------------------------------------------------------- #
    # Grasp planner (Contact-GraspNet)
    # --------------------------------------------------------------------- #
    def plan_grasp(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmentation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plan grasp candidates for a single segmented instance.

        Contact-GraspNet is run on the depth image, cropped to a depth window that always
        contains the segmented object. If it returns no candidates (typical for very thin or
        small segments), a geometric fallback derives pinch grasps from the segment's own 3D
        points: the fingers close across the object's thinnest extent (which must fit the
        gripper opening), the tool-center point sits on observed geometry, the approach comes
        from the camera's side perpendicular to the closing axis, candidates that would
        collide with the scene are dropped, and the rest are checked for IK reachability. If
        that also yields nothing, ``NoGraspCandidatesError`` is raised with the counts.

        No camera/world transform is applied: transform the poses into the robot base frame
        yourself (``camera_extrinsics @ grasp_pose``). Each pose is the pose of the
        gripper's tool-center point (TCP, the grasp point between the fingertips) with ``z``
        = approach direction and ``y`` = finger closing axis. The poses are TCP poses and
        are passed to ``solve_ik`` as-is: ``solve_ik`` applies the TCP-to-hand offset
        internally, so do NOT add or subtract any TCP / hand offset yourself.

        Args:
            depth:
                Depth image in meters.
                Shape: (H, W), dtype float32/float64.
            intrinsics:
                Camera intrinsic matrix.
                Shape: (3, 3), dtype float64.
            segmentation:
                Instance segmentation map where each integer > 0 corresponds to a
                unique object instance ID.
                Shape: (H, W) or (H, W, 1), dtype int32/int64.

        Returns:
            grasp_poses:
                np.ndarray of shape (K, 4, 4), dtype float64.
                Homogeneous transforms for each candidate grasp IN THE CAMERA FRAME.
            grasp_scores:
                np.ndarray of shape (K,), dtype float64.
                Contact-GraspNet confidence per candidate; for geometric-fallback grasps the
                alignment of the approach with the camera line of sight in (0, 1]. Candidates
                are ordered best first in both cases.

        Raises:
            ValueError: empty mask, no valid depth under the mask, or depth/mask shape mismatch.
            NoGraspCandidatesError: Contact-GraspNet returned no candidates and no
                collision-free, reachable geometric grasp exists for the segment.

        Example:
            >>> cam = obs["robot0_robotview"]
            >>> rgb = cam["images"]["rgb"]
            >>> depth = cam["images"]["depth"]
            >>> sam3_results = sam3_seg_fn(rgb, text_prompt="red mug")
            >>> best = max(sam3_results, key=lambda d: d["score"])
            >>> mask = best["mask"]
            >>> K = cam["intrinsics"]
            >>> grasp_poses, grasp_scores = plan_grasp(
            ...     depth=depth,
            ...     intrinsics=K,
            ...     segmentation=mask,
            ... )
            >>> best_idx = grasp_scores.argmax()
            >>> best_T = grasp_poses[best_idx]  # (4, 4)
            >>> camera_extrinsics = cam["pose_mat"]
            >>> grasp_sample_world_frame = camera_extrinsics @ best_T
        """
        self._log_step("Contact GraspNet", "Running grasp candidate planning …")
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[:, :, 0]

        # Fail with a description of the geometry instead of an opaque error deeper down
        # when perception did not deliver enough of the object to plan on.
        if depth.shape != segmentation.shape:
            raise ValueError(
                f"plan_grasp: depth {depth.shape} and segmentation {segmentation.shape} must have the same (H, W)"
            )
        in_mask = np.asarray(segmentation) > 0
        n_mask_px = int(np.count_nonzero(in_mask))
        if n_mask_px == 0:
            raise ValueError(
                "plan_grasp: the segmentation mask selects 0 pixels, so there is no object geometry to plan a "
                "grasp on; check the segmentation result (and its score) before calling plan_grasp."
            )
        depth_in_mask = np.asarray(depth, dtype=np.float64)[in_mask]
        depth_ok = np.isfinite(depth_in_mask) & (depth_in_mask > 0)
        if not depth_ok.any():
            raise ValueError(
                f"plan_grasp: none of the {n_mask_px} masked pixels has a valid (finite, > 0) depth value, so "
                "the segmented object has no 3D geometry."
            )
        obj_lo = float(depth_in_mask[depth_ok].min())
        obj_hi = float(depth_in_mask[depth_ok].max())
        # Contact-GraspNet crops the scene to ``z_range`` before planning. Fixed scene cameras
        # can be well over 2 m from the table, so the window is always widened to contain the
        # segmented object; otherwise the object itself is cropped away and the planner has
        # nothing to grasp (two_arm_lift 2026-10-01: handles at 2.2-2.4 m, window 0.2-2.0 m).
        default_hi = 3.5 if self.is_handover else 2.0
        z_range = [0.2, round(float(max(default_hi, obj_hi + 0.5)), 3)]
        report: dict[str, Any] = {
            "function": "plan_grasp",
            "mask_pixels": n_mask_px,
            "object_depth_range_m": [round(obj_lo, 3), round(obj_hi, 3)],
            "planner_depth_range": z_range,
            "source": None,
            "graspnet_candidates": 0,
            "fallback": None,
        }
        self.last_call_report = report

        grasps, scores, _ = self.grasp_net_plan_fn(
            depth,
            intrinsics,
            segmentation,
            1,
            z_range=z_range,
            forward_passes=1 if self.is_handover else 3,
        )
        n_graspnet = int(len(scores))
        report["graspnet_candidates"] = n_graspnet
        if n_graspnet > 0:
            self._env.grasp_sample, self._env.grasp_scores = grasps, scores
            # Contact-GraspNet's grasp frame has the fingers along its x axis (its Panda control
            # points sit at x = +-0.0527 m), while the IK target frame (URDF ``panda_hand``, the
            # frame ``solve_ik`` expects) closes the fingers along y. Rotate each grasp by -90 deg
            # about its approach axis so x_graspnet -> y_hand; without this the executed gripper is
            # yawed 90 deg against the planned grasp (invisible on cubes, fatal on bars / handles).
            # The +0.12 m shift along the approach moves the frame to the TCP between the fingertips.
            self._env.grasp_sample_tf = (
                vtf.SE3.from_matrix(self._env.grasp_sample)
                @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))
                @ vtf.SE3.from_rotation(vtf.SO3.from_z_radians(-np.pi / 2))
            ).as_matrix()
            self._env.grasp_source = report["source"] = "contact_graspnet"
            if hasattr(self._env, "viser_server"):
                self._env._update_viser_server()
            self._log_step_update(
                text=f"Contact-GraspNet: {n_graspnet} candidates, best score={float(scores.max()):.3f}"
            )
            return self._env.grasp_sample_tf, self._env.grasp_scores

        # ---- Contact-GraspNet found nothing: geometric fallback on the segment's own points ----
        graspnet_msg = (
            f"Contact-GraspNet returned no grasp candidates for the segmented object ({n_mask_px} mask "
            f"pixels, depth {obj_lo:.3f}-{obj_hi:.3f} m, planner depth range {z_range})"
        )
        try:
            poses_cam, fb_scores, fb_report = geometric_grasp_candidates(
                object_points_from_mask(depth, intrinsics, in_mask),
                scene_points_from_depth(depth, intrinsics, stride=2),
                depth=depth,
                intrinsics=intrinsics,
            )
        except InsufficientGeometryError as exc:
            report["fallback"] = {"status": "insufficient_geometry", "reason": str(exc)}
            self._log_step_update(text=f"Contact-GraspNet: 0 candidates; geometric fallback failed: {exc}")
            raise NoGraspCandidatesError(f"{graspnet_msg}; geometric fallback: {exc}", report) from exc
        fb_report["status"] = "candidates"
        report["fallback"] = fb_report

        # Reachability: solve IK for every candidate (each arm of a bimanual API) and keep the
        # ones the solver actually reaches. The camera is identified by matching the depth
        # image against the last observation, which also provides camera->base ``pose_mat``.
        pose_mat = self._camera_pose_for_depth(depth)
        if pose_mat is None:
            fb_report["ik"] = {
                "checked": False,
                "reason": "depth image does not match any camera of the last get_observation(); "
                          "reachability not verified",
            }
            keep = list(range(len(poses_cam)))
        else:
            arms = [0, 1] if self.bimanual else [0]
            keep = []
            for i, pose_cam in enumerate(poses_cam):
                pose_base = pose_mat @ pose_cam
                checks = [self._ik_reachability(pose_base, arm) for arm in arms]
                fb_report["candidates"][i]["tcp_base_frame"] = pose_base[:3, 3].round(4).tolist()
                fb_report["candidates"][i]["ik"] = checks
                if any(c["reachable"] is not False for c in checks):
                    keep.append(i)
            fb_report["ik"] = {
                "checked": True,
                "arms": arms,
                "reachable_candidates": len(keep),
                "tolerance": {"position_m": self._IK_POS_TOL, "orientation_deg": self._IK_ROT_TOL_DEG},
            }
            if not keep:
                worst = "; ".join(
                    f"cand {i}: " + ", ".join(
                        f"arm{c['arm']} pos_err={c.get('position_error_m')} m rot_err={c.get('orientation_error_deg')} deg"
                        for c in cand["ik"])
                    for i, cand in enumerate(fb_report["candidates"])
                )
                raise NoGraspCandidatesError(
                    f"{graspnet_msg}; the geometric fallback produced {len(poses_cam)} pinch candidate(s) but none "
                    f"is reachable (IK position error > {self._IK_POS_TOL} m or orientation error > "
                    f"{self._IK_ROT_TOL_DEG} deg for every arm): {worst}",
                    report,
                )
        poses_cam = poses_cam[keep]
        fb_scores = fb_scores[keep]
        fb_report["candidates"] = [fb_report["candidates"][i] for i in keep]
        fb_report["returned"] = len(keep)
        self._env.grasp_source = report["source"] = "geometric_fallback"
        self._env.grasp_sample_tf, self._env.grasp_scores = poses_cam, fb_scores
        self._env.grasp_sample = (
            vtf.SE3.from_matrix(poses_cam) @ vtf.SE3.from_translation(np.array([0, 0, -0.12]))
        ).as_matrix()
        if hasattr(self._env, "viser_server"):
            self._env._update_viser_server()
        ik_note = (f"{len(keep)} reachable" if fb_report["ik"]["checked"] else "IK not checked")
        self._log_step_update(
            text=(f"Contact-GraspNet: 0 candidates; geometric fallback: {len(poses_cam)} pinch grasp(s) across "
                  f"{fb_report['obb']['closing_extent_m']:.3f} m ({ik_note})")
        )
        return poses_cam, fb_scores

    _IK_POS_TOL = 0.02  # m: least-squares IK that lands farther than this did not reach the target
    _IK_ROT_TOL_DEG = 15.0

    def _camera_pose_for_depth(self, depth: np.ndarray) -> np.ndarray | None:
        """Camera->robot-base ``pose_mat`` of the last observation's camera whose depth image is
        ``depth`` (the agent plans from the arrays ``get_observation`` returned), else None."""
        obs = self._last_observation
        if not isinstance(obs, dict):
            return None
        depth = np.asarray(depth)
        for cam in obs.values():
            if not isinstance(cam, dict) or "pose_mat" not in cam or not isinstance(cam.get("images"), dict):
                continue
            cam_depth = cam["images"].get("depth")
            if cam_depth is None:
                continue
            cam_depth = np.squeeze(np.asarray(cam_depth))
            if cam_depth.shape == depth.shape and np.array_equal(cam_depth, depth):
                return np.asarray(cam["pose_mat"], dtype=np.float64)
        return None

    def _ik_reachability(self, pose_base: np.ndarray, arm: int) -> dict[str, Any]:
        """Solve IK for a TCP pose in robot0's base frame with ``arm`` (0 or 1) and compare the
        pose the solver reached with the request. ``reachable`` is None when it could not be
        judged (no achieved pose from the server, or the request failed)."""
        pos = np.asarray(pose_base[:3, 3], dtype=np.float64)
        quat = np.asarray(vtf.SO3.from_matrix(pose_base[:3, :3]).wxyz, dtype=np.float64)
        out: dict[str, Any] = {"arm": arm, "reachable": None}
        try:
            if arm == 1:
                pos, quat = transform_pose_arm0_to_arm1(pos, quat, self._env)
            offset_pos = apply_tcp_offset(pos, quat, self._TCP_OFFSET)
            joints, achieved = self.ik_check_fn(np.concatenate([quat, offset_pos]))
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            out["error"] = f"{type(exc).__name__}: {exc}"
            return out
        out["joints"] = np.asarray(joints, dtype=np.float64)[:7].round(4).tolist()
        if achieved is None:
            out["note"] = "IK server did not return the achieved pose; reachability unknown"
            return out
        achieved = np.asarray(achieved, dtype=np.float64)
        pos_err = float(np.linalg.norm(achieved[4:7] - offset_pos))
        rot_err = float(np.linalg.norm((vtf.SO3(wxyz=achieved[:4]).inverse() @ vtf.SO3(wxyz=quat)).log()))
        out["position_error_m"] = round(pos_err, 4)
        out["orientation_error_deg"] = round(float(np.degrees(rot_err)), 2)
        out["reachable"] = bool(pos_err <= self._IK_POS_TOL and np.degrees(rot_err) <= self._IK_ROT_TOL_DEG)
        return out

    # --------------------------------------------------------------------- #
    # IK / motion primitives
    # --------------------------------------------------------------------- #
    def solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> np.ndarray:
        """Solve inverse kinematics so that the gripper's tool-center point (TCP,
        the grasp point between the fingertips) reaches the target pose.

        The fixed TCP-to-panda_hand offset is applied internally before solving.
        Pass the desired grasp / TCP pose directly (e.g. a ``plan_grasp`` pose
        transformed into the world frame); do NOT subtract any TCP or hand
        offset yourself -- that would place the gripper about 10 cm short of
        the target.

        Args:
            position:
                Target TCP position in world frame.
                Shape: (3,), dtype float64.
            quaternion_wxyz:
                Target orientation as a unit quaternion in world frame.
                Shape: (4,), [w, x, y, z], dtype float64.

        Returns:
            joints:
                np.ndarray of shape (7,), dtype float64.
                Joint angles for the 7 DoF Franka arm.

        Example:
            >>> target_pos = np.array([0.5, 0.0, 0.3])
            >>> target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity, wxyz
            >>> joints = solve_ik(target_pos, target_quat)
            >>> move_to_joints(joints)
        """
        pos_str = np.array2string(np.asarray(position), precision=4)
        self._log_step("IK Solver", f"Solving IK for target position {pos_str} …")
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)

        if self.real:
            quat_wxyz = (vtf.SO3(wxyz=quat_wxyz) @ vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi/4+np.pi/2)).wxyz

            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
            )
            joints = extract_arm_joints(self.cfg)
            self._log_step_update(text=f"IK solved (real mode, 1 pass)")
            return joints
        else:
            self.cfg = solve_ik_with_convergence(
                self.ik_solve_fn, quat_wxyz, offset_pos, self.cfg
            )
            joints = extract_arm_joints(self.cfg)
            self._log_step_update(text=f"IK converged")
            return joints

    # Single arm control APIs

    def move_to_joints(self, joints: np.ndarray) -> None:
        """Move the robot to a given joint configuration in a blocking manner.

        Args:
            joints:
                Target joint angles for the 7-DoF Franka arm.
                Shape: (7,), dtype float64.

        Returns:
            None

        Example:
            >>> joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
            >>> move_to_joints(joints)
        """
        self._log_step("move_to_joints", "Moving robot to target joint configuration …")
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)
        self._log_step_update(text="Motion complete.")

        # self._env.move_to_joints_non_blocking(joints)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        self._log_step("open_gripper", "Opening gripper …")
        _open_gripper(self._env, steps=30)
        self._log_step_update(text="Gripper opened.")

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        self._log_step("close_gripper", "Closing gripper …")
        _close_gripper(self._env, steps=30)
        self._log_step_update(text="Gripper closed.")

        # """Plan a trajectory between two poses. This takes much longer than the IK solver (4s) but returns a trajectory of joint space waypoints which may be smoother and more continuous compared to setting joint targets directly to IK solutions.

    def traj_plan(
        self, start_pose_wxyz_xyz: np.ndarray, end_pose_wxyz_xyz: np.ndarray
    ) -> np.ndarray:
        """Plan a trajectory between two poses.
        Args:
            start_pose_wxyz_xyz:
                Start pose as a unit quaternion in world frame.
                Shape: (7,), dtype float64.
            end_pose_wxyz_xyz:
                End pose as a unit quaternion in world frame.
                Shape: (7,), dtype float64.

        Returns:
            waypoints:
                np.ndarray of shape (N, 7), dtype float64.
                Waypoints for the trajectory.
        """
        start_time = time.time()
        traj = self.trajopt_plan_fn(start_pose_wxyz_xyz, end_pose_wxyz_xyz)
        end_time = time.time()
        print(
            f"Trajectory planning time: {end_time - start_time} seconds for {len(traj)} waypoints"
        )
        return traj[:, :-1]

    def move_along_trajectory(self, trajectory: np.ndarray) -> None:
        """Move the robot along a trajectory of joint space waypoints.
        Args:
            trajectory:
                np.ndarray of shape (N, 7), dtype float64.
                Trajectory of joint space waypoints.
        """
        for waypoint in trajectory:
            self._env.move_to_joints_blocking(waypoint, tolerance=0.025, max_steps=15)

    # Dual arm control APIs
    def move_to_joints_both(self, joints0: np.ndarray, joints1: np.ndarray) -> None:
        """Move the arms 0 and 1 to a given joint configuration in a blocking manner simultaneously.

        Args:
            joints0:
                Target joint angles for the 7-DoF Franka arm 0.
                Shape: (7,), dtype float64.
            joints1:
                Target joint angles for the 7-DoF Franka arm 1.
                Shape: (7,), dtype float64.
        """
        self._env.move_to_joints_blocking_both(joints0, joints1)

    def move_to_joints_arm0(self, joints: np.ndarray) -> None:
        """Move the robot arm 0 to a given joint configuration in a blocking manner.

        Args:
            joints:
                Target joint angles for the 7-DoF Franka arm 0.
                Shape: (7,), dtype float64.
        """
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def move_to_joints_arm1(self, joints: np.ndarray) -> None:
        """Move the robot arm 1 to a given joint configuration in a blocking manner.

        Args:
            joints:
                Target joint angles for the 7-DoF Franka arm 1.
                Shape: (7,), dtype float64.
        """
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking_arm1(joints)

    def open_gripper_arm0(self) -> None:
        """Open gripper fully for Arm 0 (robot0).
        Args:
            None
        Returns:
            None
        """
        _open_gripper(self._env, steps=30)

    def close_gripper_arm0(self) -> None:
        """Close gripper fully for Arm 0 (robot0).
        Args:
            None
        Returns:
            None
        """
        _close_gripper(self._env, steps=30)

    def open_gripper_arm1(self) -> None:
        """Open gripper fully for Arm 1 (robot1).
        Args:
            None
        Returns:
            None
        """
        _open_gripper_arm1(self._env, steps=30)

    def close_gripper_arm1(self) -> None:
        """Close gripper fully for Arm 1 (robot1).
        Args:
            None
        Returns:
            None
        """
        _close_gripper_arm1(self._env, steps=30)

    def solve_ik_arm0(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Solve inverse kinematics for the gripper TCP of Arm 0 (robot0); the TCP-to-hand
        offset is applied internally (see ``solve_ik``)."""
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)

        self.cfg = solve_ik_with_convergence(
            self.ik_solve_fn, quat_wxyz, offset_pos, self.cfg
        )
        return extract_arm_joints(self.cfg)

    def solve_ik_arm1(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Solve inverse kinematics for the gripper TCP of Arm 1 (robot1); the TCP-to-hand
        offset is applied internally (see ``solve_ik``)."""
        if not hasattr(self._env, "move_to_joints_blocking_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")

        pos_arm1, quat_wxyz_arm1 = transform_pose_arm0_to_arm1(
            position, quaternion_wxyz, self._env
        )
        offset_pos = apply_tcp_offset(pos_arm1, quat_wxyz_arm1, self._TCP_OFFSET)

        self.cfg = solve_ik_with_convergence(
            self.ik_solve_fn, quat_wxyz_arm1, offset_pos, self.cfg
        )
        return extract_arm_joints(self.cfg)
