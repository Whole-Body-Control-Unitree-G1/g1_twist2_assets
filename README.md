# G1 + TWIST2 head + Dex1/D405 hand — Isaac Sim assets

Self-contained USD/URDF assets for a Unitree G1 fitted with a TWIST2 2-DoF camera head
and Dex1_1 grippers carrying RealSense D405 cameras. Meters, Z-up.

## Contents

### `g1_assembled/` — the full robot (what you load in Isaac Sim)
- `g1_29dof_with_dex1_base_fix1.usd` — Unitree G1 (29-DoF) with:
  - both **Dex1_1 hands** rebuilt with the official **D405 mount** (built-in USB-cam removed),
    all gripper meshes matte black, wrist cameras `left_D405` / `right_D405`.
  - the **TWIST2 2-DoF head** mounted on `head_link` (referenced from `twist2_head.usda`),
    merged into the single robot articulation.
  - Everything is one articulation (root `root_joint`). Meshes are inline except:
    - the head, which references `./twist2_head.usda`;
    - the hand geometry (`visuals`, `collisions` and `<side>_D405` of the hand links) and its materials
      (`<side>_hand_Looks`), which reference `../dex1_d405_hand/dex1_1_d405_{right,left}.usd`.
    Keep the folders together. Hand link prims, joints, drives and masses stay in the G1 file.
  - Hand names follow MagicSim's `g1_dex1`: links `<side>_hand_palm_link` (the Dex1-1 base),
    `<side>_dex1_finger{1,2}_{1,2,3}_link` (`_3` = pad), `<side>_hand_d405_mount_link`,
    `<side>_hand_d405_camera_link`; prismatic joints `<side>_dex1_finger_joint_{1,2}`; fixed joints
    `<side>_hand_palm_joint` (to `<side>_wrist_yaw_link`), `<side>_dex1_finger{1,2}_{2,3}_joint`,
    `<side>_hand_d405_{mount,camera}_joint`.
- `twist2_head.usda` — the head, referenced by the G1 (also opens standalone).
- `config.yaml`, `configuration/` — original Unitree asset config.

**Head joints to drive:** `joint_yaw` (yaw, centered 23°), `joint_pitch_motor` (pitch, −30°…+90°).
Gears `gear_yaw` / `gear_pitch` couple the rest.

**D405 cameras:** `<root>/<side>_hand_d405_camera_link/<side>_D405`, and per-hand optical frames.
Intrinsics: our unit's calibrated colour stream, 848×480, fx 429.97 / fy 429.45, cx 413.72 / cy 243.44 px,
plumb_bob distortion (HFOV 89.2° / VFOV 58.4°), authored as Isaac Sim's OpenCV pinhole lens model on the camera prims.

### `dex1_d405_hand/` — Dex1_1 hand with the D405 mount (single source for the hand)
- `dex1_1_d405.urdf` (+ `meshes/`) — the source of truth for ROS, planners and the USDs below.
- `dex1_1_d405.usd` — articulated standalone hand, URDF names (`hand_palm_link`, `dex1_finger_joint_1`, …),
  camera `hand_d405_camera_link/D405`, fixed to the world.
- `dex1_1_d405_right.usd`, `dex1_1_d405_left.usd` — the same hand with `right_` / `left_`
  names (e.g. `right_hand_palm_link`) and `<side>_D405` cameras; the G1 references their geometry.
- All three have the same frames as the URDF and the G1. `dex1_finger_joint_1`/`_2` are prismatic
  (−0.02…0.0245 m, + closes) and joint 2 mimics joint 1 (single motor). The D405 mount and camera are
  their own links on fixed joints. Colliders come from the visual meshes: convex decomposition for the
  base, mount and finger bodies (`dex1_finger*_2_link`), convex hulls for the rest.
- `dex1_1_d405_spheres.yml` — collision spheres for cuRobo/BODex, per URDF link in link frames
  (152 spheres; sphere 0 of `dex1_finger1_3_link`/`dex1_finger2_3_link` is the pad-face contact point).
- Built from the official `Dex1_1_Realsense_D405_Camera_Mount_M5010` mount.

### `tools/` — regenerate the hand assets (needs `usd-core trimesh numpy scipy pyyaml`)
- `build_dex1_1_usd.py` — URDF → the three hand USDs above.
- `g1_use_dex1_hand.py` — point the G1's hand geometry at the per-side hand USDs (re-runnable).
- `yaml_spheres_to_usd.py` — `dex1_1_d405_spheres.yml` → `dex1_d405_hand/dex1_1_d405_spheres_edit.usd`
  (the hand plus its spheres, one `spheres_<link>` group per link) for editing in Isaac Sim.
- `usd_spheres_to_yaml.py` — the edited USD → `dex1_1_d405_spheres.yml`.
- `make_pad_knurl_texture.py` — the pad knurl normal map (`dex1_d405_hand/textures/`).

Editing spheres: `python tools/yaml_spheres_to_usd.py`, edit and save the USD in Isaac Sim, then
`python tools/usd_spheres_to_yaml.py dex1_d405_hand/dex1_1_d405_spheres_edit.usd dex1_d405_hand/dex1_1_d405_spheres.yml`.
The hand USDs themselves are generated: change the URDF, meshes or `build_dex1_1_usd.py` and rebuild,
rather than editing them in Isaac Sim.

After editing the URDF: run `build_dex1_1_usd.py`; the G1 picks the change up through its references.

### `twist2_head/` — standalone 2-DoF head
- `twist2_head.usda` — yaw (gear-coupled) + pitch (crank/rod linkage), ZED Mini stereo pair.
  Pitch limits −30°…+90°, yaw centered 23°.

## Notes / TODO
- The head's mount transform on the G1 is a **first pass** (registered to the mid360 lidar
  hole via the 4 corner screws, ~4 mm residual). Refine with a camera calibration and update
  the transform on the `twist2_head` reference prim in the G1 USD.
- cuRobo spheres for the G1's hands: take `dex1_d405_hand/dex1_1_d405_spheres.yml` and prefix the
  link names with `left_` / `right_` (they then match the G1's link names).
