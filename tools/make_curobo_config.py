#!/usr/bin/env python3
"""Make the MagicSim cuRobo robot configs for the G1 with Dex1-1 + D405 hands and the 2-DoF head.

Starts from Haoran's magicsim_g1_fixed_hand.yml (or _mobile.yml): body collision spheres, buffers,
self-collision pairs, locked legs, cspace weights, tool frames and planner keys stay as they are.
The Dex3 hand entries are replaced with the Dex1-1 + D405 hand of the URDF from make_curobo_urdf.py:
  - collision spheres from dex1_d405_hand/dex1_1_d405_spheres.yml, prefixed "<side>_";
  - the hand links ignore self-collision with each other, the wrist links and the hip-pitch link
    (like the Dex3 palm did);
  - finger joints locked open (the planner protects the open gripper), also in ignore_joints so
    MagicSim fills them with that value;
  - the head has no collision spheres (only head_link keeps Haoran's). cuRobo drops joints that lead
    to no collision link or tool frame, so the head joints are not listed anywhere here: the planner
    never sees them (the URDF keeps the head for its camera frames).
The usd_path / usd_robot_root keys (read only by cuRobo's Isaac Sim examples) are dropped.

  python tools/make_curobo_config.py --base-config .../configs/robot/magicsim_g1_fixed_hand.yml \\
      --urdf-path robot/g1_dex1_head/g1_dex1_head.urdf --out .../configs/robot/magicsim_g1_dex1_head.yml
Requires pyyaml.
"""
import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SPHERES = REPO / "dex1_d405_hand" / "dex1_1_d405_spheres.yml"
SIDES = ("right", "left")
DEX3_LINKS = ("hand_palm_link", "hand_fingers_link")
FINGER_JOINTS = ("dex1_finger_joint_1", "dex1_finger_joint_2")
FINGER_OPEN = -0.02
HAND_ALSO_IGNORES = ("wrist_yaw_link", "wrist_pitch_link", "wrist_roll_link", "hip_pitch_link")
HAND_SPHERE_BUFFER = 0.0       # Haoran's -0.02 would shrink the Dex1-1's 3-7 mm spheres to nothing
HAND_SELF_COLLISION_BUFFER = 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", type=Path, required=True, help="magicsim_g1_fixed_hand[_mobile].yml")
    ap.add_argument("--urdf-path", required=True, help="kinematics.urdf_path, relative to cuRobo's assets")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cfg = yaml.safe_load(a.base_config.read_text())
    k = cfg["robot_cfg"]["kinematics"]
    k["urdf_path"] = a.urdf_path
    k["asset_root_path"] = str(Path(a.urdf_path).parent)
    cfg.pop("usd_path", None)
    cfg.pop("usd_robot_root", None)

    hand = yaml.safe_load(SPHERES.read_text())["collision_spheres"]
    hand_links = list(hand)  # hand_palm_link, hand_d405_*, dex1_finger*
    for side in SIDES:
        dex3 = [f"{side}_{l}" for l in DEX3_LINKS]
        ours = [f"{side}_{l}" for l in hand_links]
        # collision links: Dex3 entries -> ours, at the position of the old palm entry
        names = k["collision_link_names"]
        i = names.index(dex3[0])
        names[:] = [n for n in names if n not in dex3]
        names[i:i] = ours
        # spheres, buffers
        for d in (k["collision_spheres"], k["collision_sphere_buffer"], k["self_collision_buffer"]):
            for n in dex3:
                d.pop(n, None)
        for l in hand_links:
            k["collision_spheres"][f"{side}_{l}"] = [{"center": s["center"], "radius": s["radius"]} for s in hand[l]]
            k["collision_sphere_buffer"][f"{side}_{l}"] = HAND_SPHERE_BUFFER
            k["self_collision_buffer"][f"{side}_{l}"] = HAND_SELF_COLLISION_BUFFER
        # self-collision: drop Dex3 entries / mentions, then hand <-> hand, wrist and hip-pitch links
        sci = k["self_collision_ignore"]
        for n in dex3:
            sci.pop(n, None)
        for n, lst in sci.items():
            sci[n] = [x for x in lst if x not in dex3]
        for n in ours:
            sci[n] = [x for x in ours if x != n] + [f"{side}_{l}" for l in HAND_ALSO_IGNORES]
        # locked joints (planner) and ignore_joints (MagicSim fills them from these values)
        extra = {f"{side}_{j}": FINGER_OPEN for j in FINGER_JOINTS}
        k["lock_joints"].update(extra)
        cfg.setdefault("ignore_joints", {}).update(extra)

    # cspace: every locked joint must be listed, with a weight and a default position
    cs = k["cspace"]
    for j, v in k["lock_joints"].items():
        if j not in cs["joint_names"]:
            cs["joint_names"].append(j)
            cs["null_space_weight"].append(1.0)
            cs["cspace_distance_weight"].append(1.0)
            cs["default_joint_position"].append(v)
    n = len(cs["joint_names"])
    assert all(len(cs[key]) == n for key in ("null_space_weight", "cspace_distance_weight", "default_joint_position"))

    # every self-collision / buffer key must be a collision link, and every collision link needs spheres
    links = set(k["collision_link_names"])
    for key in ("collision_spheres", "collision_sphere_buffer", "self_collision_buffer"):  # (his ignore list also names sensor links)
        stray = [x for x in k[key] if x not in links]
        assert not stray, f"{key} has non-collision links {stray}"
    assert all(k["collision_spheres"].get(l) for l in links), "collision link without spheres"

    header = (f"# Generated by g1_twist2_assets/tools/make_curobo_config.py from {a.base_config.name} and\n"
              f"# dex1_d405_hand/dex1_1_d405_spheres.yml ({sum(len(v) for v in hand.values())} spheres per hand). Do not edit;\n"
              f"# regenerate after changing either.\n")
    a.out.write_text(header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=120))
    total = sum(len(v) for v in k["collision_spheres"].values())
    print(f"wrote {a.out}: {len(links)} collision links, {total} spheres, cspace {n} joints, "
          f"locked {len(k['lock_joints'])}, tool_frames {k['tool_frames']}")


if __name__ == "__main__":
    sys.exit(main())
