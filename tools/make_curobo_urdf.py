#!/usr/bin/env python3
"""Make the cuRobo planning URDF of the G1 with Dex1-1 + D405 hands and the TWIST2 2-DoF head.

The body is MagicSim/cuRobo's simplified G1 (robot/g1_simple/g1_fixed_hand.urdf or its _mobile
variant: pelvis as root, locked simplified legs, Haoran's waist limits), so the body collision spheres
and planner settings of magicsim_g1_fixed_hand.yml apply unchanged. Its Dex3 palm / fingers links are
replaced by:
  - the Dex1-1 + D405 hand from dex1_d405_hand/dex1_1_d405.urdf, prefixed "<side>_" (the same names
    as the G1 USD), mounted on <side>_wrist_yaw_link at the G1 USD's <side>_hand_palm_joint
    (PALM_MOUNT_* in g1_use_dex1_hand.py);
  - the 2-DoF head from the G1 USD: head_link -> head_base_link (fixed) -> head_yaw_link
    (joint_yaw) -> head_pitch_link (joint_pitch), with ROS optical frames for the two ZED cameras.
    Only the kinematic chain: the gear-coupled motor bodies are left out and the head has no meshes.

Everything is written to --out-dir: <name>.urdf and meshes/dex1_d405/ (hand STLs). Body meshes are
referenced from the base URDF's folder (--body-mesh-prefix). With --check, the link poses are compared
with the G1 USD (arms, waist, hands, head) at q = 0 and at random arm / waist / head configurations.

  python tools/make_curobo_urdf.py --base-urdf .../g1_simple/g1_fixed_hand.urdf --out-dir out --check
  python tools/make_curobo_urdf.py --base-urdf .../g1_simple/g1_fixed_hand_mobile.urdf --name g1_dex1_head_mobile --out-dir out --check
--joint-limit NAME=LOWER,UPPER (radians or metres, repeatable) overrides a joint's planning limits, e.g.
--joint-limit waist_pitch_joint=-0.1,0.1 to keep the planner from bending the torso.
Requires usd-core, numpy, scipy.
"""
import argparse
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_use_dex1_hand import PALM_MOUNT_XYZ, PALM_MOUNT_YAW  # noqa: E402

G1_USD = REPO / "g1_assembled" / "g1_29dof_with_dex1_base_fix1.usd"
HAND_URDF = REPO / "dex1_d405_hand" / "dex1_1_d405.urdf"
ROOT = "/g1_29dof_with_hand_rev_1_0"
HEAD = f"{ROOT}/twist2_head"
SIDES = ("left", "right")
# Head bodies in the G1 USD -> URDF links, and the joints between them (USD name kept for the DoFs,
# so they match the simulated articulation)
HEAD_LINKS = {"base": "head_base_link", "yaw_body": "head_yaw_link", "pitch_body": "head_pitch_link"}
HEAD_JOINTS = [("joint_yaw", "base", "yaw_body"), ("joint_pitch", "yaw_body", "pitch_body")]
HEAD_MOUNT_JOINT = "head_base_joint"
HEAD_CAMERAS = {"zed_left": "head_zed_left_optical_frame", "zed_right": "head_zed_right_optical_frame"}
# The G1 USD's head is placed at +2.29 deg pitch but base_fixed says -2.29 deg; the simulation
# follows the joint. "joint" (default) matches the simulation, "placed" the authored prim pose.
LEG_KEYS = ("hip", "knee", "ankle", "foot", "upper_leg")  # simplified in the cuRobo body: not compared
TOL_MM, TOL_DEG = 0.5, 0.1  # the Unitree body's own authored poses disagree by up to ~0.2 mm
USD_CAM_TO_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])  # USD camera (-Z fwd, +Y up) -> ROS optical (+Z fwd, +Y down)


# ----------------------------------------------------------------------------- math / URDF helpers

def tf(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    m = np.eye(4)
    m[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    m[:3, 3] = xyz
    return m


def fmt(v):
    return " ".join(f"{x:.9g}" if abs(x) > 1e-12 else "0" for x in v)


def origin_el(M):
    return ET.Element("origin", xyz=fmt(M[:3, 3]), rpy=fmt(R.from_matrix(M[:3, :3]).as_euler("xyz")))


def joint_el(name, jtype, parent, child, M, axis=None, limit=None):
    j = ET.Element("joint", name=name, type=jtype)
    j.append(origin_el(M))
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    if axis is not None:
        ET.SubElement(j, "axis", xyz=fmt(axis))
    if limit is not None:
        ET.SubElement(j, "limit", **{k: f"{v:.9g}" for k, v in limit.items()})
    return j


def urdf_fk(robot, q=None):
    """Link poses in the root link frame at joint positions q (default 0)."""
    q = q or {}
    joints = robot.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    root = next(l.get("name") for l in robot.findall("link") if l.get("name") not in children)
    pose, pending = {root: np.eye(4)}, list(joints)
    while pending:
        progressed = False
        for j in list(pending):
            parent = j.find("parent").get("link")
            if parent not in pose:
                continue
            o = j.find("origin")
            M = tf([float(v) for v in o.get("xyz", "0 0 0").split()], [float(v) for v in o.get("rpy", "0 0 0").split()]) if o is not None else np.eye(4)
            a = j.find("axis")
            axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.array([1.0, 0, 0])
            v = q.get(j.get("name"), 0.0)
            if j.get("type") in ("revolute", "continuous"):
                M = M.copy()
                M[:3, :3] = M[:3, :3] @ R.from_rotvec(axis * v).as_matrix()
            elif j.get("type") == "prismatic":
                M = M @ tf(axis * v)
            pose[j.find("child").get("link")] = pose[parent] @ M
            pending.remove(j)
            progressed = True
        assert progressed, f"URDF not a tree below {root}: {[j.get('name') for j in pending]}"
    return pose


# ----------------------------------------------------------------------------- USD

class G1Usd:
    def __init__(self, path):
        self.stage = Usd.Stage.Open(str(path))
        self.xc = UsdGeom.XformCache(0)
        self.inv_root = np.linalg.inv(self.world(ROOT))

    def world(self, path):
        return np.array(self.xc.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))).T

    def pose(self, path):
        """Prim pose in the G1 root prim frame, as authored (q = 0)."""
        return self.inv_root @ self.world(path)

    def joint(self, name, under=ROOT):
        prim = next(p for p in Usd.PrimRange(self.stage.GetPrimAtPath(under)) if p.GetName() == name and p.IsA(UsdPhysics.Joint))
        return prim, UsdPhysics.Joint(prim)

    @staticmethod
    def local(pos, rot):
        m = np.eye(4)
        m[:3, 3] = list(pos)
        m[:3, :3] = R.from_quat([*rot.GetImaginary(), rot.GetReal()]).as_matrix()
        return m


# ----------------------------------------------------------------------------- build

def add_hands(robot, mesh_dir_rel):
    hand = ET.parse(HAND_URDF).getroot()
    for side in SIDES:
        for tag in ("link", "joint"):  # remove the Dex3 hand of the base URDF
            for el in list(robot.findall(tag)):
                if el.get("name") in (f"{side}_hand_palm_link", f"{side}_hand_fingers_link",
                                      f"{side}_hand_palm_joint", f"{side}_hand_fingers_joint"):
                    robot.remove(el)
        for el in hand:
            if el.tag not in ("link", "joint"):
                continue
            el = ET.fromstring(ET.tostring(el))
            el.set("name", f"{side}_{el.get('name')}")
            for ref in el.findall("parent") + el.findall("child"):
                ref.set("link", f"{side}_{ref.get('link')}")
            for m in el.iter("mesh"):
                m.set("filename", f"{mesh_dir_rel}/{Path(m.get('filename')).name}")
            for mat in el.iter("material"):
                mat.set("name", f"{side}_{mat.get('name')}")
            robot.append(el)
        mount = tf(PALM_MOUNT_XYZ[side], (0, 0, PALM_MOUNT_YAW))
        robot.append(joint_el(f"{side}_hand_palm_joint", "fixed", f"{side}_wrist_yaw_link", f"{side}_hand_palm_link", mount))


def add_head(robot, usd, head_mount):
    """Head links in the G1 root frame at q = 0, re-rooted at the chosen head base pose."""
    prim, j = usd.joint("base_fixed", HEAD)
    head_link = usd.pose(f"{ROOT}/head_link")
    placed = usd.pose(f"{HEAD}/base")
    by_joint = head_link @ usd.local(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()) @ \
        np.linalg.inv(usd.local(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()))
    base = by_joint if head_mount == "joint" else placed
    move = base @ np.linalg.inv(placed)  # authored head pose -> chosen head pose
    frame = {"base": base}  # URDF frame of each head link, in the G1 root frame
    robot.append(ET.Element("link", name=HEAD_LINKS["base"]))
    robot.append(joint_el(HEAD_MOUNT_JOINT, "fixed", "head_link", HEAD_LINKS["base"], np.linalg.inv(head_link) @ base))
    for name, b0, b1 in HEAD_JOINTS:
        _, rj = usd.joint(name, HEAD)
        rj = UsdPhysics.RevoluteJoint(rj.GetPrim())
        assert rj.GetAxisAttr().Get() == "Z", f"{name}: axis {rj.GetAxisAttr().Get()}"
        # joint frame on the parent side = URDF frame of the child link (rotation about its +Z)
        jf = move @ usd.pose(f"{HEAD}/{b0}") @ usd.local(rj.GetLocalPos0Attr().Get(), rj.GetLocalRot0Attr().Get())
        frame[b1] = jf
        frame[f"{b1}_body"] = move @ usd.pose(f"{HEAD}/{b1}")  # the USD body frame, for cameras below
        lower, upper = math.radians(rj.GetLowerLimitAttr().Get()), math.radians(rj.GetUpperLimitAttr().Get())
        robot.append(ET.Element("link", name=HEAD_LINKS[b1]))
        robot.append(joint_el(name, "revolute", HEAD_LINKS[b0], HEAD_LINKS[b1], np.linalg.inv(frame[b0]) @ jf,
                              axis=(0, 0, 1), limit=dict(lower=lower, upper=upper, effort=10.0, velocity=3.0)))
    for cam, link in HEAD_CAMERAS.items():
        opt = move @ usd.pose(f"{HEAD}/pitch_body/{cam}") @ USD_CAM_TO_OPTICAL
        robot.append(ET.Element("link", name=link))
        robot.append(joint_el(f"{link}_joint", "fixed", HEAD_LINKS["pitch_body"], link, np.linalg.inv(frame["pitch_body"]) @ opt))
    return frame


# ----------------------------------------------------------------------------- check

def check(robot, usd, head_frame, head_mount, trials=20, seed=0):
    """Compare URDF FK with the G1 USD. At q = 0: every URDF link that has a USD prim of the same name
    (arms, waist, torso, hands, D405 links) plus the head links and camera frames. At random
    configurations: the palm / finger / camera links against the USD's joint chain."""
    fk0 = urdf_fk(robot)
    root_link = next(iter(fk0))
    # URDF root (pelvis for the fixed body, g1_mobile for _mobile) -> G1 root frame
    pelvis = "g1_29dof_with_hand_rev_1_0"
    to_g1 = usd.pose(f"{ROOT}/pelvis") @ np.linalg.inv(fk0[pelvis])
    errs = []
    for link, M in fk0.items():
        if link in (pelvis, root_link) or any(k in link for k in LEG_KEYS):
            continue
        prim = usd.stage.GetPrimAtPath(f"{ROOT}/{link}")
        if prim and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            errs.append((link, to_g1 @ M, usd.pose(f"{ROOT}/{link}")))
    # head: URDF frames against the frames built from the USD (joint frames / optical frames)
    for usd_name, link in HEAD_LINKS.items():
        errs.append((link, to_g1 @ fk0[link], head_frame[usd_name]))
    report = []
    for link, a, b in errs:
        dp = np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000
        dr = math.degrees(R.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
        report.append((link, dp, dr))
    worst = max(report, key=lambda r: r[1] + r[2])
    print(f"check q=0: {len(report)} links vs the G1 USD, max {max(r[1] for r in report):.4f} mm, "
          f"{max(r[2] for r in report):.4f} deg (worst {worst[0]})")
    for link, dp, dr in report:
        if dp > TOL_MM or dr > TOL_DEG:
            print(f"   MISMATCH {link}: {dp:.3f} mm {dr:.3f} deg")
    # random configurations: moving joints must move the links the way the USD joint frames say
    rng = np.random.default_rng(seed)
    moving = [j for j in robot.findall("joint") if j.get("type") in ("revolute", "prismatic")
              and any(k in j.get("name") for k in ("waist", "shoulder", "elbow", "wrist", "dex1_finger_joint", "joint_yaw", "joint_pitch"))]
    worst_rand = 0.0
    for _ in range(trials):
        q = {}
        for j in moving:
            lim = j.find("limit")
            q[j.get("name")] = rng.uniform(float(lim.get("lower")), float(lim.get("upper")))
        fk = urdf_fk(robot, q)
        # reference: the USD's own joints, applied along the same chains
        for side in SIDES:
            for link in ("hand_palm_link", "dex1_finger1_3_link", "dex1_finger2_3_link", "hand_d405_camera_link"):
                ref = usd_chain_pose(usd, f"{side}_{link}", q)
                a = to_g1 @ fk[f"{side}_{link}"]
                worst_rand = max(worst_rand, np.linalg.norm(a[:3, 3] - ref[:3, 3]) * 1000,
                                 math.degrees(R.from_matrix(a[:3, :3].T @ ref[:3, :3]).magnitude()))
    print(f"check random q ({trials} configurations, arms/waist/fingers): max {worst_rand:.4f} (mm or deg)")
    return max(max(r[1], r[2]) for r in report), worst_rand


def usd_chain_pose(usd, body, q):
    """Pose of `body` in the G1 root frame with the USD joints from the pelvis set to q (by name)."""
    joints = {}
    for p in usd.stage.Traverse():
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if b0 and b1:
                joints[b1[0].name] = (p, j, b0[0].name)
    chain, cur = [], body
    while cur != "pelvis":
        p, j, parent = joints[cur]
        chain.append((p, j, parent))
        cur = parent
    M = usd.pose(f"{ROOT}/pelvis")
    for p, j, parent in reversed(chain):
        L0 = usd.local(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get())
        L1 = usd.local(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())
        v = q.get(p.GetName(), 0.0)
        motion = np.eye(4)
        if p.IsA(UsdPhysics.RevoluteJoint):
            axis = {"X": 0, "Y": 1, "Z": 2}[UsdPhysics.RevoluteJoint(p).GetAxisAttr().Get()]
            motion[:3, :3] = R.from_rotvec(np.eye(3)[axis] * v).as_matrix()
        elif p.IsA(UsdPhysics.PrismaticJoint):
            axis = {"X": 0, "Y": 1, "Z": 2}[UsdPhysics.PrismaticJoint(p).GetAxisAttr().Get()]
            motion[:3, 3] = np.eye(3)[axis] * v
        M = M @ L0 @ motion @ np.linalg.inv(L1)
    return M


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-urdf", type=Path, required=True, help="cuRobo robot/g1_simple/g1_fixed_hand[_mobile].urdf")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", default="g1_dex1_head", help="robot name and URDF file name")
    ap.add_argument("--body-mesh-prefix", default="../g1_simple/",
                    help="prefix for the base URDF's mesh paths, relative to --out-dir (default: cuRobo's g1_simple)")
    ap.add_argument("--head-mount", choices=("joint", "placed"), default="joint",
                    help="head base pose: the G1 USD's base_fixed joint (as simulated) or the authored prim pose")
    ap.add_argument("--check", action="store_true", help="compare the result with the G1 USD")
    ap.add_argument("--joint-limit", action="append", default=[], metavar="NAME=LOWER,UPPER",
                    help="override a joint's planning limits (repeatable), e.g. waist_pitch_joint=-0.1,0.1")
    a = ap.parse_args()

    robot = ET.parse(a.base_urdf).getroot()
    robot.set("name", a.name)
    for m in robot.iter("mesh"):
        m.set("filename", a.body_mesh_prefix + m.get("filename"))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    mesh_rel = "meshes/dex1_d405"
    (a.out_dir / mesh_rel).mkdir(parents=True, exist_ok=True)
    for m in ET.parse(HAND_URDF).getroot().iter("mesh"):
        shutil.copy2(HAND_URDF.parent / m.get("filename"), a.out_dir / mesh_rel / Path(m.get("filename")).name)

    add_hands(robot, mesh_rel)
    usd = G1Usd(G1_USD)
    head_frame = add_head(robot, usd, a.head_mount)
    joints = {j.get("name"): j for j in robot.findall("joint")}
    for spec in a.joint_limit:
        name, _, rng = spec.partition("=")
        lower, upper = (float(v) for v in rng.split(","))
        if name not in joints or joints[name].find("limit") is None:
            sys.exit(f"--joint-limit: no joint with limits named {name!r}")
        lim = joints[name].find("limit")
        lim.set("lower", f"{lower:.9g}")
        lim.set("upper", f"{upper:.9g}")
        print(f"limit {name}: [{lower}, {upper}]")
    ET.indent(robot, "  ")
    out = a.out_dir / f"{a.name}.urdf"
    header = (f"<!-- Generated by g1_twist2_assets/tools/make_curobo_urdf.py from {a.base_urdf.name}, "
              f"dex1_d405_hand/dex1_1_d405.urdf and the G1 USD (head mount: {a.head_mount}). Do not edit. -->\n")
    out.write_text('<?xml version="1.0"?>\n' + header + ET.tostring(robot, encoding="unicode") + "\n")
    print(f"wrote {out}")
    if a.check:
        q0, qr = check(robot, usd, head_frame, a.head_mount)
        if q0 > TOL_MM or qr > TOL_MM:
            sys.exit("check FAILED")
        print("check passed")


if __name__ == "__main__":
    main()
