#!/usr/bin/env python3
"""Make MagicSim's robot assets for the G1 with Dex1-1 + D405 hands and the TWIST2 2-DoF head.

Writes into <magicsim>/Assets/Robots/:
  <name>.usd                  the G1 USD of this repo, flattened into one file (head, hands and
                              materials inlined), with a floating base like MagicSim's other G1s:
                              root_joint (pelvis fixed to the world) removed, articulation root on
                              pelvis. The pad texture goes to textures/<name>/.
  URDF/<name>.urdf            the Pink IK model, like MagicSim's g1_dex1.urdf: root "pelvis", full
                              29-DoF body, hands and head, no meshes. It is the cuRobo URDF of
                              make_curobo_urdf.py (upper body, hands, head) with the simplified legs
                              replaced by the G1 USD's legs, and every joint limit taken from the USD
                              (Pink drives the simulated robot).
Both are checked against the G1 USD: every rigid body pose (USD), and every link at q = 0 plus the
feet, palms, pads and D405 cameras at random joint configurations (URDF).

  python tools/make_magicsim_assets.py [--magicsim ~/wbcG1/MagicSim] [--name g1_dex1_head]
Requires usd-core, numpy, scipy.
"""
import argparse
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_curobo_urdf import (  # noqa: E402
    G1_USD, ROOT, SIDES, G1Usd, add_hands, add_head, joint_el, urdf_fk, usd_chain_pose,
)

REPO = Path(__file__).resolve().parent.parent
CUROBO_BODY = "Third_Party/curobo/curobo/content/assets/robot/g1_simple/g1_fixed_hand.urdf"
CUROBO_ROOT = "g1_29dof_with_hand_rev_1_0"   # the cuRobo body's pelvis link
LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
CUROBO_LEG_LINKS = ("hip_pitch_link", "upper_leg_link", "knee_link", "foot_link")
AXES = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}
TOL_MM, TOL_DEG = 0.5, 0.1


# ----------------------------------------------------------------------------- USD

def make_usd(out, textures_rel):
    stage = Usd.Stage.Open(str(G1_USD))
    layer = stage.Flatten()
    root = layer.GetPrimAtPath(ROOT)
    rj = layer.GetPrimAtPath(f"{ROOT}/root_joint")
    pelvis = layer.GetPrimAtPath(f"{ROOT}/pelvis")
    assert rj and pelvis, "expected root_joint and pelvis"
    # floating base: articulation root (and its PhysX settings) from root_joint to pelvis
    schemas = [s for s in rj.GetInfo("apiSchemas").GetAddedOrExplicitItems() if "Articulation" in s]
    ops = pelvis.GetInfo("apiSchemas") if pelvis.HasInfo("apiSchemas") else Sdf.TokenListOp()
    items = list(ops.GetAddedOrExplicitItems())
    pelvis.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=schemas + [s for s in items if s not in schemas]))
    for name, spec in rj.attributes.items():
        if name.startswith("physxArticulation:"):
            Sdf.AttributeSpec(pelvis, name, spec.typeName).default = spec.default
    del root.nameChildren["root_joint"]
    # textures: copy next to the USD, repoint the flattened (absolute) asset paths
    copied = {}

    def fix_assets(path):
        spec = layer.GetAttributeAtPath(path) if path.IsPropertyPath() else None
        if spec is None or spec.typeName != Sdf.ValueTypeNames.Asset or not spec.default:
            return
        src = Path(spec.default.resolvedPath or spec.default.path)
        if src.suffix.lower() in (".png", ".jpg", ".jpeg", ".exr") and src.is_file():
            dst = out.parent / textures_rel / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied[src.name] = dst
            spec.default = Sdf.AssetPath(f"./{textures_rel}/{src.name}")
    layer.Traverse(Sdf.Path.absoluteRootPath, fix_assets)
    out.unlink(missing_ok=True)
    layer.Export(str(out))
    print(f"wrote {out} (textures: {sorted(copied)})")


def check_usd(out):
    src, new = Usd.Stage.Open(str(G1_USD)), Usd.Stage.Open(str(out))
    xs, xn = UsdGeom.XformCache(0), UsdGeom.XformCache(0)
    bodies = [p for p in src.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    worst, missing = 0.0, []
    for p in bodies:
        q = new.GetPrimAtPath(p.GetPath())
        if not q:
            missing.append(str(p.GetPath()))
            continue
        worst = max(worst, np.abs(np.array(xs.GetLocalToWorldTransform(p)) - np.array(xn.GetLocalToWorldTransform(q))).max())
    joints = lambda st: {str(p.GetPath()) for p in st.Traverse() if p.IsA(UsdPhysics.Joint)}
    lost = joints(src) - joints(new)
    world = [str(p.GetPath()) for p in new.Traverse() if p.IsA(UsdPhysics.Joint)
             and (not UsdPhysics.Joint(p).GetBody0Rel().GetTargets() or not UsdPhysics.Joint(p).GetBody1Rel().GetTargets())]
    art = [str(p.GetPath()) for p in new.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    layers = [l for l in new.GetUsedLayers() if not l.anonymous]
    unresolved = []
    for p in new.Traverse():
        for a in p.GetAttributes():
            v = a.Get() if a.GetTypeName() == Sdf.ValueTypeNames.Asset else None
            if v and v.path and not v.path.endswith(".mdl") and not v.resolvedPath:
                unresolved.append(f"{p.GetPath()}.{a.GetName()}: {v.path}")
    cams = lambda st: sorted(str(p.GetPath()) for p in st.Traverse() if p.IsA(UsdGeom.Camera))
    print(f"check USD: {len(bodies)} bodies, max pose diff {worst:.1e}, missing {missing}; joints lost {sorted(lost)}; "
          f"world joints {world}; articulation root {art}; layers {[l.identifier.split('/')[-1] for l in layers]}; "
          f"unresolved assets {unresolved}; cameras same {cams(src) == cams(new)} ({len(cams(new))})")
    ok = not missing and worst < 1e-6 and lost == {f"{ROOT}/root_joint"} and not world and art == [f"{ROOT}/pelvis"] \
        and len(layers) == 1 and not unresolved and cams(src) == cams(new)
    return ok


# ----------------------------------------------------------------------------- Pink IK URDF

def usd_revolute(usd, name):
    prim = usd.stage.GetPrimAtPath(f"{ROOT}/joints/{name}")
    j = UsdPhysics.RevoluteJoint(prim)
    assert tuple(j.GetLocalPos1Attr().Get()) == (0, 0, 0) and j.GetLocalRot1Attr().Get().GetReal() == 1, name
    return prim, j


def make_urdf(magicsim, out, usd):
    robot = ET.parse(magicsim / CUROBO_BODY).getroot()
    robot.set("name", out.stem)
    add_hands(robot, "unused")
    add_head(robot, usd, "joint")
    # simplified cuRobo legs -> the USD's legs
    drop_links = {f"{s}_{l}" for s in SIDES for l in CUROBO_LEG_LINKS}
    for el in list(robot):
        if el.tag == "link" and el.get("name") in drop_links or \
           el.tag == "joint" and el.find("child").get("link") in drop_links:
            robot.remove(el)
    for side in SIDES:
        for name in LEG_JOINTS:
            prim, j = usd_revolute(usd, f"{side}_{name}_joint")
            parent, child = j.GetBody0Rel().GetTargets()[0].name, j.GetBody1Rel().GetTargets()[0].name
            robot.append(ET.Element("link", name=child))
            robot.append(joint_el(f"{side}_{name}_joint", "revolute", parent, child,
                                  usd.local(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()), axis=AXES[j.GetAxisAttr().Get()],
                                  limit=dict(lower=0.0, upper=0.0, effort=0.0, velocity=0.0)))
    # root link name, no meshes / visuals / collisions / materials (Pink only needs kinematics)
    for el in robot.iter():
        for attr in ("name", "link"):
            if el.get(attr) == CUROBO_ROOT:
                el.set(attr, "pelvis")
    for link in robot.findall("link"):
        for tag in ("visual", "collision"):
            for sub in link.findall(tag):
                link.remove(sub)
    for mat in robot.findall("material"):
        robot.remove(mat)
    # every movable joint's limits from the USD (the simulated robot)
    for j in robot.findall("joint"):
        if j.get("type") not in ("revolute", "prismatic"):
            continue
        prim = next((p for p in Usd.PrimRange(usd.stage.GetPrimAtPath(ROOT)) if p.GetName() == j.get("name") and p.IsA(UsdPhysics.Joint)), None)
        assert prim is not None, f"{j.get('name')} not in the G1 USD"
        uj = UsdPhysics.RevoluteJoint(prim) if prim.IsA(UsdPhysics.RevoluteJoint) else UsdPhysics.PrismaticJoint(prim)
        scale = math.pi / 180 if prim.IsA(UsdPhysics.RevoluteJoint) else 1.0
        kind = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
        lim = j.find("limit")
        if lim is None:
            lim = ET.SubElement(j, "limit")
        lim.set("lower", f"{uj.GetLowerLimitAttr().Get() * scale:.9g}")
        lim.set("upper", f"{uj.GetUpperLimitAttr().Get() * scale:.9g}")
        force = prim.GetAttribute(f"drive:{kind}:physics:maxForce").Get()
        vel = prim.GetAttribute("physxJoint:maxJointVelocity").Get()
        lim.set("effort", f"{force if force is not None else 0.0:.9g}")
        lim.set("velocity", f"{vel * scale if vel is not None else 0.0:.9g}")
    ET.indent(robot, "  ")
    header = ("<!-- Generated by g1_twist2_assets/tools/make_magicsim_assets.py (Pink IK model: no meshes). "
              "Do not edit. -->\n")
    out.write_text('<?xml version="1.0"?>\n' + header + ET.tostring(robot, encoding="unicode") + "\n")
    print(f"wrote {out}")
    return robot


def check_urdf(robot, usd, trials=20, seed=0):
    fk0 = urdf_fk(robot)
    to_g1 = usd.pose(f"{ROOT}/pelvis")
    errs = []
    for link, M in fk0.items():
        prim = usd.stage.GetPrimAtPath(f"{ROOT}/{link}")
        if prim and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            U = usd.pose(f"{ROOT}/{link}")
            A = to_g1 @ M
            errs.append((link, np.linalg.norm(A[:3, 3] - U[:3, 3]) * 1000, math.degrees(R.from_matrix(A[:3, :3].T @ U[:3, :3]).magnitude())))
    worst0 = max(max(e[1], e[2]) for e in errs)
    print(f"check URDF q=0: {len(errs)} links vs the G1 USD, max {max(e[1] for e in errs):.4f} mm {max(e[2] for e in errs):.4f} deg")
    for e in errs:
        if e[1] > TOL_MM or e[2] > TOL_DEG:
            print(f"   MISMATCH {e}")
    rng = np.random.default_rng(seed)
    moving = [j for j in robot.findall("joint") if j.get("type") in ("revolute", "prismatic") and j.get("name") not in ("joint_yaw", "joint_pitch")]
    worst = 0.0
    for _ in range(trials):
        q = {j.get("name"): rng.uniform(float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))) for j in moving}
        fk = urdf_fk(robot, q)
        for side in SIDES:
            for link in ("ankle_roll_link", "hand_palm_link", "dex1_finger1_3_link", "dex1_finger2_3_link", "hand_d405_camera_link"):
                A, U = to_g1 @ fk[f"{side}_{link}"], usd_chain_pose(usd, f"{side}_{link}", q)
                worst = max(worst, np.linalg.norm(A[:3, 3] - U[:3, 3]) * 1000, math.degrees(R.from_matrix(A[:3, :3].T @ U[:3, :3]).magnitude()))
    print(f"check URDF random q ({trials} configurations, legs/waist/arms/fingers): max {worst:.4f} (mm or deg)")
    return worst0 <= TOL_MM and worst <= TOL_MM


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--magicsim", type=Path, default=REPO.parent / "MagicSim")
    ap.add_argument("--name", default="g1_dex1_head")
    a = ap.parse_args()
    robots = a.magicsim / "Assets" / "Robots"
    usd_out, urdf_out = robots / f"{a.name}.usd", robots / "URDF" / f"{a.name}.urdf"
    make_usd(usd_out, f"textures/{a.name}")
    ok_usd = check_usd(usd_out)
    usd = G1Usd(G1_USD)
    robot = make_urdf(a.magicsim, urdf_out, usd)
    ok_urdf = check_urdf(robot, usd)
    if not (ok_usd and ok_urdf):
        sys.exit("check FAILED")
    print("checks passed")


if __name__ == "__main__":
    main()
