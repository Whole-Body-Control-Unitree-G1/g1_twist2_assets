#!/usr/bin/env python3
"""Build articulated Dex1-1 + D405 USDs from dex1_d405_hand/dex1_1_d405.urdf.

Writes three files next to the URDF, all with identical geometry, frames and colliders:
  dex1_1_d405.usd        standalone hand, plain URDF names (hand_palm_link, dex1_finger_joint_1, ...), camera "D405"
  dex1_1_d405_right.usd  names prefixed "right_", camera "right_D405"  (referenced by the G1)
  dex1_1_d405_left.usd   names prefixed "left_",  camera "left_D405"   (referenced by the G1)
Link and joint names follow MagicSim's g1_dex1 (e.g. right_hand_palm_link, right_dex1_finger1_3_link,
right_dex1_finger_joint_1); the D405 links are <side>_hand_d405_{mount,camera}_link.

Layout: one rigid body per URDF link, as in the URDF and the G1 (g1_assembled/ references link contents 1:1):
  /<root>                        ArticulationRoot, self-collision off, fixed to the world
    /<p>hand_palm_link, /<p>hand_d405_mount_link, /<p>hand_d405_camera_link,
    /<p>dex1_finger1_1_link ... /<p>dex1_finger2_3_link
                                 rigid bodies with visuals/<link>/mesh and collisions/<link>/mesh
      /<p>hand_d405_camera_link/<cam>
                                 Camera at the URDF hand_d405_optical_frame
    /joints                      <p>dex1_finger_joint_1, <p>dex1_finger_joint_2 (prismatic), fixed joints
                                 (incl. hand_d405_mount_joint, hand_d405_camera_joint), root_joint
Colliders are built from the visual meshes: convex decomposition for the base, mount and finger
bodies (a single hull of dex1_finger*_2_link would cover the pad face), convex hulls for the rest.
Everything is black like the real gripper; the pads get a matte rubber OmniPBR material with the
1 mm diamond-knurl normal map (textures/pad_knurl_normal.png, made by tools/make_pad_knurl_texture.py)
on planar UVs over the gripping face.

Requires: usd-core, trimesh, numpy, scipy.   Usage: python tools/build_dex1_1_usd.py
"""
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.spatial.transform import Rotation as R

HAND_DIR = Path(__file__).resolve().parent.parent / "dex1_d405_hand"
BASE_LINK = "hand_palm_link"           # the Dex1-1 base (URDF root)
CAMERA_LINK = "hand_d405_camera_link"
OPTICAL_FRAME = "hand_d405_optical_frame"  # frame only (no geometry): the camera prim goes under CAMERA_LINK
FINGER_JOINTS = ("dex1_finger_joint_1", "dex1_finger_joint_2")  # prismatic; the second mimics the first
DECOMPOSE = {BASE_LINK, "hand_d405_mount_link", "dex1_finger1_2_link", "dex1_finger2_2_link"}
# Materials (OmniPBR), set from a real D405 frame of the gripper: pads and fingers both render at
# ~0.2x the (sRGB) pixel value of a light-grey floor = ~0.04x in linear light, so ~0.018 albedo for a
# ~0.45 floor; only soft edge highlights (no metal reflections).
# The G1 binds its hand meshes to these materials (referenced as <side>_hand_Looks).
BODY_COLOR = PAD_COLOR = (0.018, 0.018, 0.018)   # black anodised aluminium / black rubber (linear albedo)
BODY_METALLIC, BODY_ROUGHNESS = 0.0, 0.65      # anodising reads as a satin dielectric, not bare metal
PAD_LINKS = {"dex1_finger1_3_link", "dex1_finger2_3_link"}
# Pads: matte black rubber with a 1 mm diamond knurl (normal map from tools/make_pad_knurl_texture.py).
# The pad UVs are the gripping-face plane (link y, z) divided by one texture tile.
PAD_NORMAL_MAP = "./textures/pad_knurl_normal.png"   # relative to dex1_d405_hand/
PAD_TILE_M = 0.001 * math.sqrt(2)
PAD_ROUGHNESS = 0.8
DRIVE = dict(stiffness=100.0, damping=1.0, max_force=200.0)  # same gains as the G1 hand joints
# D405 colour stream calibration of our unit (realsense2_camera color/camera_info, 848x480, plumb_bob):
# HFOV 89.2 deg, VFOV 58.4 deg. Written as Isaac Sim's OpenCV pinhole lens model (exact cx, cy and
# distortion); focal length / apertures give the same fx, fy for tools that only read those.
CAMERA = dict(
    width=848, height=480,
    fx=429.968017578125, fy=429.45074462890625, cx=413.7232666015625, cy=243.4409637451172,
    k1=-0.055683065205812454, k2=0.05787091329693794, p1=0.0005697416490875185, p2=0.00048715356388129294,
    k3=-0.018814567476511,
    focal_length=1.88,  # mm; apertures follow from fx, fy
    clip=(0.02, 10.0),
)


def T(xyz, rpy):
    m = np.eye(4)
    m[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    m[:3, 3] = xyz
    return m


def parse_urdf(path):
    root = ET.parse(path).getroot()
    vec = lambda s, d: [float(v) for v in s.split()] if s else d
    links = {}
    for l in root.findall("link"):
        vis, col, mass = l.find("visual/geometry/mesh"), l.find("collision/geometry/mesh"), l.find("inertial/mass")
        links[l.get("name")] = dict(
            visual=vis.get("filename") if vis is not None else None,
            collision=col.get("filename") if col is not None else None,
            mass=float(mass.get("value")) if mass is not None else 0.0,
        )
    joints = {}
    for j in root.findall("joint"):
        o, lim = j.find("origin"), j.find("limit")
        joints[j.get("name")] = dict(
            type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            T=T(vec(o.get("xyz") if o is not None else None, [0, 0, 0]), vec(o.get("rpy") if o is not None else None, [0, 0, 0])),
            axis=vec(j.find("axis").get("xyz") if j.find("axis") is not None else None, [1, 0, 0]),
            lower=float(lim.get("lower")) if lim is not None else None,
            upper=float(lim.get("upper")) if lim is not None else None,
        )
    return links, joints


def quatf(Rm):
    x, y, z, w = R.from_matrix(Rm).as_quat()
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def matrix(M):
    return Gf.Matrix4d(np.asarray(M).T.tolist())  # USD uses the row-vector convention


def build(links, joints, mesh_root, out, prefix, cam_name, root_name, mimic):
    pose = {BASE_LINK: np.eye(4)}  # link poses in the base link at q = 0
    pending = dict(joints)
    while pending:
        for n, j in list(pending.items()):
            if j["parent"] in pose:
                pose[j["child"]] = pose[j["parent"]] @ j["T"]
                del pending[n]

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = f"/{root_name}"
    rp = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(rp)
    UsdPhysics.ArticulationRootAPI.Apply(rp)
    rp.AddAppliedSchema("PhysxArticulationAPI")
    rp.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)

    def omnipbr(name, color, roughness, metallic=0.0, normal_map=None):
        """OmniPBR (Isaac Sim's standard MDL material), like the G1's hand materials."""
        material = UsdShade.Material.Define(stage, f"{root}/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"{root}/Looks/{name}/Shader")
        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
        if normal_map:
            tex = shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset)
            tex.Set(Sdf.AssetPath(normal_map))
            tex.GetAttr().SetColorSpace("raw")
        for output in (material.CreateSurfaceOutput("mdl"), material.CreateDisplacementOutput("mdl"),
                       material.CreateVolumeOutput("mdl")):
            output.ConnectToSource(shader.ConnectableAPI(), "out")
        return material

    body_material = omnipbr("body_anodised", BODY_COLOR, BODY_ROUGHNESS, BODY_METALLIC)
    pad_material = omnipbr("pad_rubber", PAD_COLOR, PAD_ROUGHNESS, normal_map=PAD_NORMAL_MAP)

    cache = {}

    def mesh(path, filename, M=None, color=BODY_COLOR, collider=None, pad=False):
        f = mesh_root / filename
        tm = cache.setdefault(f, trimesh.load(f))
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(tm.vertices.astype(np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tm.faces)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(tm.faces.flatten().tolist()))
        m.CreateSubdivisionSchemeAttr("none")
        m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*tm.bounds[0]), Gf.Vec3f(*tm.bounds[1])]))
        if M is not None:
            m.AddTransformOp().Set(matrix(M))
        if collider:
            m.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(collider)
        else:
            m.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
            if pad:  # planar UVs on the gripping face (link y-z plane), one knurl tile per PAD_TILE_M
                st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
                st.Set(Vt.Vec2fArray.FromNumpy((tm.vertices[:, 1:3] / PAD_TILE_M).astype(np.float32)))
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(pad_material if pad else body_material)

    bodies = [l for l in links if l != OPTICAL_FRAME]
    for link in bodies:
        path = f"{root}/{prefix}{link}"
        x = UsdGeom.Xform.Define(stage, path)
        x.AddTransformOp().Set(matrix(pose[link]))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(links[link]["mass"])
        color = PAD_COLOR if link in PAD_LINKS else BODY_COLOR
        approx = "convexDecomposition" if link in DECOMPOSE else "convexHull"
        mesh(f"{path}/visuals/{link}/mesh", links[link]["visual"], color=color, pad=link in PAD_LINKS)
        mesh(f"{path}/collisions/{link}/mesh", links[link]["visual"], collider=approx)
    base = f"{root}/{prefix}{BASE_LINK}"

    # D405: ROS optical frame (+Z forward, +Y down) -> USD camera (looks down -Z, +Y up), on the camera link
    cam = UsdGeom.Camera.Define(stage, f"{root}/{prefix}{CAMERA_LINK}/{cam_name}")
    cam.AddTransformOp().Set(matrix(np.linalg.inv(pose[CAMERA_LINK]) @ pose[OPTICAL_FRAME] @ np.diag([1, -1, -1, 1])))
    c = CAMERA
    cam.CreateFocalLengthAttr(c["focal_length"])
    cam.CreateHorizontalApertureAttr(c["width"] * c["focal_length"] / c["fx"])
    cam.CreateVerticalApertureAttr(c["height"] * c["focal_length"] / c["fy"])
    cam.CreateClippingRangeAttr(Gf.Vec2f(*c["clip"]))
    lens = cam.GetPrim()
    lens.AddAppliedSchema("OmniLensDistortionOpenCvPinholeAPI")
    lens.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    lens.CreateAttribute("omni:lensdistortion:opencvPinhole:imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(c["width"], c["height"]))
    for key in ("cx", "cy", "fx", "fy", "k1", "k2", "p1", "p2", "k3"):
        lens.CreateAttribute(f"omni:lensdistortion:opencvPinhole:{key}", Sdf.ValueTypeNames.Float).Set(c[key])

    fj = UsdPhysics.FixedJoint.Define(stage, f"{root}/joints/root_joint")  # fixed base, like the G1 hands' wrist
    fj.CreateBody1Rel().SetTargets([base])
    for name, j in joints.items():
        if j["child"] == OPTICAL_FRAME:
            continue
        path = f"{root}/joints/{prefix}{name}"
        prismatic = j["type"] == "prismatic"
        jt = UsdPhysics.PrismaticJoint.Define(stage, path) if prismatic else UsdPhysics.FixedJoint.Define(stage, path)
        jt.CreateBody0Rel().SetTargets([f"{root}/{prefix}{j['parent']}"])
        jt.CreateBody1Rel().SetTargets([f"{root}/{prefix}{j['child']}"])
        # USD prismatic axes are +X/+Y/+Z; a URDF axis of -X becomes +X with both frames turned 180 deg about Z
        flip = np.eye(3)
        if prismatic:
            ax = np.array(j["axis"])
            assert np.count_nonzero(ax) == 1 and abs(ax[0]) == 1, f"{name}: only +/-X prismatic axes handled"
            if ax[0] < 0:
                flip = R.from_euler("z", math.pi).as_matrix()
        jt.CreateLocalPos0Attr(Gf.Vec3f(*map(float, j["T"][:3, 3])))
        jt.CreateLocalRot0Attr(quatf(j["T"][:3, :3] @ flip))
        jt.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        jt.CreateLocalRot1Attr(quatf(flip))
        if prismatic:
            jt.CreateAxisAttr("X")
            jt.CreateLowerLimitAttr(j["lower"])
            jt.CreateUpperLimitAttr(j["upper"])
            d = UsdPhysics.DriveAPI.Apply(jt.GetPrim(), "linear")
            d.CreateTypeAttr("force")
            d.CreateStiffnessAttr(DRIVE["stiffness"])
            d.CreateDampingAttr(DRIVE["damping"])
            d.CreateMaxForceAttr(DRIVE["max_force"])
            d.CreateTargetPositionAttr(0.0)

    if mimic:  # single-motor gripper: finger joint 2 follows joint 1 (PhysX: q + gearing * q_ref + offset = 0)
        m = stage.GetPrimAtPath(f"{root}/joints/{prefix}{FINGER_JOINTS[1]}")
        m.AddAppliedSchema("PhysxMimicJointAPI:transX")
        m.CreateRelationship("physxMimicJoint:transX:referenceJoint").SetTargets([f"{root}/joints/{prefix}{FINGER_JOINTS[0]}"])
        m.CreateAttribute("physxMimicJoint:transX:referenceJointAxis", Sdf.ValueTypeNames.Token).Set("transX")
        m.CreateAttribute("physxMimicJoint:transX:gearing", Sdf.ValueTypeNames.Float).Set(-1.0)
        m.CreateAttribute("physxMimicJoint:transX:offset", Sdf.ValueTypeNames.Float).Set(0.0)
    stage.GetRootLayer().Save()
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", type=Path, default=HAND_DIR / "dex1_1_d405.urdf")
    ap.add_argument("--out-dir", type=Path, default=HAND_DIR)
    ap.add_argument("--no-mimic", action="store_true", help="Leave finger joint 2 independent of joint 1.")
    a = ap.parse_args()
    links, joints = parse_urdf(a.urdf)
    for suffix, prefix, cam in (("", "", "D405"), ("_right", "right_", "right_D405"), ("_left", "left_", "left_D405")):
        out = a.out_dir / f"dex1_1_d405{suffix}.usd"
        out.unlink(missing_ok=True)
        build(links, joints, a.urdf.parent, out, prefix, cam, f"dex1_1_d405{suffix}", not a.no_mimic)


if __name__ == "__main__":
    main()
