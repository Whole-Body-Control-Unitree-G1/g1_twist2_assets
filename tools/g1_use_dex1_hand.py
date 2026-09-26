#!/usr/bin/env python3
"""Point the G1's Dex1-1 hand geometry at dex1_d405_hand/dex1_1_d405_{right,left}.usd.

For each <side>_<link> hand body in the G1 USD this replaces the inline `visuals`, `collisions`
(and on the camera link the `<side>_D405` camera) with references to the same prims in the per-side
hand file. The D405 mount and camera links (<side>_hand_d405_mount_link / _camera_link), which the
original G1 did not have, are created with their fixed joints, as in the hand file and the URDF.
Hand names follow MagicSim's g1_dex1 (<side>_hand_palm_link, <side>_dex1_finger1_1_link, ...,
<side>_dex1_finger_joint_1); a G1 with the older names (<side>_hand_base_link, <side>_hand_Link1_1,
<side>_hand_Joint1_1, ...) is renamed first, with every path that points at those prims. The hand file's materials are referenced as <root>/<side>_hand_Looks and bound to the
visual meshes, so geometry, camera and look all come from the hand builder. Link prims, joints,
drives and masses of the G1 are left as they are, so prim paths, joint names and control
behaviour do not change. Re-running is safe.

Rebuild the hand files first with tools/build_dex1_1_usd.py.
Usage: python tools/g1_use_dex1_hand.py [--g1 g1_assembled/g1_29dof_with_dex1_base_fix1.usd]
"""
import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

REPO = Path(__file__).resolve().parent.parent
ROOT = "/g1_29dof_with_hand_rev_1_0"
BASE_LINK = "hand_palm_link"
LINKS = [BASE_LINK, "dex1_finger1_1_link", "dex1_finger1_2_link", "dex1_finger1_3_link", "dex1_finger2_1_link",
         "dex1_finger2_2_link", "dex1_finger2_3_link", "hand_d405_mount_link", "hand_d405_camera_link"]
# Links the original G1 lacks -> the hand-file fixed joint that attaches each one
NEW_LINKS = {"hand_d405_mount_link": "hand_d405_mount_joint", "hand_d405_camera_link": "hand_d405_camera_joint"}
CAMERA_LINK = "hand_d405_camera_link"
BODY_MATERIAL, PAD_MATERIAL = "body_anodised", "pad_rubber"  # materials in the hand file's Looks
PAD_LINKS = {"dex1_finger1_3_link", "dex1_finger2_3_link"}
# Older G1 hand names: <side>_hand_<old> -> <side>_<new> (links under ROOT, joints under ROOT/joints)
LEGACY_LINKS = {
    "base_link": BASE_LINK, "Link1_1": "dex1_finger1_1_link", "Link1_2": "dex1_finger1_2_link",
    "Link1_3": "dex1_finger1_3_link", "Link2_1": "dex1_finger2_1_link", "Link2_2": "dex1_finger2_2_link",
    "Link2_3": "dex1_finger2_3_link", "d405_mount": "hand_d405_mount_link", "d405_camera": "hand_d405_camera_link",
}
LEGACY_JOINTS = {
    "Joint1_1": "dex1_finger_joint_1", "Joint2_1": "dex1_finger_joint_2", "Joint1_2": "dex1_finger1_2_joint",
    "Joint1_3": "dex1_finger1_3_joint", "Joint2_2": "dex1_finger2_2_joint", "Joint2_3": "dex1_finger2_3_joint",
    "base_to_d405_mount": "hand_d405_mount_joint", "d405_mount_to_camera": "hand_d405_camera_joint",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g1", type=Path, default=REPO / "g1_assembled" / "g1_29dof_with_dex1_base_fix1.usd")
    a = ap.parse_args()
    layer = Sdf.Layer.FindOrOpen(str(a.g1))
    rename_legacy_hand_prims(layer)
    g1 = Usd.Stage.Open(layer)

    for side in ("right", "left"):
        hand_file = REPO / "dex1_d405_hand" / f"dex1_1_d405_{side}.usd"
        assert hand_file.exists(), f"missing {hand_file}; run tools/build_dex1_1_usd.py"
        asset = Path("..") / hand_file.relative_to(REPO)  # relative to g1_assembled/
        hand_root = f"/dex1_1_d405_{side}"
        hand = Usd.Stage.Open(str(hand_file))
        base_local = np.array(UsdGeom.Xformable(g1.GetPrimAtPath(f"{ROOT}/{side}_{BASE_LINK}")).GetLocalTransformation()).T
        for link, joint in NEW_LINKS.items():
            add_fixed_link(layer, hand, hand_root, side, link, joint, base_local)
        old_cam = layer.GetPrimAtPath(f"{ROOT}/{side}_{BASE_LINK}/{side}_D405")  # camera used to be on the base
        if old_cam:
            del layer.GetPrimAtPath(f"{ROOT}/{side}_{BASE_LINK}").nameChildren[f"{side}_D405"]
        for link in LINKS:
            body = layer.GetPrimAtPath(f"{ROOT}/{side}_{link}")
            assert body, f"G1 has no {side}_{link}"
            children = ["visuals", "collisions"] + ([f"{side}_D405"] if link == CAMERA_LINK else [])
            for name in children:
                if name in body.nameChildren:
                    del body.nameChildren[name]
                spec = Sdf.PrimSpec(body, name, Sdf.SpecifierDef, "Camera" if name.endswith("D405") else "Xform")
                spec.referenceList.Prepend(Sdf.Reference(str(asset), f"{hand_root}/{side}_{link}/{name}"))
        looks = f"{ROOT}/{side}_hand_Looks"
        if layer.GetPrimAtPath(looks):
            del layer.GetPrimAtPath(ROOT).nameChildren[f"{side}_hand_Looks"]
        spec = Sdf.PrimSpec(layer.GetPrimAtPath(ROOT), f"{side}_hand_Looks", Sdf.SpecifierDef, "Scope")
        spec.referenceList.Prepend(Sdf.Reference(str(asset), f"{hand_root}/Looks"))
        # Material overrides need the composed prims, so they are authored in a second pass below.

    layer.Save()
    stage = Usd.Stage.Open(str(a.g1))
    for side in ("right", "left"):
        for link in LINKS:
            vis = stage.GetPrimAtPath(f"{ROOT}/{side}_{link}/visuals")
            material = f"{ROOT}/{side}_hand_Looks/" + (PAD_MATERIAL if link in PAD_LINKS else BODY_MATERIAL)
            for p in Usd.PrimRange(vis):
                if p.IsA(UsdGeom.Mesh):
                    over = Sdf.CreatePrimInLayer(layer, p.GetPath())
                    over.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MaterialBindingAPI"]))
                    rel = Sdf.RelationshipSpec(over, "material:binding", custom=False) if "material:binding" not in over.relationships else over.relationships["material:binding"]
                    rel.targetPathList.explicitItems = [Sdf.Path(material)]
    layer.Save()
    print(f"updated {a.g1}")


def add_fixed_link(layer, hand, hand_root, side, link, joint, base_local):
    """Create <side>_<link> (rigid body at the hand-file pose) and its fixed joint in the G1 layer."""
    hand_link = hand.GetPrimAtPath(f"{hand_root}/{side}_{link}")
    local = base_local @ np.array(UsdGeom.Xformable(hand_link).GetLocalTransformation()).T  # the palm link is the hand root
    name = f"{side}_{link}"
    spec = layer.GetPrimAtPath(f"{ROOT}/{name}") or Sdf.PrimSpec(layer.GetPrimAtPath(ROOT), name, Sdf.SpecifierDef, "Xform")
    spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(["PhysicsRigidBodyAPI", "PhysicsMassAPI"]))
    q = Gf.Matrix4d(local.T.tolist()).ExtractRotationQuat()
    values = [
        ("physics:mass", Sdf.ValueTypeNames.Float, hand_link.GetAttribute("physics:mass").Get()),
        ("xformOp:translate", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*map(float, local[:3, 3]))),
        ("xformOp:orient", Sdf.ValueTypeNames.Quatd, Gf.Quatd(q.GetReal(), q.GetImaginary())),
        ("xformOp:scale", Sdf.ValueTypeNames.Double3, Gf.Vec3d(1, 1, 1)),
        ("xformOpOrder", Sdf.ValueTypeNames.TokenArray, ["xformOp:translate", "xformOp:orient", "xformOp:scale"]),
    ]
    for attr, typ, value in values:
        (spec.attributes.get(attr) or Sdf.AttributeSpec(spec, attr, typ, variability=Sdf.VariabilityUniform if attr == "xformOpOrder" else Sdf.VariabilityVarying)).default = value

    hj = UsdPhysics.FixedJoint(hand.GetPrimAtPath(f"{hand_root}/joints/{side}_{joint}"))
    parent = hj.GetBody0Rel().GetTargets()[0].name  # e.g. right_hand_palm_link
    joints = layer.GetPrimAtPath(f"{ROOT}/joints")
    jname = f"{side}_{joint}"
    if jname in joints.nameChildren:
        del joints.nameChildren[jname]
    js = Sdf.PrimSpec(joints, jname, Sdf.SpecifierDef, "PhysicsFixedJoint")
    for rel, target in (("physics:body0", f"{ROOT}/{parent}"), ("physics:body1", f"{ROOT}/{name}")):
        Sdf.RelationshipSpec(js, rel, custom=False).targetPathList.explicitItems = [Sdf.Path(target)]
    for attr, typ in (("physics:localPos0", Sdf.ValueTypeNames.Point3f), ("physics:localRot0", Sdf.ValueTypeNames.Quatf),
                      ("physics:localPos1", Sdf.ValueTypeNames.Point3f), ("physics:localRot1", Sdf.ValueTypeNames.Quatf)):
        Sdf.AttributeSpec(js, attr, typ).default = hj.GetPrim().GetAttribute(attr).Get()


def rename_legacy_hand_prims(layer):
    """Rename older hand prims (<side>_hand_Link1_1, ...) to the current names, and repoint every
    relationship target, attribute connection and internal reference in the layer that used them."""
    moves = []
    for side in ("right", "left"):
        moves += [(f"{ROOT}/{side}_hand_{o}", f"{ROOT}/{side}_{n}") for o, n in LEGACY_LINKS.items()]
        moves += [(f"{ROOT}/joints/{side}_hand_{o}", f"{ROOT}/joints/{side}_{n}") for o, n in LEGACY_JOINTS.items()]
    moves = [(Sdf.Path(o), Sdf.Path(n)) for o, n in moves if layer.GetPrimAtPath(o)]
    if not moves:
        return
    for o, n in moves:
        assert not layer.GetPrimAtPath(n), f"both {o} and {n} exist"
    edit = Sdf.BatchNamespaceEdit()
    for o, n in moves:
        edit.Add(o, n)
    assert layer.CanApply(edit), f"cannot rename: {layer.CanApply(edit)}"
    layer.Apply(edit)

    def remap(path):
        for o, n in moves:
            if path.HasPrefix(o):
                return path.ReplacePrefix(o, n)
        return path

    def remap_list_op(list_op, make):
        changed = False
        for field in ("explicitItems", "addedItems", "prependedItems", "appendedItems", "deletedItems", "orderedItems"):
            items = list(getattr(list_op, field))
            new = [make(i) for i in items]
            if new != items:
                setattr(list_op, field, new)
                changed = True
        return changed

    def visit(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.RelationshipSpec):
            remap_list_op(spec.targetPathList, remap)
        elif isinstance(spec, Sdf.AttributeSpec):
            remap_list_op(spec.connectionPathList, remap)
        elif isinstance(spec, Sdf.PrimSpec):
            internal = lambda r: type(r)(r.assetPath, remap(r.primPath), r.layerOffset, r.customData) \
                if not r.assetPath and not r.primPath.isEmpty else r
            remap_list_op(spec.referenceList, internal)
            remap_list_op(spec.inheritPathList, remap)
            remap_list_op(spec.specializesList, remap)
    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    print(f"renamed {len(moves)} legacy hand prims")


if __name__ == "__main__":
    main()
