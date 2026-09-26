#!/usr/bin/env python3
"""Make a sphere-editing USD: the hand plus its collision spheres as sphere prims.

Open the result in Isaac Sim, move / resize / add / delete spheres, save, then write them back:
  python tools/yaml_spheres_to_usd.py                    # -> dex1_d405_hand/dex1_1_d405_spheres_edit.usd
  python tools/usd_spheres_to_yaml.py dex1_d405_hand/dex1_1_d405_spheres_edit.usd dex1_d405_hand/dex1_1_d405_spheres.yml

The hand is referenced (not copied) from dex1_1_d405.usd. Each URDF link's spheres sit in a group
`<link>/spheres_<link>` at that link's frame, so sphere centres stay in link frames. Add new spheres
inside the group of the link they belong to (duplicating an existing sphere is easiest).
Sphere 0 of dex1_finger1_3_link / dex1_finger2_3_link is the pad contact point (red); keep it first.

Requires usd-core, numpy, pyyaml.
"""
import argparse
from pathlib import Path

import numpy as np
import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

HAND_DIR = Path(__file__).resolve().parent.parent / "dex1_d405_hand"
ROOT = "/dex1_1_d405"
CONTACT_LINKS = {"dex1_finger1_3_link", "dex1_finger2_3_link"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", type=Path, default=HAND_DIR / "dex1_1_d405.usd")
    ap.add_argument("--spheres", type=Path, default=HAND_DIR / "dex1_1_d405_spheres.yml")
    ap.add_argument("--out", type=Path, default=HAND_DIR / "dex1_1_d405_spheres_edit.usd")
    a = ap.parse_args()

    spheres = yaml.safe_load(a.spheres.read_text())["collision_spheres"]
    hand = Usd.Stage.Open(str(a.hand))
    a.out.unlink(missing_ok=True)
    st = Usd.Stage.CreateNew(str(a.out))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = st.DefinePrim(ROOT, "Xform")
    root.GetReferences().AddReference(Path(__import__("os").path.relpath(a.hand, a.out.parent)).as_posix())
    st.SetDefaultPrim(root)

    def material(name, rgb, opacity):
        mat = UsdShade.Material.Define(st, f"/SphereLooks/{name}")
        sh = UsdShade.Shader.Define(st, f"/SphereLooks/{name}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    blue, red = material("sphere", (0.1, 0.45, 1.0), 0.35), material("contact", (1.0, 0.1, 0.1), 0.6)
    count = 0
    for link, sph in spheres.items():
        if not hand.GetPrimAtPath(f"{ROOT}/{link}"):
            raise SystemExit(f"{a.hand} has no link {link}")
        body, pose = link, Gf.Matrix4d(1.0)
        group = UsdGeom.Xform.Define(st, f"{ROOT}/{body}/spheres_{link}")
        group.AddTransformOp().Set(pose)
        for k, s in enumerate(sph):
            sp = UsdGeom.Sphere.Define(st, f"{group.GetPath()}/s{k:03d}")
            sp.CreateRadiusAttr(float(s["radius"]))
            sp.AddTranslateOp().Set(Gf.Vec3d(*map(float, s["center"])))
            UsdShade.MaterialBindingAPI.Apply(sp.GetPrim()).Bind(red if (k == 0 and link in CONTACT_LINKS) else blue)
            count += 1
    st.GetRootLayer().Save()
    print(f"wrote {a.out}: {count} spheres from {a.spheres}")


if __name__ == "__main__":
    main()
