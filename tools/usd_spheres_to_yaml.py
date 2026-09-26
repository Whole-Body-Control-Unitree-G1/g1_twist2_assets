"""Read collision spheres from a sphere-editing USD and write the cuRobo/BODex collision_spheres YAML.

Usage: python tools/usd_spheres_to_yaml.py spheres.usd dex1_d405_hand/dex1_1_d405_spheres.yml [--fallback old.yml]

A link's spheres are the Sphere prims inside the group `spheres_<link>` (as written by
tools/yaml_spheres_to_usd.py), expressed in that group's frame = the link frame. Without such a group
(older files), the Sphere prims anywhere below the link prim <root>/<link> are used, in the link prim's
frame. Scale is applied to the radius. Links with no spheres keep those from --fallback.
"""
import argparse
import numpy as np
import yaml
from pxr import Usd, UsdGeom

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("usd"); ap.add_argument("out"); ap.add_argument("--fallback")
ap.add_argument("--root", default="/dex1_1_d405")
a = ap.parse_args()
ORDER = ["hand_palm_link", "hand_d405_mount_link", "hand_d405_camera_link", "dex1_finger1_1_link", "dex1_finger1_2_link",
         "dex1_finger1_3_link", "dex1_finger2_1_link", "dex1_finger2_2_link", "dex1_finger2_3_link"]
# Sphere-editing files made before the links were renamed: old group name -> link
LEGACY = {"base_link": "hand_palm_link", "d405_mount": "hand_d405_mount_link", "d405_camera": "hand_d405_camera_link",
          **{f"Link{f}_{i}": f"dex1_finger{f}_{i}_link" for f in (1, 2) for i in (1, 2, 3)}}
fallback = yaml.safe_load(open(a.fallback))["collision_spheres"] if a.fallback else {}
st = Usd.Stage.Open(a.usd); xc = UsdGeom.XformCache(0)
world = lambda p: np.array(xc.GetLocalToWorldTransform(p)).T
groups = {p.GetName()[len("spheres_"):]: p for p in st.Traverse() if p.GetName().startswith("spheres_")}
groups = {LEGACY.get(k, k): p for k, p in groups.items()}


def spheres_under(prim, frame, skip_groups):
    out = []
    inv = np.linalg.inv(world(frame))
    it = iter(Usd.PrimRange(prim))
    for p in it:
        if skip_groups and p != prim and p.GetName().startswith("spheres_"):
            it.PruneChildren(); continue
        if not p.IsA(UsdGeom.Sphere): continue
        rel = inv @ world(p)
        r = UsdGeom.Sphere(p).GetRadiusAttr().Get() * np.linalg.norm(rel[:3, :3], axis=0).max()
        out.append({"center": [round(float(v), 4) for v in rel[:3, 3]], "radius": round(float(r), 4)})
    return out


out, src = {}, {}
for link in ORDER:
    if link in groups:
        sph = spheres_under(groups[link], groups[link], skip_groups=False)
    elif (lp := st.GetPrimAtPath(f"{a.root}/{link}")):
        sph = spheres_under(lp, lp, skip_groups=True)
    else:
        sph = []
    out[link], src[link] = (sph, "usd") if sph else (fallback.get(link, []), "fallback")
with open(a.out, "w") as f:
    f.write("collision_spheres:\n")
    for link in ORDER:
        f.write(f"  {link}:  # {src[link]}, {len(out[link])} spheres\n")
        for s in out[link]:
            c = s["center"]; f.write(f'    - "center": [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]\n      "radius": {s["radius"]:.4f}\n')
for link in ORDER: print(f"{link:12s} {src[link]:8s} {len(out[link])}")
