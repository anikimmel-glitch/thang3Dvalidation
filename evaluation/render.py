"""Bounded rendering worker; uses the authors' view conventions."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def render(kind, mesh_path, output):
    if kind == "cadrille":
        import trimesh
        import open3d as o3d
        from vendor.cadrille.render import mesh_to_image
        mesh = trimesh.load(mesh_path, force="mesh")
        vertices = np.asarray(mesh.vertices) / 2 + .5
        obj = o3d.geometry.TriangleMesh()
        obj.vertices = o3d.utility.Vector3dVector(vertices)
        obj.triangles = o3d.utility.Vector3iVector(mesh.faces)
        obj.paint_uniform_color(np.array([255, 255, 136]) / 255)
        obj.compute_vertex_normals()
        views = [mesh_to_image(obj, camera_distance=-.9, front=f, img_size=128)
                 for f in ([1,1,1], [-1,-1,-1], [-1,1,-1], [1,-1,1])]
        views = [np.asarray(ImageOps.expand(x, border=3, fill="black")) for x in views]
        image = Image.fromarray(np.vstack((np.hstack(views[:2]), np.hstack(views[2:]))))
    elif kind == "cadreasoner":
        from vendor.cadreasoner.visualization import Plotter
        plotter = Plotter(scale_gt=True, scale_pred=False)
        image = plotter.get_img(str(mesh_path), None)
        plotter.plotter.close()
        plotter.iso_plotter.close()
    else:
        raise ValueError(kind)
    pixels = np.asarray(image)
    if not np.isfinite(pixels).all() or np.std(pixels) < 2:
        raise RuntimeError("Blank or invalid render")
    image.save(output)


if __name__ == "__main__":
    render(*sys.argv[1:])
