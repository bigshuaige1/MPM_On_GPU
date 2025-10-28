from pxr import Usd, UsdGeom, Gf, Sdf, UsdUtils
import os

def load_obj_vertices(path):
    verts = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.strip().split()[:4]
                verts.append(Gf.Vec3f(float(x), float(y), float(z)))
    return verts

def load_obj_topology(path):
    faceVertexCounts = []
    faceVertexIndices = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('f '):
                parts = line.strip().split()[1:]
                idxs = []
                for p in parts:
                    i = p.split('/')[0]
                    idxs.append(int(i) - 1)  # OBJ 是 1-based
                faceVertexCounts.append(len(idxs))
                faceVertexIndices.extend(idxs)
    return faceVertexCounts, faceVertexIndices

def compute_extent(points):
    if not points:
        return [Gf.Vec3f(0), Gf.Vec3f(0)]
    mins = Gf.Vec3f(points[0])
    maxs = Gf.Vec3f(points[0])
    for p in points[1:]:
        mins[0] = min(mins[0], p[0]); mins[1] = min(mins[1], p[1]); mins[2] = min(mins[2], p[2])
        maxs[0] = max(maxs[0], p[0]); maxs[1] = max(maxs[1], p[1]); maxs[2] = max(maxs[2], p[2])
    return [mins, maxs]

# 配置
res_dir = r"D:\_Projects\ailab\Port_Kernel\MPM\res"
pattern = "res_{}.obj"  # 文件名模式
first = 1
last = 60
out_usd = r"D:\_Projects\ailab\Port_Kernel\MPM\anim.usd"
prim_path = "/Mesh"

# 读取首帧拓扑
first_obj = os.path.join(res_dir, pattern.format(first))
faceCounts, faceIndices = load_obj_topology(first_obj)

# 读取所有帧顶点，若点数不一致则走 Value Clips 兜底
all_points = []
expected_len = None
topology_consistent = True
inconsistent_frame = None
for f in range(first, last + 1):
    path = os.path.join(res_dir, pattern.format(f))
    pts = load_obj_vertices(path)
    if expected_len is None:
        expected_len = len(pts)
    if len(pts) != expected_len:
        topology_consistent = False
        inconsistent_frame = f
        break
    all_points.append(pts)

if topology_consistent:
    # 创建单一 USD，写入 timeSamples（拓扑一致）
    stage = Usd.Stage.CreateNew(out_usd)
    stage.SetStartTimeCode(first)
    stage.SetEndTimeCode(last)
    stage.SetTimeCodesPerSecond(24.0)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreateFaceVertexCountsAttr(faceCounts)
    mesh.CreateFaceVertexIndicesAttr(faceIndices)
    points_attr = mesh.CreatePointsAttr()
    extent_attr = mesh.CreateExtentAttr()

    for i, pts in enumerate(all_points):
        t = first + i
        points_attr.Set(pts, time=t)
        extent_attr.Set(compute_extent(pts), time=t)

    display_color = mesh.CreateDisplayColorAttr()
    display_color.Set([Gf.Vec3f(0.45, 0.75, 0.9)])

    stage.Save()
    print(f"Saved USD to: {out_usd}")
else:
    # 兜底：为每一帧生成一个独立 USD，并用 Clips 组装到 anim.usd
    print(f"检测到拓扑不一致，首次出现于第 {inconsistent_frame} 帧。启用 Value Clips 方案……")

    clips_dir = os.path.join(res_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    clip_asset_paths = []
    # 为每一帧写一个 clip USD（绝对或相对路径均可，优先相对 anim.usd 的相对路径）
    for f in range(first, last + 1):
        obj_path = os.path.join(res_dir, pattern.format(f))

        # 逐帧读取拓扑与点
        pts = load_obj_vertices(obj_path)
        fCounts, fIndices = load_obj_topology(obj_path)

        clip_name = f"clip_{f:04d}.usd"
        clip_path = os.path.join(clips_dir, clip_name)

        clip_stage = Usd.Stage.CreateNew(clip_path)
        clip_mesh = UsdGeom.Mesh.Define(clip_stage, prim_path)
        clip_mesh.CreateFaceVertexCountsAttr(fCounts)
        clip_mesh.CreateFaceVertexIndicesAttr(fIndices)
        clip_mesh.CreatePointsAttr(pts)
        clip_mesh.CreateExtentAttr(compute_extent(pts))
        # 一点颜色，方便预览
        clip_mesh.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.75, 0.9)])
        clip_stage.Save()

        # 使 anim.usd 使用相对路径引用 clips
        rel_path = os.path.relpath(clip_path, os.path.dirname(out_usd))
        # USD 要用正斜杠
        rel_path = rel_path.replace('\\', '/')
        clip_asset_paths.append(rel_path)

    # 构建 anim.usd 并设置 Clips 元数据
    stage = Usd.Stage.CreateNew(out_usd)
    stage.SetStartTimeCode(first)
    stage.SetEndTimeCode(last)
    stage.SetTimeCodesPerSecond(24.0)

    # 创建一个占位的 Mesh prim，Clips 会在其上随时间切换几何
    mesh = UsdGeom.Mesh.Define(stage, prim_path)

    # 使用 ClipsAPI 写入剪辑信息
    clips = Usd.ClipsAPI.Apply(mesh.GetPrim())
    # 设定剪辑里的目标 primPath（各 clip 文件内也在 /Mesh）
    clips.SetClipPrimPath(prim_path)
    # 资产路径列表（按帧顺序）
    clips.SetClipAssetPaths(clip_asset_paths)
    # 激活表：在每个时间点启用对应下标的 clip
    clip_active = [(float(t), int(t - first)) for t in range(first, last + 1)]
    clips.SetClipActive(clip_active)
    # 时间映射：让每个剪辑在其局部时间 t 对应到舞台时间 t
    clip_times = [(float(t), float(t)) for t in range(first, last + 1)]
    clips.SetClipTimes(clip_times)

    stage.Save()
    print(f"Saved clip-driven USD to: {out_usd}")