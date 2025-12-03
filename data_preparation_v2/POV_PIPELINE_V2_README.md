# POV-Normalized Pipeline v2

## Overview

This update consolidates POV image rendering, layout rotation, and graph generation into a single stage (stage4_render_povs_v2.py).

**Key principle**: Layouts are rendered ONCE in stage 3, then ROTATED (not re-rendered) in stage 4 to match each POV orientation.

## What Changed

### Before (Multiple Stages)
```
stage3 → Render layouts (one per room)
stage4 → Render POV images
stage5 → Generate graphs (one per room, world-oriented)
```

### After (Consolidated)
```
stage3 → Render layouts (unchanged - one per room)
stage4_v2 → Render POVs + Rotate layouts + Generate POV graphs (all in one)
```

## File Structure

```
dataset/
├── layouts/                    ← Stage 3 output (original layouts)
│   ├── tex/{scene}_{room}_tex_layout.png
│   └── seg/{scene}_{room}_seg_layout.png
│
├── layouts_pov/                ← Stage 4 v2 output (rotated layouts)
│   ├── tex/{scene}_{room}_{pov}_tex_layout.png
│   └── seg/{scene}_{room}_{pov}_seg_layout.png
│
├── povs/                       ← Stage 4 v2 output (POV images)
│   ├── tex/{scene}_{room}_{pov}_tex_pov.png
│   ├── seg/{scene}_{room}_{pov}_seg_pov.png
│   └── pov_info.json           ← All POV metadata
│
└── graphs/                     ← Stage 4 v2 output (POV-normalized)
    ├── jsons/{scene}_{room}_{pov}_room_graph.json
    └── texts/{scene}_{room}_{pov}_room_description.txt
```

## Camera Improvements

| Parameter | Old | New |
|-----------|-----|-----|
| Position | Door center | 0.8m outside door |
| FOV | 60° | 80° |
| Height | 1.5m | 1.6m |
| Look-at | Room center | Furniture centroid |
| Tilt | None | 5° downward |

## Graph Improvements

### Object Naming
- Old: `chair_1`, `chair_2`, `table_1`
- New: `the left chair`, `the right chair`, `the table ahead`

### Spatial Relations
- Old: World-oriented (`north of`, `east of`)
- New: POV-relative (`to your left`, `ahead of`)

### Empty Room Descriptions
Old:
```
You are standing at the doorway, looking into the Corridor.
The room appears to be empty.
```

New:
```
You are standing at the doorway, looking into the Corridor.
The room is empty. It is a small, long and narrow space.
There are doors ahead and behind you.
```

## Usage

### Run Stage 4 v2 (single machine)
```bash
python stage4_render_povs_v2.py \
    --dataset-root /path/to/dataset \
    --fov 80.0 \
    --step-back 0.8 \
    --camera-height 1.6
```

### Run on HPC
```bash
sbatch hpc_scripts/run_stage4_v2_array.sh
```

### Generate Manifest
```bash
python collect_manifest_pov_normalized.py \
    --dataset-root /path/to/dataset \
    --output-dir /path/to/dataset/manifests
```

## Files Included

| File | Description |
|------|-------------|
| `stage4_render_povs_v2.py` | Main script: POV rendering + layout rotation + graphs |
| `collect_manifest_pov_normalized.py` | Manifest collector (reads from layouts_pov/) |
| `hpc_scripts/run_stage4_v2_array.sh` | SLURM array job script |

## Benefits

1. **No duplicates**: Each (POV, layout, graph) triple is unique
2. **Efficient**: Layouts rotated via PIL, not re-rendered from 3D
3. **Better camera**: See more of the room from natural standing position
4. **Natural language**: Descriptions feel natural from viewer perspective
5. **Consolidated**: One stage instead of three
