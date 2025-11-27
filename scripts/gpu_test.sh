#!/bin/bash
#BSUB -J rendering_check
#BSUB -q gpul40s
#BSUB -gpu "num=1"
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 00:30
#BSUB -o rendering_check_%J.out
#BSUB -e rendering_check_%J.err

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64
echo "===== HEADLESS RENDERING CHECK ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# ----------------------------------------------------------------------
# Activate conda environment FIRST
# ----------------------------------------------------------------------
echo "=== Activating Conda Environment ==="

# Try different conda locations
if [ -f "/work3/s233249/conda_envs/imginav/bin/activate" ]; then
    source /work3/s233249/conda_envs/imginav/bin/activate
    echo "Activated from /work3/s233249/conda_envs/imginav"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav
    echo "Activated via conda.sh"
else
    echo "ERROR: Cannot find conda environment"
    exit 1
fi

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ----------------------------------------------------------------------
# GPU Check
# ----------------------------------------------------------------------
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

# ----------------------------------------------------------------------
# Library Check
# ----------------------------------------------------------------------
echo "=== System Libraries ==="

echo "Checking for EGL..."
ldconfig -p 2>/dev/null | grep -i egl || echo "  libEGL: NOT FOUND"

echo ""
echo "Checking for OSMesa..."
ldconfig -p 2>/dev/null | grep -i osmesa || echo "  libOSMesa: NOT FOUND"

echo ""
echo "Checking for Mesa..."
ldconfig -p 2>/dev/null | grep -i "libGL" | head -3 || echo "  libGL: NOT FOUND"

echo ""

# ----------------------------------------------------------------------
# Python Package Check
# ----------------------------------------------------------------------
echo "=== Python Packages ==="
python << 'EOF'
packages = ['numpy', 'trimesh', 'PIL', 'pyrender', 'OpenGL', 'pyglet']
for pkg in packages:
    try:
        if pkg == 'PIL':
            import PIL
            print(f"  ✓ PIL (Pillow): {PIL.__version__}")
        elif pkg == 'OpenGL':
            import OpenGL
            print(f"  ✓ PyOpenGL: {OpenGL.__version__}")
        else:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {pkg}: {ver}")
    except ImportError as e:
        print(f"  ✗ {pkg}: NOT INSTALLED")
EOF
echo ""

# ----------------------------------------------------------------------
# Test 1: OSMesa Backend
# ----------------------------------------------------------------------
echo "=== Test 1: OSMesa Backend ==="
PYOPENGL_PLATFORM=osmesa python << 'EOF'
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

try:
    from OpenGL import osmesa
    print("  ✓ OpenGL.osmesa imported")
    
    ctx = osmesa.OSMesaCreateContext(osmesa.OSMESA_RGBA, None)
    if ctx:
        print("  ✓ OSMesa context created")
        osmesa.OSMesaDestroyContext(ctx)
    else:
        print("  ✗ OSMesa context creation failed")
except Exception as e:
    print(f"  ✗ OSMesa failed: {e}")
EOF
echo ""

# ----------------------------------------------------------------------
# Test 2: EGL Backend
# ----------------------------------------------------------------------
echo "=== Test 2: EGL Backend ==="
PYOPENGL_PLATFORM=egl python << 'EOF'
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

try:
    from OpenGL import EGL as egl
    print("  ✓ OpenGL.EGL imported")
    
    display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
    if display and display != egl.EGL_NO_DISPLAY:
        print("  ✓ EGL display obtained")
    else:
        print("  ✗ No EGL display available")
except Exception as e:
    print(f"  ✗ EGL failed: {e}")
EOF
echo ""

# ----------------------------------------------------------------------
# Test 3: PyRender with OSMesa
# ----------------------------------------------------------------------
echo "=== Test 3: PyRender + OSMesa ==="
PYOPENGL_PLATFORM=osmesa python << 'EOF'
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

try:
    import numpy as np
    import trimesh
    import pyrender
    
    # Create scene
    scene = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 1.0])
    box = trimesh.creation.box(extents=[1, 1, 1])
    mesh = pyrender.Mesh.from_trimesh(box)
    scene.add(mesh)
    
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 3
    scene.add(cam, pose=cam_pose)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=cam_pose)
    
    # Render
    from pyrender import OffscreenRenderer
    renderer = OffscreenRenderer(640, 480)
    color, depth = renderer.render(scene)
    renderer.delete()
    
    print(f"  ✓ OSMESA RENDERING WORKS! Shape: {color.shape}")
    
    # Save test
    from PIL import Image
    img = Image.fromarray(color)
    img.save('/tmp/test_osmesa.png')
    print("  ✓ Saved /tmp/test_osmesa.png")
    
except Exception as e:
    import traceback
    print(f"  ✗ OSMesa rendering failed: {e}")
    traceback.print_exc()
EOF
echo ""

# ----------------------------------------------------------------------
# Test 4: PyRender with EGL
# ----------------------------------------------------------------------
echo "=== Test 4: PyRender + EGL ==="
PYOPENGL_PLATFORM=egl python << 'EOF'
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

try:
    import numpy as np
    import trimesh
    import pyrender
    
    scene = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 1.0])
    box = trimesh.creation.box(extents=[1, 1, 1])
    mesh = pyrender.Mesh.from_trimesh(box)
    scene.add(mesh)
    
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 3
    scene.add(cam, pose=cam_pose)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=cam_pose)
    
    from pyrender import OffscreenRenderer
    renderer = OffscreenRenderer(640, 480)
    color, depth = renderer.render(scene)
    renderer.delete()
    
    print(f"  ✓ EGL RENDERING WORKS! Shape: {color.shape}")
    
    from PIL import Image
    img = Image.fromarray(color)
    img.save('/tmp/test_egl.png')
    print("  ✓ Saved /tmp/test_egl.png")
    
except Exception as e:
    import traceback
    print(f"  ✗ EGL rendering failed: {e}")
    traceback.print_exc()
EOF
echo ""

# ----------------------------------------------------------------------
# Test 5: Xvfb (Virtual Display)
# ----------------------------------------------------------------------
echo "=== Test 5: Xvfb (Virtual Display) ==="
if command -v xvfb-run &> /dev/null; then
    echo "  xvfb-run is available"
    xvfb-run -a python << 'EOF'
try:
    import numpy as np
    import trimesh
    import pyrender
    
    scene = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 1.0])
    box = trimesh.creation.box(extents=[1, 1, 1])
    mesh = pyrender.Mesh.from_trimesh(box)
    scene.add(mesh)
    
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 3
    scene.add(cam, pose=cam_pose)
    
    from pyrender import OffscreenRenderer
    renderer = OffscreenRenderer(640, 480)
    color, depth = renderer.render(scene)
    renderer.delete()
    
    print(f"  ✓ XVFB RENDERING WORKS! Shape: {color.shape}")
    
    from PIL import Image
    img = Image.fromarray(color)
    img.save('/tmp/test_xvfb.png')
    print("  ✓ Saved /tmp/test_xvfb.png")
    
except Exception as e:
    print(f"  ✗ Xvfb rendering failed: {e}")
EOF
else
    echo "  xvfb-run not available"
fi
echo ""

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
echo "===== SUMMARY ====="
echo ""
echo "Based on the results above, use one of these in your job scripts:"
echo ""
echo "  Option 1 (if OSMesa works):"
echo "    export PYOPENGL_PLATFORM=osmesa"
echo ""
echo "  Option 2 (if EGL works):"
echo "    export PYOPENGL_PLATFORM=egl"
echo ""
echo "  Option 3 (if Xvfb works):"
echo "    xvfb-run -a python your_script.py"
echo ""
echo "===== CHECK COMPLETE: $(date) ====="