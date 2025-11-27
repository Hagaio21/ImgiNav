# PowerShell script to run pipeline on 10 scenes from valid_scenes.txt

# Step 1: Filter scenes
Write-Host "Step 1: Filtering 10 scenes..." -ForegroundColor Cyan
python -c "import shutil; from pathlib import Path; d1=Path('D:/datasets/3D-FRONT_FUTURE/3D-FRONT'); d2=Path('test_dataset/filtered_scenes'); d2.mkdir(exist_ok=True); ids=[l.strip() for l in open('test_dataset/valid_scenes.txt')][:10]; copied=[shutil.copy2(d1/f'{i}.json', d2/f'{i}.json') for i in ids if (d1/f'{i}.json').exists()]; print(f'Copied {len(copied)} scenes')"

# Step 2: Stage 1 - Reconstruct Geometry
Write-Host "`nStep 2: Stage 1 - Reconstructing geometry..." -ForegroundColor Cyan
python data_preparation_v2/stage1_reconstruct_geometry.py `
    --scenes-dir "test_dataset/filtered_scenes" `
    --model-dir "D:\datasets\3D-FRONT_FUTURE\3D-FUTURE-model\3D-FUTURE-model" `
    --model-info "D:\datasets\3D-FRONT_FUTURE\3D-FUTURE-model\model_info.json" `
    --taxonomy "test_dataset\taxonomy\taxonomy.json" `
    --output-dir "test_dataset\geometry" `
    --texture-dir "D:\datasets\3D-FRONT_FUTURE\3D-FRONT-texture"

# Step 3: Stage 2 - Compile Metadata
Write-Host "`nStep 3: Stage 2 - Compiling metadata..." -ForegroundColor Cyan
python data_preparation_v2/stage2_compile_metadata.py `
    --scenes-dir "test_dataset/filtered_scenes" `
    --model-info "D:\datasets\3D-FRONT_FUTURE\3D-FUTURE-model\model_info.json" `
    --taxonomy "test_dataset\taxonomy\taxonomy.json" `
    --output-dir "test_dataset\metadata"

# Step 4: Stage 3 - Render Layouts
Write-Host "`nStep 4: Stage 3 - Rendering layouts..." -ForegroundColor Cyan
python data_preparation_v2/stage3_render_layouts.py `
    --geometry-dir "test_dataset\geometry" `
    --metadata-dir "test_dataset\metadata" `
    --taxonomy "test_dataset\taxonomy\taxonomy.json" `
    --output-dir "test_dataset\layouts"

# Step 5: Stage 4 - Render POVs
Write-Host "`nStep 5: Stage 4 - Rendering POVs..." -ForegroundColor Cyan
python data_preparation_v2/stage4_render_povs.py `
    --geometry-dir "test_dataset\geometry" `
    --metadata-dir "test_dataset\metadata" `
    --output-dir "test_dataset\povs" `
    --width 500 `
    --height 500

# Step 6: Stage 5 - Build Graphs
Write-Host "`nStep 6: Stage 5 - Building graphs..." -ForegroundColor Cyan
python data_preparation_v2/stage5_build_graphs.py `
    --metadata-dir "test_dataset\metadata" `
    --output-dir "test_dataset\graphs"

# Step 7: Stage 6 - Generate Manifests
Write-Host "`nStep 7: Stage 6 - Generating manifests..." -ForegroundColor Cyan
python data_preparation_v2/stage6_generate_manifests.py `
    --dataset-root "test_dataset" `
    --output-dir "test_dataset\manifests"

Write-Host "`nPipeline complete!" -ForegroundColor Green

