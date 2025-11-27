# Filter 10 scenes
python -c "import shutil; from pathlib import Path; d1=Path('D:/datasets/3D-FRONT_FUTURE/3D-FRONT'); d2=Path('test_dataset/filtered_scenes'); d2.mkdir(exist_ok=True); ids=[l.strip() for l in open('test_dataset/valid_scenes.txt')][:10]; [shutil.copy2(d1/f'{i}.json', d2/f'{i}.json') for i in ids if (d1/f'{i}.json').exists()]"

# Stage 1
conda run -n imginav python data_preparation_v2/stage1_reconstruct_geometry.py --scenes-dir "test_dataset/filtered_scenes" --model-dir "D:\datasets\3D-FRONT_FUTURE\3D-FUTURE-model\3D-FUTURE-model" --model-info "D:\datasets\3D-FRONT_FUTURE\3D-FUTURE-model\model_info.json" --taxonomy "test_dataset\taxonomy\taxonomy.json" --output-dir "test_dataset\geometry" --texture-dir "D:\datasets\3D-FRONT_FUTURE\3D-FRONT-texture"

# Stage 2
conda run -n imginav python data_preparation_v2/stage2_compile_metadata.py --scenes-dir "test_dataset/filtered_scenes" --model-info "D:\datasets\3D-FRONT_FUTURE\3D-FUTURE-model\model_info.json" --taxonomy "test_dataset\taxonomy\taxonomy.json" --output-dir "test_dataset\metadata"

# Stage 3
conda run -n imginav python data_preparation_v2/stage3_render_layouts.py --geometry-dir "test_dataset\geometry" --metadata-dir "test_dataset\metadata" --output-dir "test_dataset\layouts"

# Stage 4
conda run -n imginav python data_preparation_v2/stage4_render_povs.py --geometry-dir "test_dataset\geometry" --metadata-dir "test_dataset\metadata" --output-dir "test_dataset\povs" --width 1280 --height 720

# Stage 5
conda run -n imginav python data_preparation_v2/stage5_build_graphs.py --metadata-dir "test_dataset\metadata" --output-dir "test_dataset\graphs"

# Stage 6
conda run -n imginav python data_preparation_v2/stage6_generate_manifests.py --dataset-root "test_dataset" --output-dir "test_dataset\manifests"

