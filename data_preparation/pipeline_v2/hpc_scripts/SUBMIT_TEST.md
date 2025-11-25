# Submitting Pipeline v2 Test

## Test Script: `run_render_test.sh`

This script will process **10 scenes** using **2 jobs** to test the pipeline.

## Submission

From the HPC login node:

```bash
cd /work3/s233249/ImgiNav/data_preparation/pipeline_v2/hpc_scripts
bsub < run_render_test.sh
```

## What to Check

After submission, monitor the jobs:

```bash
# Check job status
bjobs

# Check logs (replace JOBID with actual job ID)
tail -f logs/render_test_1.JOBID.out
tail -f logs/render_test_2.JOBID.out
```

## Expected Output

The test will create:
- `dataset_v2_test/layouts/rgb/` - RGB layout images
- `dataset_v2_test/layouts/seg/` - Segmentation layout images
- `dataset_v2_test/povs/rgb/` - RGB POV images (6 per scene)
- `dataset_v2_test/povs/seg/` - Segmentation POV images (6 per scene)
- `dataset_v2_test/graphs/` - Graph JSON, text, and visualization files

## Verify Results

Check that:
1. All 10 scenes were processed
2. Layout images exist (256x256 PNG)
3. POV images exist (6 per scene, 256x256 PNG)
4. Graph files were generated
5. No errors in log files

## If Test Succeeds

Once the test completes successfully, you can run the full pipeline:

```bash
bsub < run_render.sh
```

This will process all scenes using 20 jobs.

