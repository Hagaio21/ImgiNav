#!/bin/bash
# Simple script to move all images back from to_review folder to layout_new folder

set -euo pipefail

# Configuration
LAYOUT_DIR="/work3/s233249/ImgiNav/datasets/layout_new"
TO_REVIEW_DIR="${LAYOUT_DIR}/to_review"

echo "Moving all images back from to_review folder to layout_new"
echo "Layout directory: ${LAYOUT_DIR}"
echo "To review directory: ${TO_REVIEW_DIR}"

# Verify layout directory exists
if [ ! -d "${LAYOUT_DIR}" ]; then
  echo "ERROR: Layout directory not found: ${LAYOUT_DIR}"
  exit 1
fi

# Check if to_review directory exists
if [ ! -d "${TO_REVIEW_DIR}" ]; then
  echo "WARNING: to_review directory not found: ${TO_REVIEW_DIR}"
  echo "Nothing to move back."
  exit 0
fi

# Count files in to_review
FILE_COUNT=$(find "${TO_REVIEW_DIR}" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
echo "Found ${FILE_COUNT} image files in ${TO_REVIEW_DIR}"

if [ ${FILE_COUNT} -eq 0 ]; then
  echo "No image files found in ${TO_REVIEW_DIR}"
  exit 0
fi

# Move all image files back
MOVED=0
SKIPPED=0

for img_file in "${TO_REVIEW_DIR}"/*.{png,jpg,jpeg} 2>/dev/null; do
  # Check if file exists (glob might not match)
  if [ ! -f "${img_file}" ]; then
    continue
  fi
  
  filename=$(basename "${img_file}")
  dest_path="${LAYOUT_DIR}/${filename}"
  
  # Check if destination already exists
  if [ -f "${dest_path}" ]; then
    echo "WARNING: Destination already exists: ${filename}, skipping"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  
  # Move file
  mv "${img_file}" "${dest_path}"
  MOVED=$((MOVED + 1))
done

echo ""
echo "=========================================="
echo "Move back completed"
echo "=========================================="
echo "Moved: ${MOVED} images"
if [ ${SKIPPED} -gt 0 ]; then
  echo "Skipped: ${SKIPPED} images (already exist)"
fi
echo "=========================================="

if [ ${MOVED} -gt 0 ]; then
  echo ""
  echo "All images moved back successfully"
  echo "You can now delete the to_review folder if desired:"
  echo "  rm -r ${TO_REVIEW_DIR}"
fi

