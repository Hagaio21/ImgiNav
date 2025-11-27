# PowerShell script to visualize taxonomy colors
# Uses imginav conda environment

Write-Host "Visualizing taxonomy colors..." -ForegroundColor Cyan

# Activate conda environment and run visualization
conda run -n imginav python data_preparation_v2/visualize_taxonomy_colors.py `
    --taxonomy "data_preparation_v2/taxonomy.json" `
    --mode all `
    --output-dir "data_preparation_v2/taxonomy_visualizations"

Write-Host "`nDone! Check data_preparation_v2/taxonomy_visualizations/ for output images." -ForegroundColor Green

