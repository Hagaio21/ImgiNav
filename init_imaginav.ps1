# Create folders
New-Item -ItemType Directory -Force -Path "datasets"
New-Item -ItemType Directory -Force -Path "scripts"
New-Item -ItemType Directory -Force -Path "notebooks"
New-Item -ItemType Directory -Force -Path "experiments"
New-Item -ItemType Directory -Force -Path "core\planning"
New-Item -ItemType Directory -Force -Path "core\perception"
New-Item -ItemType Directory -Force -Path "core\mapping"
New-Item -ItemType Directory -Force -Path "config"

# Create empty config files
@(
  "layout_diffusion.yaml",
  "language_grounding.yaml",
  "planner.yaml",
  "mapping.yaml"
) | ForEach-Object {
  New-Item -ItemType File -Force -Path "config\$_"
}

# Create .gitignore
@"
__pycache__/
*.pyc
*.pt
*.ckpt
*.npz
checkpoints/
results/
temp/
logs/
notebooks/__pycache__/
"@ | Set-Content ".gitignore"

# Create requirements.txt
@"
torch
diffusers
transformers
omegaconf
networkx
matplotlib
opencv-python
tqdm
numpy
scipy
"@ | Set-Content "requirements.txt"

# Create README.md
@"
# ImagiNav

**ImagiNav** is a robot navigation framework that interprets natural language instructions to generate imagined spatial layouts using learned priors. The robot plans and navigates within this imagined space, updating its beliefs through sensor feedback.

## Setup
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Components
- Imagination (diffusion/CVAE)
- Probabilistic semantic mapping
- Classical planning (e.g. A*)
- Language grounding
"@ | Set-Content "README.md"

# Create dummy scripts
@(
  "train_imagination.py",
  "gen_layout_from_text.py",
  "plan_in_imagined_layout.py",
  "update_beliefs_from_observation.py"
) | ForEach-Object {
  Set-Content "scripts\$_" "# Placeholder for $_`npass"
}
