# ImagiNav

**ImagiNav** is a robot navigation framework that interprets natural language instructions to generate imagined spatial layouts using learned priors. The robot plans and navigates within this imagined space, updating its beliefs through sensor feedback.

## Setup
\\\ash
pip install -r requirements.txt
\\\

## Components
- Imagination (diffusion/CVAE)
- Probabilistic semantic mapping
- Classical planning (e.g. A*)
- Language grounding
