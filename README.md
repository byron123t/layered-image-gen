# Layout optimization for reducing hallucination in Diffusion models

The goal of this work is to reduce (1. ) hallucination and (2.) token costs (secondary) in diffusion models via generating layouts. 

Layouts should have a stack or graph based layers.

1. Use a model, e.g., diffusion, etc. to generate a layout.
2. Use a deterministic methodology to construct a stack or layout.
3. Assign tags to each of the layers, e.g., text, graphic, etc. 
4. Iteratively use diffusion, text generation, etc. to refine image. 

Experiments to run:

1. Implement the methodology, and see if it capable of reducing hallucination.
2. Evaluate LLMs on image quality (whatever pertinent metric). 
3. insert more experiments here
