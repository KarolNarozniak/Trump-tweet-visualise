# Trump Tweet Visualize Documentation

This project builds and explores a time-dependent mention network from the Trump tweet archive CSV.

Layer 1 delivers:

- deterministic preprocessing
- weekly mention and co-mention artifacts
- a stable global animation model for time progression
- an interactive Streamlit interface for exploration and export

Use this documentation when you need to:

- run the full build and app flow from scratch
- tune graph behavior through config and environment variables
- understand how artifacts are generated and consumed
- deploy both the app and docs endpoints

Core endpoints:

- Streamlit app: `http://localhost:3001`
- Docusaurus docs: `http://localhost:3002`
