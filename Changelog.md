## Major Milestones
Started tracking: 3/5/2026

#### Architecture Overhaul: Notebook to Production
3/5/2026
Transitioned the project from experimental Jupyter Notebooks into a mature software engineering directory structure.
- Data Flow: Logic is now strictly separated into extraction (loader.py), transformation (metrics.py), and orchestration (pipeline.py).
- Machine Learning: Clustered roles and similarity searches are now handled by a dedicated PlayerModels class, which automatically scrubs corrupted data (infinities) before scaling.
- Artifact Generation: Visualizations no longer block terminal execution and instead export PNGs files directly to the visuals/ folder.
- Reproducibility: A single python main.py command now runs the entire end-to-end process, backed by a requirements.txt with all the libraries

#### LLM Agent is Operational: Additional Features
3/6/2025
Set up a Streamlit Frontend and a FastAPI backend(using uvicorn)
- Registering Custom Tools for the LLM Agents(tools and functions that were already built, just need to register the tools to the model)
    - Custom graphs, KNN machine learning model
    - feature ranking, which stats lead to more goals, etc
- Adding RAG
    - add a Documentation Retriever
    - if the user asks what a stat means
    - create a metrics_definitions.txt file that explains the custom new NBA inspired stats
- Adding Conversational Memory
    - so the model can remember or build on from the previous question
    - using WindowBufferMemory: remembering the last *k* messages, keeps VRAM low and prevents the context window from being too crowded
    - using SummaryMemory: LLM writes a tiny paragraph that summarizes the conversation so far, and uses it as the memory, much more sophisticated, but takes longer

3/6/2025
Integrated RapidFuzz
- since the model acts like a strict database, if a user types a slightly different spelling to the actual name
- then the query will return zero results, making the agent assume that the player doesn't exist
- this problem is solved by using RapidFuzz