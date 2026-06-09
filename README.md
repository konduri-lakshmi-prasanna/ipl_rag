# 🏏 IPL Cricket Q&A — LangGraph RAG System

IPL Cricket Q&A is an AI-powered cricket statistics chatbot built using Python and FastAPI. You ask natural language questions about IPL matches, players, and teams, and the system retrieves accurate answers from a ChromaDB vector knowledge base. It does not make up statistics or rely on general knowledge — every answer is grounded in the indexed IPL data.

---

## What This Project Does

The system routes your question to the most relevant specialist retrieval node using a LangGraph state machine, fetches matching chunks from ChromaDB, and synthesizes a final answer using a Groq-hosted LLM.

The following query types are supported:

**Batting** — runs, strike rates, centuries, and batting averages for any player or season.

**Bowling** — wickets, economy rates, best figures, and bowling averages.

**Head-to-Head** — win/loss records between any two IPL teams.

**Team Performance** — overall season records, win rates, and playoff history.

**Venue** — ground-specific stats, average scores, and pitch behavior.

**Player Form** — recent match-by-match performance for any player.

**Records** — all-time IPL records such as most runs, most wickets, and highest team totals.

---

## UML Diagrams

### 1. System Architecture

Shows all layers — User, API layer, LangGraph graph, specialist nodes, and ChromaDB.

```mermaid
flowchart TD
    User([User]) -->|HTTP POST /ask| API(FastAPI - app.py)
    API -->|invoke graph| Graph(graph.py - LangGraph)
    Graph -->|route intent| Router(router_node)
    Router -->|batting| Bat(batting_node)
    Router -->|bowling| Bowl(bowling_node)
    Router -->|head to head| H2H(h2h_node)
    Router -->|team| Team(team_node)
    Router -->|venue| Venue(venue_node)
    Router -->|form| Form(form_node)
    Router -->|records| Rec(records_node)
    Bat --> Val(validation_node)
    Bowl --> Val
    H2H --> Val
    Team --> Val
    Venue --> Val
    Form --> Val
    Rec --> Val
    Val -->|context passes| Syn(synthesis_node)
    Syn -->|LLM call| Groq[/Groq LLaMA 3.3 70B/]
    Groq -->|generated answer| Syn
    Syn -->|final_answer| API
    API -->|JSON response| User
    Bat & Bowl & H2H & Team & Venue & Form & Rec -->|section filter + k=4| Chroma[(ChromaDB - ipl_rag)]
```

---

### 2. Vector Store Ingestion Pipeline

Shows how IPL data is loaded, chunked, embedded, and saved to ChromaDB with section metadata.

```mermaid
flowchart TD
    Data[/IPL CSV or Text Data/] --> Detect{Detect Section Type}
    Detect -->|batting stats| BatChunk[Chunk batting records]
    Detect -->|bowling stats| BowlChunk[Chunk bowling records]
    Detect -->|match results| MatchChunk[Chunk match results]
    Detect -->|player profiles| PlayerChunk[Chunk player profiles]
    BatChunk --> Tag[Attach section metadata tag]
    BowlChunk --> Tag
    MatchChunk --> Tag
    PlayerChunk --> Tag
    Tag --> Embed[Embed with all-MiniLM-L6-v2]
    Embed --> Save[(Persist to ChromaDB - chroma_db/)]
```

---

### 3. RAG Query Flow

Shows how a user question flows through routing, section-filtered retrieval, validation, and synthesis to produce an answer.

```mermaid
flowchart TD
    Q([User Question]) --> State(IPLAgentState initialised)
    State --> Router(router_node - classifies intent and extracts entities)
    Router --> Node(Specialist retrieval node selected)
    Node --> Filter[ChromaDB section filter + semantic search k=4]
    Filter --> Ctx[Context chunks added to state]
    Ctx --> Val(validation_node - checks context quality)
    Val --> Syn(synthesis_node)
    Syn --> Prompt(Prompt assembled with context and query)
    Prompt --> LLM[/Groq LLaMA 3.3 70B/]
    LLM --> Answer([Final answer returned])
```

---

### 4. Sequence Diagram

Shows the full order of interactions between all components from query to answer.

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI app.py
    participant Graph as graph.py LangGraph
    participant Router as router_node
    participant Node as Specialist Node
    participant Chroma as ChromaDB
    participant Val as validation_node
    participant Syn as synthesis_node
    participant Groq as Groq LLM

    User->>API: POST /ask with user_query
    API->>Graph: graph.invoke(state)
    Graph->>Router: route query
    Router->>Router: classify intent and extract entities
    Router-->>Graph: set intent in state
    Graph->>Node: invoke specialist node
    Node->>Chroma: retriever.invoke(query, filter=section, k=4)
    Chroma-->>Node: return top-4 chunks
    Node-->>Graph: update state with context
    Graph->>Val: validate retrieved context
    Val-->>Graph: context quality confirmed
    Graph->>Syn: synthesize answer
    Syn->>Groq: LLM call with context and query
    Groq-->>Syn: generated answer text
    Syn-->>Graph: set final_answer in state
    Graph-->>API: return completed state
    API-->>User: JSON response with answer
```

---

### 5. Class Diagram

Shows all classes, their attributes, methods, and relationships.

```mermaid
classDiagram
    class IPLAgentState {
        +user_query : str
        +intent : str
        +entities : list
        +batting_context : list
        +bowling_context : list
        +h2h_context : list
        +team_context : list
        +venue_context : list
        +form_context : list
        +records_context : list
        +final_answer : str
    }

    class RouterNode {
        +llm : ChatGroq
        +classify_intent(state) str
        +extract_entities(state) list
        +__call__(state) IPLAgentState
    }

    class RetrievalNode {
        +vectorstore : Chroma
        +embeddings : HuggingFaceEmbeddings
        +section : str
        +k : int
        +build_retriever() Retriever
        +build_query(state) str
        +__call__(state) IPLAgentState
    }

    class ValidationNode {
        +min_chunks : int
        +check_context(state) bool
        +__call__(state) IPLAgentState
    }

    class SynthesisNode {
        +llm : ChatGroq
        +prompt : PromptTemplate
        +assemble_context(state) str
        +__call__(state) IPLAgentState
    }

    class VectorStoreSetup {
        +persist_directory : str
        +collection_name : str
        +embed_model : str
        +load_data() list
        +chunk_documents(docs) list
        +build_and_persist()
    }

    class FastAPIApp {
        +graph : CompiledGraph
        +ask(query) JSONResponse
        +health() JSONResponse
    }

    FastAPIApp --> IPLAgentState : creates
    RouterNode --> IPLAgentState : reads and updates
    RetrievalNode --> IPLAgentState : reads and updates
    ValidationNode --> IPLAgentState : reads and updates
    SynthesisNode --> IPLAgentState : reads and updates
    RetrievalNode --> VectorStoreSetup : uses persisted chroma_db
```

---

## How It Works

When you send a query, the FastAPI server passes it into the LangGraph graph as an `IPLAgentState` object. The router node reads the query, classifies the intent (batting, bowling, h2h, etc.), and extracts named entities like player or team names.

The graph then routes to the matching specialist node. Each node connects to ChromaDB and filters by its own `section` metadata tag, so a batting question only searches batting chunks and a venue question only searches venue chunks. The node runs a semantic search with `k=4` and writes the retrieved documents back into the state.

After retrieval, the validation node checks that enough context was returned. The synthesis node then assembles all retrieved chunks, builds a prompt, and calls the Groq API to generate a final grounded answer.

---

## Technologies Used

- Python 3.11
- FastAPI for the API server
- LangGraph for the agentic graph orchestration
- LangChain for retriever and LLM wrappers
- Groq API with the LLaMA 3.3 70B model as the language model
- ChromaDB for storing and searching document embeddings
- HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`) for generating embeddings

---

## Project Structure

The `app.py` file is the FastAPI application. It defines the `/ask` endpoint and invokes the LangGraph graph per request.

The `main.py` file is the entry point that starts the Uvicorn server.

The `graph.py` file defines the LangGraph state machine — all nodes, edges, and conditional routing logic.

The `state.py` file defines the `IPLAgentState` TypedDict that is passed between every node.

The `nodes/` folder contains all specialist nodes. `router.py` classifies intent and extracts entities. `batting.py`, `bowling.py`, `h2h.py`, `team.py`, `venue.py`, `form.py`, and `records.py` each handle retrieval for their respective domain. `validation.py` checks retrieved context quality. `synthesis.py` assembles context and calls the LLM.

The `vectorstore/` folder contains `setup.py` which loads IPL data, chunks it, attaches section metadata, and persists it to ChromaDB.

The `chroma_db/` folder is where the persisted ChromaDB vector store is saved. This is auto-generated when you run the setup script.

The `requirements.txt` file lists all Python dependencies.

---

## How to Run the Project

Clone the repository and go into the project folder.

```bash
git clone https://github.com/konduri-lakshmi-prasanna/ipl_rag.git
cd ipl_rag
```

Create a virtual environment and activate it.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the Python packages.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project folder and add your Groq API key like this.

```
GROQ_API_KEY=your_key_here
```

Build the ChromaDB vector store from the IPL data.

```bash
python vectorstore/setup.py
```

Finally, run the API server.

```bash
python main.py
```

Then open your browser or API client and send requests to `http://localhost:8000`.

---

## How to Use the API

Send a POST request to `/ask` with your question as JSON.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the head-to-head record between CSK and MI?"}'
```

Example questions you can ask:

- "How many centuries has Virat Kohli scored in IPL?"
- "What is the head-to-head record between CSK and MI?"
- "Which venues have the highest average first innings score?"
- "Who has the most wickets at the Wankhede Stadium?"
- "What is RCB's win rate in playoff matches?"
- "Who has the best economy rate among all IPL bowlers?"

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key for LLM inference |

---

## What I Learned

This project helped me understand how agentic RAG systems work in practice. I learned how to design a multi-node LangGraph state machine, how to use ChromaDB section metadata for domain-filtered retrieval, how to pass state cleanly between nodes, how to integrate FastAPI with a compiled LangGraph graph, and how routing logic differs from a simple single-chain RAG pipeline.

---

## Author

Built by Prasanna Konduri as part of an AI/ML project at SVECW, exploring LangGraph agentic architectures, ChromaDB vector retrieval, and Groq-powered LLM inference for domain-specific Q&A.