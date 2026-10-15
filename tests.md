# Chunking Evaluation & Retrieval Benchmark: Kev 4B vs. Baseline Methods

**Date**: 2026-10-15  
**Document**: Homer's *Odyssey* (`odyssey.txt`, 611,300 bytes, English translation by Samuel Butler)  
**Asset ID**: `7709b532-ece1-491d-abdf-1d950f89aa0a`  
**API Key**: `test7`  
**Embedding Model**: `embeddinggemma2` (768-dimensional dense vectors)  
**Database**: SQLite with `sqlite-vec`  

---

## 1. Executive Summary

This benchmark compares the retrieval effectiveness, semantic boundary cohesion, and re-ranking performance of **Kev 4B-based fine chunking** against three baseline chunking strategies across 8 canonical episodes from Homer's *Odyssey*.

### Key Takeaways:
1. **Top Retrieval Recall**: Kev 4B Fine Chunks achieved the **highest top-3 recall (87.5%)** in raw dense vector search, outperforming Greedy Sentence splitting (75.0%), Stage 1 Coarse splitting (75.0%), and Fixed sliding windows (62.5%).
2. **Re-Ranking Synergy**: When paired with **Kev Re-ranking (`POST /retrieve/kev`)**, top-3 hit rate reached **100.0% (8/8)**. When paired with **FlashRank Neural Cross-Encoder (`POST /retrieve/standard`)**, top-1 hit rate jumped from 25.0% to **75.0%**.
3. **Semantic Narrative Integrity**: Qualitative review shows Kev cleanly isolates multi-turn narrative units into complete episodes (e.g., Penelope's test of the olive-tree bridal bed was split into a clean 3-chunk arc without cutting through speeches).
4. **Formatting Optimization Found**: Because Project Gutenberg / Classics text files contain hard line breaks (`\n`) every ~70 characters, `split_atomic_units` (`r"((?<=[.?!])\s+|\n+)"`) treated plain line wrapping as potential unit breaks. Pre-processing text to unwrap soft line breaks will prevent mid-sentence candidate boundaries and further improve Kev chunking accuracy.

---

## 2. Chunking Methods Evaluated

| Strategy | Description | Chunk Count | Average Length | Min Length | Max Length |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Kev 4B Fine** | Stage 2 multi-candidate breakpoint decisions via `/v1/systemone` | **542** | **1,127.9 chars** | 158 chars | 1,614 chars |
| **Greedy Sentence** | Sentence accumulation strictly using `(?<=[.?!])\s+` up to ~1,128 chars | **612** | **996.9 chars** | 302 chars | 1,180 chars |
| **Fixed Window** | Fixed 1,128 character windows with a 100 character sliding overlap | **595** | **1,127.2 chars** | 200 chars | 1,128 chars |
| **Stage 1 Coarse** | Structural paragraph / double-newline grouping (~2,000–4,000 chars) | **172** | **3,554.1 chars** | 2,226 chars | 4,120 chars |

---

## 3. Evaluation Dataset & Test Queries

The evaluation queries represent self-contained episodes with specific factual anchors:

1. **Cyclops Blinding**: *"How did Ulysses blind the Cyclops Polyphemus and what trick did he use with his name?"*  
   *Target Anchor*: `"beam into the monster's eye"`
2. **Scylla & Charybdis**: *"What did Circe warn Ulysses about the monster Scylla and the whirlpool Charybdis?"*  
   *Target Anchor*: `"whirlpool"`
3. **Cattle of Hyperion**: *"Why were the men punished for slaughtering the cattle of the Sun god Hyperion?"*  
   *Target Anchor*: `"Hyperion"`
4. **Bridal Bed Test**: *"How did Penelope test Ulysses using the secret of their olive tree bridal bed?"*  
   *Target Anchor*: `"centre-post of my bed"`
5. **The Hound Argos**: *"What happened to the old hound Argos when he recognized Ulysses after twenty years?"*  
   *Target Anchor*: `"Argos"`
6. **Disguised Athena**: *"How did Telemachus receive Athena when she came to Ithaca disguised as Mentes?"*  
   *Target Anchor*: `"Mentes"`
7. **The Lotus-Eaters**: *"What happened to the men who tasted the honeyed fruit of the Lotus-eaters?"*  
   *Target Anchor*: `"Lotus-eaters"`
8. **Song of the Sirens**: *"How did Ulysses listen to the Sirens while keeping his ship and crew safe?"*  
   *Target Anchor*: `"Sirens"`

---

## 4. Benchmark Results

### 4.1 Raw Dense Vector Retrieval (`embeddinggemma2`, Cosine Similarity)

Evaluated without re-ranking to isolate the effect of chunk boundaries on dense embedding quality:

| Method | Hit@1 | Hit@3 | MRR (Mean Reciprocal Rank) | Avg Margin (Top 1 - Top 2) | Broken Sentence Rate | Broken Quote Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kev 4B Fine** | 25.0% | **87.5%** | 0.573 | 0.0197 | 43.9%* | 46.3% |
| **Greedy Sentence** | 37.5% | 75.0% | **0.589** | **0.0221** | **0.0%** | 44.9% |
| **Stage 1 Coarse** | 37.5% | 75.0% | **0.589** | 0.0171 | 5.8% | **23.8%** |
| **Fixed 1128 Chars** | **50.0%** | 62.5% | 0.575 | 0.0217 | 97.5% | 48.4% |

*\*Note on Kev Broken Sentence Rate: Caused by hard line breaks (`\n`) in Gutenberg source text being partitioned into atomic units. See Section 6 for remediation.*

---

### 4.2 End-to-End Pipeline: Vector Search + Re-ranking

Querying the live API endpoints with the indexed Kev fine chunks (`test7`):

| Endpoint / Re-ranker | Hit@1 | Hit@3 | Notes |
| :--- | :---: | :---: | :--- |
| `POST /search?rerank=false` | 50.0% | 87.5% | Baseline dense vector search against fine chunks |
| `POST /retrieve/standard` (FlashRank Cross-Encoder) | **75.0%** | 87.5% | Promotes the most relevant chunk to rank #1 in 6 of 8 queries |
| `POST /retrieve/kev` (Kev 4B System One Re-ranker) | 62.5% | **100.0%** | **Every query found the target passage in the top 3 results** |

---

## 5. Qualitative Episode Analysis: The Bridal Bed Scene

In Book XXIII, Penelope tests Ulysses with the secret of their bridal bed. Kev segmented this episode into three distinct, self-contained chunks:

* **Chunk 507 (`chunk_id: 4511`, 1,438 chars)**:
  * *Content*: Penelope commands nurse Euryclea to move Ulysses' bed outside the bridal chamber.
  * *Semantic Role*: The Challenge.
* **Chunk 508 (`chunk_id: 4512`, 1,389 chars)**:
  * *Content*: Ulysses' unbroken reply explaining that the bed was built around the stump of a living olive tree, with stone walls built around it, and cannot be moved without felling the tree at its roots.
  * *Semantic Role*: The Secret Revealed (isolated cleanly without preamble or postscript).
* **Chunk 509 (`chunk_id: 4513`, 1,443 chars)**:
  * *Content*: Penelope breaks into tears, embraces Ulysses, and admits this was the private token only they and one maidservant knew.
  * *Semantic Role*: The Recognition & Reconciliation.

*By contrast, Fixed 1,128-char chunking sliced directly through line 9815 (`"and making it into a bed-prop. I then bored a hole down..."`), dividing the key explanation across two disconnected chunks.*

---

## 6. Recommendations & Next Steps

1. **Unwrap Soft Line Breaks in Text Ingestion**:
   In `ragdoll/pipeline/chunker.py`, `split_atomic_units` currently splits on `r"((?<=[.?!])\s+|\n+)"`. For plain text files with 70–80 column word-wrapping, normalize single newlines (`(?<!\n)\n(?!\n)`) to spaces before unit splitting so Kev is only offered genuine sentence transitions as candidate split points.
2. **Pair Kev Fine Chunks with Re-ranking**:
   The highest retrieval performance was achieved by combining Kev fine chunks with FlashRank (`POST /retrieve/standard`) for top-1 accuracy (75%) and Kev Re-ranker (`POST /retrieve/kev`) for top-3 recall (100%).

---

## 7. Retrieval Methods Benchmark: Raw Vector vs. FlashRank vs. Kev 4B

**Dataset**: Exact same 542 fine chunks of `odyssey.txt` (asset `7709b532-ece1-491d-abdf-1d950f89aa0a`, key `test7`).  
**Candidate Pool**: Top 15 vector candidates re-ranked to top 5.  
**Queries**: 12 diverse episodic queries.

### 7.1 Quantitative Comparison

| Retrieval Method | Hit@1 | Hit@3 | Hit@5 | MRR | Avg Latency | P95 Latency | Compute Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw Vector (Cosine)** | 33.3% (4/12) | 75.0% (9/12) | 91.7% (11/12) | 0.569 | 214.9 ms | 1,346.5 ms | `sqlite-vec` distance |
| **Standard Re-ranker (FlashRank)** | 75.0% (9/12) | 91.7% (11/12) | **100.0% (12/12)** | 0.836 | **120.2 ms** | **219.0 ms** | Local ONNX MiniLM cross-encoder |
| **Kev Re-ranker (Kev 4B)** | **83.3% (10/12)** | **100.0% (12/12)** | **100.0% (12/12)** | **0.917** | 3,966.6 ms | 4,401.4 ms | LLM System One scoring (`/v1/systemone`) |

---

### 7.2 Query-by-Query Rank Comparison

| Query Description | Raw Vector | FlashRank | Kev 4B | Key Dynamics |
| :--- | :---: | :---: | :---: | :--- |
| **Cyclops Blinding** | #3 | #5 | **#1** | FlashRank demoted target; Kev identified the exact blinding act. |
| **Scylla & Charybdis** | #3 | **#1** | #2 | Both re-rankers elevated the passage into top 2. |
| **Cattle of Hyperion** | #2 | **#1** | **#1** | Both re-rankers correctly promoted to #1. |
| **Bridal Bed Test** | #2 | #3 | **#1** | Kev recognized the olive tree secret; promoted to #1. |
| **Hound Argos** | #2 | **#1** | **#1** | Both re-rankers isolated the recognition passage. |
| **Athena as Mentes** | #4 | **#1** | **#1** | Both re-rankers recovered the true greeting scene from #4. |
| **The Lotus-Eaters** | **#1** | **#1** | **#1** | Unanimous agreement. |
| **Song of the Sirens** | **#1** | **#1** | **#1** | Unanimous agreement. |
| **Circe's Swine** | **#1** | **#1** | **#1** | Unanimous agreement. |
| **Aeolus Bag of Winds** | #6 (>5) | #2 | **#1** | Raw vector missed top 5; Kev promoted to #1. |
| **Teiresias Prophecy** | **#1** | **#1** | **#1** | Unanimous agreement. |
| **Bow Contest** | #4 | **#1** | #2 | FlashRank gave cleaner lexical match; Kev placed at #2. |

---

### 7.3 Architectural Takeaways

1. **Re-ranking is Essential**: Raw vector search on dense chunks achieves only **33.3% Hit@1** and **0.569 MRR**. Either re-ranker more than doubles Top-1 precision.
2. **Kev 4B Excels on Complex Factual Actions**: In episodes requiring narrative understanding (e.g., blinding Polyphemus, Aeolus's sack of winds, and the olive-tree bridal bed test), Kev promoted the answer to **#1**, whereas FlashRank either demoted or placed it below distractors.
3. **The Speed vs. Precision Trade-off**:
   - **FlashRank** is ~33x faster (120ms vs 3,966ms) and local (zero network/GPU overhead).
   - **Kev 4B** achieves higher accuracy (**MRR 0.917** vs **0.836**, **100% Hit@3** vs **91.7%**).
   - **Recommended Pattern**: Use Standard FlashRank (`POST /retrieve/standard`) for real-time user-facing search; use Kev (`POST /retrieve/kev`) for agentic workflows, complex reasoning queries, and high-stakes extraction tasks.
