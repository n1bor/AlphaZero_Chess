# AlphaZero Chess — Project Guide

A from-scratch implementation of the AlphaZero reinforcement learning framework applied to chess, using Python and PyTorch. The system learns to play chess entirely through self-play, without requiring any human game data.

---

## How it works

AlphaZero combines two ideas:

1. **Monte Carlo Tree Search (MCTS)** — a game-tree search algorithm that explores possible moves by simulation. Instead of evaluating positions with hand-crafted rules, it uses a neural network to guide the search.

2. **Deep residual neural network** — takes a board position as input and outputs two things:
   - A **policy**: a probability distribution over all 4,672 possible moves, suggesting which moves are worth exploring.
   - A **value**: a single number between -1 and +1 estimating the probability of winning from this position.

These two components improve each other iteratively:
- MCTS uses the network to produce better self-play games.
- The network is trained on those games to produce better move predictions.

---

## Neural network architecture

- **Input**: 22-channel 8×8 tensor encoding piece positions, castling rights, en passant state, move counts, and repetition counters.
- **Body**: One convolutional block (22→256 filters) followed by 19 residual blocks (256 channels each), all with batch normalisation.
- **Policy head**: Outputs 4,672 values (8×8×73) covering all queen moves, knight moves, and pawn promotions.
- **Value head**: Single output with tanh activation, predicting win (+1), loss (-1), or draw (0).
- **Loss function**: Combined policy cross-entropy + value mean-squared error.

---

## Board representation

The board is an 8×8 numpy array of single-character strings. White pieces are uppercase (`R N B Q K P`), black pieces are lowercase (`r n b q k p`), and empty squares are `" "`. The board state is tracked alongside castling flags, en passant state, and move counts.

---

## Directory structure

All data is stored under a root directory configured in `src/config.py` (default: `/home/owensr/chess`).

```
<rootDir>/
├── data/
│   ├── games/          # Raw self-play game data from generate.py
│   ├── train/          # De-duplicated training positions (from game_de_dup.py)
│   ├── test/           # Test positions
│   ├── validate/       # Validation positions
│   ├── model_data/     # Saved network checkpoints (.gz files)
│   │   ├── latest.gz   # Current best network (used by train_one.py)
│   │   ├── best.gz     # Best evaluated network (used by tournament.py)
│   │   └── random.gz   # Randomly initialised network (starting point)
│   ├── graphs/         # Loss plots saved during training
│   └── analysis/       # Board images saved by analyze_games.py
└── stockfish/
    └── stockfish-ubuntu-x86-64-avx2   # Stockfish binary (required by generate.py)
```

---

## Configuration

**`src/config.py`** — edit this before running any script:

```python
rootDir = '/home/owensr/chess'   # path to data directory
lr = 0.0003                       # learning rate
batch_size = 3072                  # training batch size
```

---

## Source files

| File | Purpose |
|---|---|
| `chess_board.py` | Chess rules, move generation, board state |
| `encoder_decoder.py` | Convert board state ↔ neural network tensors |
| `alpha_net.py` | Neural network definition and loss function |
| `MCTS_chess.py` | MCTS search algorithm and self-play |
| `Player.py` | Abstract base class for players |
| `AlphaZero_player.py` | AlphaZero player wrapping MCTS + network |
| `Stockfish.py` | Stockfish engine player (via UCI subprocess) |
| `config.py` | Shared configuration |
| `visualize_board.py` | Render board as a matplotlib figure |

---

## Runnable scripts

### `generate.py` — Generate training data using Stockfish

Plays games using Stockfish (with randomised move selection weighted by score), recording board states and evaluations as training data. Saves compressed `.gz` files to `<rootDir>/data/games/`.

```bash
python3 src/generate.py <runId> <runtimeSeconds>
```

| Argument | Description |
|---|---|
| `runId` | Identifier appended to the output filename |
| `runtimeSeconds` | How long to run before stopping (seconds) |

**Example:**
```bash
python3 src/generate.py 1 3600    # run for 1 hour, save as data_1.gz
```

**Requires:** Stockfish binary at `<rootDir>/stockfish/stockfish-ubuntu-x86-64-avx2`

---

### `game_de_dup.py` — De-duplicate and split game data

Reads all raw game files from `<rootDir>/data/games/`, removes duplicate board positions using SHA-256 hashing (backed by a persistent `dbm` database), and splits unique positions 80/10/10 into train, test, and validate sets.

```bash
python3 src/game_de_dup.py
```

No arguments. Paths are read from `config.py`. Run this after `generate.py` and before training.

---

### `train_one.py` — Train on a single dataset (one pass)

Loads game data from `<rootDir>/data/<trainDir>/`, loads the current `latest.gz` network, trains for one pass through the data, then saves the result as a timestamped file and overwrites `latest.gz`.

```bash
python3 src/train_one.py <runId> <trainDir> <runtimeSeconds>
```

| Argument | Description |
|---|---|
| `runId` | Identifier included in the saved filename |
| `trainDir` | Subdirectory under `<rootDir>/data/` containing training files |
| `runtimeSeconds` | Time limit for training |

**Example:**
```bash
python3 src/train_one.py 1 train 7200
```

**Requires:** `<rootDir>/data/model_data/latest.gz` to exist.

---

### `train_all_games.py` — Train on all games in the train directory

Similar to `train_one.py` but with hardcoded paths. Loads from `<rootDir>/data/train`, trains using the `start_net.gz` model, and saves a timestamped output to `<rootDir>/data/model_data/`.

```bash
python3 src/train_all_games.py
```

No arguments. Edit the script directly to change the source network or data path.

---

### `lr_search.py` — Learning rate search

Trains the network across a list of candidate learning rates to find the best value. Saves loss plots to `<rootDir>/data/graphs/`. Useful before committing to a full training run.

```bash
python3 src/lr_search.py
```

No arguments. Edit `lrList` inside the script to change the learning rates to test. Loads `random.gz` as the starting network.

---

### `MCTS_chess.py` — MCTS self-play data generation

When run directly, performs MCTS self-play using a randomly initialised network, generating game datasets. Primarily used as a library by other scripts.

```bash
python3 src/MCTS_chess.py
```

No arguments. The number of games and parallel processes are set by constants near the bottom of the file.

---

### `evaluator.py` — Evaluate two networks against each other

Pits a "current" network against a "best" network in a series of games using MCTS (777 simulations per move). Runs 6 parallel processes playing 50 games each (300 total). Prints the win ratio of the current network.

```bash
python3 src/evaluator.py
```

No arguments. Edit `current_net` and `best_net` filenames at the bottom of the script to point to the networks to compare. Results are saved to `./evaluator_data/`.

---

### `match.py` — Play a single game between two players

Plays one game between any combination of AlphaZero and Stockfish players. Validates that both players produce identical board states throughout. Returns result as +1 (player A wins), -1 (player B wins), or 0 (draw).

```bash
python3 src/match.py --aType <type> --bType <type> [options]
```

| Argument | Values | Description |
|---|---|---|
| `--aType` | `alpha` \| `stockfish` | Player A type |
| `--bType` | `alpha` \| `stockfish` | Player B type |
| `--aAnetwork` | path | Path to `.gz` model file (if `--aType alpha`) |
| `--aAsteps` | int | MCTS simulations per move (if `--aType alpha`) |
| `--aSFHash` | int | Stockfish hash size in MB (if `--aType stockfish`) |
| `--aSFDepth` | int | Stockfish search depth (if `--aType stockfish`) |
| `--bAnetwork` | path | Path to `.gz` model file (if `--bType alpha`) |
| `--bAsteps` | int | MCTS simulations per move (if `--bType alpha`) |
| `--bSFHash` | int | Stockfish hash size in MB (if `--bType stockfish`) |
| `--bSFDepth` | int | Stockfish search depth (if `--bType stockfish`) |

**Examples:**
```bash
# AlphaZero vs Stockfish
python3 src/match.py \
  --aType alpha --aAnetwork /path/to/best.gz --aAsteps 777 \
  --bType stockfish --bSFHash 256 --bSFDepth 5

# AlphaZero vs AlphaZero
python3 src/match.py \
  --aType alpha --aAnetwork /path/to/model_a.gz --aAsteps 200 \
  --bType alpha --bAnetwork /path/to/model_b.gz --bAsteps 200
```

---

### `tournament.py` — Run an ongoing tournament with Elo ratings

Runs a continuous round-robin tournament between a configurable list of players, updating Elo ratings (starting at 1000, K=32) after each game. Uses a thread pool to run matches in parallel. Prints the leaderboard after each round.

```bash
python3 src/tournament.py
```

No arguments. Edit the `players` list near the bottom of the script to configure which networks and Stockfish instances participate.

**Example player configuration (in script):**
```python
players.append(Entry(Stockfish(100, 1, elo=1320)))          # Stockfish depth 1
players.append(Entry(AlphaZero('/path/to/best.gz', 777)))   # AlphaZero 777 steps
```

Runs indefinitely until interrupted.

---

### `pipeline.py` — Full AlphaZero iteration pipeline

Runs a complete training iteration: loads existing game data, trains the network, and saves the result. Originally intended to also run MCTS self-play (that code is currently commented out).

```bash
python3 src/pipeline.py
```

No arguments. Edit hardcoded paths and network names inside the script.

---

### `analyze_games.py` — Visualise a saved game

Loads a single game dataset file and saves each board position as a PNG image to `<rootDir>/data/analysis/`. Useful for inspecting what games the network played.

```bash
python3 src/analyze_games.py
```

No arguments. Edit `data_path` and `file` at the top of the script to point to the game file to analyse.

---

## Typical workflow

```
1. generate.py          → produce raw game data in data/games/
         ↓
2. game_de_dup.py       → deduplicate and split into data/train/, data/test/
         ↓
3. train_one.py         → train network on data/train/, update latest.gz
         ↓
4. evaluator.py         → compare latest.gz vs best.gz, keep the better one
         ↓
5. tournament.py        → benchmark against Stockfish with Elo tracking
         ↓
   repeat from step 1
```
