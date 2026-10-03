# Third-Party Material

The datasets and frozen QA evidence contain public GitHub metadata and code
excerpts from the following repositories. Their original licenses and notices
continue to apply; this release does not relicense upstream code:

- https://github.com/ethereum/go-ethereum
- https://github.com/bnb-chain/bsc
- https://github.com/0xPolygon/bor
- https://github.com/celo-org/celo-blockchain

NiCad-Go uses the official comparison engine from
https://github.com/CordyJ/Open-NiCad at the commit specified in BASELINES.md.
The build script retrieves the upstream license with the engine source.
The Go extraction frontend and the Go path-context encoder are adaptations;
they are not official Go releases of NiCad or code2vec.

The code2vec architecture is described by Alon et al., "code2vec: Learning
Distributed Representations of Code," POPL 2019 (PACMPL 3, Article 40).
The released checkpoint was trained for the Go implementation described in
models/go_code2vec_training.json.
