package main

import (
	"crypto/md5"
	"encoding/hex"
)

// MD5DocID returns the hex-encoded MD5 of the document content.
// Upstream LightRAG uses MD5 for doc IDs (lightrag/utils.py compute_mdhash_id);
// we mirror it exactly so re-inserting the same text is a no-op (the docID
// collides with the existing entry in DocStatusStore.Enqueue).
//
// Why MD5 and not sha256:
//   - Upstream parity: matches `compute_mdhash_id(content, prefix="doc-")` keys
//     in the on-disk JSON files. Switching to sha256 would break docID
//     compatibility if a learner pointed s03 at upstream's working_dir.
//   - This is a content-addressing hash, not a security hash — collision
//     resistance under deliberate attack is not relevant to the use case.
//
// We omit upstream's "doc-" prefix because in s03 the type system already
// disambiguates DocID from chunkID; the prefix is a vestige from the days
// when both shared one KV namespace.
func MD5DocID(content string) string {
	sum := md5.Sum([]byte(content))
	return hex.EncodeToString(sum[:])
}
