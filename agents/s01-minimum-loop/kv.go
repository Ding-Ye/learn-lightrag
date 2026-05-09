package main

// KVStore mirrors upstream BaseKVStorage (lightrag/base.py:308). In s01 it's
// a sync.Map-backed wrapper — no persistence, no per-key locking. s05 ships
// the real JSON-on-disk implementation.

import (
	"context"
	"sync"
)

// KVStore is the canonical contract; FilterMissing is the load-bearing
// primitive that drives "which new chunks need extracting" in s09.
type KVStore interface {
	Get(ctx context.Context, id string) (map[string]any, bool, error)
	GetByIDs(ctx context.Context, ids []string) (map[string]map[string]any, error)
	Upsert(ctx context.Context, items map[string]map[string]any) error
	FilterMissing(ctx context.Context, ids []string) ([]string, error)
	Delete(ctx context.Context, ids []string) error
	Persist(ctx context.Context) error
}

// MemoryKVStore is the in-process implementation.
type MemoryKVStore struct {
	data sync.Map // map[string]map[string]any
}

// NewMemoryKVStore returns an empty store.
func NewMemoryKVStore() *MemoryKVStore { return &MemoryKVStore{} }

func (m *MemoryKVStore) Get(ctx context.Context, id string) (map[string]any, bool, error) {
	v, ok := m.data.Load(id)
	if !ok {
		return nil, false, nil
	}
	return v.(map[string]any), true, nil
}

func (m *MemoryKVStore) GetByIDs(ctx context.Context, ids []string) (map[string]map[string]any, error) {
	out := make(map[string]map[string]any, len(ids))
	for _, id := range ids {
		if v, ok := m.data.Load(id); ok {
			out[id] = v.(map[string]any)
		}
	}
	return out, nil
}

func (m *MemoryKVStore) Upsert(ctx context.Context, items map[string]map[string]any) error {
	for k, v := range items {
		m.data.Store(k, v)
	}
	return nil
}

func (m *MemoryKVStore) FilterMissing(ctx context.Context, ids []string) ([]string, error) {
	missing := make([]string, 0, len(ids))
	for _, id := range ids {
		if _, ok := m.data.Load(id); !ok {
			missing = append(missing, id)
		}
	}
	return missing, nil
}

func (m *MemoryKVStore) Delete(ctx context.Context, ids []string) error {
	for _, id := range ids {
		m.data.Delete(id)
	}
	return nil
}

// Persist is a no-op in s01; s05 ships the real JSON file dump.
func (m *MemoryKVStore) Persist(ctx context.Context) error { return nil }
