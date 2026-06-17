//! Provides a least-recently-used cache with async access.
//!
//! To operate, this cache is bound to an I/O back-end object that provides the loading and
//! flushing of cache entries.
//!
//! Also supports inter-cache dependency, e.g. for when the qcow2 L2 table cache needs to be
//! flushed before the refblock cache, because some clusters were freed (so the L2 references need
//! to be cleared before the clusters are deallocated).

#![allow(dead_code)]

use crate::vector_select::FutureVector;
use async_trait::async_trait;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::{io, mem};
use tokio::sync::{Mutex, MutexGuard, RwLock, RwLockWriteGuard};
use tracing::{error, instrument, trace};

/// Cache entry structure, wrapping the cached object.
pub(crate) struct AsyncLruCacheEntry<V> {
    /// Cached object.
    ///
    /// Always set during operation, only cleared when trying to unwrap the `Arc` on eviction.
    value: Option<Arc<V>>,

    /// When this entry was last accessed.
    last_used: AtomicUsize,
}

/// Least-recently-used cache with async access.
struct AsyncLruCacheInner<
    Key: Clone + Copy + Debug + PartialEq + Eq + Hash + Send + Sync,
    Value: Send + Sync,
    IoBackend: AsyncLruCacheBackend<Key = Key, Value = Value>,
> {
    /// I/O back-end that performs loading and flushing of cache entries.
    backend: IoBackend,

    /// Cache entries.
    map: RwLock<HashMap<Key, AsyncLruCacheEntry<Value>>>,

    /// Flush dependencies (flush these first).
    flush_before: Mutex<Vec<Arc<dyn FlushableCache>>>,

    /// Monotonically increasing counter to generate “timestamps”.
    lru_timer: AtomicUsize,

    /// Upper limit of how many entries to cache.
    limit: usize,
}

/// Least-recently-used cache with async access.
///
/// Keeps the least recently used entries up to a limited count.  Accessing and flushing is
/// async-aware.
///
/// `K` is the key used to uniquely identify cache entries, `V` is the cached data.
pub(crate) struct AsyncLruCache<
    K: Clone + Copy + Debug + PartialEq + Eq + Hash + Send + Sync,
    V: Send + Sync,
    B: AsyncLruCacheBackend<Key = K, Value = V>,
>(Arc<AsyncLruCacheInner<K, V, B>>);

/// Internal trait used to implement inter-cache flush dependencies.
#[async_trait(?Send)]
trait FlushableCache: Send + Sync {
    /// Flush the cache.
    async fn flush(&self) -> io::Result<()>;

    /// Check of circular dependencies.
    ///
    /// Return `true` if (and only if) `other` is already a transitive dependency of `self`.
    async fn check_circular(&self, other: &Arc<dyn FlushableCache>) -> bool;
}

/// Provides loading and flushing for cache entries.
pub(crate) trait AsyncLruCacheBackend: Send + Sync {
    /// Key type.
    type Key: Clone + Copy + Debug + PartialEq + Eq + Hash + Send + Sync;
    /// Value (object) type.
    type Value: Send + Sync;

    /// Load the given object.
    #[allow(async_fn_in_trait)] // No need for Send
    async fn load(&self, key: Self::Key) -> io::Result<Self::Value>;

    /// Flush the given object.
    ///
    /// The implementation should itself check whether the object is dirty; `flush()` is called for
    /// all evicted cache entries, regardless of whether they actually are dirty or not.
    #[allow(async_fn_in_trait)] // No need for Send
    async fn flush(&self, key: Self::Key, value: &Self::Value) -> io::Result<()>;

    /// Drop the given object without flushing.
    ///
    /// The cache owner is invalidating the cache, evicting all objects without flushing them.  If
    /// dropping the object as-is would cause problems (e.g. because it is verified not to be
    /// dirty), those problems need to be resolved here.
    ///
    /// # Safety
    /// Depending on the nature of the cache, this operation may be unsafe.  Must only be performed
    /// if the cache owner requested it and guarantees it is safe.
    unsafe fn evict(&self, key: Self::Key, value: Self::Value);
}

impl<
        K: Clone + Copy + Debug + PartialEq + Eq + Hash + Send + Sync,
        V: Send + Sync,
        B: AsyncLruCacheBackend<Key = K, Value = V>,
    > AsyncLruCache<K, V, B>
{
    /// Create a new cache.
    ///
    /// `size` is the maximum number of entries to keep in the cache.
    pub fn new(backend: B, size: usize) -> Self {
        AsyncLruCache(Arc::new(AsyncLruCacheInner {
            backend,
            map: Default::default(),
            flush_before: Default::default(),
            lru_timer: AtomicUsize::new(0),
            limit: size,
        }))
    }

    /// Retrieve an entry from the cache.
    ///
    /// If there is no entry yet, run `read()` to generate it.  If then there are more entries in
    /// the cache than its limit, flush out the oldest entry via `flush()`.
    pub async fn get_or_insert(&self, key: K) -> io::Result<Arc<V>> {
        self.0.get_or_insert(key).await
    }

    /// Force-insert the given object into the cache.
    ///
    /// If there is an existing object under that key, it is flushed first.
    pub async fn insert(&self, key: K, value: Arc<V>) -> io::Result<()> {
        self.0.insert(key, value).await
    }

    /// Flush all cache entries.
    ///
    /// Those entries are not evicted, but remain in the cache.
    pub async fn flush(&self) -> io::Result<()> {
        self.0.flush().await
    }

    /// Evict all cache entries.
    ///
    /// Evicts all cache entries without flushing them.
    ///
    /// # Safety
    /// Depending on the nature of the cache, this operation may be unsafe.  Perform at your own
    /// risk.
    pub async unsafe fn invalidate(&self) -> io::Result<()> {
        unsafe { self.0.invalidate() }.await
    }
}

impl<
        K: Clone + Copy + Debug + PartialEq + Eq + Hash + Send + Sync + 'static,
        V: Send + Sync + 'static,
        B: AsyncLruCacheBackend<Key = K, Value = V> + 'static,
    > AsyncLruCache<K, V, B>
{
    /// Set up a flush dependency.
    ///
    /// Ensure that before anything in this cache is flushed, `flush_before` is flushed first.
    #[instrument(
        level = "trace",
        name = "AsyncLruCache::depend_on",
        skip_all,
        fields(
            self = Arc::as_ptr(&self.0) as usize,
            other = Arc::as_ptr(&other.0) as usize,
        )
    )]
    pub async fn depend_on<
        K2: Clone + Copy + Debug + PartialEq + Eq + Hash + Send + Sync + 'static,
        V2: Send + Sync + 'static,
        B2: AsyncLruCacheBackend<Key = K2, Value = V2> + 'static,
    >(
        &self,
        other: &AsyncLruCache<K2, V2, B2>,
    ) -> io::Result<()> {
        let cloned: Arc<AsyncLruCacheInner<K2, V2, B2>> = Arc::clone(&other.0);
        let cloned: Arc<dyn FlushableCache> = cloned;

        loop {
            {
                let mut locked = self.0.flush_before.lock().await;
                // Shouldn’t be long, so linear search seems fine
                if locked.iter().any(|x| Arc::ptr_eq(x, &cloned)) {
                    break;
                }

                let self_arc: Arc<AsyncLruCacheInner<K, V, B>> = Arc::clone(&self.0);
                let self_arc: Arc<dyn FlushableCache> = self_arc;
                if !other.0.check_circular(&self_arc).await {
                    trace!("No circular dependency, entering new dependency");
                    locked.push(cloned);
                    break;
                }
            }

            trace!("Circular dependency detected, flushing other cache first");

            other.0.flush().await?;
        }

        Ok(())
    }
}

impl<
        K: Clone + Copy + Debug + PartialEq + Eq + Hash + Send + Sync,
        V: Send + Sync,
        B: AsyncLruCacheBackend<Key = K, Value = V>,
    > AsyncLruCacheInner<K, V, B>
{
    /// Flush all dependencies.
    ///
    /// Flush all caches that must be flushed before this one.  Remove all successfully flushed
    /// caches from our dependency list.
    ///
    /// Call with a guard that should be dropped only after this cache is flushed, so that no new
    /// dependencies can enter while we are still flushing this cache.
    #[instrument(level = "trace", name = "AsyncLruCache::flush_dependencies", skip_all)]
    async fn flush_dependencies(
        flush_before: &mut MutexGuard<'_, Vec<Arc<dyn FlushableCache>>>,
    ) -> io::Result<()> {
        while let Some(dep) = flush_before.pop() {
            trace!("Flushing dependency {:?}", Arc::as_ptr(&dep) as *const _);
            if let Err(err) = dep.flush().await {
                flush_before.push(dep);
                return Err(err);
            }
        }
        Ok(())
    }

    /// Ensure there is at least one free entry in the cache.
    ///
    /// Do this by evicting (flushing) existing entries, if necessary.
    #[instrument(
        level = "trace",
        name = "AsyncLruCache::ensure_free_entry",
        skip_all,
        fields(self = &self as *const _ as usize),
    )]
    async fn ensure_free_entry(
        &self,
        map: &mut RwLockWriteGuard<'_, HashMap<K, AsyncLruCacheEntry<V>>>,
    ) -> io::Result<()> {
        while map.len() >= self.limit {
            trace!("{} / {} used", map.len(), self.limit);

            let now = self.lru_timer.load(Ordering::Relaxed);
            let oldest = map
                .iter()
                .filter(|(_key, entry)| Arc::strong_count(entry.value()) == 1)
                .fold((0, None), |oldest, (key, entry)| {
                    // Users must not create weak references, and so we know that with a `strong_count`
                    // of 1 (while holding the map’s write lock), no one can access this entry anymore
                    // and we could safely drop it.
                    assert_eq!(Arc::weak_count(entry.value()), 0);

                    let age = now.wrapping_sub(entry.last_used.load(Ordering::Relaxed));
                    if age >= oldest.0 {
                        (age, Some(*key))
                    } else {
                        oldest
                    }
                });

            let Some(oldest_key) = oldest.1 else {
                error!("Cannot evict entry from cache; everything is in use");
                return Err(io::Error::other(
                    "Cannot evict entry from cache; everything is in use",
                ));
            };

            trace!("Removing entry with key {oldest_key:?}, aged {}", oldest.0);

            let oldest_entry = map.remove(&oldest_key).unwrap();

            // We checked `strong_count` above to be 1, and there are no weak references, so the
            // only reference to this entry must have been the one in the map.  We held the write
            // lock throughout, there was no await point between the check and here, so the
            // `strong_count` must still be 1 and we can thus safely unwrap the `Arc`.
            let evicted_object = Arc::try_unwrap(oldest_entry.value.unwrap())
                .unwrap_or_else(|_| panic!("entry has gained external references"));

            let mut dep_guard = self.flush_before.lock().await;
            Self::flush_dependencies(&mut dep_guard).await?;
            trace!("Flushing {oldest_key:?}");
            if let Err(err) = self.backend.flush(oldest_key, &evicted_object).await {
                map.insert(
                    oldest_key,
                    AsyncLruCacheEntry {
                        value: Some(Arc::new(evicted_object)),
                        last_used: oldest_entry.last_used.load(Ordering::Relaxed).into(),
                    },
                );
                return Err(err);
            }
        }

        Ok(())
    }

    /// Retrieve an entry from the cache.
    ///
    /// If there is no entry yet, run `read()` to generate it.  If then there are more entries in
    /// the cache than its limit, flush out the oldest entry via `flush()`.
    ///
    /// Users must not create weak references to the returned `Arc`.
    async fn get_or_insert(&self, key: K) -> io::Result<Arc<V>> {
        {
            let map = self.map.read().await;
            if let Some(entry) = map.get(&key) {
                entry.last_used.store(
                    self.lru_timer.fetch_add(1, Ordering::Relaxed),
                    Ordering::Relaxed,
                );
                return Ok(Arc::clone(entry.value()));
            }
        }

        let mut map = self.map.write().await;
        if let Some(entry) = map.get(&key) {
            entry.last_used.store(
                self.lru_timer.fetch_add(1, Ordering::Relaxed),
                Ordering::Relaxed,
            );
            return Ok(Arc::clone(entry.value()));
        }

        self.ensure_free_entry(&mut map).await?;

        let object = Arc::new(self.backend.load(key).await?);

        let new_entry = AsyncLruCacheEntry {
            value: Some(Arc::clone(&object)),
            last_used: AtomicUsize::new(self.lru_timer.fetch_add(1, Ordering::Relaxed)),
        };
        map.insert(key, new_entry);

        Ok(object)
    }

    /// Force-insert the given object into the cache.
    ///
    /// If there is an existing object under that key, it is flushed first.
    async fn insert(&self, key: K, value: Arc<V>) -> io::Result<()> {
        let mut map = self.map.write().await;
        if let Some(entry) = map.get_mut(&key) {
            entry.last_used.store(
                self.lru_timer.fetch_add(1, Ordering::Relaxed),
                Ordering::Relaxed,
            );
            let mut dep_guard = self.flush_before.lock().await;
            Self::flush_dependencies(&mut dep_guard).await?;
            self.backend.flush(key, entry.value()).await?;
            entry.value = Some(value);
        } else {
            self.ensure_free_entry(&mut map).await?;

            let new_entry = AsyncLruCacheEntry {
                value: Some(value),
                last_used: AtomicUsize::new(self.lru_timer.fetch_add(1, Ordering::Relaxed)),
            };
            map.insert(key, new_entry);
        }

        Ok(())
    }

    /// Flush all cache entries.
    ///
    /// Those entries are not evicted, but remain in the cache.
    #[instrument(
        level = "trace",
        name = "AsyncLruCache::flush",
        skip_all,
        fields(self = &self as *const _ as usize)
    )]
    async fn flush(&self) -> io::Result<()> {
        let mut futs = FutureVector::new();

        let mut dep_guard = self.flush_before.lock().await;
        Self::flush_dependencies(&mut dep_guard).await?;

        let map = self.map.read().await;
        for (key, entry) in map.iter() {
            let key = *key;
            let object = Arc::clone(entry.value());
            trace!("Flushing {key:?}");
            futs.push(Box::pin(
                async move { self.backend.flush(key, &object).await },
            ));
        }

        let mut first_err = None;
        while let Err(e) = futs.discarding_join().await {
            first_err.get_or_insert(e);
        }
        if let Some(e) = first_err {
            Err(e)
        } else {
            Ok(())
        }
    }

    /// Evict all cache entries.
    ///
    /// Evicts all cache entries without flushing them.
    ///
    /// # Safety
    /// Depending on the nature of the cache, this operation may be unsafe.  Perform at your own
    /// risk.
    #[instrument(
        level = "trace",
        name = "AsyncLruCache::invalidate",
        skip_all,
        fields(self = &self as *const _ as usize)
    )]
    async unsafe fn invalidate(&self) -> io::Result<()> {
        let mut in_use = Vec::new();

        let mut map = self.map.write().await;
        // Clear the map; we could use `.drain()`, but doing this allows the following loop to put
        // objects back into the new map in case they cannot be evicted.
        let old_map = mem::take(&mut *map);
        for (key, mut entry) in old_map {
            let object = entry.value.take().unwrap();
            trace!("Evicting {key:?}");
            match Arc::try_unwrap(object) {
                Ok(object) => {
                    // Caller guarantees this is safe
                    unsafe { self.backend.evict(key, object) };
                }

                Err(arc) => {
                    trace!("Entry is still in use, retaining it");
                    entry.value = Some(arc);
                    map.insert(key, entry);
                    in_use.push(key);
                }
            }
        }

        if in_use.is_empty() {
            self.flush_before.lock().await.clear();
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "Cannot invalidate cache, entries still in use: {}",
                in_use
                    .iter()
                    .map(|key| format!("{key:?}"))
                    .collect::<Vec<String>>()
                    .join(", "),
            )))
        }
    }
}

impl<V> AsyncLruCacheEntry<V> {
    /// Return the cached object.
    fn value(&self) -> &Arc<V> {
        self.value.as_ref().unwrap()
    }
}

#[async_trait(?Send)]
impl<
        K: Clone + Copy + Debug + PartialEq + Eq + Hash + Send + Sync,
        V: Send + Sync,
        B: AsyncLruCacheBackend<Key = K, Value = V>,
    > FlushableCache for AsyncLruCacheInner<K, V, B>
{
    async fn flush(&self) -> io::Result<()> {
        AsyncLruCacheInner::<K, V, B>::flush(self).await
    }

    async fn check_circular(&self, other: &Arc<dyn FlushableCache>) -> bool {
        let deps = self.flush_before.lock().await;
        for dep in deps.iter() {
            if Arc::ptr_eq(dep, other) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Minimal backend for testing: load returns the key, flush is a no-op
    struct DummyBackend;

    impl AsyncLruCacheBackend for DummyBackend {
        type Key = usize;
        type Value = usize;

        async fn load(&self, key: usize) -> io::Result<usize> {
            Ok(key)
        }

        async fn flush(&self, _key: usize, _value: &usize) -> io::Result<()> {
            Ok(())
        }

        unsafe fn evict(&self, _key: usize, _value: usize) {}
    }

    /// Backend that records flush calls in order
    #[derive(Default)]
    struct RecordingBackend {
        flushed: std::sync::Mutex<Vec<(usize, usize)>>,
    }

    impl AsyncLruCacheBackend for RecordingBackend {
        type Key = usize;
        type Value = usize;

        async fn load(&self, key: usize) -> io::Result<usize> {
            Ok(key)
        }

        async fn flush(&self, key: usize, value: &usize) -> io::Result<()> {
            self.flushed.lock().unwrap().push((key, *value));
            Ok(())
        }

        unsafe fn evict(&self, _key: usize, _value: usize) {}
    }

    impl<B: AsyncLruCacheBackend> AsyncLruCacheBackend for Arc<B> {
        type Key = <B as AsyncLruCacheBackend>::Key;
        type Value = <B as AsyncLruCacheBackend>::Value;

        async fn load(&self, key: Self::Key) -> io::Result<Self::Value> {
            (**self).load(key).await
        }

        async fn flush(&self, key: Self::Key, value: &Self::Value) -> io::Result<()> {
            (**self).flush(key, value).await
        }

        unsafe fn evict(&self, key: Self::Key, value: Self::Value) {
            unsafe { (**self).evict(key, value) }
        }
    }

    /// `flush()` must continue past individual entry errors and report the first one, not stop at
    /// the first failure
    #[tokio::test]
    async fn test_flush_continues_past_errors() {
        #[derive(Default)]
        struct FailOddBackend {
            flush_count: AtomicUsize,
        }

        impl AsyncLruCacheBackend for FailOddBackend {
            type Key = usize;
            type Value = usize;

            async fn load(&self, key: usize) -> io::Result<usize> {
                Ok(key)
            }

            async fn flush(&self, key: usize, _value: &usize) -> io::Result<()> {
                self.flush_count.fetch_add(1, Ordering::Relaxed);
                if key % 2 == 1 {
                    Err(io::Error::other("odd key"))
                } else {
                    Ok(())
                }
            }

            unsafe fn evict(&self, _key: usize, _value: usize) {}
        }

        const ENTRIES: usize = 42;

        let backend = Arc::new(FailOddBackend::default());
        let cache = AsyncLruCache::new(Arc::clone(&backend), ENTRIES);

        for i in 0..ENTRIES {
            cache.get_or_insert(i).await.unwrap();
        }

        let err = cache.flush().await.unwrap_err();
        assert!(err.to_string().contains("odd key"));

        assert_eq!(backend.flush_count.load(Ordering::Relaxed), ENTRIES);
    }

    /// Eviction must remove the least-recently-used entry
    #[tokio::test]
    async fn test_lru_eviction_order() {
        const ENTRIES: usize = 3;

        let backend = Arc::new(RecordingBackend::default());
        let cache = AsyncLruCache::new(Arc::clone(&backend), ENTRIES);

        for i in 0..ENTRIES {
            cache.get_or_insert(i).await.unwrap();
        }

        // Touch key 0 so it becomes most-recently-used
        cache.get_or_insert(0).await.unwrap();

        // Insert one more key — must evict key 1 (the oldest untouched)
        cache.get_or_insert(ENTRIES).await.unwrap();

        assert_eq!(*backend.flushed.lock().unwrap(), [(1, 1)]);
    }

    /// Entries with external `Arc` references must not be evicted
    #[tokio::test]
    async fn test_in_use_entries_not_evicted() {
        let backend = Arc::new(RecordingBackend::default());
        let cache = AsyncLruCache::new(Arc::clone(&backend), 2);

        let held = cache.get_or_insert(0).await.unwrap();
        cache.get_or_insert(1).await.unwrap();

        // Insert key 2 — key 0 is oldest but in use, so key 1 must be evicted
        cache.get_or_insert(2).await.unwrap();

        assert_eq!(*backend.flushed.lock().unwrap(), [(1, 1)]);
        assert_eq!(*held, 0);
    }

    /// When all entries are in use, eviction must fail with an error
    #[tokio::test]
    async fn test_cache_full_all_in_use() {
        const ENTRIES: usize = 23;

        let cache = AsyncLruCache::new(DummyBackend, ENTRIES);

        let mut held = vec![];
        for i in 0..ENTRIES {
            held.push(cache.get_or_insert(i).await.unwrap());
        }

        let err = cache.get_or_insert(ENTRIES).await.unwrap_err();
        assert!(err.to_string().contains("everything is in use"));
    }

    /// `invalidate()` must retain entries that are still in use and evict the rest
    #[tokio::test]
    async fn test_invalidate_retains_in_use() {
        let cache = AsyncLruCache::new(DummyBackend, 16);

        let held = cache.get_or_insert(0).await.unwrap();
        cache.get_or_insert(1).await.unwrap();
        cache.get_or_insert(2).await.unwrap();

        let err = unsafe { cache.invalidate() }.await.unwrap_err();
        assert!(err.to_string().contains("still in use"));

        let from_cache = cache.get_or_insert(0).await.unwrap();
        assert!(Arc::ptr_eq(&from_cache, &held));

        assert_eq!(cache.0.map.read().await.len(), 1);
    }

    /// When eviction flush fails, the entry must be re-inserted and remain accessible
    #[tokio::test]
    async fn test_eviction_flush_failure_reinserts_entry() {
        struct FailFlushBackend;

        impl AsyncLruCacheBackend for FailFlushBackend {
            type Key = usize;
            type Value = usize;

            async fn load(&self, key: usize) -> io::Result<usize> {
                Ok(key)
            }

            async fn flush(&self, _key: usize, _value: &usize) -> io::Result<()> {
                Err(io::Error::other("flush failed"))
            }

            unsafe fn evict(&self, _key: usize, _value: usize) {}
        }

        const ENTRIES: usize = 2;

        let cache = AsyncLruCache::new(FailFlushBackend, ENTRIES);

        for i in 0..ENTRIES {
            cache.get_or_insert(i).await.unwrap();
        }

        // Eviction flush fails
        let err = cache.get_or_insert(ENTRIES).await.unwrap_err();
        assert!(err.to_string().contains("flush failed"));

        // All original entries must still be in the cache
        assert_eq!(cache.0.map.read().await.len(), ENTRIES);
        for i in 0..ENTRIES {
            let entry = cache.get_or_insert(i).await.unwrap();
            assert_eq!(*entry, i);
        }

        // New entry was never inserted
        let err = cache.get_or_insert(ENTRIES).await.unwrap_err();
        assert!(err.to_string().contains("flush failed"));
    }

    /// `insert()` over an existing key must flush the old value first
    #[tokio::test]
    async fn test_insert_flushes_existing() {
        let backend = Arc::new(RecordingBackend::default());
        let cache = AsyncLruCache::new(Arc::clone(&backend), 16);

        cache.get_or_insert(5).await.unwrap();
        cache.insert(5, Arc::new(55)).await.unwrap();

        assert_eq!(*backend.flushed.lock().unwrap(), [(5, 5)]);
        assert_eq!(*cache.get_or_insert(5).await.unwrap(), 55);
        assert_eq!(cache.0.map.read().await.len(), 1);
    }
}
