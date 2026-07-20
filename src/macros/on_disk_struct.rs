//! Macro to allow binary structures be read/stored from/to disk.

#![allow(dead_code, unused_imports, unused_macros)] // TODO: Use and remove

use std::io;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// A structure to be loaded from or stored on disk (e.g. qcow2 image header), in binary format.
pub(crate) trait OnDiskStruct: Sized {
    /// The on-disk size of this structure, in bytes
    const ON_DISK_SIZE: usize;

    /// Convenience alias for [`Self::ON_DISK_SIZE`] for objects
    fn on_disk_size(&self) -> usize {
        Self::ON_DISK_SIZE
    }

    /// Load this structure from the given slice.
    ///
    /// Use little endian, unless the structure has a specific inherent endianness.
    #[allow(dead_code)]
    fn load_from_le(bytes: &[u8]) -> io::Result<Self> {
        Self::load_from(bytes)
    }

    /// Store this structure in the given slice.
    ///
    /// Use little endian, unless the structure has a specific inherent endianness.
    #[allow(dead_code)]
    fn store_to_le(&self, bytes: &mut [u8]) -> io::Result<()> {
        self.store_to(bytes)
    }

    /// Load this structure from the given slice.
    ///
    /// Use big endian, unless the structure has a specific inherent endianness.
    fn load_from_be(bytes: &[u8]) -> io::Result<Self> {
        Self::load_from(bytes)
    }

    /// Store this structure in the given slice.
    ///
    /// Use big endian, unless the structure has a specific inherent endianness.
    fn store_to_be(&self, bytes: &mut [u8]) -> io::Result<()> {
        self.store_to(bytes)
    }

    /// Load this structure from the given slice, in default inherent endianness.
    fn load_from(bytes: &[u8]) -> io::Result<Self>;

    /// Store this structure in the given slice, in default inherent endianness.
    fn store_to(&self, bytes: &mut [u8]) -> io::Result<()>;
}

/// Implement [`OnDiskStruct`] for a type wrapping a primitive type.
///
/// First argument: The wrapping type, e.g. `AtomicU16`; second argument: The inner primitive type
/// (e.g. [`std::sync::atomic::AtomicPrimitive::Storage`], which just is not yet stable); third
/// argument: The wrapping function (for loading); fourth argument: The unwrapping function (for
/// storing).
macro_rules! impl_on_disk_struct_wrapped_primitive {
    ($type:ty, $raw_type:ty, $wrap:expr, $unwrap:expr) => {
        impl $crate::macros::on_disk_struct::OnDiskStruct for $type {
            const ON_DISK_SIZE: usize = std::mem::size_of::<$raw_type>();

            fn load_from_le(bytes: &[u8]) -> std::io::Result<Self> {
                Ok($wrap(<$raw_type>::load_from_le(bytes)?))
            }

            fn store_to_le(&self, bytes: &mut [u8]) -> std::io::Result<()> {
                let raw = $unwrap(self);
                raw.store_to_le(bytes)
            }

            fn load_from_be(bytes: &[u8]) -> std::io::Result<Self> {
                Ok($wrap(<$raw_type>::load_from_be(bytes)?))
            }

            fn store_to_be(&self, bytes: &mut [u8]) -> std::io::Result<()> {
                let raw = $unwrap(self);
                raw.store_to_be(bytes)
            }

            fn load_from(bytes: &[u8]) -> std::io::Result<Self> {
                Ok($wrap(<$raw_type>::load_from(bytes)?))
            }

            fn store_to(&self, bytes: &mut [u8]) -> std::io::Result<()> {
                let raw = $unwrap(self);
                raw.store_to(bytes)
            }
        }
    };
}

/// Implement [`OnDiskStruct`] for a primitive integer type.
macro_rules! impl_on_disk_struct_primitive {
    ($type:ty) => {
        impl $crate::macros::on_disk_struct::OnDiskStruct for $type {
            const ON_DISK_SIZE: usize = std::mem::size_of::<$type>();

            fn load_from_le(bytes: &[u8]) -> std::io::Result<Self> {
                Ok(<$type>::from_le_bytes(bytes.try_into().map_err(|_| {
                    $crate::misc_helpers::invalid_data(format!(
                        "Cannot load type {} (length {}) from buffer of length {}",
                        std::any::type_name::<$type>(),
                        Self::ON_DISK_SIZE,
                        bytes.len(),
                    ))
                })?))
            }

            fn store_to_le(&self, bytes: &mut [u8]) -> std::io::Result<()> {
                if bytes.len() != Self::ON_DISK_SIZE {
                    return Err($crate::misc_helpers::invalid_data(format!(
                        "Cannot write type {} (length {}) into buffer of length {}",
                        std::any::type_name::<$type>(),
                        Self::ON_DISK_SIZE,
                        bytes.len(),
                    )));
                }

                bytes.copy_from_slice(&self.to_le_bytes());
                Ok(())
            }

            fn load_from_be(bytes: &[u8]) -> std::io::Result<Self> {
                Ok(<$type>::from_be_bytes(bytes.try_into().map_err(|_| {
                    $crate::misc_helpers::invalid_data(format!(
                        "Cannot load type {} (length {}) from buffer of length {}",
                        std::any::type_name::<$type>(),
                        Self::ON_DISK_SIZE,
                        bytes.len(),
                    ))
                })?))
            }

            fn store_to_be(&self, bytes: &mut [u8]) -> std::io::Result<()> {
                if bytes.len() != Self::ON_DISK_SIZE {
                    return Err($crate::misc_helpers::invalid_data(format!(
                        "Cannot write type {} (length {}) into buffer of length {}",
                        std::any::type_name::<$type>(),
                        Self::ON_DISK_SIZE,
                        bytes.len(),
                    )));
                }

                bytes.copy_from_slice(&self.to_be_bytes());
                Ok(())
            }

            fn load_from(bytes: &[u8]) -> std::io::Result<Self> {
                Ok(<$type>::from_ne_bytes(bytes.try_into().map_err(|_| {
                    $crate::misc_helpers::invalid_data(format!(
                        "Cannot load type {} (length {}) from buffer of length {}",
                        std::any::type_name::<$type>(),
                        Self::ON_DISK_SIZE,
                        bytes.len(),
                    ))
                })?))
            }

            fn store_to(&self, bytes: &mut [u8]) -> std::io::Result<()> {
                if bytes.len() != Self::ON_DISK_SIZE {
                    return Err($crate::misc_helpers::invalid_data(format!(
                        "Cannot write type {} (length {}) into buffer of length {}",
                        std::any::type_name::<$type>(),
                        Self::ON_DISK_SIZE,
                        bytes.len(),
                    )));
                }

                bytes.copy_from_slice(&self.to_ne_bytes());
                Ok(())
            }
        }
    };
}

impl_on_disk_struct_primitive!(u16);
impl_on_disk_struct_primitive!(u32);
impl_on_disk_struct_primitive!(u64);
impl_on_disk_struct_wrapped_primitive!(AtomicU32, u32, Into::into, |x: &AtomicU32| x
    .load(Ordering::Relaxed));
impl_on_disk_struct_wrapped_primitive!(AtomicU64, u64, Into::into, |x: &AtomicU64| x
    .load(Ordering::Relaxed));

/// Implement [`OnDiskStruct`] for the contained `struct` definition.
///
/// The struct name must be followed by a specification of endianness and whether to allow gaps in
/// the field offsets, i.e.: `struct <Struct>/<LE, BE, NE>, <no_gaps, allow_gaps>`.
///
/// All field types must be annotated with their byte offset in the structure, e.g. `foo: u32[42]`.
macro_rules! on_disk_struct {
    (
        $(#[$attr:meta])*
        struct $struct_name:ident/$endianness:ident, $packed:ident {
            $(
                $(#[$id_attr:meta])*
                $identifier:ident: $type:ty[$offset:literal],
            )+
        }
    ) => {
        $(#[$attr])*
        struct $struct_name {
            $(
                $(#[$id_attr])*
                $identifier: $type,
            )+
        }

        // Verify strict field ordering
        const _: () = const {
            let mut next = 0;
            $(
                $crate::macros::on_disk_struct::on_disk_struct_helper!(@check_layout $packed, $offset, next);
                next = $offset + <$type>::ON_DISK_SIZE;
            )+
            let _ = next;
        };

        impl $crate::macros::on_disk_struct::OnDiskStruct for $struct_name {
            const ON_DISK_SIZE: usize = $crate::macros::last_element!($($offset + <$type>::ON_DISK_SIZE),+);

            fn load_from(bytes: &[u8]) -> std::io::Result<Self> {
                if bytes.len() < Self::ON_DISK_SIZE {
                    return Err($crate::misc_helpers::invalid_data(format!(
                        "Cannot read struct {} (length {}) from buffer of length {}",
                        std::any::type_name::<$struct_name>(),
                        Self::ON_DISK_SIZE,
                        bytes.len(),
                    )));
                }

                Ok($struct_name {
                    $(
                        $identifier: $crate::macros::on_disk_struct::on_disk_struct_helper!(
                            @load $endianness,
                            $type,
                            &bytes[$offset..($offset + <$type>::ON_DISK_SIZE)]
                        )?,
                    )+
                })
            }

            fn store_to(&self, bytes: &mut [u8]) -> std::io::Result<()> {
                if bytes.len() < Self::ON_DISK_SIZE {
                    return Err($crate::misc_helpers::invalid_data(format!(
                        "Cannot write struct {} (length {}) into buffer of length {}",
                        std::any::type_name::<$struct_name>(),
                        Self::ON_DISK_SIZE,
                        bytes.len(),
                    )));
                }

                $(
                    $crate::macros::on_disk_struct::on_disk_struct_helper!(
                        @store $endianness,
                        self.$identifier,
                        &mut bytes[$offset..($offset + <$type>::ON_DISK_SIZE)]
                    )?;
                )+

                Ok(())
            }
        }
    }
}

pub(crate) use on_disk_struct;

/// Helper macro for [`on_disk_struct`].
///
/// Various functionalities, selected by the first parameter.
macro_rules! on_disk_struct_helper {
    // Check a field offset against the actual offset within the struct, either allowing gaps or not
    (@check_layout no_gaps, $field_offset:literal, $actual_offset:expr) => {
        assert!($field_offset == $actual_offset);
    };
    (@check_layout allow_gaps, $field_offset:literal, $actual_offset:expr) => {
        assert!($field_offset >= $actual_offset);
    };

    // Load with an endianness specified
    (@load NE, $type:ty, $slice:expr) => {
        <$type>::load_from($slice)
    };
    (@load LE, $type:ty, $slice:expr) => {
        <$type>::load_from_le($slice)
    };
    (@load BE, $type:ty, $slice:expr) => {
        <$type>::load_from_be($slice)
    };

    // Store with an endianness specified
    (@store NE, $value:expr, $slice:expr) => {
        $value.store_to($slice)
    };
    (@store LE, $value:expr, $slice:expr) => {
        $value.store_to_le($slice)
    };
    (@store BE, $value:expr, $slice:expr) => {
        $value.store_to_be($slice)
    };
}

pub(crate) use on_disk_struct_helper;
