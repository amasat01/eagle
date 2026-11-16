/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The device fixture's meta: units over the one kernel of fixture_device.cu, whose fatbin (sm_61) the build
 * embeds as fixture__image / fixture__image_size.
 *
 *   axpy     y = fma(a, x, v[0]) + n * v[1]: MUTABLE y, PER_SAMPLE x, VEC_IN v (2 components), UNIFORM a,
 *            NSAMPLES n; no block stated (eagle's choice).
 *   axpy64   the same with launch.block 64.
 *   boxdev   the same with a device thread box (refused on DEVICE).
 *   banded   the same as a BANDED unit (refused on DEVICE).
 *   nofn     a device function the image does not have (eg_unit_load: FAULT).
 *   noimg    a device image symbol the library does not export (eg_unit_load: FAULT).
 */
#include "eagle/door/eg_door.h"
#include "eagle/door/eg_unit_meta.h"

#define SLOT(var, role_, dtype_, rank_, shape_, name_)                                                      \
    static const struct eg_unit_slot var = { .struct_size = sizeof(struct eg_unit_slot), .role = role_,    \
                                             .dtype = dtype_, .rank = rank_, .like_slot = -1,             \
                                             .shape = shape_, .name = name_ }

static const int64_t v_shape[] = { 2 };
SLOT(slot_y, EG_ROLE_MUTABLE, EG_DTYPE_REAL, 0, NULL, "y");
SLOT(slot_x, EG_ROLE_PER_SAMPLE, EG_DTYPE_REAL, 0, NULL, "x");
SLOT(slot_v, EG_ROLE_VEC_IN, EG_DTYPE_REAL, 1, v_shape, "v");
SLOT(slot_a, EG_ROLE_UNIFORM, EG_DTYPE_REAL, 0, NULL, "a");
SLOT(slot_n, EG_ROLE_NSAMPLES, EG_DTYPE_UINT32, 0, NULL, "n");

static const struct eg_unit_slot* const slots[] = { &slot_y, &slot_x, &slot_v, &slot_a, &slot_n };

static const struct eg_unit_body body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 5, .slots = slots,
};

static const struct eg_unit_box box = { .struct_size = sizeof(struct eg_unit_box), .axis = 0, .t = 2, .stride = 1 };

#define LAUNCH(var, block_, device_box_)                                                                    \
    static const struct eg_unit_launch var = { .struct_size = sizeof(struct eg_unit_launch), .block = block_, \
                                               .active_map_slot = -1, .active_count_slot = -1,             \
                                               .index_map_slot = -1, .index_count_slot = -1, .group = 1,   \
                                               .device_box = device_box_ }
LAUNCH(launch_any, 0, NULL);
LAUNCH(launch_64, 64, NULL);
LAUNCH(launch_box, 0, &box);

#define DIGEST(c) { c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c }

#define UNIT(var, name_, byte, scalar_, image_, function_, launch_)                                         \
    static const struct eg_unit_meta var = {                                                                \
        .struct_size = sizeof(struct eg_unit_meta), .name = name_, .digest = DIGEST(byte),                 \
        .pattern = EG_PATTERN_PURE, .scalar_type = scalar_, .device_image = image_ "__image",               \
        .device_function = function_, .device_image_format = EG_IMAGE_FATBIN, .body = &body, .launch = &launch_, \
        .device_image_size = image_ "__image_size" }
UNIT(unit_axpy, "axpy", 0xC1, EG_SCALAR_FLOAT64, "fixture", "axpy", launch_any);
UNIT(unit_axpy64, "axpy64", 0xC2, EG_SCALAR_FLOAT64, "fixture", "axpy", launch_64);
UNIT(unit_boxdev, "boxdev", 0xC3, EG_SCALAR_FLOAT64, "fixture", "axpy", launch_box);
UNIT(unit_banded, "banded", 0xC4, EG_SCALAR_BANDED, "fixture", "axpy", launch_any);
UNIT(unit_nofn, "nofn", 0xC5, EG_SCALAR_FLOAT64, "fixture", "missing", launch_any);
UNIT(unit_noimg, "noimg", 0xC6, EG_SCALAR_FLOAT64, "absent", "axpy", launch_any);

static const struct eg_unit_meta* const units[] = { &unit_axpy, &unit_axpy64, &unit_boxdev,
                                                    &unit_banded, &unit_nofn, &unit_noimg };

static const struct eg_library_meta library = {
    .struct_size = sizeof(struct eg_library_meta), .schema_major = EG_META_SCHEMA_MAJOR, .n_units = 6, .units = units,
};

__attribute__((visibility("default"))) const struct eg_library_meta* eg_unit_library_meta(void) { return &library; }
