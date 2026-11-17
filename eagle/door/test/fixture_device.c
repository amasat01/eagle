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
 *   dcells   MIN and MAX cells over rows of 4 samples (slots tot, mx, x).
 *   dkeyed   a keyed accumulate with a companion and its device fold function (slots acc, acc_c, x).
 *   dkeyednf the same naming a fold function the image does not have (eg_unit_load: FAULT).
 *   dauto    an automatic unit with a persist entry; dautorange the same with a range entry .
 *   dkeyedh  the same with a host fold entry and no device fold function (a DEVICE plan: NOT_WRITTEN).
 *   dcount   a finishing unit, one step per launch (slots k, limit, mask, finished, geom); geom holds the launch's
 *            threads per block and blocks as the kernel saw them.
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

/* cells and a companion: ACCUM_OUT slots, cell rows of 4 samples, a keyed accumulate over the lane axis "key" */
#define ACC(var, name_) SLOT(var, EG_ROLE_ACCUM_OUT, EG_DTYPE_REAL, 0, NULL, name_)
ACC(slot_tot, "tot");
ACC(slot_mx, "mx");
ACC(slot_acc, "acc");
ACC(slot_acc_c, "acc_c");
static const int32_t cell_tot_slots[] = { 0 }, cell_mx_slots[] = { 1 };
static const struct eg_unit_cell cell_tot = { .struct_size = sizeof(struct eg_unit_cell), .op = EG_REDUCE_MIN, .columns = 1,
                                              .slots = cell_tot_slots, .name = "tot" };
static const struct eg_unit_cell cell_mx = { .struct_size = sizeof(struct eg_unit_cell), .op = EG_REDUCE_MAX, .columns = 1,
                                             .slots = cell_mx_slots, .name = "mx" };
static const struct eg_unit_cell* const cell_list[] = { &cell_tot, &cell_mx };
static const struct eg_unit_slot* const cells_slots[] = { &slot_tot, &slot_mx, &slot_x };
static const struct eg_unit_body cells_body = { .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL,
    .n_slots = 3, .n_cells = 2, .cell_row = 4, .slots = cells_slots, .cells = cell_list };
#define COMP(var, host_, device_) static const struct eg_unit_companion var = { .struct_size = sizeof(struct eg_unit_companion), \
    .slot = 1, .target_slot = 0, .host_fold_entry = host_, .device_fold_function = device_ }
COMP(comp_ok, NULL, "dkeyed__fold_acc");
COMP(comp_nofn, NULL, "absent__fold");
COMP(comp_host, "absent__fold_host", NULL);
static const struct eg_unit_companion* const comp_ok_list[] = { &comp_ok };
static const struct eg_unit_companion* const comp_nofn_list[] = { &comp_nofn };
static const struct eg_unit_companion* const comp_host_list[] = { &comp_host };
static const struct eg_unit_slot* const keyed_slots[] = { &slot_acc, &slot_acc_c, &slot_x };
#define KEYED_BODY(var, list_) static const struct eg_unit_body var = { .struct_size = sizeof(struct eg_unit_body), \
    .access = EG_ACCESS_CROSS_SAMPLE_WRITE, .n_slots = 3, .n_companions = 1, .slots = keyed_slots, .companions = list_ }
KEYED_BODY(keyed_ok_body, comp_ok_list);
KEYED_BODY(keyed_nofn_body, comp_nofn_list);
KEYED_BODY(keyed_host_body, comp_host_list);
static const char* const key_lane_labels[] = { "key", NULL };
static const int64_t key_lane_extents[] = { -1, 4 };
static const char* const key_axis_labels[] = { "key" };
static const int64_t key_axis_extents[] = { -1 };
static const struct eg_unit_plane_axes key_axes = { .struct_size = sizeof(struct eg_unit_plane_axes), .slot = 0, .n_axes = 1,
                                                    .labels = key_axis_labels, .extents = key_axis_extents };
static const struct eg_unit_plane_axes* const key_axes_list[] = { &key_axes };
static const struct eg_unit_launch launch_keyed = {
    .struct_size = sizeof(struct eg_unit_launch), .n_lane_axes = 2, .active_map_slot = -1, .active_count_slot = -1,
    .index_map_slot = -1, .index_count_slot = -1, .n_plane_axes = 1, .group = 1, .lane_labels = key_lane_labels,
    .lane_extents = key_lane_extents, .plane_axes = key_axes_list };
#define MADE_UNIT(var, name_, byte, function_, body_, launch_)                                              \
    static const struct eg_unit_meta var = {                                                                \
        .struct_size = sizeof(struct eg_unit_meta), .name = name_, .digest = DIGEST(byte),                 \
        .pattern = EG_PATTERN_PURE, .scalar_type = EG_SCALAR_FLOAT64, .device_image = "fixture__image",    \
        .device_function = function_, .device_image_format = EG_IMAGE_FATBIN, .body = &body_, .launch = &launch_, \
        .device_image_size = "fixture__image_size" }
/* The images of nvcc 12.9 (the R4 rows): a fatbin (sm_61 code and compute_61 PTX) that loads, PTX only (the 12.6 driver refuses
 * it: PTX 8.8), and a cubin for sm_70 alone (no machine code for an sm_61 device). Same kernel, function "axpy". */
#define IMAGE_UNIT(var, name_, byte, sym, format_, isa_)                                                    \
    static const struct eg_unit_meta var = {                                                                \
        .struct_size = sizeof(struct eg_unit_meta), .name = name_, .digest = DIGEST(byte),                 \
        .pattern = EG_PATTERN_PURE, .scalar_type = EG_SCALAR_FLOAT64, .device_image = sym "__image",         \
        .device_function = "axpy", .device_image_format = format_, .body = &body, .launch = &launch_any,     \
        .device_image_size = sym "__image_size", .device_compiler = "nvcc 12.9", .device_ptx_isa = isa_ }
IMAGE_UNIT(unit_fat129, "fat129", 0xE1, "fat129", EG_IMAGE_FATBIN, 88);
IMAGE_UNIT(unit_ptx129, "ptx129", 0xE2, "ptx129", EG_IMAGE_PTX, 88);
IMAGE_UNIT(unit_sass70, "sass70", 0xE3, "sass70", EG_IMAGE_FATBIN, 0);

MADE_UNIT(unit_dcells, "dcells", 0xD2, "dcells", cells_body, launch_any);
MADE_UNIT(unit_dkeyed, "dkeyed", 0xD3, "dkeyed", keyed_ok_body, launch_keyed);
MADE_UNIT(unit_dkeyednf, "dkeyednf", 0xD4, "dkeyed", keyed_nofn_body, launch_keyed);
MADE_UNIT(unit_dkeyedh, "dkeyedh", 0xD5, "dkeyed", keyed_host_body, launch_keyed);

SLOT(slot_icount, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, 0, NULL, "index_count");
static const struct eg_unit_slot* const prefix_slots[] = { &slot_y, &slot_x, &slot_icount };
static const struct eg_unit_body prefix_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 3, .slots = prefix_slots };
static const struct eg_unit_launch launch_prefix = {
    .struct_size = sizeof(struct eg_unit_launch), .active_map_slot = -1, .active_count_slot = -1, .index_map_slot = -1,
    .index_count_slot = 2, .group = 1 };
MADE_UNIT(unit_dprefix, "dprefix", 0xD6, "dprefix", prefix_body, launch_prefix);

SLOT(slot_k, EG_ROLE_MUTABLE, EG_DTYPE_REAL, 0, NULL, "k");
SLOT(slot_limit, EG_ROLE_PER_SAMPLE, EG_DTYPE_REAL, 0, NULL, "limit");
SLOT(slot_mask, EG_ROLE_TERMINATED, EG_DTYPE_BOOL, 0, NULL, "mask");
SLOT(slot_finished, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, 0, NULL, "finished");
SLOT(slot_geom, EG_ROLE_LOOKUP, EG_DTYPE_REAL, 0, NULL, "geom");
static const struct eg_unit_slot* const count_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_geom };
static const struct eg_unit_body count_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 5, .slots = count_slots };
static const struct eg_unit_finish count_finish = {
    .struct_size = sizeof(struct eg_unit_finish), .flags = 0, .mask_slot = 2, .counter_slot = 3, .steps_slot = -1,
    .steps = 1, .steps_max = 0, .step_ops = -1, .fp64_ops_ir = -1, .host_steps_reverse = NULL, .device_steps_reverse = NULL };
static const struct eg_unit_meta unit_dcount = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "dcount", .digest = DIGEST(0xD7), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .device_image = "fixture__image", .device_function = "dcount",
    .device_image_format = EG_IMAGE_FATBIN, .body = &count_body, .launch = &launch_any, .finish = &count_finish,
    .device_image_size = "fixture__image_size" };

SLOT(slot_map, EG_ROLE_LOOKUP, EG_DTYPE_INT32, 0, NULL, "active_map");
SLOT(slot_live, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, 0, NULL, "active_count");
static const struct eg_unit_slot* const count1_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished };
static const struct eg_unit_slot* const count1_active_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_map, &slot_live };
static const struct eg_unit_body count1_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 4, .slots = count1_slots };
static const struct eg_unit_body count_active_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 6, .slots = count1_active_slots };
static const struct eg_unit_launch launch_count_active = {
    .struct_size = sizeof(struct eg_unit_launch), .active_map_slot = 4, .active_count_slot = 5, .index_map_slot = -1,
    .index_count_slot = -1, .group = 1 };
static const struct eg_unit_finish finish_one = {
    .struct_size = sizeof(struct eg_unit_finish), .mask_slot = 2, .counter_slot = 3, .steps_slot = -1, .steps = 1,
    .step_ops = -1, .fp64_ops_ir = -1 };
#define COUNT_UNIT(var, name_, byte, function_, body_, launch_)                                             \
    static const struct eg_unit_meta var = {                                                                \
        .struct_size = sizeof(struct eg_unit_meta), .name = name_, .digest = DIGEST(byte),                 \
        .pattern = EG_PATTERN_PURE, .scalar_type = EG_SCALAR_FLOAT64, .device_image = "fixture__image",    \
        .device_function = function_, .device_image_format = EG_IMAGE_FATBIN, .body = &body_, .launch = &launch_, \
        .finish = &finish_one, .device_image_size = "fixture__image_size" }
COUNT_UNIT(unit_dcount1, "dcount1", 0xD9, "dcount1", count1_body, launch_any);
COUNT_UNIT(unit_dcount1a, "dcount1a", 0xDA, "dcount1a", count_active_body, launch_count_active);

/* dauto / dautorange: an automatic unit (finish steps AUTO) with a persist entry (dauto), and with a range entry (dautorange).
 * Slots: hits (uint32 per sample), geo (int64 per sample: the persist launch's grid, block, chunk), done (the mask), the
 * finished counter and the steps word (eagle-made). */
SLOT(slot_hits, EG_ROLE_MUTABLE, EG_DTYPE_UINT32, 0, NULL, "hits");
SLOT(slot_geo, EG_ROLE_MUTABLE, EG_DTYPE_INT64, 0, NULL, "geo");
SLOT(slot_done, EG_ROLE_TERMINATED, EG_DTYPE_BOOL, 0, NULL, "done");
SLOT(slot_stepword, EG_ROLE_LOOKUP, EG_DTYPE_INT64, 0, NULL, "step_word");
static const struct eg_unit_slot* const auto_slots[] = { &slot_hits, &slot_geo, &slot_done, &slot_finished, &slot_stepword };
static const struct eg_unit_body auto_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 5, .slots = auto_slots };
static const struct eg_unit_finish auto_finish = {
    .struct_size = sizeof(struct eg_unit_finish), .flags = EG_FINISH_ENTRY_COUNTS, .mask_slot = 2, .counter_slot = 3,
    .steps_slot = 4, .steps = EG_STEPS_AUTO, .steps_max = 1, .step_ops = -1, .fp64_ops_ir = -1 };
#define AUTO_UNIT(var, name_, byte, function_)                                                              \
    static const struct eg_unit_meta var = {                                                                \
        .struct_size = sizeof(struct eg_unit_meta), .name = name_, .digest = DIGEST(byte),                 \
        .pattern = EG_PATTERN_PURE, .scalar_type = EG_SCALAR_FLOAT64, .device_image = "fixture__image",    \
        .device_function = function_, .device_image_format = EG_IMAGE_FATBIN, .body = &auto_body,           \
        .launch = &launch_any, .finish = &auto_finish, .device_image_size = "fixture__image_size" }
AUTO_UNIT(unit_dauto, "dauto", 0xDB, "dauto");
AUTO_UNIT(unit_dautorange, "dautorange", 0xDC, "dautorange");

static const struct eg_unit_meta* const units[] = { &unit_axpy, &unit_axpy64, &unit_boxdev, &unit_banded, &unit_nofn,
                                                    &unit_noimg, &unit_dcells, &unit_dkeyed, &unit_dkeyednf, &unit_dkeyedh, &unit_fat129, &unit_ptx129, &unit_sass70, &unit_dprefix, &unit_dcount, &unit_dcount1, &unit_dcount1a, &unit_dauto, &unit_dautorange };

static const struct eg_library_meta library = {
    .struct_size = sizeof(struct eg_library_meta), .schema_major = EG_META_SCHEMA_MAJOR, .n_units = 19, .units = units,
};

__attribute__((visibility("default"))) const struct eg_library_meta* eg_unit_library_meta(void) { return &library; }
