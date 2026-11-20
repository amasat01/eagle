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
#include "auto_costs.h"

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

/* dprefixsum: a prefix set with a SUM cell of rows of 4 samples (slots tot, tot_comp, x, index_count). */
ACC(slot_tot_c, "tot_comp");
static const int32_t cell_psum_comp[] = { 1 };
static const struct eg_unit_cell cell_psum = { .struct_size = sizeof(struct eg_unit_cell), .op = EG_REDUCE_SUM, .columns = 1,
                                               .slots = cell_tot_slots, .name = "tot", .comp_slots = cell_psum_comp };
static const struct eg_unit_cell* const cell_psum_list[] = { &cell_psum };
static const struct eg_unit_slot* const psum_slots[] = { &slot_tot, &slot_tot_c, &slot_x, &slot_icount };
static const struct eg_unit_body psum_body = { .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL,
    .n_slots = 4, .n_cells = 1, .cell_row = 4, .slots = psum_slots, .cells = cell_psum_list };
static const struct eg_unit_launch launch_psum = {
    .struct_size = sizeof(struct eg_unit_launch), .active_map_slot = -1, .active_count_slot = -1, .index_map_slot = -1,
    .index_count_slot = 3, .group = 1 };
MADE_UNIT(unit_dprefixsum, "dprefixsum", 0xDB, "dprefixsum", psum_body, launch_psum);

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

SLOT(slot_imap, EG_ROLE_LOOKUP, EG_DTYPE_INT32, 0, NULL, "index_map");
static const struct eg_unit_slot* const lookup_slots[] = { &slot_y, &slot_x, &slot_imap, &slot_icount };
static const struct eg_unit_body lookup_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 4, .slots = lookup_slots };
static const struct eg_unit_launch launch_lookup = {
    .struct_size = sizeof(struct eg_unit_launch), .active_map_slot = -1, .active_count_slot = -1, .index_map_slot = 2,
    .index_count_slot = 3, .group = 1 };
MADE_UNIT(unit_dlookup, "dlookup", 0xDD, "dlookup", lookup_body, launch_lookup);


/* SEGMENTED groups, as the host fixture's "sg" (static selector) and "dg" (dynamic selector, class = k >= 4): see test/fixture_unit.c. */
SLOT(slot_sg_offsets, EG_ROLE_LOOKUP, EG_DTYPE_INT64, 0, NULL, "sg_offsets");
SLOT(slot_sg_class, EG_ROLE_MUTABLE, EG_DTYPE_INT64, 0, NULL, "sg_class");
SLOT(slot_dg_offsets, EG_ROLE_LOOKUP, EG_DTYPE_INT64, 0, NULL, "dg_offsets");
SLOT(slot_dg_class, EG_ROLE_MUTABLE, EG_DTYPE_INT64, 0, NULL, "dg_class");
SLOT(slot_dg_stale, EG_ROLE_MUTABLE, EG_DTYPE_BOOL, 0, NULL, "dg_stale");
SLOT(slot_pos, EG_ROLE_MUTABLE, EG_DTYPE_REAL, 0, NULL, "pos");
static const struct eg_unit_slot* const sg_k_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_sg_offsets };
static const struct eg_unit_slot* const sg_cls_slots[] = { &slot_limit, &slot_sg_class };
static const struct eg_unit_slot* const dg_k_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_dg_offsets,
                                                         &slot_dg_class, &slot_dg_stale, &slot_pos };
static const struct eg_unit_slot* const dg_sweep_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_map, &slot_live };
#define SEG_REF(var, name_, byte) static const struct eg_unit_ref var = { .struct_size = sizeof(struct eg_unit_ref), .name = name_, .digest = DIGEST(byte) }
SEG_REF(ref_sg_cls, "dsg__cls", 0xF1);
SEG_REF(ref_sg_k0, "dsg__k0", 0xF2);
SEG_REF(ref_sg_k1, "dsg__k1", 0xF3);
SEG_REF(ref_dg_k0, "ddg__k0", 0xF5);
SEG_REF(ref_dg_k1, "ddg__k1", 0xF6);
SEG_REF(ref_dg_sweep, "ddg__sweep", 0xF7);
static const struct eg_unit_ref* const sg_branches[] = { &ref_sg_k0, &ref_sg_k1 };
static const struct eg_unit_ref* const dg_branches[] = { &ref_dg_k0, &ref_dg_k1 };
#define DISPATCH(var, strategy_, class_, stale_, offsets_, classify_, name_, branches_, sweep_)                     \
    static const struct eg_unit_dispatch var = { .struct_size = sizeof(struct eg_unit_dispatch), .strategy = strategy_, \
        .mix_slot = -1, .class_slot = class_, .stale_slot = stale_, .offsets_slot = offsets_, .classify = classify_,   \
        .name = name_, .k = 2, .branches = branches_, .sweep = sweep_ }
DISPATCH(disp_sg_cls, EG_DISPATCH_SEGMENTED, 1, -1, -1, NULL, "dsg", NULL, NULL);
DISPATCH(disp_sg_branch, EG_DISPATCH_SEGMENTED, -1, -1, 4, &ref_sg_cls, "dsg", sg_branches, NULL);
DISPATCH(disp_dg_branch, EG_DISPATCH_SEGMENTED, 5, 6, 4, NULL, "ddg", dg_branches, &ref_dg_sweep);
DISPATCH(disp_dg_sweep, EG_DISPATCH_SWITCH, -1, -1, -1, NULL, "ddg", NULL, NULL);
static const struct eg_unit_dispatch* const sg_cls_list[] = { &disp_sg_cls };
static const struct eg_unit_dispatch* const sg_branch_list[] = { &disp_sg_branch };
static const struct eg_unit_dispatch* const dg_branch_list[] = { &disp_dg_branch };
static const struct eg_unit_dispatch* const dg_sweep_list[] = { &disp_dg_sweep };
#define SEG_BODY(var, slots_, n_, list_) static const struct eg_unit_body var = { .struct_size = sizeof(struct eg_unit_body), \
    .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = n_, .slots = slots_, .n_dispatches = 1, .dispatches = list_ }
SEG_BODY(sg_cls_body, sg_cls_slots, 2, sg_cls_list);
SEG_BODY(sg_branch_body, sg_k_slots, 5, sg_branch_list);
SEG_BODY(dg_branch_body, dg_k_slots, 8, dg_branch_list);
SEG_BODY(dg_sweep_body, dg_sweep_slots, 6, dg_sweep_list);
#define SEG_UNIT(var, name_, byte, function_, body_, launch_, finish_)                                              \
    static const struct eg_unit_meta var = {                                                                       \
        .struct_size = sizeof(struct eg_unit_meta), .name = name_, .digest = DIGEST(byte),                        \
        .pattern = EG_PATTERN_PURE, .scalar_type = EG_SCALAR_FLOAT64, .device_image = "fixture__image",           \
        .device_function = function_, .device_image_format = EG_IMAGE_FATBIN, .body = &body_, .launch = &launch_, \
        .finish = finish_, .device_image_size = "fixture__image_size" }
SEG_UNIT(unit_sg_cls, "dsg__cls", 0xF1, "sgCls", sg_cls_body, launch_any, NULL);
SEG_UNIT(unit_sg_k0, "dsg__k0", 0xF2, "sgK0", sg_branch_body, launch_any, &finish_one);
SEG_UNIT(unit_sg_k1, "dsg__k1", 0xF3, "sgK1", sg_branch_body, launch_any, &finish_one);
COUNT_UNIT(unit_sgf, "dsgf", 0xF4, "sgf", count1_body, launch_any);
SEG_UNIT(unit_dg_k0, "ddg__k0", 0xF5, "dgK0", dg_branch_body, launch_any, &finish_one);
SEG_UNIT(unit_dg_k1, "ddg__k1", 0xF6, "dgK1", dg_branch_body, launch_any, &finish_one);
SEG_UNIT(unit_dg_sweep, "ddg__sweep", 0xF7, "dgSweep", dg_sweep_body, launch_count_active, &finish_one);
COUNT_UNIT(unit_dgf, "ddgf", 0xF8, "dgf", count1_body, launch_any);


/* The AUTO group "dsga": the host fixture's "sga" on the device (see test/fixture_unit.c). */
SLOT(slot_sga_offsets, EG_ROLE_LOOKUP, EG_DTYPE_INT64, 0, NULL, "sga_offsets");
SLOT(slot_sga_class, EG_ROLE_MUTABLE, EG_DTYPE_INT64, 0, NULL, "sga_class");
SLOT(slot_sga_mix, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, 0, NULL, "sga_mix");
SLOT(slot_how, EG_ROLE_MUTABLE, EG_DTYPE_REAL, 0, NULL, "how");
static const struct eg_unit_slot* const sga_k_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_sga_offsets, &slot_how };
static const struct eg_unit_slot* const sga_cls_slots[] = { &slot_limit, &slot_sga_class };
static const struct eg_unit_slot* const sga_auto_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_sga_mix, &slot_how };
SEG_REF(ref_sga_cls, "dsga__cls", 0x91);
SEG_REF(ref_sga_k0, "dsga__k0", 0x92);
SEG_REF(ref_sga_k1, "dsga__k1", 0x93);
static const struct eg_unit_ref* const sga_branches[] = { &ref_sga_k0, &ref_sga_k1 };
DISPATCH(disp_sga_cls, EG_DISPATCH_SEGMENTED, 1, -1, -1, NULL, "dsga", NULL, NULL);
DISPATCH(disp_sga_branch, EG_DISPATCH_SEGMENTED, -1, -1, 4, &ref_sga_cls, "dsga", sga_branches, NULL);
static const struct eg_unit_dispatch* const sga_cls_list[] = { &disp_sga_cls };
static const struct eg_unit_dispatch* const sga_branch_list[] = { &disp_sga_branch };
SEG_BODY(sga_cls_body, sga_cls_slots, 2, sga_cls_list);
SEG_BODY(sga_branch_body, sga_k_slots, 6, sga_branch_list);
SEG_UNIT(unit_sga_cls, "dsga__cls", 0x91, "sgCls", sga_cls_body, launch_any, NULL);
SEG_UNIT(unit_sga_k0, "dsga__k0", 0x92, "sgaK0", sga_branch_body, launch_any, &finish_one);
SEG_UNIT(unit_sga_k1, "dsga__k1", 0x93, "sgaK1", sga_branch_body, launch_any, &finish_one);
#define COST(var, ...) static const struct eg_branch_cost var = { sizeof(struct eg_branch_cost), __VA_ARGS__ }
#define AUTO_SET(name_, byte)                                                                                            \
    COST(cost_##name_##_0, AUTO_COST_##name_##_0);                                                                       \
    COST(cost_##name_##_1, AUTO_COST_##name_##_1);                                                                       \
    static const struct eg_branch_cost* const costs_##name_[] = { &cost_##name_##_0, &cost_##name_##_1 };                 \
    static const struct eg_unit_dispatch disp_sga_##name_ = { .struct_size = sizeof(struct eg_unit_dispatch),           \
        .strategy = EG_DISPATCH_AUTO, .mix_slot = 4, .class_slot = -1, .stale_slot = -1, .offsets_slot = -1,           \
        .classify = &ref_sga_cls, .name = "dsga", .k = 2, .branches = sga_branches, .branch_costs = costs_##name_,      \
        .cost_flags = AUTO_FLAGS_##name_ };                                                                              \
    static const struct eg_unit_dispatch* const sga_##name_##_list[] = { &disp_sga_##name_ };                            \
    SEG_BODY(sga_##name_##_body, sga_auto_slots, 6, sga_##name_##_list);                                                 \
    SEG_UNIT(unit_sga_##name_, "dsga_" #name_, byte, "sga", sga_##name_##_body, launch_any, &finish_one)
AUTO_SET(rare, 0x94);
AUTO_SET(even, 0x95);
AUTO_SET(scalar, 0x96);
AUTO_SET(heavy, 0x97);

/* dscat / dscatp: the checked and unchecked builds of one data-indexed accumulate (slots idx, jdx, val, acc, tbl, fault; see fixture_device.cu). */
static const int64_t scat_tbl_shape[] = { 8 }, scat_fault_shape[] = { 2 };
SLOT(slot_sidx, EG_ROLE_PER_SAMPLE, EG_DTYPE_INT32, 0, NULL, "idx");
SLOT(slot_sjdx, EG_ROLE_PER_SAMPLE, EG_DTYPE_INT32, 0, NULL, "jdx");
SLOT(slot_sval, EG_ROLE_PER_SAMPLE, EG_DTYPE_REAL, 0, NULL, "val");
SLOT(slot_sacc, EG_ROLE_ACCUM_OUT, EG_DTYPE_REAL, 0, NULL, "acc");
static const struct eg_unit_slot slot_stbl = { .struct_size = sizeof(struct eg_unit_slot), .flags = EG_SLOT_SHARED, .role = EG_ROLE_LOOKUP, .dtype = EG_DTYPE_REAL,
                                               .rank = 1, .like_slot = -1, .shape = scat_tbl_shape, .name = "tbl" };
static const struct eg_unit_slot slot_sfault = { .struct_size = sizeof(struct eg_unit_slot), .flags = EG_SLOT_SHARED, .role = EG_ROLE_LOOKUP, .dtype = EG_DTYPE_INT64,
                                                 .rank = 1, .like_slot = -1, .shape = scat_fault_shape, .name = "fault" };
static const struct eg_unit_slot* const scat_slots[] = { &slot_sidx, &slot_sjdx, &slot_sval, &slot_sacc, &slot_stbl, &slot_sfault };
static const struct eg_unit_body scat_body = { .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 6, .slots = scat_slots };
static const struct eg_unit_body scatp_body = { .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 5, .slots = scat_slots };
MADE_UNIT(unit_dscatp, "dscatp", 0xF2, "dscatp", scatp_body, launch_any);
static const struct eg_unit_meta unit_dscat = {
    .struct_size = sizeof(struct eg_unit_meta), .flags = EG_UNIT_CHECKED, .name = "dscat", .digest = DIGEST(0xF1), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .device_image = "fixture__image", .device_function = "dscat", .device_image_format = EG_IMAGE_FATBIN, .body = &scat_body,
    .launch = &launch_any, .device_image_size = "fixture__image_size", .fault_slot = 5 };
static const struct eg_unit_meta unit_dscat_badshape = {
    .struct_size = sizeof(struct eg_unit_meta), .flags = EG_UNIT_CHECKED, .name = "dscat_badshape", .digest = DIGEST(0xF3), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .device_image = "fixture__image", .device_function = "dscat", .device_image_format = EG_IMAGE_FATBIN, .body = &scat_body,
    .launch = &launch_any, .device_image_size = "fixture__image_size", .fault_slot = 4 };

/* dbrkat: sets the uint32 cell `stop` to 1 when the uint32 cell `counter` (a WHILE's iteration index) reads 3. */
SLOT(slot_ctr, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, 0, NULL, "counter");
SLOT(slot_stop, EG_ROLE_MUTABLE, EG_DTYPE_UINT32, 0, NULL, "stop");
static const struct eg_unit_slot* const brkat_slots[] = { &slot_ctr, &slot_stop };
static const struct eg_unit_body brkat_body = { .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 2, .slots = brkat_slots };
MADE_UNIT(unit_dbrkat, "dbrkat", 0xDE, "dbrkat", brkat_body, launch_any);

static const struct eg_unit_meta* const units[] = { &unit_axpy, &unit_axpy64, &unit_boxdev, &unit_banded, &unit_nofn,
                                                    &unit_noimg, &unit_dcells, &unit_dkeyed, &unit_dkeyednf, &unit_dkeyedh, &unit_fat129, &unit_ptx129, &unit_sass70, &unit_dprefix, &unit_dprefixsum, &unit_dcount, &unit_dcount1, &unit_dcount1a, &unit_dauto, &unit_dautorange, &unit_dlookup,
                                                    &unit_sg_cls, &unit_sg_k0, &unit_sg_k1, &unit_sgf, &unit_dg_k0, &unit_dg_k1, &unit_dg_sweep, &unit_dgf,
                                                    &unit_sga_cls, &unit_sga_k0, &unit_sga_k1, &unit_sga_rare, &unit_sga_even, &unit_sga_scalar, &unit_sga_heavy, &unit_dscat, &unit_dscatp, &unit_dscat_badshape, &unit_dbrkat };

static const struct eg_library_meta library = {
    .struct_size = sizeof(struct eg_library_meta), .schema_major = EG_META_SCHEMA_MAJOR, .n_units = 40, .units = units,
};

__attribute__((visibility("default"))) const struct eg_library_meta* eg_unit_library_meta(void) { return &library; }
