// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Groups.h
 * @brief The capability groups of ``eagle-backend/1`` (bit -> 1.0 symbols) and
 *        the mandatory and experimental symbol lists, as the core binds them.
 *
 * The table mirrors the one in ``eagle_backend.h``. A group's 1.0 symbol list
 * never changes; symbols added within major 1 are bound one by one (their
 * presence is the capability) and never join a group's list.
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "eagle_backend.h"

namespace eagle_seam {

/** @brief A capability group: its bit and its 1.0 symbols. */
struct GroupSpec {
    const char* name;
    std::uint64_t bit;
    std::vector<const char*> symbols;
};

/** @brief The symbols every backend exports. */
inline const std::vector<const char*>& mandatorySymbols()
{
    static const std::vector<const char*> all = {
        "eagle_backend_version",
        "eagle_backend_last_error",
        "eagle_backend_build_info",
        "eagle_backend_probe",
        "eagle_backend_free",
        "eagle_backend_capabilities",
        "eagle_backend_device_types",
    };
    return all;
}

/** @brief Every capability group, in bit order. */
inline const std::vector<GroupSpec>& groups()
{
    static const std::vector<GroupSpec> all = {
        { "stream", EAGLE_BACKEND_CAP_STREAM,
            { "eagle_backend_stream_create", "eagle_backend_stream_ptr", "eagle_backend_stream_synchronize",
                "eagle_backend_stream_release" } },
        { "capturer", EAGLE_BACKEND_CAP_CAPTURER,
            { "eagle_backend_capturer_create", "eagle_backend_capturer_begin", "eagle_backend_capturer_end",
                "eagle_backend_capturer_release" } },
        { "captured", EAGLE_BACKEND_CAP_CAPTURED,
            { "eagle_backend_captured_is_valid", "eagle_backend_captured_release" } },
        { "fork", EAGLE_BACKEND_CAP_FORK,
            { "eagle_backend_fork_create", "eagle_backend_fork_fork", "eagle_backend_fork_join",
                "eagle_backend_fork_branch", "eagle_backend_fork_origin", "eagle_backend_fork_forked",
                "eagle_backend_fork_size", "eagle_backend_fork_release" } },
        { "conditional", EAGLE_BACKEND_CAP_CONDITIONAL,
            { "eagle_backend_conditional_create", "eagle_backend_conditional_begin",
                "eagle_backend_conditional_body_stream", "eagle_backend_conditional_is_loop",
                "eagle_backend_conditional_end", "eagle_backend_conditional_release" } },
        { "attribution", EAGLE_BACKEND_CAP_ATTRIBUTION,
            { "eagle_backend_capture_snapshot_nodes", "eagle_backend_is_node_toggleable" } },
        { "graph", EAGLE_BACKEND_CAP_GRAPH,
            { "eagle_backend_graph_create", "eagle_backend_graph_from_captured", "eagle_backend_graph_stream",
                "eagle_backend_graph_add_node", "eagle_backend_graph_launcher", "eagle_backend_graph_last_node",
                "eagle_backend_graph_release" } },
        { "launcher", EAGLE_BACKEND_CAP_LAUNCHER,
            { "eagle_backend_launcher_launch", "eagle_backend_launcher_synchronize", "eagle_backend_launcher_stream",
                "eagle_backend_launcher_set_logical_size", "eagle_backend_launcher_kernel_node_count",
                "eagle_backend_launcher_set_node_enabled", "eagle_backend_launcher_release" } },
        { "exec", EAGLE_BACKEND_CAP_EXEC, { "eagle_backend_run_device" } },
        { "device", EAGLE_BACKEND_CAP_DEVICE, { "eagle_backend_device_props" } },
        { "interop", EAGLE_BACKEND_CAP_INTEROP, { "eagle_backend_fence", "eagle_backend_event_pool_created" } },
        { "filtering", EAGLE_BACKEND_CAP_FILTERING, { "eagle_backend_compact_device" } },
    };
    return all;
}

/** @brief The group named @p name (it must exist). */
inline const GroupSpec& group(const std::string& name)
{
    for (const GroupSpec& g : groups())
        if (name == g.name)
            return g;
    throw std::logic_error("eagle_seam: no group named " + name);
}

/** @brief The experimental (``eagle_backend_x_*``) symbols, each optional. */
inline const std::vector<const char*>& experimentalSymbols()
{
    static const std::vector<const char*> all = {
        "eagle_backend_x_captured_debug_dot",
        "eagle_backend_x_capture_guard_depth",
        "eagle_backend_x_capture_guard_pending_count",
        "eagle_backend_x_restore_device",
        "eagle_backend_x_composer_create",
        "eagle_backend_x_composer_register_launcher",
        "eagle_backend_x_composer_register_callable",
        "eagle_backend_x_composer_register_nested",
        "eagle_backend_x_composer_build",
        "eagle_backend_x_composer_set_routing",
        "eagle_backend_x_composer_launch",
        "eagle_backend_x_composer_fired_history",
        "eagle_backend_x_composer_reset_fired_history",
        "eagle_backend_x_composer_mode",
        "eagle_backend_x_composer_num_members",
        "eagle_backend_x_composer_member_name",
        "eagle_backend_x_composer_release",
    };
    return all;
}

} // namespace eagle_seam
