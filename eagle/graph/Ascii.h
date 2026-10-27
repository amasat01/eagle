// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file graph/Ascii.h
 * @brief The box block diagram behind ``ascii()`` (see Render.h): one box per
 *        node on its layout row, edges routed between the boxes (the hazard
 *        kind as line style), a frame around the nodes of a conditional, a
 *        stage tag in the left gutter.
 *
 * The diagram is built in small passes over one working state (items, rows,
 * columns, frame extents, a character canvas); each pass is a method.
 */

#include "eagle/graph/Description.h"
#include "eagle/graph/Layout.h"

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace eagle {
namespace graph {
namespace detail {

/** @brief The display cells of a UTF-8 string, one entry per code point. */
inline std::vector<std::string> cellsOf(const std::string& s)
{
    std::vector<std::string> out;
    for (std::size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        const std::size_t len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

/** @brief ``text`` with every non-ASCII code point replaced for a terminal
 *  without UTF-8: the few symbols eagle writes get a spelling, the rest ``?``. */
inline std::string asciiFold(const std::string& text)
{
    static const std::map<std::string, std::string> spelling = {
        { "\xE2\x89\xA4", "<=" }, { "\xE2\x89\xA5", ">=" },
        { "\xC2\xB7", "-" },      { "\xE2\x80\xA6", "..." } };
    std::string out;
    for (const std::string& c : cellsOf(text)) {
        const auto found = spelling.find(c);
        out += c.size() == 1 ? c : found != spelling.end() ? found->second : "?";
    }
    return out;
}

/** @brief ``text`` cut to ``cap`` cells, ending in an ellipsis when cut. */
inline std::vector<std::string> truncated(const std::string& text, std::size_t cap, bool unicode)
{
    auto cells = cellsOf(unicode ? text : asciiFold(text));
    if (cells.size() <= cap)
        return cells;
    const std::size_t dots = unicode ? 1 : 3;
    cells.resize(cap > dots ? cap - dots : 0);
    for (std::size_t i = 0; i < dots; ++i)
        cells.push_back(unicode ? "\xE2\x80\xA6" : ".");
    return cells;
}

enum Bits : unsigned { kU = 1, kD = 2, kL = 4, kR = 8 };

/** @brief The look of an edge run: the hazard kind, solid for ``RAW``. */
enum Style : unsigned char { kSolid = 0, kDashed = 1, kDotted = 2, kNoStyle = 255 };

inline Style styleOf(EdgeKind k) { return k == EdgeKind::RAW ? kSolid : k == EdgeKind::WAR ? kDashed : kDotted; }

/** @brief A straight vertical (or horizontal) line piece in the given style. */
inline const char* lineChar(bool vertical, Style st, bool unicode)
{
    // Light box-drawing (U+2500 block) for solid runs only; the dashed and
    // dotted styles are ASCII: the heavy/dashed glyphs (U+2504.., U+254C..) are
    // missing from common browser monospace fonts, and the fallback font's
    // different advance width shears the diagram in Jupyter and Sphinx.
    static const char* uni[2][3] = { { "\xE2\x94\x80", ".", "~" }, { "\xE2\x94\x82", ":", "!" } };
    static const char* ascii[2][3] = { { "-", ".", "~" }, { "|", ":", "!" } };
    return (unicode ? uni : ascii)[vertical][st];
}

/** @brief The box-drawing character for the lines meeting at a cell. */
inline const char* maskChar(unsigned m, bool unicode, Style st = kSolid)
{
    if (!unicode)
        return (m & (kL | kR)) == 0 ? lineChar(true, st, false)
            : (m & (kU | kD)) == 0 ? lineChar(false, st, false) : "+";
    switch (m) {
    case kU: case kD: case kU | kD: return lineChar(true, st, true);
    case kL: case kR: case kL | kR: return lineChar(false, st, true);
    case kU | kR: return "\xE2\x94\x94";
    case kU | kL: return "\xE2\x94\x98";
    case kD | kR: return "\xE2\x94\x8C";
    case kD | kL: return "\xE2\x94\x90";
    case kU | kD | kR: return "\xE2\x94\x9C";
    case kU | kD | kL: return "\xE2\x94\xA4";
    case kL | kR | kD: return "\xE2\x94\xAC";
    case kL | kR | kU: return "\xE2\x94\xB4";
    default: return "\xE2\x94\xBC";
    }
}

class BoxDiagram {
public:
    BoxDiagram(const Description& d, const Layout& lay, bool unicode, std::size_t maxLabel)
        : d_(d), lay_(lay), unicode_(unicode), maxLabel_(maxLabel)
    {
    }

    std::string render()
    {
        d_.validate();
        if (d_.nodes.empty())
            return "";
        for (gidx l : lay_.layer)
            layers_ = std::max(layers_, l + 1);
        stages_ = stagesOf(d_);
        buildItems();
        buildSegments();
        orderLayers();
        widenForFrames();
        placeColumns();
        measureFrames();
        stackRows();
        paint();
        return text();
    }

private:
    /** @brief A node box, or the pass-through column of an edge that skips the layer. */
    struct Item {
        gidx node = -1;                    // -1: a pass-through
        std::vector<gidx> chain;           // enclosing frames, outermost first
        std::vector<std::string> label;    // label cells (nodes)
        std::size_t width = 1;
        std::size_t x = 0;                 // left column
        bool out = false, in = false;      // has an outgoing / incoming segment
        std::vector<std::string> planes;   // names shown under the source
        gidx layer = 0;
        Style style = kSolid;              // of the run a pass-through carries
        std::size_t center() const { return x + width / 2; }
        bool within(gidx frame) const
        {
            return std::find(chain.begin(), chain.end(), frame) != chain.end();
        }
    };

    struct Segment {
        std::size_t from, to;              // items in adjacent layers
        std::vector<std::string> names;    // planes, shown under `from` when it is a node
        Style style = kSolid;
    };

    /** @brief Where a frame sits: its first and last layer, the rows of border
     *  it needs above and below, and its box (rows r0..r1, columns c0..c1). */
    struct FrameSpan {
        gidx top = 0, bottom = -1;         // bottom < 0: no node inside
        std::size_t extTop = 1, extBottom = 1;
        std::size_t r0 = 0, r1 = 0, c0 = 0, c1 = 0;
        bool used() const { return bottom >= 0; }
    };

    const Description& d_;
    const Layout& lay_;
    bool unicode_;
    std::size_t maxLabel_;
    gidx layers_ = 0;

    std::vector<Item> items_;
    std::vector<std::size_t> nodeItem_;
    std::vector<Segment> segments_;
    std::vector<std::vector<std::size_t>> rows_;   // item ids per layer, left to right
    std::vector<FrameSpan> frames_;
    std::vector<gidx> stages_;                     // per node, empty without stage frames
    std::vector<std::string> gutter_;
    std::vector<std::size_t> boxY_, labelRow_, busRow_, layerWidth_;
    std::size_t left_ = 0, maxWidth_ = 0, width_ = 0, height_ = 0;
    std::vector<std::vector<std::string>> cells_;
    std::vector<std::vector<unsigned>> lines_;
    std::vector<std::vector<Style>> styles_;
    bool kindSeen_[3] = { false, false, false };

    // ---- items and segments ----
    std::vector<gidx> chainOf(gidx frame) const
    {
        std::vector<gidx> c;
        for (gidx f = frame; f != -1; f = d_.frames[f].parent)
            if (d_.frames[f].kind != "stage")      // a stage is a row, never drawn
                c.push_back(f);
        std::reverse(c.begin(), c.end());
        return c;
    }

    void buildItems()
    {
        for (const Node& nd : d_.nodes) {
            Item it;
            it.node = nd.id;
            it.chain = chainOf(nd.frame);
            it.label = truncated(nd.label, maxLabel_, unicode_);
            if (nd.opaque) {   // own code: tagged, since the box glyphs stay solid
                it.label.insert(it.label.begin(), "<");
                it.label.push_back(">");
            }
            it.width = it.label.size() + 4 + (it.label.size() % 2 == 0 ? 1 : 0);   // odd: a centre column
            it.layer = lay_.layer[nd.id];
            nodeItem_.push_back(items_.size());
            items_.push_back(it);
        }
    }

    static std::string joinPlanes(const std::vector<std::string>& planes)
    {
        std::string out;
        for (const std::string& p : planes)
            out += (out.empty() ? "" : ",") + p;
        return out;
    }

    /** @brief One segment per layer step of every node pair joined by an edge;
     *  a step through a layer with no node of the pair gets a pass-through
     *  item. A pair is one run: its plane names once, in the style of its
     *  first hazard kind (read-after-write over write-after-read over
     *  write-after-write). */
    void buildSegments()
    {
        struct Run {
            std::vector<std::string> names;
            Style style = kNoStyle;
        };
        std::map<std::pair<gidx, gidx>, Run> pairs;
        for (const Edge& e : d_.edges) {
            Run& run = pairs[{ e.src, e.dst }];
            run.style = std::min(run.style, styleOf(e.kind));
            if (!e.plane.empty() && std::find(run.names.begin(), run.names.end(), e.plane) == run.names.end())
                run.names.push_back(e.plane);
        }
        for (const auto& [ends, run] : pairs) {
            kindSeen_[run.style] = true;
            const std::size_t first = nodeItem_[ends.first], last = nodeItem_[ends.second];
            std::vector<gidx> shared;      // frames holding both ends
            for (std::size_t i = 0; i < items_[first].chain.size() && i < items_[last].chain.size()
                 && items_[first].chain[i] == items_[last].chain[i]; ++i)
                shared.push_back(items_[first].chain[i]);
            std::size_t prev = first;
            for (gidx l = lay_.layer[ends.first] + 1; l <= lay_.layer[ends.second]; ++l) {
                std::size_t next = last;
                if (l < lay_.layer[ends.second]) {
                    Item through;
                    through.chain = shared;
                    through.layer = l;
                    through.style = run.style;
                    next = items_.size();
                    items_.push_back(through);
                }
                segments_.push_back({ prev, next, prev == first ? run.names : std::vector<std::string>(), run.style });
                prev = next;
            }
        }
        for (const Segment& s : segments_) {
            items_[s.from].out = true;
            items_[s.to].in = true;
            if (items_[s.from].node == -1)
                continue;
            auto& names = items_[s.from].planes;
            for (const std::string& n : s.names)
                if (std::find(names.begin(), names.end(), n) == names.end())
                    names.push_back(n);
        }
    }

    // ---- order inside each layer ----
    /** @brief ``in`` regrouped so the members of each frame sit together
     *  (first-seen order), recursing into nested frames. */
    std::vector<std::size_t> gather(const std::vector<std::size_t>& in, std::size_t depth) const
    {
        std::vector<std::size_t> out;
        std::vector<bool> used(in.size(), false);
        for (std::size_t i = 0; i < in.size(); ++i) {
            if (used[i])
                continue;
            if (items_[in[i]].chain.size() <= depth) {
                out.push_back(in[i]);
                continue;
            }
            const gidx f = items_[in[i]].chain[depth];
            std::vector<std::size_t> group;
            for (std::size_t j = i; j < in.size(); ++j)
                if (!used[j] && items_[in[j]].chain.size() > depth && items_[in[j]].chain[depth] == f) {
                    group.push_back(in[j]);
                    used[j] = true;
                }
            for (std::size_t g : gather(group, depth + 1))
                out.push_back(g);
        }
        return out;
    }

    void orderLayers()
    {
        rows_.assign(layers_, {});
        std::vector<double> position(items_.size(), 0.0);
        for (gidx l = 0; l < layers_; ++l) {
            // nodes by the layout's order; a pass-through just right of what feeds it
            std::vector<std::pair<std::pair<double, int>, std::size_t>> key;
            for (std::size_t i = 0; i < items_.size(); ++i) {
                if (items_[i].layer != l)
                    continue;
                if (items_[i].node != -1) {
                    key.push_back({ { double(lay_.order[items_[i].node]), 0 }, i });
                    continue;
                }
                double k = -1.0;
                for (const Segment& s : segments_)
                    if (s.to == i)
                        k = position[s.from] + 0.5;
                key.push_back({ { k, 1 }, i });
            }
            std::stable_sort(key.begin(), key.end());
            std::vector<std::size_t> seq;
            for (const auto& kv : key)
                seq.push_back(kv.second);
            rows_[l] = gather(seq, 0);
            for (std::size_t i = 0; i < rows_[l].size(); ++i)
                position[rows_[l][i]] = double(i);
        }
    }

    // ---- columns ----
    std::size_t labelCells(gidx frame) const
    {
        return truncated(d_.frames[frame].text(), maxLabel_, unicode_).size();
    }

    /** @brief A frame is at least as wide as its label: widen its first item per layer. */
    void widenForFrames()
    {
        for (gidx f = 0; f < gidx(d_.frames.size()); ++f) {
            const std::size_t need = labelCells(f) + 2;
            for (const auto& row : rows_) {
                std::size_t sum = 0, count = 0, first = 0;
                for (std::size_t i : row)
                    if (items_[i].within(f)) {
                        first = count == 0 ? i : first;
                        sum += items_[i].width + (count ? 3 : 0);
                        ++count;
                    }
                if (count && sum < need)
                    items_[first].width += 2 * ((need - sum + 1) / 2);
            }
        }
    }

    /** @brief Space between two neighbours: 3, plus 2 for each frame border crossed. */
    std::size_t gap(std::size_t a, std::size_t b) const
    {
        const auto& ca = items_[a].chain;
        const auto& cb = items_[b].chain;
        std::size_t shared = 0;
        while (shared < ca.size() && shared < cb.size() && ca[shared] == cb[shared])
            ++shared;
        return 3 + 2 * ((ca.size() - shared) + (cb.size() - shared));
    }

    std::size_t edgePad(std::size_t item) const { return 2 * items_[item].chain.size(); }

    std::size_t planeCells(std::size_t item) const
    {
        return cellsOf(unicode_ ? joinPlanes(items_[item].planes) : asciiFold(joinPlanes(items_[item].planes))).size();
    }

    /** @brief The offsets of a row's items from its left edge, and its width.
     *  A plane name is written from its source's centre, so the next line in
     *  the row starts a column after it: the row grows to fit the names. */
    std::pair<std::vector<std::size_t>, std::size_t> rowOffsets(gidx l) const
    {
        const auto& r = rows_[l];
        std::vector<std::size_t> at;
        std::size_t x = edgePad(r.front()), need = 0;
        for (std::size_t i = 0; i < r.size(); ++i) {
            const Item& it = items_[r[i]];
            x += i ? gap(r[i - 1], r[i]) : 0;
            if (it.out) {
                const std::size_t half = it.width / 2;
                x = std::max(x, need > half ? need - half : 0);
                need = it.planes.empty() ? 0 : x + half + planeCells(r[i]) + 1;
            }
            at.push_back(x);
            x += it.width;
        }
        return { at, std::max(x + edgePad(r.back()), need + 1) };
    }

    void placeColumns()
    {
        layerWidth_.assign(layers_, 0);
        for (gidx l = 0; l < layers_; ++l) {
            layerWidth_[l] = rowOffsets(l).second;
            maxWidth_ = std::max(maxWidth_, layerWidth_[l]);
        }
        gutter_.assign(layers_, "");
        std::size_t gutterWidth = 0;
        gidx shown = -1;
        for (gidx l = 0; l < layers_; ++l) {
            std::string schedule;
            bool same = true, any = false;
            gidx stage = l;
            for (std::size_t i : rows_[l])
                if (items_[i].node != -1) {
                    const std::string& s = d_.nodes[items_[i].node].stage;
                    same = same && (!any || s == schedule);
                    schedule = any ? schedule : s;
                    stage = stages_.empty() ? l : stages_[items_[i].node];
                    any = true;
                }
            if (stage != shown)                  // a stage's later rows carry no tag
                gutter_[l] = "stage " + std::to_string(stage) + (same && !schedule.empty() ? " " + schedule : "");
            shown = stage;
            gutterWidth = std::max(gutterWidth, cellsOf(gutter_[l]).size());
        }
        left_ = gutterWidth + 3;
        width_ = left_ + maxWidth_ + 2;
        for (gidx l = 0; l < layers_; ++l) {
            const auto [at, total] = rowOffsets(l);
            for (std::size_t i = 0; i < at.size(); ++i)
                items_[rows_[l][i]].x = left_ + (maxWidth_ - total) / 2 + at[i];
        }
    }

    // ---- frames and rows ----
    /** @brief Border rows a frame needs above (or below) its nodes: one, plus
     *  one more than any child frame that starts (or ends) in the same layer. */
    std::size_t extent(gidx f, bool top) const
    {
        std::size_t e = 1;
        for (gidx c = 0; c < gidx(frames_.size()); ++c)
            if (d_.frames[c].parent == f && frames_[c].used()
                && (top ? frames_[c].top == frames_[f].top : frames_[c].bottom == frames_[f].bottom))
                e = std::max(e, extent(c, top) + 1);
        return e;
    }

    void measureFrames()
    {
        frames_.assign(d_.frames.size(), FrameSpan{});
        for (FrameSpan& fs : frames_)
            fs.top = layers_;
        for (const Item& it : items_)
            for (gidx f : it.chain) {
                frames_[f].top = std::min(frames_[f].top, it.layer);
                frames_[f].bottom = std::max(frames_[f].bottom, it.layer);
            }
        for (gidx f = 0; f < gidx(frames_.size()); ++f)
            if (frames_[f].used()) {
                frames_[f].extTop = extent(f, true);
                frames_[f].extBottom = extent(f, false);
            }
    }

    void stackRows()
    {
        std::vector<std::size_t> topPad(layers_, 0), bottomPad(layers_, 0);
        for (const FrameSpan& fs : frames_)
            if (fs.used()) {
                topPad[fs.top] = std::max(topPad[fs.top], fs.extTop);
                bottomPad[fs.bottom] = std::max(bottomPad[fs.bottom], fs.extBottom);
            }
        boxY_.assign(layers_, 0);
        labelRow_.assign(layers_, 0);
        busRow_.assign(layers_, 0);
        std::size_t cur = 0;
        for (gidx l = 0; l < layers_; ++l) {
            boxY_[l] = cur + topPad[l];
            cur = boxY_[l] + 3 + bottomPad[l];
            if (l + 1 < layers_) {
                labelRow_[l] = cur;
                busRow_[l] = cur + 1;
                cur += 2;
            }
        }
        height_ = cur;
        for (gidx f = 0; f < gidx(frames_.size()); ++f)       // boxes, children first
            if (frames_[f].used())
                placeFrame(f);
    }

    /** @brief The frame's box: its items' columns padded by 2 for it and for
     *  each frame nested between it and the item. */
    void placeFrame(gidx f)
    {
        FrameSpan& fs = frames_[f];
        std::size_t c0 = width_, c1 = 0;
        for (const Item& it : items_)
            if (it.within(f)) {
                const std::size_t pad = 2 * (it.chain.end() - std::find(it.chain.begin(), it.chain.end(), f));
                c0 = std::min(c0, it.x - pad);
                c1 = std::max(c1, it.x + it.width - 1 + pad);
            }
        fs.r0 = boxY_[fs.top] - fs.extTop;
        fs.r1 = boxY_[fs.bottom] + 2 + fs.extBottom;
        fs.c0 = c0;
        fs.c1 = c1;
    }

    // ---- painting ----
    void put(std::size_t r, std::size_t c, const std::string& s)
    {
        if (r < height_ && c < width_)
            cells_[r][c] = s;
    }

    void putText(std::size_t r, std::size_t c, const std::vector<std::string>& text)
    {
        for (std::size_t i = 0; i < text.size(); ++i)
            put(r, c + i, text[i]);
    }

    void line(std::size_t r, std::size_t c, unsigned bits, Style st)
    {
        lines_[r][c] |= bits;
        styles_[r][c] = std::min(styles_[r][c], st);
    }

    void paint()
    {
        cells_.assign(height_, std::vector<std::string>(width_, " "));
        lines_.assign(height_, std::vector<unsigned>(width_, 0));
        styles_.assign(height_, std::vector<Style>(width_, kNoStyle));
        routeSegments();
        for (std::size_t r = 0; r < height_; ++r)
            for (std::size_t c = 0; c < width_; ++c)
                if (lines_[r][c])
                    cells_[r][c] = maskChar(lines_[r][c], unicode_, styles_[r][c]);
        drawFrames();           // over the lines: a line crossing a border makes a junction
        drawFrameLabels();
        drawBoxes();
        drawPlaneNames();
        for (gidx l = 0; l < layers_; ++l)
            putText(boxY_[l], 2, cellsOf(gutter_[l]));
    }

    /** @brief A frame border cell: a junction where an edge line already crosses it. */
    void border(std::size_t r, std::size_t c, unsigned bits)
    {
        if (r < height_ && c < width_)
            put(r, c, lines_[r][c] ? maskChar(lines_[r][c] | bits, unicode_)
                : !unicode_ && bits == (kL | kR) ? "=" : maskChar(bits, unicode_));
    }

    void drawFrames()
    {
        static const char* corner[] = { "\xE2\x94\x8C", "\xE2\x94\x90", "\xE2\x94\x94", "\xE2\x94\x98" };   // light corners: rounded ones are missing from common fonts
        for (const FrameSpan& fs : frames_) {
            if (!fs.used())
                continue;
            for (std::size_t c = fs.c0 + 1; c < fs.c1; ++c) {
                border(fs.r0, c, kL | kR);
                border(fs.r1, c, kL | kR);
            }
            for (std::size_t r = fs.r0 + 1; r < fs.r1; ++r) {
                border(r, fs.c0, kU | kD);
                border(r, fs.c1, kU | kD);
            }
            put(fs.r0, fs.c0, unicode_ ? corner[0] : "+");
            put(fs.r0, fs.c1, unicode_ ? corner[1] : "+");
            put(fs.r1, fs.c0, unicode_ ? corner[2] : "+");
            put(fs.r1, fs.c1, unicode_ ? corner[3] : "+");
        }
    }

    void drawFrameLabels()
    {
        for (gidx f = 0; f < gidx(frames_.size()); ++f) {
            const FrameSpan& fs = frames_[f];
            if (!fs.used())
                continue;
            const auto cells = truncated(d_.frames[f].text(), fs.c1 - fs.c0 > 5 ? fs.c1 - fs.c0 - 5 : 1, unicode_);
            put(fs.r0, fs.c0 + 2, " ");
            putText(fs.r0, fs.c0 + 3, cells);
            put(fs.r0, fs.c0 + 3 + cells.size(), " ");
        }
    }

    /** @brief The lines of every segment: down from the source, across the bus
     *  row, down to the destination; a pass-through carries the line through. */
    void routeSegments()
    {
        for (const Segment& s : segments_) {
            const Item &a = items_[s.from], &b = items_[s.to];
            const gidx l = a.layer;
            const std::size_t from = a.center(), to = b.center(), bus = busRow_[l];
            const Style st = s.style;
            for (std::size_t r = boxY_[l] + 3; r <= labelRow_[l]; ++r)
                line(r, from, kU | kD, st);
            if (from == to) {
                line(bus, from, kU | kD, st);
            } else {
                line(bus, from, kU | (to > from ? kR : kL), st);
                line(bus, to, kD | (to > from ? kL : kR), st);
                for (std::size_t c = std::min(from, to) + 1; c < std::max(from, to); ++c)
                    line(bus, c, kL | kR, st);
            }
            for (std::size_t r = bus + 1; r < boxY_[l + 1]; ++r)
                line(r, to, kU | kD, st);
        }
        for (const Item& it : items_)
            if (it.node == -1)
                for (std::size_t r = boxY_[it.layer]; r < boxY_[it.layer] + 3; ++r)
                    line(r, it.center(), kU | kD, it.style);
    }

    void drawBoxes()
    {
        static const char* square[] = { "\xE2\x94\x8C", "\xE2\x94\x90", "\xE2\x94\x94", "\xE2\x94\x98" };
        for (const Item& it : items_) {
            if (it.node == -1)
                continue;
            const std::string h = unicode_ ? "\xE2\x94\x80" : "-";
            const std::string v = unicode_ ? "\xE2\x94\x82" : "|";
            const std::size_t y = boxY_[it.layer], right = it.x + it.width - 1;
            for (std::size_t c = it.x + 1; c < right; ++c) {
                put(y, c, h);
                put(y + 1, c, " ");
                put(y + 2, c, h);
            }
            put(y, it.x, unicode_ ? square[0] : "+");
            put(y, right, unicode_ ? square[1] : "+");
            put(y + 2, it.x, unicode_ ? square[2] : "+");
            put(y + 2, right, unicode_ ? square[3] : "+");
            put(y + 1, it.x, v);
            put(y + 1, right, v);
            putText(y + 1, it.x + 1 + (it.width - 2 - it.label.size()) / 2, it.label);
            if (it.in)
                put(y, it.center(), unicode_ ? "\xE2\x94\xB4" : "+");
            if (it.out)
                put(y + 2, it.center(), unicode_ ? "\xE2\x94\xAC" : "+");
        }
    }

    /** @brief Plane names under their source, whole: the columns leave room for them. */
    void drawPlaneNames()
    {
        for (gidx l = 0; l + 1 < layers_; ++l)
            for (std::size_t i : rows_[l])
                if (items_[i].out && !items_[i].planes.empty())
                    putText(labelRow_[l], items_[i].center(),
                        cellsOf(unicode_ ? joinPlanes(items_[i].planes) : asciiFold(joinPlanes(items_[i].planes))));
    }

    /** @brief One line naming the style of each non-solid run drawn. */
    std::string legend() const
    {
        if (!kindSeen_[kDashed] && !kindSeen_[kDotted])
            return "";
        std::string out = "  edges:";
        for (EdgeKind k : { EdgeKind::RAW, EdgeKind::WAR, EdgeKind::WAW })
            if (kindSeen_[styleOf(k)]) {
                const char* g = lineChar(false, styleOf(k), unicode_);
                out += std::string("  ") + g + g + " " + kindName(k) + " " + kindMeaning(k);
            }
        return out + "\n";
    }

    std::string text() const
    {
        std::string out;
        for (const auto& row : cells_) {
            std::size_t end = row.size();
            while (end > 0 && row[end - 1] == " ")
                --end;
            for (std::size_t c = 0; c < end; ++c)
                out += row[c];
            out += "\n";
        }
        return out + legend();
    }
};

}  // namespace detail
}  // namespace graph
}  // namespace eagle
