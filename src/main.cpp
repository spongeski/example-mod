// Click Buffer
// ---------------------------------------------------------------------------
// A real-time input buffer: early clicks snap to the nearest safe frame.
//
//   1. You press jump. The press is NOT applied right away -- it is parked.
//   2. On the next physics step (from processCommands, NEVER from inside the game's own
//      input handling) the mod snapshots the game and simulates a short way ahead, once for
//      "click now" and then for "click 1 frame later", "2 frames later", ...
//   3. If clicking now already works, the click is applied as normal (nothing changes).
//   4. If clicking now fails but a later frame works, the click is moved to the NEAREST frame
//      that keeps you alive -- but only if that frame is within your chosen window N
//      (2 / 4 / 8 / 16), i.e. you were at most N frames early.
//   5. (Optional, experimental) If NO early fix exists, the mod also checks whether pressing up
//      to N frames EARLIER would have survived. If so it rewinds the game to that frame, applies
//      the click there, and replays forward to where you were. Needs a snapshot of every frame.
//   6. The live game is restored to exactly where it was before the lookahead.
//
// Tap modes (cube, ball, UFO, spider): the lookahead assumes a short tap.
// Wave: there is no "tap" -- every press AND every release is a direction change, so the
//       lookahead assumes the new direction is held for WAVE_HORIZON frames instead.
// Ship / swing / robot: left completely untouched.
//
// A label says this is NOT a real run, and (optionally) completions with a retimed click
// are not saved.
// ---------------------------------------------------------------------------

#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

using namespace geode::prelude;

namespace {

// ----- tuning knobs ---------------------------------------------------------
constexpr double TPS = 240.0;          // physics steps per second (stock GD)
constexpr int HORIZON = 110;           // tap modes: how many frames ahead each lookahead runs
constexpr int TAP_FRAMES = 6;          // tap modes: the click is assumed released after this many frames
constexpr int MIN_IMPROVEMENT = 8;     // tap modes: a later click must survive this many frames longer to count
constexpr int SCORE_TOL = 3;           // tap modes: tolerance when comparing "frames survived"

// Wave has no "tap": we assume the new direction is held for at least this many frames.
// Keep it short -- a long hold would flag perfectly fine presses as deadly.
constexpr int WAVE_HORIZON = 12;
constexpr int WAVE_MIN_IMPROVEMENT = 3;
constexpr int WAVE_SCORE_TOL = 1;

// ----- settings -------------------------------------------------------------
bool bufferEnabled() { return Mod::get()->getSettingValue<bool>("enabled"); }
bool blockSaving() { return Mod::get()->getSettingValue<bool>("block-saving"); }
bool rescueLate() { return Mod::get()->getSettingValue<bool>("rescue-late"); }
double lookaheadMs() { return static_cast<double>(Mod::get()->getSettingValue<int64_t>("lookahead-ms")); }
int windowFrames() {
    int n = 4;
    try { n = std::stoi(Mod::get()->getSettingValue<std::string>("window")); } catch (...) {}
    return std::clamp(n, 1, 64);
}

// ----- state ----------------------------------------------------------------
enum class Mode { None, Tap, Wave };

struct Event {
    int step;     // game step (frame) on which to apply it
    bool down;
    bool p1;
};

struct Intent {   // a click the game reported, parked until we are at a safe point to plan it
    bool down;
    bool p1;
};

struct Snap {
    int frame;               // physics frame this snapshot was taken at (state BEFORE that frame ran)
    double t;                // level time at that frame
    CheckpointObject* cp;    // retained snapshot of the game
};

struct Plan {
    int delay = 0;           // frames to delay the click (0 = leave it alone)
    bool failsNow = false;   // clicking now fails and no EARLY fix exists (candidate for late rescue)
};

struct State {
    std::deque<Intent> pending;  // parked clicks, waiting to be planned
    std::deque<Event> sched;     // retimed clicks waiting for their frame (sorted by step)

    bool simulating = false;     // true while WE are driving the game for a lookahead / replay
    bool firing = false;         // true while WE are applying a click (so our hook lets it through)
    bool simDied = false;
    int simDeathFrame = -1;

    int lastInputFrame = -1000000;  // last frame any input was applied (late rescue must not rewind past it)
    std::deque<Snap> history;       // rolling snapshots of the last few frames (late-click rescue only)

    int retimed = 0;
};

State& st() {
    static State s;
    return s;
}

int frameOf(GJBaseGameLayer* l) {
    // NOTE: assumes m_gameState.m_levelTime exists and counts seconds of level time.
    return static_cast<int>(std::llround(l->m_gameState.m_levelTime * TPS));
}

void clearHistory() {
    auto& g = st();
    for (auto& sn : g.history)
        if (sn.cp) sn.cp->release();
    g.history.clear();
}

void resetRuntime() {
    auto& g = st();
    clearHistory();
    g.pending.clear();
    g.sched.clear();
    g.lastInputFrame = -1000000;
}

PlayLayer* livePlay(GJBaseGameLayer* layer) {
    if (st().simulating || !bufferEnabled()) return nullptr;
    auto pl = PlayLayer::get();
    if (!pl || static_cast<GJBaseGameLayer*>(pl) != layer) return nullptr;
    if (pl->m_levelSettings && pl->m_levelSettings->m_platformerMode) return nullptr;  // classic mode only
    if (pl->m_levelEndAnimationStarted) return nullptr;
    if (pl->m_player1 && pl->m_player1->m_isDead) return nullptr;
    return pl;
}

Mode modeOf(PlayLayer* pl) {
    auto p = pl->m_player1;
    if (!p) return Mode::None;
    if (p->m_isDart) return Mode::Wave;                                  // wave
    if (p->m_isShip || p->m_isSwing || p->m_isRobot) return Mode::None;  // untouched
    return Mode::Tap;                                                    // cube / ball / UFO / spider
}

void addEvent(Event e) {
    auto& q = st().sched;
    auto it = std::upper_bound(q.begin(), q.end(), e, [](Event const& x, Event const& y) { return x.step < y.step; });
    q.insert(it, e);
}

// Last step already scheduled for this player, or -1 if nothing is waiting.
int lastScheduled(bool p1) {
    int last = -1;
    for (auto const& e : st().sched)
        if (e.p1 == p1) last = std::max(last, e.step);
    return last;
}

// Apply one click to the live game, right now.
void applyInput(PlayLayer* pl, bool down, bool p1) {
    auto& g = st();
    g.lastInputFrame = frameOf(pl);
    g.firing = true;
    pl->handleButton(down, 1, p1);
    g.firing = false;
}

// Apply every retimed click that is due.
void fireDue(PlayLayer* pl) {
    auto& g = st();
    int k = frameOf(pl);
    while (!g.sched.empty() && g.sched.front().step <= k) {
        Event e = g.sched.front();
        g.sched.pop_front();
        applyInput(pl, e.down, e.p1);
    }
}

// Used when the mod gets switched off mid-level: hand everything we are holding back to the game.
void flushAll(PlayLayer* pl) {
    auto& g = st();
    auto s = std::move(g.sched);
    auto p = std::move(g.pending);
    g.sched.clear();
    g.pending.clear();
    g.firing = true;
    for (auto const& e : s) pl->handleButton(e.down, 1, e.p1);
    for (auto const& i : p) pl->handleButton(i.down, 1, i.p1);
    g.firing = false;
}

// ----- lookahead simulation -------------------------------------------------
std::vector<Event> makeEvents(std::vector<Event> const& base, int step, bool down, bool p1, bool wave) {
    auto ev = base;
    ev.push_back({step, down, p1});
    // Tap modes: assume a short tap. Wave: the new direction is simply held.
    if (!wave && down) ev.push_back({step + TAP_FRAMES, false, p1});
    std::stable_sort(ev.begin(), ev.end(), [](Event const& x, Event const& y) { return x.step < y.step; });
    return ev;
}

// Restore the snapshot, apply `evs` at their frames, run up to `horizon` frames.
// Returns how many frames the player survived (== horizon if it never died).
int simulate(PlayLayer* pl, CheckpointObject* cp, double t0, int c0, std::vector<Event> const& evs, int horizon) {
    auto& g = st();
    g.simulating = true;
    g.simDied = false;
    g.simDeathFrame = -1;

    pl->loadFromCheckpoint(cp);
    pl->m_gameState.m_levelTime = t0;

    size_t i = 0;
    for (int guard = 0; guard < horizon * 4 && !g.simDied && frameOf(pl) < c0 + horizon; ++guard) {
        int k = frameOf(pl);
        for (; i < evs.size() && evs[i].step <= k; ++i)
            pl->handleButton(evs[i].down, 1, evs[i].p1);  // passes straight through while simulating
        pl->update(1.f / static_cast<float>(TPS));        // so does our update hook
    }

    g.simulating = false;
    return g.simDied ? std::max(0, g.simDeathFrame - c0) : horizon;
}

// How many frames should this click be delayed? (0 = leave it alone)
Plan planDelay(PlayLayer* pl, bool down, bool p1, bool wave) {
    auto& g = st();
    Plan out;
    const int N = windowFrames();
    const int H = wave ? WAVE_HORIZON : HORIZON;
    const int minImprove = wave ? WAVE_MIN_IMPROVEMENT : MIN_IMPROVEMENT;
    const int tol = wave ? WAVE_SCORE_TOL : SCORE_TOL;
    const int c0 = frameOf(pl);

    CheckpointObject* cp = pl->createCheckpoint();
    if (!cp) return out;
    cp->retain();
    const double t0 = pl->m_gameState.m_levelTime;

    const auto start = std::chrono::steady_clock::now();
    const double budget = lookaheadMs();
    auto outOfTime = [&]() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() > budget;
    };

    std::vector<Event> base(g.sched.begin(), g.sched.end());
    auto run = [&](int d) { return simulate(pl, cp, t0, c0, makeEvents(base, c0 + d, down, p1, wave), H); };

    do {
        int s0 = run(0);
        if (s0 >= H) break;       // clicking right now is already fine
        out.failsNow = true;      // clicking now fails; stays true unless an early fix is found

        // Phase 1: first delay (within the buffer window) that works.
        std::vector<int> sc(N + 1, 0);
        int a = -1;
        int bestPartial = s0;
        bool timedOut = false;
        for (int d = 1; d <= N; ++d) {
            if (outOfTime()) { timedOut = true; break; }
            sc[d] = run(d);
            if (sc[d] >= H) { a = d; break; }
            bestPartial = std::max(bestPartial, sc[d]);
        }
        if (timedOut) {
            log::debug("lookahead ran out of time, click left alone");
            out.failsNow = false;
            break;
        }

        // If nothing survives the whole horizon (e.g. a later obstacle needs another click),
        // fall back to "the delays that survive longest".
        if (a < 0) {
            if (bestPartial < s0 + minImprove) break;  // no delay helps
            const int threshold = bestPartial - tol;
            for (int d = 1; d <= N; ++d)
                if (sc[d] >= threshold) { a = d; break; }
            if (a < 0) break;
        }

        out.delay = a;
        out.failsNow = false;
        log::debug("buffered {}: nearest safe frame is +{} (N={})", wave ? "wave edge" : "click", a, N);
    } while (false);

    // Put the live game back exactly where it was before the lookahead.
    pl->loadFromCheckpoint(cp);
    pl->m_gameState.m_levelTime = t0;
    cp->release();
    return out;
}

// ----- late-click rescue (experimental) -------------------------------------
// Take a snapshot of the game before every physics step (only while the setting is on),
// keeping the last N+2 of them.
void recordSnapshot(PlayLayer* pl) {
    auto& g = st();
    int f = frameOf(pl);
    if (!g.history.empty()) {
        if (g.history.back().frame == f) return;         // processCommands can run more than once per frame
        if (g.history.back().frame > f) clearHistory();  // time went backwards (respawn / rewind)
    }
    CheckpointObject* cp = pl->createCheckpoint();
    if (!cp) return;
    cp->retain();
    g.history.push_back({f, pl->m_gameState.m_levelTime, cp});
    while (g.history.size() > static_cast<size_t>(windowFrames()) + 2) {
        g.history.front().cp->release();
        g.history.pop_front();
    }
}

Snap* findSnap(int frame) {
    for (auto& sn : st().history)
        if (sn.frame == frame) return &sn;
    return nullptr;
}

// Would the click have survived if it had been pressed j frames ago? Returns the smallest such j
// (1..N), or 0 if none. We never rewind past a frame where input was applied, because the replay
// would not contain that input.
int planLate(PlayLayer* pl, bool down, bool p1, bool wave) {
    auto& g = st();
    if (!g.sched.empty() || g.history.empty()) return 0;
    const int N = windowFrames();
    const int H = wave ? WAVE_HORIZON : HORIZON;
    const int c0 = frameOf(pl);

    CheckpointObject* cp0 = pl->createCheckpoint();
    if (!cp0) return 0;
    cp0->retain();
    const double t0 = pl->m_gameState.m_levelTime;

    const auto start = std::chrono::steady_clock::now();
    const double budget = lookaheadMs();

    int found = 0;
    for (int j = 1; j <= N; ++j) {
        if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() > budget) break;
        if (c0 - j <= g.lastInputFrame) break;  // older snapshots would drop a real input
        Snap* sn = findSnap(c0 - j);
        if (!sn) continue;
        auto ev = makeEvents({}, sn->frame, down, p1, wave);
        if (simulate(pl, sn->cp, sn->t, sn->frame, ev, H) >= H) { found = j; break; }
    }

    // Always leave the live game exactly as we found it; applyLate does the actual rewind.
    pl->loadFromCheckpoint(cp0);
    pl->m_gameState.m_levelTime = t0;
    cp0->release();
    return found;
}

// Rewind the live game j frames, apply the click on that frame, then run forward again to where we were.
// If the replay somehow dies, the game is put back exactly as it was and false is returned.
bool applyLate(PlayLayer* pl, bool down, bool p1, int j) {
    auto& g = st();
    const int c0 = frameOf(pl);
    Snap* sn = findSnap(c0 - j);
    if (!sn) return false;

    CheckpointObject* now = pl->createCheckpoint();
    if (!now) return false;
    now->retain();
    const double tNow = pl->m_gameState.m_levelTime;

    g.simulating = true;  // our hooks stay out of the way (and no death is triggered) during the replay
    g.simDied = false;
    g.simDeathFrame = -1;

    pl->loadFromCheckpoint(sn->cp);
    pl->m_gameState.m_levelTime = sn->t;
    const int pressFrame = sn->frame;

    pl->handleButton(down, 1, p1);  // passes straight through while simulating
    for (int guard = 0; guard < j * 4 + 8 && !g.simDied && frameOf(pl) < c0; ++guard)
        pl->update(1.f / static_cast<float>(TPS));

    g.simulating = false;

    const bool ok = !g.simDied;
    if (ok) {
        clearHistory();  // those snapshots belong to the old timeline
        g.lastInputFrame = pressFrame;
    } else {
        pl->loadFromCheckpoint(now);
        pl->m_gameState.m_levelTime = tNow;
    }
    now->release();
    return ok;
}

// ----- UI -------------------------------------------------------------------
void refreshLabel(PlayLayer* pl) {
    if (!pl || !pl->m_uiLayer) return;
    auto label = static_cast<CCLabelBMFont*>(pl->m_uiLayer->getChildByID("assist-label"_spr));
    if (!label) return;
    int n = st().retimed;
    if (n > 0) {
        std::string s = "ASSISTED: " + std::to_string(n) + " CLICKS RETIMED - NOT A REAL RUN";
        label->setString(s.c_str());
        label->setColor({255, 80, 80});
    } else {
        std::string s = "CLICK BUFFER (" + std::to_string(windowFrames()) + "F) - NOT A REAL RUN";
        label->setString(s.c_str());
        label->setColor({255, 255, 255});
    }
}

void showBanner() {
    auto scene = CCDirector::get()->getRunningScene();
    if (!scene) return;
    scene->removeChildByID("assist-banner"_spr);
    auto win = CCDirector::get()->getWinSize();
    auto l = CCLabelBMFont::create("ASSISTED RUN - NOT A REAL RUN", "bigFont.fnt");
    l->setID("assist-banner"_spr);
    l->setScale(0.6f);
    l->setColor({255, 70, 70});
    l->setPosition({win.width / 2, win.height - 40});
    scene->addChild(l, 99999);
}

// ----- click handling -------------------------------------------------------
// Decide what happens to one parked click. Runs from processCommands, between physics steps.
void resolve(PlayLayer* pl, Intent in) {
    auto& g = st();
    const int c0 = frameOf(pl);
    const Mode mode = modeOf(pl);

    // 1) A retimed click for this player is still waiting: queue behind it so press/release order never flips.
    int last = lastScheduled(in.p1);
    if (last >= 0) {
        addEvent({std::max(c0, last + 1), in.down, in.p1});
        return;
    }

    // 2) Ship / swing / robot, or a tap-mode release: nothing to fix.
    if (mode == Mode::None || (mode == Mode::Tap && !in.down)) {
        applyInput(pl, in.down, in.p1);
        return;
    }

    // 3) Plan it.
    const bool wave = (mode == Mode::Wave);
    Plan plan = planDelay(pl, in.down, in.p1, wave);

    if (plan.delay <= 0 && plan.failsNow && rescueLate()) {
        int j = planLate(pl, in.down, in.p1, wave);
        if (j > 0 && applyLate(pl, in.down, in.p1, j)) {
            g.retimed++;
            refreshLabel(pl);
            return;  // the click was applied inside the rewind
        }
    }

    if (plan.delay <= 0) {
        applyInput(pl, in.down, in.p1);  // nothing to fix: the click goes through as normal
        return;
    }

    addEvent({c0 + plan.delay, in.down, in.p1});
    g.retimed++;
    refreshLabel(pl);
}

void drainPending(PlayLayer* pl) {
    auto& g = st();
    while (!g.pending.empty()) {
        Intent in = g.pending.front();  // copy: a death inside resolve() may clear the queue
        g.pending.pop_front();
        resolve(pl, in);
    }
}

}  // namespace

// ============================================================================
// Intercept jump clicks
// ============================================================================
class $modify(BufferBGL, GJBaseGameLayer) {
    // handleButton(down, button, isPlayer1) is where the game applies every jump press / release.
    //
    // IMPORTANT: the game calls this from inside its own input processing. Running a lookahead
    // here (which calls PlayLayer::update() over and over) re-enters the game while it is still
    // walking its input queue -- that is what crashed on every cube jump. So this hook only PARKS
    // the click; the planning happens in processCommands below.
    void handleButton(bool down, int button, bool isPlayer1) {
        auto& g = st();
        if (g.firing || g.simulating || button != 1) {
            GJBaseGameLayer::handleButton(down, button, isPlayer1);
            return;
        }
        auto pl = livePlay(this);
        if (!pl) {
            GJBaseGameLayer::handleButton(down, button, isPlayer1);
            return;
        }

        const Mode m = modeOf(pl);
        const bool idle = g.sched.empty() && g.pending.empty();
        // Tap modes only retime presses; wave retimes presses AND releases.
        const bool plannable = (m == Mode::Tap && down) || m == Mode::Wave;
        if (idle && !plannable) {
            g.lastInputFrame = frameOf(pl);
            GJBaseGameLayer::handleButton(down, button, isPlayer1);  // zero added lag
            return;
        }

        g.pending.push_back({down, isPlayer1});  // parked: planned on the next physics step
    }

    // Runs once per physics step, BEFORE the game processes that step.
    void processCommands(float dt, bool isHalfTick, bool isLastTick) {
        auto& g = st();
        if (auto pl = livePlay(this)) {
            if (rescueLate() && modeOf(pl) != Mode::None) recordSnapshot(pl);  // state before this frame's inputs
            if (!g.sched.empty()) fireDue(pl);
            if (!g.pending.empty()) drainPending(pl);
        }
        GJBaseGameLayer::processCommands(dt, isHalfTick, isLastTick);
    }
};

// ============================================================================
// Lifecycle + UI
// ============================================================================
class $modify(BufferPL, PlayLayer) {
    void setupHasCompleted() {
        PlayLayer::setupHasCompleted();
        resetRuntime();
        st().retimed = 0;
        if (!bufferEnabled()) return;

        auto win = CCDirector::get()->getWinSize();
        std::string text = "CLICK BUFFER (" + std::to_string(windowFrames()) + "F) - NOT A REAL RUN";
        auto label = CCLabelBMFont::create(text.c_str(), "bigFont.fnt");
        label->setID("assist-label"_spr);
        label->setScale(0.35f);
        label->setOpacity(150);
        label->setPosition({win.width / 2, win.height - 10});
        m_uiLayer->addChild(label, 100);
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        if (st().simulating) return;
        resetRuntime();
        if (!m_isPracticeMode) {
            st().retimed = 0;
            refreshLabel(this);
        }
        if (auto scene = CCDirector::get()->getRunningScene()) scene->removeChildByID("assist-banner"_spr);
    }

    void onQuit() {
        resetRuntime();
        PlayLayer::onQuit();
    }

    // Only used to detect death inside lookahead simulations. Real deaths are untouched.
    void destroyPlayer(PlayerObject* player, GameObject* object) {
        auto& g = st();
        if (g.simulating) {
            if (!g.simDied) {
                g.simDied = true;
                g.simDeathFrame = frameOf(this);
            }
            return;
        }
        resetRuntime();  // dead players don't need buffered clicks
        PlayLayer::destroyPlayer(player, object);
    }

    void update(float dt) {
        auto& g = st();
        // Mod switched off mid-level: give any held-back clicks back to the game instead of dropping them.
        if (!g.simulating && !bufferEnabled() && (!g.sched.empty() || !g.pending.empty())) flushAll(this);
        PlayLayer::update(dt);
    }

    void levelComplete() {
        auto& g = st();
        if (g.simulating) return;
        resetRuntime();
        bool assisted = g.retimed > 0;
        bool oldTest = m_isTestMode;
        // Assumption: test-mode completions are not saved / submitted.
        if (assisted && blockSaving()) m_isTestMode = true;
        PlayLayer::levelComplete();
        m_isTestMode = oldTest;
        if (assisted) showBanner();
    }
};
