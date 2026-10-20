// Click Buffer
// ---------------------------------------------------------------------------
// A real-time input buffer: early clicks snap to a safe frame.
//
//   1. You press jump. The press is NOT applied right away -- it is parked.
//   2. On the next physics step (from processCommands, NEVER from inside the game's own
//      input handling) the mod snapshots the game and simulates a short way ahead, once for
//      "click now" and then for "click 1 frame later", "2 frames later", ...
//   3. If clicking now already works, the click is applied as normal (nothing changes, zero lag).
//   4. If clicking now fails but later frames work, the click is moved to the MIDDLE of the
//      safe window (the centre of the run of frames that keep you alive, within your window
//      N = 2 / 4 / 8 / 16) -- not the first, edge-of-death frame -- so you get margin on both sides.
//   5. (Optional, experimental) If NO early fix exists, the mod also checks whether pressing up
//      to N frames EARLIER would have survived. If so it rewinds the game to that frame, applies
//      the click there, and replays forward to where you were. Needs a snapshot of every frame,
//      which is expensive -- leave it off if you care about FPS.
//   6. The live game is restored to exactly where it was before the lookahead.
//
// Tap modes (cube, ball, UFO, spider): the lookahead assumes a short tap.
// Wave: there is no "tap" -- every press AND every release is a direction change, so the
//       lookahead assumes the new direction is held for WAVE_HORIZON frames instead.
// Ship / swing / robot: left completely untouched.
//
// PERFORMANCE: the lookahead has to run on the main thread (GD's objects are not thread-safe),
// so it is kept cheap and bounded:
//   - simulated updates are batched SIM_CHUNK steps at a time, like a normal 60 fps frame
//     (the game's per-frame work used to run once per simulated step);
//   - the first lookahead starts from the live state instead of restoring a snapshot;
//   - every click has a hard time limit (lookahead-ms) that is enforced INSIDE the simulation;
//   - planning is capped to ~PLAN_DUTY of wall-clock time. When the budget is spent the click
//     simply goes through untouched, so a burst of clicks can never tank the frame rate.
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

using Clock = std::chrono::steady_clock;

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

// Performance.
constexpr int SIM_CHUNK = 4;           // physics steps per simulated update() call (4 = one frame at 60 fps)
constexpr double PLAN_DUTY = 0.25;     // planning may use at most ~25% of wall-clock time on average
constexpr double MIN_PLAN_MS = 1.0;    // with less budget than this a lookahead is pointless: click passes through

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
    int step;     // live schedule: absolute game step. Simulations: steps RELATIVE to the first simulated step.
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

    // simulation bookkeeping
    bool simulating = false;     // true while WE are driving the game for a lookahead / replay
    bool firing = false;         // true while WE are applying a click (so our hook lets it through)
    bool simDied = false;
    int simDeathFrame = -1;
    std::vector<Event> simEvents;  // clicks to apply during the current simulation (relative steps)
    size_t simIdx = 0;
    int simRel = -1;               // how many physics steps into the simulation we are
    int simLastK = 0;
    bool simSeen = false;

    int lastInputFrame = -1000000;  // last frame any input was applied (late rescue must not rewind past it)
    std::deque<Snap> history;       // rolling snapshots of the last few frames (late-click rescue only)

    // planning time budget (ms we may still spend right now)
    double planCredit = 0.0;
    Clock::time_point creditTick{};
    bool creditInit = false;

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
    g.creditInit = false;  // fresh attempt: full planning budget again
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

// ----- planning time budget -------------------------------------------------
// A leaky bucket: it refills at PLAN_DUTY ms per ms of wall-clock time, up to lookahead-ms.
double planBudgetMs() {
    auto& g = st();
    const double cap = lookaheadMs();
    const auto now = Clock::now();
    if (!g.creditInit) {
        g.creditInit = true;
        g.planCredit = cap;
    } else {
        const double elapsed = std::chrono::duration<double, std::milli>(now - g.creditTick).count();
        g.planCredit = std::min(cap, g.planCredit + elapsed * PLAN_DUTY);
    }
    g.creditTick = now;
    return g.planCredit;
}

// Charges the time spent planning against the bucket when it goes out of scope.
struct PlanCharge {
    Clock::time_point start = Clock::now();
    ~PlanCharge() {
        auto& g = st();
        const auto end = Clock::now();
        g.planCredit = std::max(0.0, g.planCredit - std::chrono::duration<double, std::milli>(end - start).count());
        g.creditTick = end;
    }
};

// ----- lookahead simulation -------------------------------------------------
// Event steps here are RELATIVE to the first simulated step, so the simulation does not depend on
// exactly how the game counts frames.
std::vector<Event> makeEvents(std::vector<Event> const& base, int d, bool down, bool p1, bool wave) {
    auto ev = base;
    ev.push_back({d, down, p1});
    // Tap modes: assume a short tap. Wave: the new direction is simply held.
    if (!wave && down) ev.push_back({d + TAP_FRAMES, false, p1});
    std::stable_sort(ev.begin(), ev.end(), [](Event const& x, Event const& y) { return x.step < y.step; });
    return ev;
}

void beginSim(std::vector<Event> evs) {
    auto& g = st();
    g.simulating = true;
    g.simDied = false;
    g.simDeathFrame = -1;
    g.simEvents = std::move(evs);
    g.simIdx = 0;
    g.simRel = -1;
    g.simSeen = false;
}

void endSim() {
    auto& g = st();
    g.simulating = false;
    g.simEvents.clear();
    g.simIdx = 0;
}

// Called from processCommands while simulating: apply the simulated clicks on exactly their step.
void applySimClicks(GJBaseGameLayer* l) {
    auto& g = st();
    const int k = frameOf(l);
    if (!g.simSeen || k != g.simLastK) {  // processCommands can run more than once per step
        g.simSeen = true;
        g.simLastK = k;
        ++g.simRel;
    }
    while (g.simIdx < g.simEvents.size() && g.simEvents[g.simIdx].step <= g.simRel) {
        Event e = g.simEvents[g.simIdx++];
        l->handleButton(e.down, 1, e.p1);  // our handleButton hook passes it straight through while simulating
    }
}

// Restore the snapshot (unless fromLive), apply `evs`, run up to `horizon` frames.
// Returns how many frames the player survived (== horizon if it never died),
// or -1 if the deadline hit first (result unknown).
int simulate(PlayLayer* pl, CheckpointObject* cp, double t0, int c0, std::vector<Event> evs, int horizon,
             Clock::time_point deadline, bool fromLive = false) {
    auto& g = st();
    beginSim(std::move(evs));

    if (!fromLive) {
        pl->loadFromCheckpoint(cp);
        pl->m_gameState.m_levelTime = t0;
    }

    bool aborted = false;
    for (int guard = 0; guard < horizon * 2 && !g.simDied && frameOf(pl) < c0 + horizon; ++guard) {
        if (Clock::now() >= deadline) { aborted = true; break; }
        pl->update(static_cast<float>(SIM_CHUNK) / static_cast<float>(TPS));  // our update hook passes straight through
    }

    const bool died = g.simDied;
    const int deathFrame = g.simDeathFrame;
    endSim();

    if (aborted) return -1;
    if (!died) return horizon;
    return std::min(horizon, std::max(0, deathFrame - c0));  // a death beyond the horizon is not a failure
}

// How many frames should this click be delayed? (0 = leave it alone)
// If clicking now fails, every delay in the window is tried and the click goes to the MIDDLE of the
// first run of safe delays.
Plan planDelay(PlayLayer* pl, bool down, bool p1, bool wave, Clock::time_point deadline) {
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

    // clicks that are already scheduled, relative to now
    std::vector<Event> base;
    for (auto const& e : g.sched) base.push_back({std::max(0, e.step - c0), e.down, e.p1});

    bool first = true;
    auto run = [&](int d) {
        const bool live = first;  // the very first run starts from the live state: saves one full restore
        first = false;
        return simulate(pl, cp, t0, c0, makeEvents(base, d, down, p1, wave), H, deadline, live);
    };

    do {
        const int s0 = run(0);
        if (s0 < 0) break;   // out of time: leave the click alone
        if (s0 >= H) break;  // clicking right now is already fine
        out.failsNow = true; // clicking now fails; stays true unless a fix is found

        std::vector<int> sc(N + 1, -1);  // frames survived per delay (-1 = not tried)
        int bestPartial = s0;
        int runStart = -1, runEnd = -1;  // the first contiguous run of fully safe delays
        bool timedOut = false;
        for (int d = 1; d <= N; ++d) {
            const int s = run(d);
            if (s < 0) { timedOut = true; break; }
            sc[d] = s;
            bestPartial = std::max(bestPartial, s);
            if (s >= H) {
                if (runStart < 0) runStart = d;
                runEnd = d;
            } else if (runStart >= 0) {
                break;  // the safe run is over: both of its edges are known
            }
        }

        int pick = -1;
        if (runStart >= 0) {
            // Middle of the safe window. (If time ran out mid-run this is the middle of what we verified.)
            pick = (runStart + runEnd) / 2;
        } else if (timedOut) {
            log::debug("lookahead ran out of time, click left alone");
            out.failsNow = false;
            break;
        } else {
            // Nothing survives the whole horizon (e.g. a later obstacle needs another click):
            // fall back to the delays that survive longest, and take the middle of those.
            if (bestPartial < s0 + minImprove) break;  // no delay helps
            const int threshold = bestPartial - tol;
            int a = -1, e = -1;
            for (int d = 1; d <= N; ++d) {
                if (sc[d] >= threshold) {
                    if (a < 0) a = d;
                    e = d;
                } else if (a >= 0) {
                    break;
                }
            }
            if (a < 0) break;
            pick = (a + e) / 2;
        }

        out.delay = pick;
        out.failsNow = false;
        log::debug("buffered {}: middle of the safe window is +{} (N={})", wave ? "wave edge" : "click", pick, N);
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
int planLate(PlayLayer* pl, bool down, bool p1, bool wave, Clock::time_point deadline) {
    auto& g = st();
    if (!g.sched.empty() || g.history.empty()) return 0;
    const int N = windowFrames();
    const int H = wave ? WAVE_HORIZON : HORIZON;
    const int c0 = frameOf(pl);

    CheckpointObject* cp0 = pl->createCheckpoint();
    if (!cp0) return 0;
    cp0->retain();
    const double t0 = pl->m_gameState.m_levelTime;

    int found = 0;
    for (int j = 1; j <= N; ++j) {
        if (Clock::now() >= deadline) break;
        if (c0 - j <= g.lastInputFrame) break;  // older snapshots would drop a real input
        Snap* sn = findSnap(c0 - j);
        if (!sn) continue;
        const int s = simulate(pl, sn->cp, sn->t, sn->frame, makeEvents({}, 0, down, p1, wave), H, deadline);
        if (s < 0) break;  // out of time
        if (s >= H) { found = j; break; }
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
    const int pressFrame = sn->frame;

    std::vector<Event> evs{Event{0, down, p1}};  // the click lands on the first replayed step
    beginSim(std::move(evs));

    pl->loadFromCheckpoint(sn->cp);
    pl->m_gameState.m_levelTime = sn->t;

    // Single steps so we land exactly back on c0.
    for (int guard = 0; guard < j * 4 + 8 && !g.simDied && frameOf(pl) < c0; ++guard)
        pl->update(1.f / static_cast<float>(TPS));

    const bool died = g.simDied;
    endSim();

    const bool ok = !died;
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

    // 3) Out of planning budget: never stall the frame, just let the click through.
    const double allowMs = planBudgetMs();
    if (allowMs < MIN_PLAN_MS) {
        applyInput(pl, in.down, in.p1);
        return;
    }

    // 4) Plan it, with a hard time limit.
    PlanCharge charge;  // bills the time below against the budget
    const auto deadline = charge.start + std::chrono::duration_cast<Clock::duration>(
                                             std::chrono::duration<double, std::milli>(allowMs));
    const bool wave = (mode == Mode::Wave);
    Plan plan = planDelay(pl, in.down, in.p1, wave, deadline);

    if (plan.delay <= 0 && plan.failsNow && rescueLate()) {
        int j = planLate(pl, in.down, in.p1, wave, deadline);
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
    // walking its input queue. So this hook only PARKS the click; the planning happens in
    // processCommands below.
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
        if (g.simulating) {
            applySimClicks(this);  // simulated clicks land on exactly their step
        } else if (auto pl = livePlay(this)) {
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
