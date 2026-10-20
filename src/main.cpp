// Click Buffer
// ---------------------------------------------------------------------------
// A real-time input buffer with a distribution, instead of a rescue.
//
//   1. You press jump. The press is NOT applied right away.
//   2. The mod snapshots the game and simulates a short way ahead, once for
//      "click now" and then for "click 1 frame later", "2 frames later", ...
//   3. If clicking now already works, the click is applied as normal (nothing changes).
//   4. If clicking now fails but a later frame works, we have found the true window
//      [a, b] (in frames after your click). You are only buffered if `a` is within
//      your chosen window N (2 / 4 / 8 / 16) -- i.e. you were at most N frames early.
//   5. The click is held in the buffer and released at a frame drawn from a
//      distribution inside [a, b]:
//          clicked as early as the buffer tolerates -> released on `a` (earliest frame)
//          clicked just 1 frame early               -> somewhere in the early-middle
//      so there is no single fixed "guardian angel" frame to find in a completion.
//   6. The live game is restored to exactly where it was, and your click is released
//      at the chosen frame. The matching release is delayed by the same amount.
//
// A label says this is NOT a real run, and (optionally) completions with a retimed
// click are not saved.
//
// "Frames" are 240 Hz physics steps. Only tap-style modes are buffered (cube, ball,
// UFO, spider); ship / wave / swing / robot clicks are left completely untouched.
// ---------------------------------------------------------------------------
 
#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
 
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <vector>
 
using namespace geode::prelude;
 
namespace {
 
// ----- tuning knobs ---------------------------------------------------------
constexpr double TPS = 240.0;          // physics steps per second (stock GD)
constexpr int HORIZON = 110;           // how many frames ahead each lookahead simulation runs
constexpr int TAP_FRAMES = 6;          // lookahead assumes the click is released after this many frames
constexpr int MIN_IMPROVEMENT = 8;     // a later click must survive at least this many frames longer to count
constexpr int SCORE_TOL = 3;           // tolerance when comparing "frames survived"
constexpr int MAX_WINDOW_SCAN = 2;     // look at most MAX_WINDOW_SCAN * N frames past `a` for the end of the window
constexpr double MEAN_FRAC = 0.40;     // 0 = earliest frame of window, 0.5 = dead middle. "early-middle" = 0.4
constexpr double SPREAD_FRAC = 0.20;   // randomness of the landing spot (fraction of the window width)
 
// ----- settings -------------------------------------------------------------
bool bufferEnabled() { return Mod::get()->getSettingValue<bool>("enabled"); }
bool blockSaving() { return Mod::get()->getSettingValue<bool>("block-saving"); }
double lookaheadMs() { return static_cast<double>(Mod::get()->getSettingValue<int64_t>("lookahead-ms")); }
int windowFrames() {
    int n = 4;
    try { n = std::stoi(Mod::get()->getSettingValue<std::string>("window")); } catch (...) {}
    return std::clamp(n, 1, 64);
}
 
// ----- state ----------------------------------------------------------------
struct Event {
    int step;     // game step (frame) on which to apply it
    bool down;
    bool p1;
};
 
struct State {
    std::deque<Event> sched;   // buffered clicks waiting for their frame (sorted by step)
    double owed = 0.0;         // time we've taken from the engine while stepping manually
    int lastStep[2] = {-1000000, -1000000};
    int lastDelay[2] = {0, 0};
 
    bool simulating = false;
    bool firing = false;       // true while WE are applying a buffered click (so our hook lets it through)
    bool simDied = false;
    int simDeathFrame = -1;
 
    int retimed = 0;
    std::mt19937 rng{std::random_device{}()};
};
 
State& st() {
    static State s;
    return s;
}
 
int frameOf(GJBaseGameLayer* l) {
    // NOTE: assumes m_gameState.m_levelTime exists and counts seconds of level time.
    return static_cast<int>(std::llround(l->m_gameState.m_levelTime * TPS));
}
 
void resetRuntime() {
    auto& g = st();
    g.sched.clear();
    g.owed = 0.0;
    g.lastStep[0] = g.lastStep[1] = -1000000;
    g.lastDelay[0] = g.lastDelay[1] = 0;
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
 
bool tapMode(PlayLayer* pl) {
    auto p = pl->m_player1;
    if (!p) return false;
    return !(p->m_isShip || p->m_isDart || p->m_isSwing || p->m_isRobot);
}
 
void addEvent(Event e) {
    auto& q = st().sched;
    auto it = std::upper_bound(q.begin(), q.end(), e, [](Event const& x, Event const& y) { return x.step < y.step; });
    q.insert(it, e);
}
 
// Apply every buffered event that is due. Called between physics steps.
void fireDue(PlayLayer* pl) {
    auto& g = st();
    int k = frameOf(pl);
    while (!g.sched.empty() && g.sched.front().step <= k) {
        Event e = g.sched.front();
        g.sched.pop_front();
        g.firing = true;
        pl->handleButton(e.down, 1, e.p1);
        g.firing = false;
    }
}
 
// Pick the frame inside [a, b] the click is released on.
// early = how many frames before the true window the player clicked (1..N).
int pickTarget(int a, int b, int early, int N) {
    if (b <= a) return a;
    double W = static_cast<double>(b - a);
    double frac = static_cast<double>(N - early) / static_cast<double>(N);  // 0 when clicked as early as allowed
    double mean = a + frac * MEAN_FRAC * W;
    double sd = frac * SPREAD_FRAC * W;
    if (sd <= 0.0) return a;  // earliest possible click -> earliest frame
    std::normal_distribution<double> dist(mean, sd);
    for (int i = 0; i < 16; ++i) {
        int r = static_cast<int>(std::llround(dist(st().rng)));
        if (r >= a && r <= b) return r;
    }
    return std::clamp(static_cast<int>(std::llround(mean)), a, b);
}
 
// ----- lookahead simulation -------------------------------------------------
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
            pl->handleButton(evs[i].down, 1, evs[i].p1);
        pl->update(1.f / static_cast<float>(TPS));  // our update hook passes straight through while simulating
    }
 
    g.simulating = false;
    return g.simDied ? std::max(0, g.simDeathFrame - c0) : horizon;
}
 
// How many frames should the click that just arrived be delayed? (0 = leave it alone)
int planDelay(PlayLayer* pl, bool p1) {
    auto& g = st();
    const int N = windowFrames();
    const int H = HORIZON;
    const int c0 = frameOf(pl);
 
    CheckpointObject* cp = pl->createCheckpoint();
    if (!cp) return 0;
    cp->retain();
    const double t0 = pl->m_gameState.m_levelTime;
 
    const auto start = std::chrono::steady_clock::now();
    const double budget = lookaheadMs();
    auto outOfTime = [&]() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() > budget;
    };
 
    std::vector<Event> base(g.sched.begin(), g.sched.end());
    auto run = [&](int d) {
        auto ev = base;
        ev.push_back({c0 + d, true, p1});
        ev.push_back({c0 + d + TAP_FRAMES, false, p1});
        std::stable_sort(ev.begin(), ev.end(), [](Event const& x, Event const& y) { return x.step < y.step; });
        return simulate(pl, cp, t0, c0, ev, H);
    };
 
    int result = 0;
    do {
        int s0 = run(0);
        if (s0 >= H) break;  // clicking right now is already fine
 
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
            break;
        }
 
        // If nothing survives the whole horizon (e.g. a later obstacle needs another click),
        // fall back to "the delays that survive longest".
        int threshold = H;
        if (a < 0) {
            if (bestPartial < s0 + MIN_IMPROVEMENT) break;  // no delay helps
            threshold = bestPartial - SCORE_TOL;
            for (int d = 1; d <= N; ++d)
                if (sc[d] >= threshold) { a = d; break; }
            if (a < 0) break;
        }
 
        // Phase 2: find where the window ends.
        int b = a;
        for (int d = a + 1; d <= a + MAX_WINDOW_SCAN * N; ++d) {
            if (outOfTime()) break;  // keep the part of the window we already know
            if (run(d) >= threshold) b = d;
            else break;
        }
 
        result = pickTarget(a, b, a, N);
        log::debug("buffered click: window {}..{} frames after click, released on +{} (N={})", a, b, result, N);
    } while (false);
 
    // Put the live game back exactly where it was before the lookahead.
    pl->loadFromCheckpoint(cp);
    pl->m_gameState.m_levelTime = t0;
    cp->release();
    return result;
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
void onPress(PlayLayer* pl, bool p1) {
    auto& g = st();
    int idx = p1 ? 0 : 1;
    int c0 = frameOf(pl);
 
    int delay = 0;
    if (g.lastStep[idx] >= c0) {
        // an earlier buffered click for this player is still pending: just queue behind it
        delay = g.lastStep[idx] + 1 - c0;
    } else if (tapMode(pl)) {
        delay = planDelay(pl, p1);
    }
 
    g.lastDelay[idx] = delay;
    g.lastStep[idx] = c0 + delay;
    addEvent({c0 + delay, true, p1});
 
    if (delay > 0) {
        g.retimed++;
        refreshLabel(pl);
    }
}
 
void onRelease(PlayLayer* pl, bool p1) {
    auto& g = st();
    int idx = p1 ? 0 : 1;
    int c0 = frameOf(pl);
    // keep the hold length: release is delayed by the same amount as its press, never before it
    int step = std::max(c0 + g.lastDelay[idx], g.lastStep[idx] + 1);
    g.lastStep[idx] = step;
    addEvent({step, false, p1});
}
 
}  // namespace
 
// ============================================================================
// Intercept jump clicks
// ============================================================================
class $modify(BufferBGL, GJBaseGameLayer) {
    // handleButton(down, button, isPlayer1) is where the game applies every jump press / release.
    // (pushButton / releaseButton no longer exist on GJBaseGameLayer in current bindings.)
    void handleButton(bool down, int button, bool isPlayer1) {
        // Our own scheduled clicks, lookahead simulations and non-jump buttons pass straight through.
        if (st().firing || button != 1) {
            GJBaseGameLayer::handleButton(down, button, isPlayer1);
            return;
        }
        auto pl = livePlay(this);
        if (!pl || (!tapMode(pl) && st().sched.empty())) {
            GJBaseGameLayer::handleButton(down, button, isPlayer1);
            return;
        }
        if (down) onPress(pl, isPlayer1);
        else onRelease(pl, isPlayer1);
    }
};
 
// ============================================================================
// Step-accurate release of buffered clicks + UI
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
        if (g.simulating) {
            PlayLayer::update(dt);
            return;
        }
        if (!bufferEnabled()) g.sched.clear();
 
        if (g.sched.empty()) {
            // normal frame: hand back any time we were holding and run as usual
            dt += static_cast<float>(g.owed);
            g.owed = 0.0;
            PlayLayer::update(dt);
            return;
        }
 
        // A click is buffered: advance one physics step at a time so it lands on the exact frame.
        g.owed += dt;
        const double step = 1.0 / TPS;
        for (int guard = 0; guard < 512 && !g.sched.empty() && g.owed >= step; ++guard) {
            fireDue(this);
            PlayLayer::update(static_cast<float>(step));
            g.owed -= step;
        }
        if (g.sched.empty() && g.owed > 0.0) {
            float rest = static_cast<float>(g.owed);
            g.owed = 0.0;
            PlayLayer::update(rest);
        }
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
