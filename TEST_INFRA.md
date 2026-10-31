# ClipForge E2E Test Infrastructure & Verification Architecture

## 1. Test Philosophy

ClipForge's testing architecture is built upon four foundational pillars:

1. **Opaque-Box & Requirement-Driven**: Tests are derived strictly from authoritative user requirements (R1 through R8 in `ORIGINAL_REQUEST.md`) and interface contracts in `PROJECT.md`. Tests verify external observable behavior, component state transitions, DOM mutations, and API responses rather than coupling to internal implementation details.
2. **Strict Sovereign Local-First & Zero External Dependencies**: 100% of workflows, AI services, and UI components must operate completely offline. Every outgoing network attempt is treated as a critical security and architectural violation.
3. **Deterministic & Isolated Execution**: Every test is self-contained, initializes its own state, executes deterministically without race conditions or order dependencies, and cleans up after itself.
4. **Adversarial & Boundary Rigor**: Systems are verified against malformed inputs, out-of-range bounds, rapid concurrent user actions, and edge cases (empty lists, extreme durations, NaN/infinite coordinates).

---

## 2. Feature Inventory Derived from User Requirements

| ID | Feature Name | Requirement | Core Behavioral Contract |
|---|---|---|---|
| **F-01** | **Live Interactive Video Canvas** | R1 | On-canvas draggable facecam & crop bounding boxes with 8-point resize handles, normalized `[0, 1]` coordinates, boundary clamping, and aspect ratio locking. |
| **F-02** | **WYSIWYG Caption Dragging & Inline Editing** | R1 | Direct vertical dragging of caption overlay with safe-zone snapping (`y_frac`), click-to-edit inline words, Enter/blur commit, and full `useUndo` integration. |
| **F-03** | **Storyboard & Hover Thumbnail Scrubbing** | R2 | Smooth hover scrubbing across clip storyboard frames with frame index mapping, hover timestamp calculation, and instant playback mounting. |
| **F-04** | **Color-Coded Waveform & Climax Heatmap** | R2 | Interactive audio waveform rendering amplitude envelope (100–200 bins), decibel bands (Blue/Green/Amber/Red), and highlight events: screams (`#FF4500`), crowd roars (`#9D00FF`), kill streaks (`#00F0FF`). |
| **F-05** | **AI Suggestions Co-Pilot & One-Click Apply** | R3 | Contextual insight cards (hook trimming with silence detection & retention estimates, hook ratings 0–100, virality teaser) with one-click atomic application. |
| **F-06** | **Smart Montage Timeline** | R4 | Drag-and-drop clip reordering track, magnetic snapping ($\pm 0.15$s threshold), transition selectors (`cut`, `crossfade`, `wipe`, `zoom`, `dip_black`), and multi-clip gapless playback. |
| **F-07** | **Multi-Platform Phone Simulator** | R5 | Pixel-accurate UI overlays for TikTok, YouTube Shorts, and Instagram Reels with real-time AABB collision warnings and auto-nudge repositioning. |
| **F-08** | **Streamer Hotkeys Navigation** | R6 | NLE keyboard navigation: J-K-L shuttle (reverse step/rewind, pause/stop, forward acceleration), Space (play/pause), I/O (mark in/out), 1–5 (quick ratings). |
| **F-09** | **Hotkey Input Isolation & Tactile HUD** | R6 | Strict suppression of hotkeys during text/caption editing (`input`, `textarea`, `select`, `contenteditable`, `.caption-inline-edit`), HUD badge display, and haptic/audio blip. |
| **F-10** | **One-Click Batch Action Toolbar** | R6 | Multi-clip selection with one-click batch rating, batch exporting, batch tagging, and batch deletion. |
| **F-11** | **Strict Local-First Offline Operation** | R7 | Verification of zero outbound network connections via socket audit hooks and frontend fetch interceptors. |
| **F-12** | **Automated E2E Test Suite & API Contracts** | R8 | Comprehensive frontend Vitest suite and backend FastAPI test suite verifying all studio interactions and endpoints. |

---

## 3. Test Architecture & Runner Instructions

### 3.1 Test Architecture Overview

```
clipforge/
├── frontend/
│   ├── src/
│   │   ├── tests/
│   │   │   └── interactive_studio.test.tsx  # Vitest + Happy-DOM + Testing Library E2E suite
│   │   ├── lib/api.test.ts                  # API client unit tests
│   │   ├── lib/useUndo.test.ts              # Undo/redo state machine tests
│   │   └── ...
│   └── vite.config.ts                       # Test runner config (Happy-DOM environment)
└── backend/
    ├── tests/
    │   ├── test_e2e_studio.py               # Self-executing E2E studio API & socket audit suite
    │   ├── test_units.py                    # Comprehensive backend unit test suite (247 tests)
    │   └── ...
```

### 3.2 Running the Frontend Test Suite

The frontend test suite runs under Vitest 4.1.9 with the Happy-DOM headless browser environment:

```powershell
# Navigate to frontend directory and execute tests
cd C:\Users\benni\Documents\clipforge\frontend
npm test
```

To run specifically the Interactive Studio E2E test suite:
```powershell
cd C:\Users\benni\Documents\clipforge\frontend
npx vitest run src/tests/interactive_studio.test.tsx
```

### 3.3 Running the Backend Test Suite

The backend test suite executes natively with Python using `starlette.testclient.TestClient` and Python `sys.addaudithook`:

```powershell
# Navigate to backend directory and run the E2E studio test suite
cd C:\Users\benni\Documents\clipforge\backend
..\.venv\Scripts\python.exe -m tests.test_e2e_studio

# Run the complete backend test suite
..\.venv\Scripts\python.exe -m tests.test_units
```

---

## 4. 4-Tier Test Strategy

The testing matrix is divided into four rigorous tiers covering all 12 features:

### 4.1 Tier 1: Feature Coverage (Happy Path, $\ge 5$ test cases per feature)

Each feature includes at least 5 happy-path test cases exercising nominal workflows:

| Feature | Test Case ID | Test Description | Expected Observable Output |
|---|---|---|---|
| **F-01 Canvas Drag/Resize** | `T1-CANVAS-01` | Center drag facecam bounding box | Normalized `(x, y)` updates smoothly clamped within `[0, 1]`. |
| | `T1-CANVAS-02` | Drag bottom-right (`se`) resize handle | Width and height increase proportionally with `(x, y)` origin fixed. |
| | `T1-CANVAS-03` | Drag top-left (`nw`) resize handle | `(x, y)` and `(w, h)` update simultaneously keeping opposite corner fixed. |
| | `T1-CANVAS-04` | Drag gameplay 9:16 crop window | Center coordinate `cx` translates horizontally within frame bounds. |
| | `T1-CANVAS-05` | Aspect ratio lock toggle | Resizing with 1:1 or 16:9 lock preserves width-to-height ratio. |
| **F-02 WYSIWYG Captions** | `T1-CAPT-01` | Vertical position drag handle | Updates `style.y_frac` reflecting new vertical placement. |
| | `T1-CAPT-02` | Safe-zone vertical snap | Snaps to safe vertical guideline (e.g., `0.78`) when within threshold. |
| | `T1-CAPT-03` | Inline word click-to-edit | Clicking caption word replaces text element with editable input field. |
| | `T1-CAPT-04` | Commit inline word on Enter | Pressing Enter saves modified word text and updates transcript words. |
| | `T1-CAPT-05` | Undo/redo caption word edit | Pressing `Ctrl+Z` reverts word text to previous snapshot; `Ctrl+Y` restores. |
| **F-03 Scrubbing & Thumbnails** | `T1-SCRUB-01` | Mouse hover over storyboard | Calculates frame index corresponding to cursor X coordinate. |
| | `T1-SCRUB-02` | Storyboard sprite manifest fetch | Endpoint returns correct frame count, tile width/height, and sprite URL. |
| | `T1-SCRUB-03` | Steady hover mounts video preview | Sustained hover activates lightweight `<video>` preview element. |
| | `T1-SCRUB-04` | Mouse leave resets preview | Cursor departure restores static high-res poster thumbnail. |
| | `T1-SCRUB-05` | Hover timestamp badge display | Tooltip shows formatted timestamp (`mm:ss.ff`) corresponding to cursor. |
| **F-04 Waveform & Heatmaps** | `T1-WAVE-01` | Amplitude envelope rendering | Waveform renders 100–200 normalized amplitude bars across duration. |
| | `T1-WAVE-02` | Decibel heatmap color bands | Samples $< -25$dB render Blue, $-25..-12$dB Green, $-12..-4$dB Amber, $> -4$dB Red. |
| | `T1-WAVE-03` | Scream event highlight | Loud scream event renders marker with color `#FF4500` at climax time. |
| | `T1-WAVE-04` | Crowd roar & kill streak events | Events render `#9D00FF` (crowd) and `#00F0FF` (kill streak) markers. |
| | `T1-WAVE-05` | Click-to-seek playback | Clicking waveform seeks playback `currentTime` to clicked timestamp. |
| **F-05 AI Suggestions Co-Pilot** | `T1-SUGG-01` | Suggestions retrieval endpoint | `GET /suggestions` returns list of hook, rating, and reframe cards. |
| | `T1-SUGG-02` | Dead air hook trim card | Displays silence trim suggestion with estimated retention impact (+18%). |
| | `T1-SUGG-03` | Hook strength rating card | Displays hook score (0–100) with descriptive assessment. |
| | `T1-SUGG-04` | Split facecam reframe suggestion | Suggests switching to split layout during high-reaction kill streak. |
| | `T1-SUGG-05` | One-click suggestion apply | Clicking apply patches clip metadata atomically and returns updated clip. |
| **F-06 Montage Timeline** | `T1-TIME-01` | Timeline clip block render | Renders sequential clip items with titles, scores, and durations. |
| | `T1-TIME-02` | Drag-and-drop clip reorder | Reorders clip array when clip block is dragged past adjacent block. |
| | `T1-TIME-03` | Magnetic snapping | Snaps clip boundaries when within $\pm 0.15$s snap threshold. |
| | `T1-TIME-04` | Transition selector node | Clicking transition selector allows choosing `cut`, `crossfade`, `wipe`, `zoom`. |
| | `T1-TIME-05` | Multi-clip sequential playback | Continuous playback advances automatically from clip $N$ to clip $N+1$. |
| **F-07 Phone Simulator** | `T1-SIM-01` | TikTok overlay mode | Renders TikTok right-rail icons, bottom sound bar, and creator text zone. |
| | `T1-SIM-02` | YouTube Shorts overlay mode | Renders Shorts like/dislike right rail, channel badge, and sound link. |
| | `T1-SIM-03` | Instagram Reels overlay mode | Renders Reels audio ticker, like/comment buttons, and description area. |
| | `T1-SIM-04` | Visual collision warning | Flags warning when caption bounding box intersects right-rail buttons. |
| | `T1-SIM-05` | Auto-nudge repositioning | Recommends or applies coordinate nudge to resolve UI zone collision. |
| **F-08 Hotkeys Navigation** | `T1-HOT-01` | Space play/pause | Space bar toggles playback between playing and paused. |
| | `T1-HOT-02` | J-K-L shuttle playback | J steps backward/rewinds, K pauses, L accelerates forward (1x, 2x, 4x). |
| | `T1-HOT-03` | I / O mark in / out | `KeyI` sets in-point to `currentTime`; `KeyO` sets out-point. |
| | `T1-HOT-04` | 1–5 Quick clip ratings | Number keys `1`–`5` set clip rating (1=reject, 3=neutral, 5=viral banger). |
| | `T1-HOT-05` | Rating feedback persistence | Number key triggers feedback endpoint call with score and weight. |
| **F-09 Input Isolation & HUD** | `T1-ISO-01` | Text input hotkey suppression | Typing in `<input>` does NOT trigger Space, J-K-L, or number hotkeys. |
| | `T1-ISO-02` | Textarea hotkey suppression | Typing in `<textarea>` does NOT trigger hotkeys. |
| | `T1-ISO-03` | Inline caption edit suppression | Typing inside `.caption-inline-edit` element suppresses all hotkeys. |
| | `T1-ISO-04` | Visual HUD indicator | Triggering hotkey displays animated HUD badge for 800ms. |
| | `T1-ISO-05` | Tactile & audio feedback | Hotkey trigger invokes haptic vibration and audio feedback pulse. |
| **F-10 Batch Action Toolbar** | `T1-BAT-01` | Multi-clip selection toggle | Clicking clip checkboxes updates `selected: string[]` list. |
| | `T1-BAT-02` | Select all / deselect all | Toolbar buttons select all clips or clear selection in one click. |
| | `T1-BAT-03` | Batch rate selected | One click rates all selected clips as approved or rejected. |
| | `T1-BAT-04` | Batch export ZIP | Triggers batch export packaging for all selected clips. |
| | `T1-BAT-05` | Batch build montage | Opens montage builder pre-populated with selected clips in order. |
| **F-11 Strict Offline Operation** | `T1-OFF-01` | Socket audit hook initialization | Python socket audit hook intercepts all `socket.connect` calls. |
| | `T1-OFF-02` | Loopback connection allowed | Connections to `127.0.0.1` and `localhost` proceed normally. |
| | `T1-OFF-03` | Frontend relative API calls | All frontend API calls use relative paths (`/api/...`, `/media/...`). |
| | `T1-OFF-04` | No external CDN scripts/styles | `index.html` and styles contain zero external CDN URLs. |
| | `T1-OFF-05` | Offline AI models local | All ONNX models and Whisper weights reside in local directories. |
| **F-12 API Contracts & E2E** | `T1-API-01` | Project retrieval `GET /api/projects/{id}` | Returns project JSON with clips, transcripts, and metadata. |
| | `T1-API-02` | Waveform endpoint `GET /waveform` | Returns samples, dbfs, climax_t, and color-coded event list. |
| | `T1-API-03` | Suggestions endpoint `GET /suggestions` | Returns structured suggestions list with patch dictionaries. |
| | `T1-API-04` | Suggestion apply `POST /suggestions/{id}/apply` | Atomically patches clip and returns updated object. |
| | `T1-API-05` | Batch endpoint `POST /clips/batch` | Executes action across clip IDs and reports count. |

---

### 4.2 Tier 2: Boundary & Corner Cases ($\ge 5$ test cases per feature)

Each feature includes at least 5 boundary, extreme, zero-value, negative, or overflow test cases:

| Feature | Test Case ID | Boundary Condition Tested | Expected Robust Handling |
|---|---|---|---|
| **F-01 Canvas Drag/Resize** | `T2-CANVAS-01` | Drag facecam beyond canvas boundary ($x < 0, y < 0, x > 1, y > 1$) | Coordinates clamped strictly to $[0, 1 - w]$ and $[0, 1 - h]$. |
| | `T2-CANVAS-02` | Resize facecam below minimum threshold ($w < 0.05$ or $h < 0.05$) | Dimensions clamped to minimum safe size of $0.05 \times 0.05$. |
| | `T2-CANVAS-03` | Zero/negative initial coordinates ($x=0, y=0, w=0$) | Fallback to default safe centered bounding box without NaN errors. |
| | `T2-CANVAS-04` | Extreme container aspect ratio ($100 \times 1000$ or $1000 \times 100$) | Coordinate normalization remains mathematically stable. |
| | `T2-CANVAS-05` | Gameplay crop snap edge cases ($cx = 0$ and $cx = 1.0$) | Crop window clamped so entire viewport remains within 16:9 frame. |
| **F-02 WYSIWYG Captions** | `T2-CAPT-01` | Empty caption text (delete all word characters) | Handles empty string gracefully; preserves word duration slot. |
| | `T2-CAPT-02` | Overlong word string (1000+ characters or emoji spam) | Word renders with CSS overflow wrap without breaking layout. |
| | `T2-CAPT-03` | Vertical drag to extreme top/bottom ($y\_frac = 0.0$ or $1.0$) | Clamped to safe visible vertical zone ($0.05 \le y\_frac \le 0.95$). |
| | `T2-CAPT-04` | Rapid rapid undo/redo at stack boundaries | No-op when undoing at bottom or redoing at top of stack. |
| | `T2-CAPT-05` | HTML/Script injection in caption word (`<script>alert(1)</script>`) | Sanitized as plain text string; zero script execution. |
| **F-03 Scrubbing & Thumbnails** | `T2-SCRUB-01` | Hover mouse at negative or overflow X coordinate ($x < 0$ or $x > \text{width}$) | Clamped to first (0) or last ($N-1$) storyboard frame. |
| | `T2-SCRUB-02` | Zero-duration or sub-second clip ($duration = 0.1s$) | Generates single-frame fallback storyboard gracefully. |
| | `T2-SCRUB-03` | Missing storyboard sprite image (404 / IO error) | Gracefully falls back to static clip poster without crash. |
| | `T2-SCRUB-04` | Ultra-rapid hover enter/leave (100 events in 10ms) | Debounces video preview mounting without orphaned video tags. |
| | `T2-SCRUB-05` | Extreme 60-minute clip storyboard scrubbing | Maintains linear timestamp distribution without memory leak. |
| **F-04 Waveform & Heatmaps** | `T2-WAVE-01` | Pure silence audio file ($-\infty$ dBFS or 0 amplitude) | Renders flat baseline at $-60$dBFS with quiet blue color. |
| | `T2-WAVE-02` | Digital clipping audio (0 dBFS max peak everywhere) | Renders saturated red peaks without arithmetic overflow or NaN. |
| | `T2-WAVE-03` | Zero detected events (no screams, crowd, or streaks) | Renders clean waveform without event overlays; no null errors. |
| | `T2-WAVE-04` | Overlapping simultaneous events at same timestamp | Stacks or priorities event markers gracefully without visual clutter. |
| | `T2-WAVE-05` | Waveform canvas resized to zero width/height | Handles zero-dimension canvas without divide-by-zero errors. |
| **F-05 AI Suggestions Co-Pilot** | `T2-SUGG-01` | Clip with zero silence (speech starts at $t=0.0s$) | Omits dead air trim suggestion; reports strong hook start. |
| | `T2-SUGG-02` | Apply suggestion with invalid/missing suggestion ID | Returns 404 Not Found with descriptive error message. |
| | `T2-SUGG-03` | Suggestion trim proposing start $>$ end | Rejected with 400 Bad Request; clip state unmodified. |
| | `T2-SUGG-04` | Apply suggestion concurrently to deleted clip | Returns 404 Not Found cleanly without database lock. |
| | `T2-SUGG-05` | Empty suggestion patch dictionary `{}` | Idempotent no-op returning 200 OK with unchanged clip. |
| **F-06 Montage Timeline** | `T2-TIME-01` | Drag single clip timeline (length = 1) | Reorder drag disabled or no-op; no index out of bounds. |
| | `T2-TIME-02` | Extreme snap threshold ($\Delta t = 0.149s$ vs $0.151s$) | Deterministic snapping at $\le 0.15s$; free movement at $> 0.15s$. |
| | `T2-TIME-03` | Drag clip to extreme timeline start or end | Clamped cleanly to index 0 or index $K-1$. |
| | `T2-TIME-04` | 0-second transition duration | Instant hard cut between clips without visual stutter. |
| | `T2-TIME-05` | 50-clip mega-montage drag-and-drop | Maintains 60fps drag rendering performance without DOM lag. |
| **F-07 Phone Simulator** | `T2-SIM-01` | Bounding box exactly touching UI zone border (0px gap) | Evaluates non-colliding if strict $<$; triggers warning if $\le$. |
| | `T2-SIM-02` | Full-screen bounding box ($x=0, y=0, w=1, h=1$) | Detects multiple collisions across top, bottom, and right rails. |
| | `T2-SIM-03` | Auto-nudge when no safe zone exists | Clamps to center safe zone with warning badge. |
| | `T2-SIM-04` | Switch platform simulator while collision active | Recalculates collisions instantly for new platform dimensions. |
| | `T2-SIM-05` | Negative dimensions bounding box ($w < 0$ or $h < 0$) | Normalized to positive bounding box before collision check. |
| **F-08 Hotkeys Navigation** | `T2-HOT-01` | Press J when video already at 0.00s | Clamped to 0.00s without negative playback time error. |
| | `T2-HOT-02` | Press L repeatedly up to maximum speed | Clamped at maximum 4x or 8x speed ceiling. |
| | `T2-HOT-03` | Mark Out before Mark In ($O < I$) | Rejects or clamps out-point to $\ge in + 0.5s$. |
| | `T2-HOT-04` | Press invalid number key (e.g., `0`, `6`, `9`) | Ignored; only 1–5 trigger rating curation actions. |
| | `T2-HOT-05` | Rapid concurrent keypresses (J + L + Space together) | Predictable last-write-wins state machine without lockup. |
| **F-09 Input Isolation & HUD** | `T2-ISO-01` | Press Space while typing in caption edit box | Inserts space character into text; does NOT pause video. |
| | `T2-ISO-02` | Press J-K-L while typing title in input field | Inserts letters 'j', 'k', 'l'; does NOT change playback rate. |
| | `T2-ISO-03` | Press Escape while editing caption | Cancels editing mode, blurs input, and re-enables hotkeys. |
| | `T2-ISO-04` | Rapid hotkey triggers (20 presses in 1 second) | HUD badge updates smoothly without animation queue pileup. |
| | `T2-ISO-05` | Device with haptic feedback unsupported (`vibrate=null`) | Falls back silently without JavaScript exceptions. |
| **F-10 Batch Action Toolbar** | `T2-BAT-01` | Batch action with 0 clips selected | Toolbar hidden or action buttons disabled. |
| | `T2-BAT-02` | Batch delete all clips in project | Shows confirmation modal; updates project clip count to 0. |
| | `T2-BAT-03` | Batch action on non-existent clip ID | Server filters invalid IDs; processes valid IDs and reports 200. |
| | `T2-BAT-04` | Batch export with partial rendering failure | Returns ZIP with ready clips and error summary for failed clips. |
| | `T2-BAT-05` | 100+ clips selected for batch rating | Single atomic database update without transaction timeout. |
| **F-11 Strict Offline Operation** | `T2-OFF-01` | Attempt socket connection to `8.8.8.8` (Google DNS) | Intercepted by audit hook; raises `RuntimeError` immediately. |
| | `T2-OFF-02` | Attempt socket connection to `1.1.1.1` (Cloudflare) | Intercepted by audit hook; raises `RuntimeError` immediately. |
| | `T2-OFF-03` | Frontend `window.fetch("https://api.openai.com")` | Rejected by offline guard; request blocked before network. |
| | `T2-OFF-04` | Attempt WebSocket connection to external server | Blocked by offline network policy. |
| | `T2-OFF-05` | Localhost loopback connection (`127.0.0.1:8000`) | Permitted without restriction. |
| **F-12 API Contracts & E2E** | `T2-API-01` | Request non-existent project `GET /api/projects/unknown` | Returns 404 Not Found with JSON error detail. |
| | `T2-API-02` | Request non-existent clip `GET /waveform` | Returns 404 Not Found with JSON error detail. |
| | `T2-API-03` | Malformed JSON in `POST /clips/batch` | Returns 422 Unprocessable Entity with validation details. |
| | `T2-API-04` | Negative rating value in feedback endpoint | Returns 422 validation error or 400 Bad Request. |
| | `T2-API-05` | Large payload batch operations (1000 IDs) | Processes within request limit or rejects cleanly. |

---

### 4.3 Tier 3: Cross-Feature Combinations

Pairwise interactions across features verifying that independent subsystems integrate flawlessly:

1. **`T3-COMB-01`: Canvas Reframing + Streamer Hotkeys**
   - While dragging facecam bounding box on canvas, pressing Space or J-K-L updates video playback while maintaining active pointer capture and drag position without dropping the frame.
2. **`T3-COMB-02`: WYSIWYG Inline Caption Editing + Hotkey Input Isolation**
   - Streamer double-clicks caption word to edit text. Streamer types `"JUST KILLED 5 ENEMIES!"` containing letters J, K, L, numbers 1–5, and Spaces. All characters appear in the text box. Zero hotkeys trigger. Pressing Enter commits text and instantly re-activates hotkey listeners.
3. **`T3-COMB-03`: Audio Waveform Scrubbing + Live Canvas Preview**
   - Hovering or clicking on audio waveform climax spike at $t=18.2s$ seeks video player instantly. Video canvas updates rendered gameplay frame and facecam box to $t=18.2s$ synchronously.
4. **`T3-COMB-04`: Montage Timeline Reordering + Phone Simulator Collision Engine**
   - Changing clip sequence in montage timeline recalculates platform collision warnings for newly ordered clips against active platform overlay (TikTok/Shorts/Reels).
5. **`T3-COMB-05`: AI Suggestions One-Click Apply + Audio Waveform Update**
   - Clicking "Apply Dead Air Hook Trim" modifies clip in-point from $t=0.0s$ to $t=0.8s$. Waveform component immediately rescales its time axis and shifts climax event markers to reflect the updated in-point.
6. **`T3-COMB-06`: Phone Simulator Platform Switch + Canvas Safe Zone Guides**
   - Switching phone simulator from TikTok to YouTube Shorts updates on-canvas safe zone boundary guides and repositions auto-nudged caption bounding box.
7. **`T3-COMB-07`: Batch Selection Toolbar + Hotkey Ratings**
   - Selecting 3 clips in grid view and pressing hotkey `5` batch-rates all 3 selected clips as viral bangers (score 1.0) simultaneously.
8. **`T3-COMB-08`: Streamer J-K-L Shuttle + Waveform Playhead Tracking**
   - Shuttling with L key at 2x forward speed updates the waveform playhead scrubber line smoothly at 2x rate matching audio/video playback.

---

### 4.4 Tier 4: Real-World Streamer Workloads

End-to-end user journeys replicating genuine streaming workflows:

#### Workload 1: "The Clutch Ace Review" (High-Action Valorant/CS2 Clip Curation)
1. Streamer loads newly detected clip `"5K Ace Climax"` ($duration = 32.4s$).
2. Waveform renders loud scream event (`#FF4500` at $t=21.0s$) and kill streak event (`#00F0FF` at $t=18.5s$).
3. Streamer presses Space to play; hits L for 2x fast-forward through opening setup.
4. AI Suggestion Card recommends: *"Trim 1.2s dead air opener (+22% retention)"*. Streamer clicks **Apply**. Clip in-point snaps to $t=1.2s$.
5. Streamer drags facecam bounding box to top-left corner, resizes handle to $0.28 \times 0.28$, and locks 1:1 aspect ratio.
6. Streamer toggles TikTok phone simulator. Platform simulator flags a collision: caption intersects TikTok creator description zone.
7. Streamer clicks **Auto-Nudge Safe**. Caption box moves upward to $y\_frac = 0.72$. Collision warning clears.
8. Streamer double-clicks first caption word, edits text to `"UNBELIEVABLE ACE"`, and presses Enter.
9. Streamer hits hotkey `5` (Viral Banger). HUD flashes `⭐ 5 Stars` with haptic pulse.
10. Streamer clicks **Export Clip**. Render initiates completely offline.

#### Workload 2: "Speed Montage Assembly" (Multi-Clip Highlight Reel Sequencing)
1. Streamer selects 4 top-rated clips from stream review session.
2. Batch action toolbar appears: streamer clicks **Build Montage**.
3. Montage timeline loads with 4 sequential clip blocks.
4. Streamer drags Clip 4 to the first position. Timeline clips magnetically snap into place with $\pm 0.15$s threshold.
5. Streamer clicks transition node between Clip 1 and Clip 2, selecting `crossfade` ($0.5s$).
6. Multi-clip preview plays sequence continuously across transitions.
7. Streamer triggers hotkey `O` at $t=48.0s$ to clamp total montage duration.
8. Streamer exports montage MP4. All processing executes 100% offline.

---

## 5. Verification & Integrity Assertions

1. **Deterministic Execution**: Zero flaky timers or network race conditions.
2. **Audit Hook Enforcement**: Every test session verifies that `sys.addaudithook` traps and halts any unauthorized outbound socket connection.
3. **Adversarial Verification**:
   - Special characters & XSS payload injection in caption words (`<svg/onload=alert(1)>`, `"; DROP TABLE clips;--`).
   - Boundary coordinates ($x < 0, w > 1$).
   - Concurrent hotkey bursts and fast pointer movements outside the canvas element.
