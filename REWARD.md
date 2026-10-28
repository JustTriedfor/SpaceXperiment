# Reward and merit system (v2)

Code: `src/training/reward.py` (reward), `src/training/merit.py` (merit score and curriculum gate).
Every number below can be changed in the app's Parameters tab ("Custom reward" group).

## Why the first points system failed (measured on 32 agents, 250k steps)

| Finding | Evidence |
|---|---|
| Crashes paid. The dead-center and upright bonuses were paid when the legs touched, even if the lander then crashed. | 72% of crash episodes had positive total reward. Agents learned to hit the pad centre hard (median 0.2 m off, about 15 m/s). |
| Gravity was paid as "progress". The drift term gave 10 points per metre toward the pad whatever the attitude. | A tumbling lander still collected about +13 per few steps (+400 for a 40 m fall). |
| No signal about WHEN to brake. | Touchdown speed barely moved (22.9 to 14.7 m/s in 250k steps). No agent ever survived a landing. |
| Survival is a cliff. | Touchdown at 1.5 m/s or less survives 98%, about 2.5 m/s survives 23%, 3 m/s or more never survives. So almost every episode looked equally bad. |
| The simulated lander cannot hover and brakes at about 8.9 m/s^2 (thrust/weight 1.9, minimum throttle 0.95). | Measured in the simulator. A real-speed start needs a proper suicide burn. |

## The new reward (points, before the 0.01 global scale)

**1. Dense guidance: potential-based shaping.** Each step pays `phi(next state) - phi(this state)`. A potential cannot
be farmed (any loop returns to the same value), unlike a per-step bonus.

| Part of phi | Value | Why |
|---|---|---|
| Distance to pad centre | -10 per m | get to the pad |
| Speed above the safe-descent envelope | -20 per m/s of excess | the braking signal |
| Tilt | -100 per rad | stay upright (a tumbling lander is charged, never paid) |
| Spin | -10 per rad/s | no tumbling |
| Each leg on the ground | +20 | touch down on the legs |

The envelope is the fastest speed from which the lander can still stop: `sqrt(2 a h) + 1.5 m/s`, with `a = 6 m/s^2`
(about 65% of the measured 8.9) and 1.5 m/s the measured safe touchdown speed. Falling faster than that lowers
phi, braking raises it, so the agent is paid exactly when it should burn. A free fall nets negative; flying along the
envelope nets positive (both tested).

**2. Small costs.** Time -0.05 per step, fuel -0.05 per unit burned. Hovering can never pay.

**3. Landing score, by zone, paid only for a survived landing** (the episode ends with the lander at rest). The green
zone is the pad (radius 2 m). A *soft* landing is inside the zone, at most 1.5 m/s, at most 0.15 rad.

| Landing | Points |
|---|---|
| Survived, outside the green zone | +100 |
| Survived, inside the zone, not soft | +500 |
| Soft landing, inside the zone, every parameter at its limit (zone edge, 1.5 m/s or 0.15 rad) | +700 |
| Soft landing, bullseye, 0 m/s, perfectly upright | +10,000 |

Between those, the score rises **exponentially** with an accuracy index `A` in [0, 1]: `score = 700 x (10000/700)^A`.
`A` is the weighted geometric mean of how far inside each limit the landing is: `(1 - miss/2 m)^0.35 x
(1 - speed/1.5 m/s)^0.45 x (1 - tilt/0.15 rad)^0.2`. Any single parameter at its limit gives `A = 0` (700), and all
three perfect gives `A = 1` (10,000). Examples:

| Miss | Speed | Tilt | Points |
|---|---|---|---|
| 1.0 m | 1.0 m/s | 0.10 rad | 1,944 |
| 0.5 m | 0.5 m/s | 0.05 rad | 4,441 |
| 0.2 m | 0.2 m/s | 0.02 rad | 7,233 |
| 0.1 m | 0.1 m/s | 0.01 rad | 8,506 |
| 0 | 0 | 0 | 10,000 |

The global reward scale is 0.005 (was 0.01) because the landing score now reaches 10,000.

**4. Other penalties.** Timeout (never landed): -500. Fuel runs out: -150 once. A crash or leaving the bounds is scored by the two crash scores in section 6.

**5. Slow-approach charge (scales exponentially with altitude) and ascent.** Added after the first v2 runs, where agents
bought safety by braking immediately and creeping down (episodes grew from 3.7 s to 5.4 s, fuel per survived landing about
350 units, about 7 s of full thrust). A fixed "slower than 3 m/s" rule never fired because they crawled at 5-10 m/s.

(The numbers in this section are the first setting; the current, stricter setting is in section 8.)

The charge per second is `R(h) x shortfall`:
- `R(h) = 2 x 100^((h - 10 m) / 30 m)` points per second: 2/s at 10 m, 20/s at about 20 m, 200/s at 40 m for a stopped lander.
- `shortfall = max(0, 1 - speed / floor(h))`, where the speed floor is 75% of the safe-descent envelope at that height
  (about 18 m/s at 40 m, 9 m/s at 10 m). Above the floor nothing is charged, so a corridor remains between the floor and the envelope.
- Exempt below 2 m (the final approach) and whenever a leg is down.

| Speed \ height | 10 m | 20 m | 30 m | 40 m |
|---|---|---|---|---|
| 0 m/s | 2.0/s | 9.3/s | 43/s | 200/s |
| 5 m/s | 1.0/s | 5.9/s | 30/s | 147/s |
| 10 m/s | 0 | 2.5/s | 17/s | 93/s |
| 15 m/s | 0 | 0 | 3.6/s | 40/s |

Ascent: -1000, once per episode, when the lander climbs faster than 1 m/s with no leg on the ground (a small bounce after
landing does not count).

**6. Crash scoring (two independent scores added together) and the off-course charge.**
A crash earns no landing score. Instead it gets a *position* score and a *speed* score, which are independent, so the agent can tell
which of the two it is improving.

Position score (where it hit: the first both-legs touch if there was one, otherwise one step before the impact):

| Miss from the pad centre | Points |
|---|---|
| 0 m (dead centre) | +1000 |
| 1 m | +707 |
| 2 m (the zone edge) | +500 |
| just past 2 m | -1000 (a cliff) |
| 3 m | -1350 |
| 4 m | -1823 |
| 6 m | -3322 |
| 10 m or more | -10,000 (the floor) |

Inside the zone the score rises exponentially (`500 x 2^(1 - miss/2 m)`). Outside it grows by a factor of 1.35 for every metre of miss.

Speed score (how hard it hit, the speed just before impact): 0 at 1.5 m/s or less, then **-100 for every m/s over 1.5**:
3 m/s is -150, 10 m/s is -850, a 30 m/s free fall is -2,850.

Examples (position + speed): dead centre at 3 m/s = +850, dead centre at 10 m/s = +150, dead centre in a 30 m/s free fall = -1,850,
3 m outside at 3 m/s = -1,500. Leaving the bounds is scored the same way (a miss of 8+ m).

An optional extra for a crash inside the zone within 5 degrees of vertical exists (`CRASH_VERTICAL_BONUS`) and is off (0) by default.

Quirks of the numbers, documented in tests: a slow dead-centre crash (about 2 m/s, +950) scores above a survived but not-soft landing in the zone (+500),
while a good soft landing (700 up to 10,000) still beats it. A plain free fall can no longer pay: it is dead centre but about 30 m/s (-1,850).

**Off-course charge: -10 points per second** while the lander's predicted touchdown point is outside 2 m of the pad centre
(where a ballistic path from the current velocity, gravity only, would land). The spawn is aimed at the pad, so it starts on course. It goes off
course when the lander brakes vertically (a longer fall) without cancelling its sideways speed, so this term teaches lateral control. It is a
flat -10/s however far off, and is not charged once a leg is down.

**7. Tilt limit (an environment rule, plus a penalty).** At any time or altitude, a lander tilted more than **32.5 degrees from vertical**
is terminated instantly (the episode ends on the very step it crosses the line; a full turn counts as upright, but it cannot
get there without crossing the limit). The rule applies to both reward types: the default-reward agent receives its own crash
reward (-100), the custom agent receives **-1000**.

- **Projected crash on top of the -1000.** A flat -1000 is exploitable: a free fall scores about -1850, so throwing the lander
  over on purpose (-1000) would beat it. So a tilt-out pays -1000 *plus* the crash it was heading for, scored from its ballistic
  path (position and speed), or from its landing if it had already touched down. It is always exactly 1000 worse than carrying on.
  Set "Tilt-out: also score the crash" to 0 for the flat -1000 (not recommended). Tested both ways.
- **Spawns are capped at 20 degrees of tilt** (12.5 degrees of margin), otherwise some starts would be illegal. The plan's classic
  start tilts of 60 and 90 degrees (levels 2 and 3) and the steep tail-first alignment of slow side entries are clamped to 20 degrees.
- A tilt-out counts as a failure (`crashed`, merit 0, `end = tilt`) and is logged in the `tilt_out` column.

**8. Environment rules added after the tilt-rule run** (both reward types; the run showed 31% survived landings but 76% of
episodes longer than 4.5 s, a 6.6 s median, 12% running to the 20 s cap: agents survived by creeping and hovering).

| Rule | What happens |
|---|---|
| Descent clock | Both legs must touch within **4.5 s** of the start, otherwise the episode is cut off with a **-1000** timeout. Once down, the lander gets 2 s more to come to rest (the simulator needs about 0.5 s of stillness to count it as at rest). Setting the settle time to 0 makes 4.5 s a hard cap on the whole episode. |
| Main-engine ignitions | The first ignition is free, a real Falcon 9 lights its landing engine once. Each re-ignition costs `25 x 2^(n-2)`: 25, 50, 100, 200, 400, 800, 1600, 3200, **6400** for the tenth. After **10** ignitions the engine cannot be lit again (a burn in progress may finish). A burn is one continuous stretch of main thrust, so flickering counts every flicker. |
| Stall (zero tolerance) | Airborne, higher than 0.5 m and slower than **0.5 m/s** (after a 0.5 s grace period at the start) = terminated at once, **-1500** (plus the projected crash, like the tilt-out). The last half metre is exempt so a gentle touchdown is not a stall. |
| Stricter slow-approach floor | The speed floor is now **90%** of the safe-descent speed (was 75%) and the charge is **4 points/s** at 10 m for a stopped lander (was 2). A stopped lander pays about 4/s at 10 m, 19/s at 20 m, 86/s at 30 m, 400/s at 40 m. |

- **Why 0.5 m/s and not exactly 0:** a lander never sits at exactly 0.000 m/s in a simulation with a 20 ms step; anything that slow in
  mid-air is a stall.
- **Interpretation to note:** "4.5 s" is the time to touch down, with a 2 s settle allowance, because a lander that touches down at
  4.2 s could not be at rest by 4.5 s. Set the settle time to 0 for the literal whole-episode cap.
- **The projected-crash guard** (see the tilt rule above) is applied to the stall too: stalling high up is always worse than a free-fall crash.
- **Merit** now also rewards a single burn: `0.35 softness + 0.20 centering + 0.15 upright + 0.10 fuel left + 0.20 single burn`.

**9. First-ignition bonus (+500) and ignition observations.**
Why: in the first run with the ignition rules (about 100k steps), the agents had learned to avoid the engine, not to use it. Median touchdown
speed was 28 m/s (a free fall), 97% of crashes were inside the zone, 38% of episodes never fired the main engine, and the average number of
ignitions per episode fell from 6.3 to 1.4 as the exponential re-ignition penalty punished experimenting.

- **+500 once per episode**, when the main engine is lit for the first time. Re-ignitions are not paid and still cost 25, 50, 100 ... as before,
  so the best ignition pattern is one clean burn: 1 ignition = +500, 2 = +475, 3 = +425, 4 = +325, 10 = about -12,300.
- **Cheapest way to farm it** is a one-step engine blip at the start and then a free fall. Measured: free fall scores about -1,730, a blip then free fall
  about -1,650, so with the bonus about -1,150. Far below any landing (at least +700), so it cannot become a stable strategy. A test guards this.
- **The agent now observes its ignition state.** Two values are appended to the observation (11 values instead of 9): the share of the ignition
  budget used so far, and whether the main engine is burning right now. Without them the first-ignition bonus, the re-ignition penalties and the
  ignition limit all depend on hidden state a feed-forward policy cannot see. Older saved models do not fit the new observation size.

**10. Staged rules, and one shared policy with 8 environments.**
The ignition-rule runs learned nothing at 100k samples (a pure 28 m/s free fall), while the run before them (tilt rule only) reached 17% survived
landings at 100k and 36% at 250k. The settings of the optimizer were not the cause (same 3e-4 learning rate as the first run; the policy was
updating normally). Switching every new rule on from step 0 made trying the engine too expensive.

| Stage | Rules active |
|---|---|
| 0 | tilt limit, slow-approach charge, crash scoring, landing score, first-ignition bonus |
| 1 | + the 4.5 s descent clock (-1000) and the stall termination (-1500) |
| 2 | + the 10-ignition limit and the re-ignition penalty (every rule) |

- **The gate:** a stage advances when the share of episodes that end with the lander safely at rest, over the last **200 episodes played at the
  current stage**, reaches **20%**. Or after 1,000,000 steps in the stage (0 = never). Episodes that started under an older stage are ignored.
  A stage change takes effect at each environment's next reset, so no episode changes its rules half way through.
- **Logged:** `stages.csv` (when each stage switched and why), a `rules_stage` column in `episodes.csv`, and the stage in the viewer header.
- **The app default is staged on** ("Staged rules" in the Run group). Off = every rule from step 0, as before. Gate settings are in the Curriculum group.
- **Shared policy, 8 environments** (sweep mode off): all 8 environments feed one policy, so each update uses 8 x 1024 = 8,192 samples (the sweep's
  agents used 1,024 each), the value estimate is less noisy (explained variance 0.35 vs 0.22 at 50k samples) and a run is several times faster in
  wall-clock time. The learning-rate sweep stays available for comparing rates.

## Merit score (0-100), the same for every agent

Used to compare agents. Never compare training reward values. 0 unless the lander touched down and the episode ended at rest. Otherwise
`100 x (0.40 softness + 0.25 centering + 0.15 uprightness + 0.20 fuel left)`, where softness is `1 - speed/2 m/s`,
centering is `1 - miss/2 m`, uprightness is `1 - tilt/0.35 rad`, and fuel left is the unused fraction of the budget.
A perfect landing with a full tank scores 100.

## Success definitions

| Name | Rule | Used for |
|---|---|---|
| Strict success | at rest, within 2 m, at most **1.5 m/s** (was 0.5), at most 0.15 rad, fuel never ran out | the final comparison |
| Soft landing | at rest, within 2 m, at most 1.5 m/s, at most 0.15 rad | the curriculum gate and the "landed" columns |

The strict success used to need 0.5 m/s, below the only published real touchdown (1.47 m/s, B1038, Aug 2017) and below the simulator's survival
cliff (98% survive at 1.5 m/s or less), so it was almost never met: 0.1% of the finished 8-env run's episodes against 6.7% at 1.5 m/s.
The strict success only changes the reported success number. It is not part of the reward, the stage gate or the level gate. Strict and soft
now share the 1.5 m/s limit and differ only in the fuel condition.

## Start conditions (the "spacex" profile)

Descent speed from real Falcon 9 figures (approximate, see `src/utils/spacex_profile.py`), Froude-scaled to the
simulator, entering from a side on a ballistic path aimed at the pad, tail-first with a small angle of attack. The
speed ramps with the curriculum (25% at level 0 up to 100% at the last level).

## 11. Revision: centring must be slow, and a flat crash penalty

Finding (run `custom_s0_app`, ~930k steps, later half): 100% of episodes ended within 2 m of the centre (median miss 0.44 m) but 0% landed softly and the median touchdown speed was 12.7 m/s. A crash in the middle still paid about +2,000 net (distance shaping ~+1,990, crash position ~+850, first ignition +500, crash speed ~-1,130), so the agents optimised accuracy first.

Changes:
- **Crash-position bonus fades with speed.** Inside the zone the bonus (+500 edge to +1000 centre) is multiplied by max(0, 1 - overspeed / 15 m/s), overspeed = speed above the 1.5 m/s goal (`CRASH_POS_FADE_MPS`). A centred crash at 30 m/s earns 0 position points; at 3 m/s it still earns 90%. Penalties outside the zone are never reduced. The tilt-out / stall projected crashes use the same rule.
- **Flat crash penalty -250** (`CRASH_PENALTY`, term `crash_pen`), independent of the position and speed scores, for every crash or out-of-bounds (not for tilt-out / stall, which have their own penalties).
- **Metric:** `brake_peak` (peak braking demand v^2/2h against 7.45 m/s^2) is now measured only above 3 m. It used to saturate at its cap for any crash because the demand explodes in the last half metre.

## 12. Revision: the clock follows the spawn, and gates follow performance

Finding (run `custom_s0_app`, 1.0M-1.16M steps): both gates fired at 1.0M because their step caps ran out (survival 4%, success 1%): stage 1 (4.5 s clock, stall rule) and level 1 -> 2 (wind 10, larger offsets) started together. 62% of episodes then timed out at exactly 4.5 s with the lander 70-130 m up, and 38% left the side of the world at 100-150 m with the throttle stuck near full. The fastest possible soft landing from the 200 m spawn (about 37 m/s down) is about 7.3 s (a 1.1 s free fall, then about 6.2 s of braking at the measured 7.45 m/s^2), so a 4.5 s clock could never be met.

Changes:
- **Clock scaled with the spawn.** At every reset the limit is `max(TIME_LIMIT_S, 1.3 x fastest soft landing + 1 s)` (`min_landing_time` in wrappers.py; `TIME_LIMIT_SCALE`, `TIME_LIMIT_PAD_S`). 4.5 s remains the floor, so low spawns behave as before. `info["time_limit_s"]` carries the value.
- **A step cap only advances an agent that has started to succeed.** A level advances at 80% soft landings, or at the 500k-step cap only if the soft-landing rate over the window is at least 10% (`ADVANCE_CAP_MIN_RATE`); a stage advances at 20% survival, or at its 1M-step cap only with at least 10% survival (`RULES_GATE_CAP_MIN_RATE`). Below that the level or stage holds.
- **Cooldown.** After any level or stage advance neither gate can fire for 150k steps (`GATE_COOLDOWN_STEPS`), so two harder things never start together.


## 13. SpaceXperiment: the same reward at real scale

The design above is unchanged; the numbers were rescaled for a 3 km, 300 m/s descent (see src/training/reward.py): W_DIST 0.5/m, W_VEL 10 per m/s over the envelope, W_SPIN 100, ENVELOPE_DECEL 14 m/s^2 (65% of the ~22 m/s^2 net braking at full throttle), ENVELOPE_FLOOR 3 m/s, slow-approach charge 0.5 points/s at a 100 m reference height with a 300 m growth span (exempt below 30 m, capped at 400 m, so at most ~50 points/s). The first run showed why: with 4 points/s at 150 m and a 900 m cap the charge cost -24,000 per episode (the level-0 start is slower than the full-speed envelope) and the agents learned to free-fall, ASCEND_SPEED 5 m/s, SOFT_SPEED 3 m/s, zone radius 10 m, CRASH_OUT_GROWTH 1.012 per metre, CRASH_SPEED_PER_MPS -100 (goal 3 m/s), CRASH_POS_FADE_MPS 15 (first tried -25 and 60: a PPO run started from the cloned expert then traded touchdown speed for accuracy, 13 -> 20 m/s while the miss fell to 7 m), OFFCOURSE_TOLERANCE_M 30. Landing scores (100 / 500 / 700 ... 10,000), the first-ignition bonus (+500), the re-ignition penalties, the flat crash penalty (-250), tilt-out (-1000), stall (-1500) and timeout (-1000) keep their values.
