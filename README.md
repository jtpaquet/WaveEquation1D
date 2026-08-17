# WaveEquation1D

1D transmission-line (telegrapher's equations) simulator that animates signal
propagation and reflections on a 10 m RG58 BNC cable, for two real setups:

1. **Rogowski coil -> 10 cm lead -> 10 m RG58 -> RC low-pass filter**,
   comparing a strongly mismatched filter (10 kΩ / 1 nF) against a near
   impedance-matched one (50 Ω / 5 µF).
2. **250 kHz / 500 mVpp square wave (op-amp output) -> 10 m RG58 -> 10 cm
   lead -> 20 MΩ DAQ input**, comparing a bare op-amp output (~5 Ω, no added
   resistor) against a 50 Ω (matched) resistor and a 40 Ω (partial match)
   resistor.

Each scenario includes a short 10 cm stub of ordinary (non-coax) lead wire --
at the coil's own leads for the Rogowski scenarios, or at the DAQ probe lead
for the square-wave scenarios -- so you can see the extra small reflection
where that stub's impedance meets the 50 Ω coax, not just at the two main
interfaces.

For each case it produces an animation of the wave bouncing back and forth
between the two interfaces until it settles, an animation of the exact
steady-state (looping), and a static comparison of the measured steady-state
waveform against the original source signal (faded in the background).

## Physics / how it works

`waveline/lines.py` implements the telegrapher's equations

```
L' dI/dt = -dV/dx - R' I
C' dV/dt = -dI/dx
```

on a lossy line with a leapfrog (Yee-type) finite-difference time-domain
scheme. A line can be a single uniform cable, or a *cascade* of segments of
different characteristic impedance (used for the 10 cm lead-wire stubs) --
segments share one spatial grid, and the impedance step at a junction
produces a real, physically correct partial reflection automatically, no
extra machinery needed. The two true ends are closed by lumped elements (the
coil's own R+L, the RC filter, the DAQ's input resistance, etc.). A
subtlety: the boundary node's own capacitance (half of one grid cell) forms
a *fast* local RC/RL time constant with the attached lumped resistor/
inductor -- often faster than the simulation time step even though the step
size satisfies the line's own CFL condition. Plain forward-Euler blows up
there, so each boundary is advanced with an *exact* zero-order-hold
discretization of its local linear ODE (`scipy.linalg.expm`), which is
unconditionally stable regardless of the ratio between dt and the
boundary's own RC/RL time constant.

`waveline/freqdomain.py` is an independent, closed-form phasor solver
(standard transmission-line input-impedance / reflection-coefficient chain
algebra, also generalized to a cascade of segments) used to compute the
*exact* periodic steady state -- for the square wave this is done by taking
the FFT of the actual (finite-rise-time) source waveform and running each
harmonic through the line's transfer function, then re-summing in time. This
is what the FDTD transient settles down to, and is much cheaper/exact to
compute directly rather than by waiting out the transient. The two solvers
are cross-checked against each other in `tests/test_physics.py` (agreement
to <1%), which also checks the cascade math directly: a step reflecting off
an internal Z0 junction between two matched terminations settles, with no
ringing, to the exact classic resistive-divider value.

Both solvers are built from the *same* scenario parameters in
`waveline/scenarios.py`, so the transient animation and the steady-state
comparison are always describing the same circuit.

## Key modeling assumptions (edit `waveline/scenarios.py` to change them)

- **Cable**: RG58, Z0 = 50 Ω, velocity factor 0.66 (v ≈ 1.98e8 m/s), length
  10 m. Attenuation is modeled from a datasheet-typical figure of ~4.3 dB per
  100 m at 1 MHz, scaled as sqrt(f) (skin effect) for the frequency-domain
  solver; the FDTD solver uses a single value evaluated at the fundamental
  frequency (a line-loss model can't easily be frequency-dependent in a
  fixed-timestep FDTD scheme without convolution, so this is a small
  simplification -- it mainly affects how many round trips it takes for
  ringing to visibly die out, not the reflection physics itself).
- **Rogowski coil**: 80 turns, 1 cm² cross-section, 23.6 µH self-inductance,
  2 Ω DC resistance. The quoted "200 mV/A" is taken as the coil's *calibrated
  sensitivity at 250 kHz* (as commercial current-probe sensitivities usually
  are), so the Thevenin EMF is 0.2 V/A x 30 A = 6 V at 250 kHz, in series with
  the coil's own R + jωL as the source impedance the cable sees. (A raw,
  uncalibrated Rogowski coil's output is actually proportional to dI/dt, not
  I; if you intended 200 mV/A as a raw uncompensated-coil mutual-inductance
  figure instead, this would need reinterpreting -- see the note in
  `scenarios.py`.)
- **RC filter measurement point**: the filter is cable-end -> R -> node -> C
  -> ground, and the measured voltage is at the R/C node (not the cable-end
  voltage) -- i.e. a real single-pole RC low-pass, not just a resistor load.
- **Square wave**: "500 mV square wave" is taken as 500 mVpp (±250 mV,
  bipolar about 0 V). A real op-amp doesn't produce an infinitely sharp edge
  (and a mathematically discontinuous step would also excite spurious
  numerical dispersion on any finite grid), so a realistic ~20 ns rise/fall
  time is used -- still >100x faster than the 4 µs period, so it still reads
  as a clean square wave, but fast enough that the classic reflection/ringing
  behavior on the edges is fully visible.
- **DAQ input**: modeled as a simple 20 MΩ resistor to ground (no input
  capacitance specified, so none assumed).
- **Op-amp output**: modeled as an ideal, non-inductive voltage source in
  series with whichever external resistor (5, 40 or 50 Ω) is being tested.
  The "no added resistor" case uses 5 Ω rather than a literal 0 Ω, since a
  real op-amp's closed-loop output stage always has *some* small nonzero
  output impedance at this frequency -- 0 Ω isn't physically achievable, and
  changes the ringing decay rate a little (see below).
- **Lead-wire stubs**: 10 cm of generic unshielded two-conductor wire (not
  coax), assumed Z0 = 200 Ω, velocity factor 0.7, somewhat lossier than the
  coax -- a placeholder for whatever the coil's own pigtail or the DAQ probe
  lead actually is, since that wasn't specified. Change `LEAD_Z0`/`LEAD_VF`/
  `LEAD_LENGTH` in `scenarios.py` if you know the real figures. At these
  frequencies 10 cm is a very small fraction of a wavelength even for the
  square wave's harmonics, so its effect is a small, fast blip right at the
  edge/transition rather than a major reshaping of the signal -- but it's a
  real, physically distinct reflection point and is marked as such
  ("Z0 step") in the plots.

## Usage

```
pip install -r requirements.txt
python run_simulation.py --list                     # see all scenarios
python run_simulation.py                              # generate everything (~5-10 min)
python run_simulation.py --scenario square_low_5      # just one
python run_simulation.py --fast                          # coarser/quicker preview
```

Outputs go to `outputs/<scenario>_reflections.gif`,
`outputs/<scenario>_steady_loop.gif` and `outputs/<scenario>_steady.png`.

Run the physics sanity checks with `python -m pytest tests/` (or
`python tests/test_physics.py`).

## Reading the plots

- **`*_reflections.gif`**: top panel is a snapshot of V(x) along the physical
  cable at each instant (source at the red dashed line, load at the blue
  one, with their reflection coefficients Γ annotated); bottom panel is the
  actually-measured signal building up over time, with a cursor showing
  where the top panel currently is. Watch the bottom trace for beating /
  ringing that decays (or doesn't) as multiple reflections superpose.
- **`*_steady_loop.gif`**: the same two panels, but computed exactly (no
  transient) and looped over one period.
- **`*_steady.png`**: top panel overlays the measured steady-state signal
  (solid, left axis) against the original source waveform (faded, right
  axis) on the *same time axis*, so you can read off the phase shift and any
  edge distortion/ringing directly. Bottom panel is a filmstrip of the
  spatial profile at several instants across one period.

**Why the spatial profile often looks almost flat for the Rogowski (sine)
cases:** at 250 kHz the signal's wavelength on this cable is about 790 m --
the 10 m cable is "electrically short" (a few degrees of phase, annotated on
each plot), so a single snapshot in space barely varies across the cable's
length even though there's a real, physically meaningful reflection
happening. The interesting effect is in *time*: multiple round trips
(~101 ns each) interfere and build up (or cancel) before the received signal
settles to its final amplitude and phase -- that's what the bottom time-trace
panel shows, and it's exactly where impedance matching changes the result:
compare the near-doubled, strongly-shifted output of the mismatched
10 kΩ/1 nF filter to the small, more predictable output of the matched
50 Ω/5 µF filter. The square-wave scenarios are electrically "faster" (a fast
edge has significant harmonic content up into the MHz range, where the cable
is a much larger fraction of a wavelength), so their spatial panels do show
a visibly traveling wavefront.

**Ringing on the square wave:** with no added series resistor (bare ~5 Ω
op-amp output), both ends are close to total reflectors (Γ_source ≈ -0.82,
Γ_load ≈ +1 from the high-Z DAQ input), so each edge rings for many round
trips (period ≈ 2 x cable transit time ≈ 200 ns) before settling -- a
textbook illustration of why series/source termination matters. A 50 Ω
series resistor makes the source match the line exactly (Γ_source = 0), so
each edge reflects once off the open load and is fully absorbed on the way
back -- no ringing. 40 Ω is a partial match: a small, quickly-decaying
overshoot instead of sustained ringing.
