Subject: VULCAN-JAX update: VULCAN 2 parity vs VULCAN 3 split, and a correction on conver_ignore

Hi Shami,

Short version: I went back through your list from July 14 and checked each item
against the actual repositories rather than against my local copy. One of my
earlier changes was wrong, I have fixed it, and the fix changes a number in the
paper. Two figures are attached.

**The conver_ignore correction (your item 6)**

You asked why I had a long species list when you normally only have HC3N. You
were right and I was wrong. The long list was mine, left over from early tests. I
had told you I would go back to HC3N, but a later commit put the 13-species list
back into the shipped configs with a comment crediting it to master. That
attribution was false. I fetched the files to check:

  exoclime master vulcan_cfg.py           conver_ignore = []
  exoclime master cfg_examples HD189      conver_ignore = []
  your master vulcan_cfg.py               conver_ignore = []
  your vm_branch vulcan_cfg.py            conver_ignore = ['HC3N']

The 13-species list is in no upstream repository. The reason my check "passed"
is embarrassing: the list had been written into my local VULCAN-master copy, and
I then read it back as proof that it came from you. That copy also had my own
stall detector and my conv_stall_window knob pasted into op.py and store.py, so
it was not an independent code at all. I have left that copy untouched, added a
guard so the parity tool now refuses to run against any checkout containing my
own identifiers, and rewritten the tests that depended on it.

The measured effect, with everything else held fixed:

  HD189     []  1495 steps    ['HC3N']  1495 steps    13-species  1296 steps
  HD209     []  1206          ['HC3N']  1206          13-species  1206
  WASP-39b  []  1202          ['HC3N']  1202          13-species  1202

So [] and ['HC3N'] are identical everywhere I tested. On this network HC3N is
not the species that gates convergence, atomic C is. The 13-species list only
moves HD189, and it does that by dropping species out of the convergence test
rather than by changing the physics.

This matters for the paper, so I have already fixed it. Our HD189 number of 1296
accepted steps came from that list, and so did the VULCAN 2.0 number of 1396,
because my local copy had the same list in it. I re-measured both against a
clean clone of your repository, three runs each, on a quiet machine:

  VULCAN 2.0   217 s   1590 steps  (1580 / 1590 / 1602 across the three runs)
  VULCAN 3.0    38.7 s 1495 steps  (1495 in all three)

Table 1 now carries those, and I removed the claim that we match the published
step count exactly, because that was never an independent match. The speedup
range in the abstract does not change: HD189 moves from 5.3x to 5.6x, which is
still inside the 4.4-6.7x set by HD209 and WASP-39b. HD209 and WASP-39b did not
need re-measuring, since both are step-matched rows and our step count for each
is insensitive to this setting.

The agreement numbers actually got better. Our median difference against
VULCAN 2.0 on HD189 improved from 7.3e-6 to 3.3e-6, and the worst single cell
above a 1e-10 mixing ratio dropped from 2.5e-2 to 5.2e-4.

**VULCAN 2 parity and VULCAN 3 are now separate presets**

The shipped planet configs (default, HD189, HD209, W39b) are now VULCAN 2
parity: their convergence settings match fetched exoclime master exactly, which
means empty conver_ignore, central-difference molecular diffusion, high_temp_cut
off, adaptive rtol off, and my stall detector off.

HD189_vulcan3.yaml is new and is the explicit VULCAN 3 preset. It is the same
planet with your vm_branch numerics: hybrid molecular diffusion, high_temp_cut,
adaptive rtol, conver_ignore = ['HC3N'], count_max 20000, mtol_conv 1e-18. Every
one of those lines cites the vm_branch line it came from. You can now run the two
configs on the same planet and compare the schemes directly.

**Your item 7, conv_stall_window**

You asked whether this was a new convergence condition. It is, and it is mine,
not VULCAN's. It declares convergence when longdy has sat near its running
minimum for 200 accepted steps without a 5 percent improvement. The problem was
that it had no off switch, so it could quietly affect a run that was supposed to
be VULCAN 2 comparable. There is now an explicit use_conv_stall flag, off in all
parity configs, and when it is off the test is compiled out entirely. Runs also
now report why they stopped: end_case still reports 1 for any convergence, so I
added a separate termination_reason field that separates a normal convergence
from a stall convergence. None of the parity runs above needed the fallback.

**Figures**

figure_1_zhang_molecular_diffusion: the benchmark against the analytic
diffusive-separation solution from Zhang, Shia and Yung (2013). Central
difference matches the analytic profile to 0.8 percent. First-order upwind is off
by 46 percent, which is the dissipation you flagged. Our operator reproduces
VULCAN 2.0's to zero difference, so the port is faithful. The right panel is the
stability reason for the hybrid: central difference goes negative above cell
Peclet 2, upwind never does.

figure_2_hd189_hybrid_validation: the hybrid on a full HD189 run. It converges
under upwind, switches at step 1500, and finishes central at 2102 steps. The
important panel is the third: compared with a pure central run, the hybrid's
final state differs by at most 1.7e-4 relative, while pure upwind differs by up
to 2.6. That is the direct check that the hybrid gives back the central answer
rather than the upwind one.

**Other items from your list**

gs is gone, replaced by Rp and Mp. hycean_pin_time is confirmed to do nothing
unless use_fix_H2He is true. high_temp_cut is in and I confirmed it raises the
bottom boundary and regrids rather than clipping temperature; the values 3500 K
and 1e6 dyn/cm2 match your branch. The YAML conversion is done and each run
writes its fully resolved config next to the output.

Two smaller things I noticed while checking the shipped networks, both in files
no default config uses, so nothing of ours is affected:
SNCHO_photo_network_C3.txt still has the old 1.00E-20 low-pressure rate for
CH2CN + H + M, and in SNCHO_DMS_photo_network_Tsai2024.txt that reaction's k0 and
k_infinity look swapped relative to the column header.

**A real cross-code comparison, finally**

Because I could not trust my local copy, I cloned exoclime/VULCAN fresh and ran
it. Two things came out of that.

First, the good news. On HD189, with both codes converged and everything matched,
VULCAN-JAX and VULCAN 2.0 agree to a median relative difference of 3.8e-6 across
all species above a 1e-12 abundance floor. In the deep well-mixed part of the
column the agreement is at the 1e-9 to 1e-6 level: helium 1.3e-9, water 1.2e-7,
methane 1.5e-6. This is the first cross-code check I have that is actually
independent, and I think it is a much better validation statement than anything
in the current draft. Step counts still differ, 1495 versus 1600, which is the
roundoff-driven step-sequence difference we already discuss.

Second, the catch, and this is the part I would most like your opinion on. Out
of the box the two codes differ by 20 percent, not 3.8e-6. It took four
experiments to find out why, and it is not the solver. Our FastChem input file
is different from yours. We use Lodders 2019 with the rocky elements suppressed
to -3.0, and you use Lodders 2009 with them at solar. For a C-H-N-O network the
only value that actually changes is helium, 10.9864 to 10.9232. Helium is inert,
which is why it showed up as a clean 11.6 percent offset everywhere. The rocky
suppression is the bigger effect: keeping Mg, Si and Fe at solar locks up some
oxygen, so your water and CO2 come out lower than ours by 25 and 42 percent in
the deep atmosphere.

I think the rocky suppression is right for these truncated networks, and it is
what I described to you back in July when I was comparing against photochem. But
it does mean that any "VULCAN-JAX versus VULCAN 2.0" number is only meaningful if
that file is matched first, and none of my earlier comparisons matched it. I have
written this up properly now.

While checking this I also found that our NCHO network has one reaction yours
does not, NH3 + CH -> NH2 + CH2. It is in your SNCHO and NCHO_full networks but
not in NCHO_photo_network.txt on any branch. It has been in my repo since the
first commit and it only changes results at the 7e-6 level, so it is not urgent,
but I would like to know whether the omission upstream is deliberate.

**What is still open**

Earth runs but does not converge, and its atom conservation is far off at the
step cap, so I would not trust it as a case yet. K2-18b runs end to end but does
not converge either, and I have not yet traced the original configuration you
sent, so I am not calling it validated.

**Questions**

1. Can you confirm the split I chose: empty conver_ignore for VULCAN 2 parity
   configs, and ['HC3N'] for the VULCAN 3 preset, matching vm_branch?

2. Adaptive rtol. Your email suggested inc_period 500, dec 0.5, inc 1.5. Current
   vm_branch op.py has dec 0.5 but inc_period 1000 and inc 1.25. Which do you
   want? I have used the branch values in the VULCAN 3 preset.

3. Do you want to keep the stall detector at all? It is easy to delete. I kept it
   as a VULCAN 3 option because it helps on trace radicals, but it is not yours
   and it is not in VULCAN 2.

4. Is NH3 + CH -> NH2 + CH2 missing from NCHO_photo_network.txt on purpose? It
   is present in your SNCHO and NCHO_full networks. Related: a fresh clone cannot
   run cfg_examples/vulcan_cfg_HD189.py as shipped, because it does not define
   use_adapt_rtol, rtol_min, rtol_max or use_fix_all_bot but op.py reads them.
   Happy to send a small pull request for both.

Best,
Isaac
