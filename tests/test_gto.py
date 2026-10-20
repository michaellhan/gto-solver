import unittest
import numpy as np
from gto.cards import cards, ALL_COMBOS, hand_class
from gto.ranges import parse_range
from gto.evaluator import eval_board
from gto import preflop as pf
from gto.solver import solve, Config, Ctx, Solver, TreeBuilder, MEMBER_FULL, N_FULL, match_action
from gto.advisor import Advisor, preflop_investments

MW_TRIALS = 1200   # multiway tests: trials per call, no time caps (see Advisor.mw_trials)


class TestCardsRanges(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_range("AA").sum(), 6)
        self.assertEqual(parse_range("AKs").sum(), 4)
        self.assertEqual(parse_range("AKo").sum(), 12)
        self.assertEqual(parse_range("AK").sum(), 16)
        self.assertEqual(parse_range("22-44").sum(), 18)
        self.assertEqual(parse_range("A5s:0.5").sum(), 2)
        self.assertEqual(parse_range("T9s-76s").sum(), 16)
        self.assertEqual(hand_class(tuple(cards("AhKh"))), "AKs")

    def test_parse_case_spaces_and_bad_classes(self):
        # any case: "AKS" was read as offsuit (12 combos), "aks" was an error
        for t in ("AKS", "aks", "aKs"):
            self.assertEqual(parse_range(t).sum(), 4, t)
        self.assertEqual(parse_range("t9s-76S").sum(), 16)
        self.assertEqual(parse_range("ahkh").sum(), 1)
        # spaces between hands: "AA KK" was AA alone
        self.assertEqual(parse_range("AA KK\nQQ  JJ").sum(), 24)
        self.assertEqual(parse_range("AA : 0.5, KK").sum(), 9)
        # a suffix other than s/o, or on a pair, is an error, not offsuit / the pair
        for t in ("AKx", "AAs", "AKso"):
            with self.assertRaises(ValueError, msg=t):
                parse_range(t)

    def test_eval(self):
        st = eval_board(cards("AhKd7c2s9h"))
        i_aa = ALL_COMBOS.index(tuple(sorted(cards("AsAc"))))
        i_kk = ALL_COMBOS.index(tuple(sorted(cards("KsKc"))))
        self.assertGreater(st[i_aa], st[i_kk])
        self.assertEqual(st[ALL_COMBOS.index(tuple(sorted(cards("AhAc"))))], -1)


class TestSolverMath(unittest.TestCase):
    def test_flop_allin_equity_near_exact(self):
        """A flop all-in's equity matrix covers every turn card (4 sampled turns x 4 rivers, the browser's Fast
        tree, had flush draws off by up to 24% equity: JsTs called a flop jam at +19.5 bb, exactly -34)."""
        import itertools
        board = tuple(cards("Ks7s2d"))
        r = parse_range("JsTs, AsQs, Qs9s, AhKd, AcKc, 77, 22, K7s, 8s8d, 9h9d, QhJh, 6s5s, AhQd")
        bm = np.ones(N_FULL)
        for c in board:
            bm *= 1 - MEMBER_FULL[c]
        active = np.nonzero(r * bm > 0)[0]
        for cfg in (Config(leaf_turns=4, leaf_rivers_flop=4), Config()):
            ctx = Ctx(active, cfg)
            L = ctx.leaf_matrix(board, set())
            E = np.zeros_like(L, dtype=np.float64); C = np.zeros_like(E)
            for t, rv in itertools.combinations([c for c in range(52) if c not in board], 2):
                s = ctx.strength(board + (t, rv)).astype(float)
                ok = (s >= 0).astype(float)
                E += np.sign(s[:, None] - s[None, :]) * np.outer(ok, ok); C += np.outer(ok, ok)
            E = E / np.maximum(C, 1) * ctx.compat
            eq = lambda M: 0.5 + 0.5 * M.sum(1) / np.maximum(ctx.compat.sum(1), 1)   # each vs the rest, evenly
            self.assertLess(np.abs(eq(L) - eq(E)).max(), 0.04)

    def test_turn_leaf_in_flop_solve_uses_every_river(self):
        """A turn leaf inside a flop solve (an all-in or check-down on the turn) averages every river, not
        leaf_rivers_flop random ones (4 in the browser's Fast tree moved a flush draw's equity by tens of points)."""
        board = tuple(cards("Ks7s2d4h"))
        r = parse_range("JsTs, AsQs, AhKd, 77, 8s8d, QhJh, 6s5s")
        bm = np.ones(N_FULL)
        for c in board:
            bm *= 1 - MEMBER_FULL[c]
        active = np.nonzero(r * bm > 0)[0]
        ctx = Ctx(active, Config(leaf_rivers_flop=4))
        ctx.root_len = 3
        L = ctx.leaf_matrix(board, set())
        E = np.zeros_like(L, dtype=np.float64); C = np.zeros_like(E)
        for rv in [c for c in range(52) if c not in board]:
            s = ctx.strength(board + (rv,)).astype(float)
            ok = (s >= 0).astype(float)
            E += np.sign(s[:, None] - s[None, :]) * np.outer(ok, ok); C += np.outer(ok, ok)
        self.assertLess(np.abs(L - E / np.maximum(C, 1) * ctx.compat).max(), 1e-5)

    def test_showdown_matrix_matches_bruteforce(self):
        board = cards("AhKd7c2s9h")
        r = parse_range("22+, A2s+, KTs+, ATo+")
        bm = np.ones(N_FULL)
        for c in board:
            bm *= 1 - MEMBER_FULL[c]
        active = np.nonzero(r * bm > 0)[0]
        ctx = Ctx(active, Config())
        M = ctx.showdown_matrix(tuple(board))
        st = eval_board(board)
        i, j = 5, 40
        gi, gj = active[i], active[j]
        a, b = ALL_COMBOS[gi]; c, d = ALL_COMBOS[gj]
        expected = 0 if {a, b} & {c, d} else np.sign(st[gi] - st[gj])
        self.assertEqual(M[i, j], expected)
        self.assertTrue(np.allclose(M, -M.T))

    def test_river_solve_converges_and_is_sane(self):
        board = cards("Ks7d2cTh4s")
        r_oop = parse_range("22+, A2s+, KTs+, QTs+, JTs, T9s, 98s, ATo+, KQo")
        r_ip = parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, A8o+, KTo+, QTo+, JTo")
        # a fixed iteration count, not a time budget: on a loaded machine (the pre-deploy checks run beside
        # other work) a budget gets fewer iterations; speed has its own benchmark (scripts/bench_solver.py)
        sol = solve(board, r_oop, r_ip, pot=6.5, stack=97.5, time_budget=1e9, max_iters=150)
        self.assertEqual(sol.iters, 150)
        # nuts-ish (KK top set) should never be folding / mostly betting; EVs all positive
        kk = sol.for_combo(tuple(cards("KhKc")))
        self.assertGreater(kk["bet 100%"][0] + kk["bet 50%"][0], 0.5)
        self.assertTrue(all(ev > 0 for _, ev in kk.values()))
        # frequencies sum to one
        self.assertTrue(np.allclose(sol.strategy[:, sol.ctx.active].sum(axis=0), 1.0))

    def test_turn_picks_subset_is_random(self):
        """Turn picks are cached per flop; a call for fewer than are cached gets a random subset of them
        (the first drawn), not the lowest cards."""
        flop = tuple(cards("Ks7d2c"))
        active = np.array([i for i, (a, b) in enumerate(ALL_COMBOS) if a not in flop and b not in flop])
        ctx = Ctx(active, Config(seed=3))
        dead = set(flop)
        many = ctx.turn_picks(flop, dead, 20)
        few = ctx.turn_picks(flop, dead, 8)
        self.assertEqual(len(many), 20)
        self.assertEqual(few, sorted(few))
        self.assertTrue(set(few) <= set(many))
        self.assertNotEqual(few, many[:8])
        self.assertEqual(ctx.turn_picks(flop, dead, 8), few)   # the same picks every time
        self.assertEqual(ctx.turn_picks(flop, dead, 20), many)
        # every card cached first, then fewer asked for: a random subset too
        ctx = Ctx(active, Config(seed=3))
        all_ = ctx.turn_picks(flop, dead, 52)
        self.assertEqual(len(all_), 49)
        few = ctx.turn_picks(flop, dead, 8)
        self.assertNotEqual(few, all_[:8])
        self.assertEqual(ctx.turn_picks(flop, dead, 8), few)
        self.assertEqual(ctx.turn_picks(flop, dead, 52), all_)

    def test_warm_start_copies_matching_nodes(self):
        # a turn solve over 8 rivers seeds one over every river: same rivers copied, new ones start at zero
        board = cards("Ks7d2cTh")
        r_oop = parse_range("22+, A2s+, KTs+, QTs+, JTs, T9s, ATo+, KQo")
        r_ip = parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, A8o+, KTo+, QTo+")
        src = solve(board, r_oop, r_ip, pot=6.5, stack=97.5, cfg=Config(n_rivers=8), time_budget=0.3)
        dst = solve(board, r_oop, r_ip, pot=6.5, stack=97.5, cfg=Config(), max_iters=0, warm=src)
        self.assertEqual(dst.iters, 0)
        self.assertTrue(np.array_equal(dst.root.regret, src.root.regret))
        self.assertTrue(np.allclose(dst.strategy, src.strategy))
        s_river = src.root.children[0].children[0].child   # check, check, then the river
        d_river = dst.root.children[0].children[0].child
        self.assertEqual((len(s_river.boards), len(d_river.boards)), (8, 48))
        for i, b in enumerate(d_river.boards):
            j = s_river.boards.index(b) if b in s_river.boards else None
            if j is None:
                self.assertFalse(d_river.ssum[i].any())
            else:
                self.assertTrue(np.array_equal(d_river.ssum[i], s_river.ssum[j]))

    def _plain_iterations(self, sol, n):
        """n iterations walking the original tree node by node (no merged subtrees, no strategy reuse)."""
        s = sol.solver
        a, b = sol.r0[None, :].astype(np.float32), sol.r1[None, :].astype(np.float32)
        for _ in range(n):
            for p in (0, 1):
                s._cfr(s.root, p, a, b, update=True, use_avg=False)
            s.iters += 1
            s._discount()

    def test_merged_subtrees_match_plain_traversal(self):
        # same-shaped subtrees below the root street are batched together; the result must be the same
        # numbers as walking every node on its own (flop: repeated turn boards; turn: different rivers)
        r_oop = parse_range("22+, A2s+, KTs+, QTs+, JTs, T9s, ATo+, KQo")
        r_ip = parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, A8o+, KTo+, QTo+")
        small = {"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]}
        for board, cfg in ((cards("Ks7d2c"), Config(n_turns=2, leaf_turns=2, leaf_rivers_flop=2, n_rivers_from_flop=2,
                                                     bet_sizes=small, raise_sizes=small)),
                           (cards("Ks7d2cTh"), Config(n_rivers=4, bet_sizes=small, raise_sizes=small))):
            fast = solve(board, r_oop, r_ip, pot=5.5, stack=97.5, cfg=cfg, max_iters=0)
            plain = solve(board, r_oop, r_ip, pot=5.5, stack=97.5, cfg=cfg, max_iters=0)
            self.assertTrue(any(len(m) > 1 for _, m in fast.solver._groups))
            fast.solver.batch_leaves = False      # matrix-matrix products round differently (see the next test)
            fast.solver.run(fast.r0, fast.r1, time_budget=1e9, max_iters=12)
            self._plain_iterations(plain, 12)
            for x, y in zip(fast.solver._nodes, plain.solver._nodes):
                self.assertTrue(np.array_equal(x.ssum, y.ssum))
                self.assertTrue(np.array_equal(x.regret, y.regret))

    def test_batched_leaves_match_one_by_one(self):
        # each pass's equity leaves computed together up front (one matrix-matrix product per matrix stack):
        # the same updates as computing them leaf by leaf, up to float32 rounding. Regret matching amplifies
        # rounding (an action tied to the last bit gets all of a combo's weight or none), so the regrets are
        # compared after one iteration and the solves after many by exploitability
        from gto.solver import Terminal
        r_oop = parse_range("22+, A2s+, KTs+, QTs+, JTs, T9s, ATo+, KQo")
        r_ip = parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, A8o+, KTo+, QTo+")
        small = {"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]}
        # (the flop with random turns and the river checked down, and with the default turn classes and river play,
        # whose deeper trees sum more leaves: more rounding)
        for board, cfg, tol in ((cards("Ks7d2c"), Config(n_turns=2, leaf_turns=2, leaf_rivers_flop=2, n_rivers_from_flop=2,
                                                          bet_sizes=small, raise_sizes=small, river_play=False,
                                                          turn_classes=False), 1e-4),
                                (cards("Ks7d2c"), Config(bet_sizes=small, raise_sizes=small), 3e-4),
                                (cards("Ks7d2cTh"), Config(n_rivers=4, bet_sizes=small, raise_sizes=small, allin_spr=20), 1e-4)):
            a = solve(board, r_oop, r_ip, pot=5.5, stack=97.5, cfg=cfg, max_iters=0)
            b = solve(board, r_oop, r_ip, pot=5.5, stack=97.5, cfg=cfg, max_iters=0)
            a.solver.batch_leaves, b.solver.batch_leaves = True, False
            for s in (a, b):
                s.solver.run(s.r0, s.r1, time_budget=1e9, max_iters=1)
            self.assertTrue(any(isinstance(t, Terminal) and t.kind == "leaf" for n in a.solver._nodes for t in n.children))
            for x, y in zip(a.solver._nodes, b.solver._nodes):
                for u, v in ((x.ssum, y.ssum), (x.regret, y.regret)):
                    self.assertLess(np.abs(u - v).max(), tol * max(np.abs(v).max(), 1.0))
            for s in (a, b):
                s.extend(1e9, max_iters=150)
            self.assertLess(abs(a.exploitability() - b.exploitability()), 0.1 * b.exploitability())

    def test_compat_sum_kernels(self):
        # the opponent mass sharing no card with each combo: per-card sums (NO_BLAS) = the member product = brute force
        import gto.solver as S
        board = cards("Ks7d2c")
        r = parse_range("22+, A2s+, K9s+, QTs+, JTs, T9s, ATo+, KQo") * np.prod([1 - MEMBER_FULL[c] for c in board], axis=0)
        sol = solve(board, r, r, pot=5.5, stack=97.5, max_iters=0)
        ro = np.random.default_rng(1).random((3, sol.ctx.n)).astype(np.float32)
        brute = ro @ sol.ctx.compat.T.astype(np.float32)
        dense = sol.solver.compat_sum(ro)
        S.NO_BLAS = True
        try:
            sums = sol.solver.compat_sum(ro)
        finally:
            S.NO_BLAS = False
        self.assertLess(np.abs(dense - brute).max(), 1e-3)
        self.assertLess(np.abs(sums - brute).max(), 1e-3)

    def test_no_blas_kernels(self):
        # the browser's path (einsum matvecs, leaf matrices trimmed to each range) solves the same game
        import gto.solver as S
        board = cards("Ks7d2c")
        r_oop = parse_range("22+, A2s+, KTs+, QTs+, JTs, T9s, ATo+, KQo")
        r_ip = parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, A8o+, KTo+, QTo+")
        small = {"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]}
        # (the browser's Accurate flop tree, the default, and its Fast one: random turns, the river checked down)
        for kw in ({}, dict(river_play=False, turn_classes=False)):
            cfg = Config(n_turns=2, leaf_turns=2, leaf_rivers_flop=2, n_rivers_from_flop=2, bet_sizes=small,
                         raise_sizes=small, **kw)
            # float32 sums in a different order drift apart early on (60 iterations: up to 0.08% of the pot), less
            # so later; the default's deeper tree drifts more (150 iterations: 0.71% and 0.77% of the pot)
            ref = solve(board, r_oop, r_ip, pot=5.5, stack=97.5, cfg=cfg, max_iters=150, time_budget=1e9)
            S.NO_BLAS = True
            try:
                sol = solve(board, r_oop, r_ip, pot=5.5, stack=97.5, cfg=cfg, max_iters=150, time_budget=1e9)
                self.assertIsNotNone(sol.solver._live)
                self.assertLess(abs(sol.exploitability() - ref.exploitability()), max(0.05, 0.12 * ref.exploitability()))
                live = sol.r0 > 0
                self.assertLess(np.abs(sol.evs[:, sol.ctx.active][:, live] - ref.evs[:, ref.ctx.active][:, live]).max(), 0.2)
            finally:
                S.NO_BLAS = False

    def test_match_action(self):
        names = ["check", "bet 33%", "bet 75%"]
        self.assertEqual(match_action(names, "bet 40%"), 1)
        self.assertEqual(match_action(names, "bet 90%"), 2)
        self.assertEqual(match_action(["fold", "call", "raise to 12", "all-in"], "jam"), 3)


class TestPreflop(unittest.TestCase):
    def test_rfi_and_defence(self):
        s = pf.hand_strategy("UTG", tuple(cards("7h2c")), [])
        self.assertEqual(s["raise"], 0.0)
        s = pf.hand_strategy("BTN", tuple(cards("AhAc")), [{"pos": "UTG", "action": "raise"}])
        self.assertEqual(s["raise"], 1.0)
        acts = [{"pos": p, "action": "fold"} for p in ("UTG", "HJ", "CO")] + [{"pos": "BTN", "action": "raise"}]
        self.assertEqual(pf.next_to_act(acts), "SB")
        acts += [{"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        self.assertTrue(pf.is_preflop_closed(acts))
        rs = pf.replay_ranges(acts)
        self.assertEqual(set(rs), {"BTN", "BB"})
        inv, active = preflop_investments(acts, {p: 100.0 for p in pf.POSITIONS})
        self.assertAlmostEqual(sum(inv.values()), 5.5)


    def test_iso_raise_ranges(self):
        """Iso-raising over one limper is a little tighter than the seat's open (published: BTN ~30-45% of
        hands, CO ~22-35%, HJ ~15-22%, SB ~18-25%, BB ~15%; Preflop Wizard, Grindlab), tighter again over
        two or more limpers, and never overlaps the over-limps; the old charts raised under half the
        button's opens."""
        F = lambda *ps: [{"pos": p, "action": "fold"} for p in ps]
        C = lambda *ps: [{"pos": p, "action": "call"} for p in ps]
        w = lambda v: float(np.sum(v)) / 1326
        bands = {"HJ": (0.15, 0.22), "CO": (0.22, 0.32), "BTN": (0.30, 0.42), "SB": (0.17, 0.25)}
        seats = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
        for hero, (lo, hi) in bands.items():
            before = seats[:seats.index(hero)]
            one = pf.strategy_for(hero, C("UTG") + F(*before[1:]))
            two = pf.strategy_for(hero, C("UTG", "HJ") + F(*before[2:])) if hero != "HJ" else None
            opened = w(pf.strategy_for(hero, F(*before))["raise"])
            iso = w(one["raise"])
            self.assertTrue(lo <= iso <= hi, (hero, iso))
            self.assertLess(iso, opened, hero)
            if hero != "SB":   # in position: at least 70% of the open
                self.assertGreater(iso, 0.7 * opened, hero)
            self.assertTrue(np.all(one["raise"] + one["call"] <= 1 + 1e-9), hero)
            if two is not None:
                self.assertLess(w(two["raise"]), 0.75 * iso, hero)
                self.assertTrue(np.all(two["raise"] + two["call"] <= 1 + 1e-9), hero)
            # linear: every hand that iso-raises over two limpers raises over one
            if two is not None:
                self.assertTrue(np.all(one["raise"] >= two["raise"] - 1e-9), hero)
        # the big blind raises one limper ~15%, several limpers less, the small blind's limp most
        bb = lambda acts: w(pf.strategy_for("BB", acts)["raise"])
        one_bb, two_bb, sb_bb = bb(F("UTG", "HJ") + C("CO") + F("BTN", "SB")), bb(F("UTG", "HJ") + C("CO", "BTN") + F("SB")), bb(F("UTG", "HJ", "CO", "BTN") + C("SB"))
        self.assertTrue(0.12 <= one_bb <= 0.18, one_bb)
        self.assertLess(two_bb, one_bb)
        self.assertGreater(sb_bb, one_bb)

    @staticmethod
    def _all_charts():
        """Every preflop chart: name -> {action: per-combo frequency}."""
        out = {f"VS_RFI{k}": v for k, v in pf.VS_RFI.items()}
        out.update({f"SHORT_CHARTS{k}": v for k, v in pf.SHORT_CHARTS.items()})
        out.update({f"VS_3BET_OOP[{k}]": v for k, v in pf.VS_3BET_OOP.items()})
        for n in dir(pf):
            v = getattr(pf, n)
            if (n.isupper() and isinstance(v, dict) and v and set(v) <= {"raise", "3bet", "4bet", "jam", "call", "check"}
                    and all(isinstance(x, np.ndarray) and x.shape == (pf.N_COMBOS,) for x in v.values())):
                out[n] = v
        for k in pf.ISO:
            out[f"ISO[{k}]"] = {"raise": pf.ISO[k], "call": pf.OVERLIMP[k]}
            out[f"ISO_MULTI[{k}]"] = {"raise": pf.ISO_MULTI[k], "call": pf.OVERLIMP_MULTI[k]}
        for o in ("UTG", "HJ", "CO", "BTN", "SB", "EP", "HU_SB"):
            for b in ("SB", "BB", "BTN", "CO", "HJ"):
                out[f"vs_3bet({o},{b})"] = pf.vs_3bet(o, b)
        for o in ("UTG", "HJ", "CO", "BTN", "SB"):
            out[f"_vs_3bet({o})"] = pf._vs_3bet(o)
            out[f"_ip_vs_blind_3bet({o})"] = pf._ip_vs_blind_3bet(o)
        return out

    def test_chart_frequencies_sum_to_at_most_one(self):
        """A hand's actions in a chart sum to at most 1: strategy_for rescales a sum above 1, so a hand
        listed in both raises and calls at 1.0 would raise less often than written."""
        for name, chart in self._all_charts().items():
            vals = [v for k, v in chart.items() if k not in ("check", "fold")]
            for v in vals:
                self.assertTrue(np.all(v >= -1e-12) and np.all(v <= 1 + 1e-9), name)
            tot = sum(vals)
            bad = sorted({hand_class(ALL_COMBOS[i]) for i in np.nonzero(tot > 1 + 1e-9)[0]})
            self.assertEqual(bad, [], name)

    def test_chart_strings_list_each_hand_once(self):
        """No range string in the charts lists a hand twice with different weights (parse_range keeps
        the larger, so the other entry is dead and misleading)."""
        import ast, inspect
        from gto.ranges import _expand_token
        tree = ast.parse(inspect.getsource(pf))
        n_strings = 0
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("R", "parse_range")
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                continue
            n_strings += 1
            seen = {}
            for part in node.args[0].value.replace("\n", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                tok, w = (part.split(":")[0], float(part.split(":")[1])) if ":" in part else (part, 1.0)
                for cb in _expand_token(tok):
                    seen.setdefault(hand_class(cb), set()).add(w)
            dups = {c: sorted(ws) for c, ws in seen.items() if len(ws) > 1}
            self.assertEqual(dups, {}, f"line {node.lineno}")
        self.assertGreater(n_strings, 50)

    def test_heads_up_mixed_raises_played_as_written(self):
        """Heads-up the big blind 3bets QTs and JTs always and 66 / A5s half the time (as the chart
        writes them), and the button 4bets A5s 40% against the 3bet."""
        T = pf.Table(pf.SEATS[2])
        s = pf.strategy_for("BB", [{"pos": "SB", "action": "raise", "to": 2.5}], T)
        for h, r in (("QTs", 1.0), ("JTs", 1.0), ("66", 0.5), ("A5s", 0.5), ("A9o", 0.5)):
            m = parse_range(h) > 0
            self.assertAlmostEqual(float(s["raise"][m].mean()), r, places=6, msg=h)
        s = pf.strategy_for("SB", [{"pos": "SB", "action": "raise", "to": 2.5},
                                   {"pos": "BB", "action": "raise", "to": 9}], T)
        self.assertAlmostEqual(float(s["raise"][parse_range("A5s") > 0].mean()), 0.4, places=6)

    def test_bets_all_in_sizes(self):
        """An all-in goes to its `to` when given, else to the effective stack (the table depth)."""
        acts = [{"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "BB", "action": "all-in"}]
        self.assertEqual(pf._bets(acts)["BB"], 100.0)
        self.assertEqual(pf._bets(acts, 40.0)["BB"], 40.0)
        acts[1]["to"] = 25.0
        self.assertEqual(pf._bets(acts, 40.0)["BB"], 25.0)
        acts[1]["to"] = 60.0
        self.assertEqual(pf._bets(acts, 40.0)["BB"], 40.0)

    def test_5bet_jam_priced_at_table_depth(self):
        """Facing a 5bet jam at 40bb deep the 4bettor calls QQ, JJ and AKo (it needs ~27% equity, not
        the 41% a 100bb jam would ask); at 100bb it folds them."""
        R_ = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        pre = [R_("UTG", "raise", 2.5), R_("HJ", "fold"), R_("CO", "fold"), R_("BTN", "raise", 7.5),
               R_("SB", "fold"), R_("BB", "fold"), R_("UTG", "raise", 18), R_("BTN", "all-in")]
        for depth, lo, hi in ((40, 0.9, 1.0), (100, 0.0, 0.1)):
            s = pf.strategy_for("UTG", pre, pf.Table(pf.SEATS[6], depth=depth))
            for h in ("QQ", "JJ", "AKo"):
                c = float(s["call"][parse_range(h) > 0].mean())
                self.assertTrue(lo <= c <= hi, (depth, h, c))
        # an all-in with its size given is priced at that size whatever the table depth
        pre[-1]["to"] = 40.0
        s = pf.strategy_for("UTG", pre, pf.Table(pf.SEATS[6]))
        self.assertGreater(float(s["call"][parse_range("QQ") > 0].mean()), 0.9)


class TestAdvisor(unittest.TestCase):
    def test_end_to_end(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"hero": "BB", "hole": "AhKh", "board": "", "preflop": pre[:-1]})
        self.assertEqual(r["street"], "preflop")
        r = adv.advise({"hero": "CO", "hole": "AhKh", "board": "Kd7s2c", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "check"}]})
        self.assertEqual(r["position"], "IP")
        self.assertGreaterEqual(r["solver"]["exploitability_pct"], 0.0)
        self.assertAlmostEqual(sum(a["freq"] for a in r["actions"]), 1.0, places=2)
        r = adv.advise({"hero": "BB", "hole": "9h8h", "board": "Kd7s2cTh", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 2},
                                 {"pos": "BB", "action": "call"}], "turn": []})
        self.assertEqual(r["street"], "turn")
        self.assertAlmostEqual(r["pot_bb"], 9.5)

    def test_preflop_vs_open_jam(self):
        adv = Advisor()
        pre = [{"pos": p, "action": "fold"} for p in ("UTG", "HJ", "CO", "BTN")] + [{"pos": "SB", "action": "all-in"}]
        st = {"players": 6, "hero": "BB", "board": "", "stack": 100, "preflop": pre}
        r = adv.advise(dict(st, hole="Kd5d"))
        self.assertEqual(r["recommended"], "fold")
        self.assertEqual({a["action"] for a in r["actions"]}, {"call", "fold"})
        self.assertEqual(adv.advise(dict(st, hole="QsQh"))["recommended"], "call")
        # short stacks: the SB shoves wide (push/fold equilibrium), so K5s calls 10bb but folds 20bb
        self.assertEqual(adv.advise(dict(st, hole="Kd5d", stack=10))["recommended"], "call")
        self.assertEqual(adv.advise(dict(st, hole="Kd5d", stack=20))["recommended"], "fold")

    def test_preflop_price_and_callers(self):
        adv = Advisor()
        st = {"players": 6, "hero": "UTG", "hole": "6s6d", "board": "", "stack": 100}
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        # a standard 3bet heads-up is the chart's spot: 66 folds, no price note
        std = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 7.5)] + [R(p, "fold") for p in ("BTN", "SB", "BB")]
        r = adv.advise(dict(st, preflop=std))
        self.assertEqual(r["recommended"], "fold")
        self.assertEqual(r["notes"], ["preflop: chart approximation"])
        # a min-3bet with two callers: 2.5bb into 16bb, 66 calls (from a fold at the standard price)
        multi = [R("UTG", "raise", 2.5), R("HJ", "call"), R("CO", "raise", 5), R("BTN", "fold"), R("SB", "call"), R("BB", "fold")]
        r = adv.advise(dict(st, preflop=multi))
        # (66 called 100% here when HJ's flat was 8% of hands; the published HJ vs UTG chart flats 2.5%,
        # mostly pairs, and against them 66 mixes)
        self.assertGreater(r["actions"][1]["freq"], 0.3)
        self.assertTrue(any("price" in n for n in r["notes"]))
        # a huge 3bet: a chart call (99 calls 90% vs a standard 3bet) folds at the worse price
        big = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 20)] + [R(p, "fold") for p in ("BTN", "SB", "BB")]
        self.assertGreater(adv.advise(dict(st, hole="9s9d", preflop=std))["actions"][1]["freq"], 0.5)
        self.assertEqual(adv.advise(dict(st, hole="9s9d", preflop=big))["recommended"], "fold")

    def test_preflop_squeeze(self):
        adv = Advisor()
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        st = {"players": 6, "hero": "BB", "board": "", "stack": 100}
        alone = [R("UTG", "raise", 2.5)] + [R(p, "fold") for p in ("HJ", "CO", "BTN", "SB")]
        two = [R("UTG", "raise", 2.5), R("HJ", "call"), R("CO", "call"), R("BTN", "fold"), R("SB", "fold")]
        size = lambda r: next(a["amount_bb"] for a in r["actions"] if a["action"] == "raise")
        freq = lambda r, k: next(a["freq"] for a in r["actions"] if a["action"] == k)
        # one more open-size per caller: 10bb alone, 15bb over two callers
        a, b = adv.advise(dict(st, hole="AhKd", preflop=alone)), adv.advise(dict(st, hole="AhKd", preflop=two))
        self.assertEqual((size(a), size(b)), (10.0, 15.0))
        self.assertEqual((freq(a, "raise"), freq(b, "raise")), (1.0, 1.0))
        # a bluff squeeze shrinks: everyone has to fold
        a, b = adv.advise(dict(st, hole="Ah5h", preflop=alone)), adv.advise(dict(st, hole="Ah5h", preflop=two))
        self.assertLess(freq(b, "raise"), freq(a, "raise"))
        self.assertTrue(any("raises re-drawn" in n for n in b["notes"]))

    def test_preflop_off_chart_spots(self):
        adv = Advisor()
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        def go(hero, hole, pre, **kw):
            r = adv.advise(dict({"players": 6, "hero": hero, "hole": hole, "board": "", "stack": 100, "preflop": pre}, **kw))
            return r, {a["action"]: a for a in r.get("actions", [])}
        # limps: the big blind raises AA (checks 72o), the button isolates to 3x + 1 per limper, and
        # over-limps 65s behind two limpers (it iso-raises it over one)
        limps = [R("UTG", "call"), R("HJ", "call")] + F("CO", "BTN") + [R("SB", "call")]
        self.assertEqual(go("BB", "AhAd", limps)[0]["recommended"], "raise")
        self.assertEqual(go("BB", "7h2d", limps)[0]["recommended"], "check")
        r, a = go("BTN", "AhAd", [R("UTG", "call")] + F("HJ", "CO"))
        self.assertEqual((r["recommended"], a["raise"]["amount_bb"]), ("raise", 4.0))
        self.assertEqual(go("BTN", "6h5h", [R("UTG", "call"), R("HJ", "call"), R("CO", "fold")])[0]["recommended"], "call")
        self.assertEqual(go("BTN", "6h5h", [R("UTG", "call")] + F("HJ", "CO"))[0]["recommended"], "raise")
        self.assertEqual(go("BB", "KhKd", [R("SB", "call")], players=2)[0]["recommended"], "raise")
        # short stacks: a 3bet at 25bb is all-in; QQ never folds to a 3bet at 40bb
        r, a = go("CO", "AhKd", [R("UTG", "raise", 2.2), R("HJ", "fold")], stack=25)
        self.assertEqual((r["recommended"], a["all-in"]["amount_bb"]), ("all-in", 25.0))
        r, a = go("UTG", "QsQd", [R("UTG", "raise", 2.2), R("HJ", "fold"), R("CO", "raise", 6.5)] + F("BTN", "SB", "BB"), stack=40)
        self.assertLess(a["fold"]["freq"], 0.05)
        # deep stacks: 55 calls a 3bet at 200bb (implied odds), folds at 100bb
        pre = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 7.5)] + F("BTN", "SB", "BB")
        self.assertEqual(go("UTG", "5s5d", pre)[0]["recommended"], "fold")
        self.assertEqual(go("UTG", "5s5d", pre, stack=200)[0]["recommended"], "call")
        # the opener calls wider in position (vs the big blind's 3bet) than out of position (vs the button's)
        ip = go("CO", "6s6d", F("UTG", "HJ") + [R("CO", "raise", 2.5)] + F("BTN", "SB") + [R("BB", "raise", 10)])[1]
        oop = go("CO", "6s6d", F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "raise", 7.5)] + F("SB", "BB"))[1]
        self.assertGreater(ip["call"]["freq"], oop["call"]["freq"])
        # heads-up the button defends a 3bet wide (in position postflop)
        r = go("SB", "Qh8h", [R("SB", "raise", 2.5), R("BB", "raise", 8)], players=2)[0]
        self.assertEqual(r["recommended"], "call")
        # 100bb: the button 4bets the big blind's 11bb 3bet to 25, not all-in (A5s jammed 100bb under JAM_SPR),
        # and the big blind calls that 4bet with JJ (every call had become a jam); at 40bb the 3bet still jams
        bb3 = F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)] + F("SB") + [R("BB", "raise", 11)]
        r, a = go("BTN", "As5s", bb3)
        self.assertEqual(a["raise"]["amount_bb"], 25.5)
        self.assertNotIn("all-in", a)
        r, a = go("BB", "JsJd", bb3 + [R("BTN", "raise", 25)])
        self.assertEqual(r["recommended"], "call")
        r, a = go("BB", "AhKd", F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)] + F("SB"), stack=40)
        self.assertIn("all-in", a)
        # a short-stack rejam over an open competes with flatting it, not just folding: suited connectors and
        # small pairs flat (push/fold against folding alone jammed 76s at 25bb and 55 at 40bb, 30% and 23% of
        # hands); AQo and KK still jam
        bo = F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.3)] + F("SB")
        self.assertEqual(go("BB", "7h6h", bo, stack=25)[0]["recommended"], "call")
        self.assertEqual(go("BB", "AhQd", bo, stack=25)[0]["recommended"], "all-in")
        bo = F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)] + F("SB")
        self.assertEqual(go("BB", "5h5d", bo, stack=40)[0]["recommended"], "call")
        self.assertEqual(go("BB", "KhKd", bo, stack=40)[0]["recommended"], "all-in")
        # ...and the big blind's iso-jam over limpers competes with its free check, against the limpers' ranges
        # (not any two cards): at 12bb K5o and J5s checked behind two limpers only after this (55% jammed)
        lim = [R("UTG", "call")] + F("HJ", "CO") + [R("BTN", "call")] + F("SB")
        self.assertEqual(go("BB", "Kh5d", lim, stack=12)[0]["recommended"], "check")
        self.assertEqual(go("BB", "Jh5h", lim, stack=12)[0]["recommended"], "check")
        self.assertEqual(go("BB", "AhTd", lim, stack=12)[0]["recommended"], "all-in")
        # closing a five-way pot at 10:1 the big blind calls suited connectors (multiway equity per combo from
        # 60-75 boards had 76s at 5% where a dealt-out simulation has 14%, and they folded 70-77%); 72o folds
        mw5 = [R("UTG", "raise", 2.5), R("HJ", "call"), R("CO", "call"), R("BTN", "call"), R("SB", "call")]
        for h in ("7h6h", "6h5h", "5h4h"):
            self.assertEqual(go("BB", h, mw5)[0]["recommended"], "call", h)
        self.assertEqual(go("BB", "7h2d", mw5)[0]["recommended"], "fold")
        # facing a 5bet with chips behind, AA can shove
        r, a = go("UTG", "AhAd", [R("UTG", "raise", 2.5), R("HJ", "raise", 8)] + F("CO", "BTN", "SB", "BB")
                  + [R("UTG", "raise", 20), R("HJ", "raise", 45)], stack=200)
        self.assertEqual(r["recommended"], "all-in")
        # antes widen opens; a straddle makes HJ first to act and the opens 2.5x the straddle
        self.assertEqual(go("UTG", "Ah9d", [])[0]["recommended"], "fold")
        self.assertGreater(go("UTG", "Ah9d", [], ante=1 / 6)[1]["raise"]["freq"], 0)
        r, a = go("HJ", "AhJd", [], straddle=True)
        self.assertEqual((r["recommended"], a["raise"]["amount_bb"], r["pot_bb"]), ("raise", 5.0, 3.5))
        self.assertIn("error", go("UTG", "AhJd", [], straddle=True)[0])
        # heads-up the big blind continues wider against a 4bet than 6-max
        hu4 = [R("SB", "raise", 2.5), R("BB", "raise", 8), R("SB", "raise", 20)]
        self.assertEqual(go("BB", "KhQh", hu4, players=2)[0]["recommended"], "call")
        # a limp-reraise is a trapped range: TT folds most of the time, KK goes in
        lrr = [R("UTG", "call")] + F("HJ", "CO") + [R("BTN", "raise", 4)] + F("SB", "BB") + [R("UTG", "raise", 14)]
        self.assertEqual(go("BTN", "TsTd", lrr)[0]["recommended"], "fold")
        self.assertEqual(go("BTN", "KsKd", lrr)[0]["recommended"], "all-in")
        # 9-handed early opens are tighter, so KJs continues less than against a 6-max UTG (where it now
        # mostly 3bets: the 6-max chart is 3bet-or-fold, matched to published solver output)
        cont = lambda a: a["call"]["freq"] + a["raise"]["freq"]
        self.assertLess(cont(go("UTG1", "KhJh", [R("UTG", "raise", 2.5)], players=9)[1]),
                        cont(go("HJ", "KhJh", [R("UTG", "raise", 2.5)])[1]))
        # facing a 3bet cold: in position calls more than the big blind out of position, whose premiums 4bet
        early3 = [R("UTG", "raise", 2.5), R("HJ", "raise", 7.5)]
        btn = go("BTN", "QsQd", early3 + [R("CO", "fold")])[1]
        bb = go("BB", "QsQd", early3 + F("CO", "BTN", "SB"))[1]
        self.assertGreater(btn["call"]["freq"], bb["call"]["freq"])
        self.assertGreater(bb["raise"]["freq"], btn["raise"]["freq"])
        self.assertEqual(bb["fold"]["freq"], 0.0)
        # ...and continues more against a 3bet over a late open (a wider 3bet)
        late3 = F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "raise", 7.5), R("SB", "fold")]
        self.assertLess(go("BB", "JhJd", late3)[1]["fold"]["freq"], go("BB", "JhJd", early3 + F("CO", "BTN", "SB"))[1]["fold"]["freq"])
        # 7+ handed blinds defend tighter against an early open
        ep = lambda n, seats: pf.strategy_for("BB", [R("UTG", "raise", 2.5)] + F(*seats), pf.Table(pf.SEATS[n]))
        cont = lambda s: float((s["call"] + s["raise"]).sum())
        self.assertLess(cont(ep(9, ("UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB"))), cont(ep(6, ("HJ", "CO", "BTN", "SB"))))

    def test_preflop_continue_rates(self):
        """Chart continue rates stay near typical 6-max 100bb solver output (share of the range that
        reaches the spot and doesn't fold), so a chart edit can't drift far without notice."""
        from gto.advisor import preflop_strategy, preflop_ranges
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        T = pf.Table(pf.SEATS[6])
        def rate(hero, pre):
            s, w = preflop_strategy(hero, pre, T, 100), preflop_ranges(pre, T, 100)[hero]
            return float((w * (1 - s["fold"])).sum() / w.sum())
        utg, co, btn = [R("UTG", "raise", 2.5)], F("UTG", "HJ") + [R("CO", "raise", 2.5)], F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)]
        bands = [
            ("BB", utg + F("HJ", "CO", "BTN", "SB"), 0.30, 0.42),
            ("BB", co + F("BTN", "SB"), 0.42, 0.55),
            ("BB", btn + F("SB"), 0.60, 0.72),
            ("BB", F("UTG", "HJ", "CO", "BTN") + [R("SB", "raise", 3)], 0.65, 0.80),
            # published solver: BTN vs a 2.5bb CO open folds 84.5% (calls 3.0%, 3bets 12.5%; Poker Academy)
            ("BTN", co, 0.12, 0.20),
            ("HJ", utg, 0.08, 0.14),
            ("UTG", utg + [R("HJ", "raise", 7.5)] + F("CO", "BTN", "SB", "BB"), 0.35, 0.50),
            ("BTN", btn + [R("SB", "raise", 10), R("BB", "fold")], 0.38, 0.55),
            # published: BTN continues ~45.4% (calls 33.9%, 4bets 11.5%) vs a BB 3bet to 11bb (888poker)
            ("BTN", btn + [R("SB", "fold"), R("BB", "raise", 11)], 0.40, 0.52),
            # published solver: CO continues 36.3% (calls 17.2%, 4bets 19.1%) vs a BTN 3bet to 7.5bb
            ("CO", co + [R("BTN", "raise", 7.5)] + F("SB", "BB"), 0.32, 0.45),
        ]
        # a hand never continues less often than one it dominates (JJ vs TT facing a 4bet used to flip on
        # board-sampling noise; AKo folded where AQo called)
        from gto.pf_classes import CLASS_OF, CLASS_INDEX, N_IN
        lines = [("HJ", utg), ("CO", [R("HJ", "raise", 2.5), R("CO", "raise", 7.5)] + F("BTN", "SB", "BB") + [R("HJ", "raise", 22)]),
                 ("CO", co + [R("BTN", "raise", 7.5)] + F("SB", "BB")), ("BB", btn + F("SB"))]
        for hero, pre in lines:
            s = preflop_strategy(hero, pre, T, 100)
            c = np.bincount(CLASS_OF, weights=1 - s["fold"], minlength=len(N_IN)) / N_IN
            for chain in pf._dominance_chains():
                for weak, strong in zip(chain, chain[1:]):
                    self.assertLessEqual(c[CLASS_INDEX[weak]], c[CLASS_INDEX[strong]] + 1e-6, f"{hero}: {weak} > {strong}")
        for hero, pre, lo, hi in bands:
            r = rate(hero, pre)
            self.assertTrue(lo <= r <= hi, f"{hero} after {[(a['pos'], a['action']) for a in pre]}: {r:.2f} not in [{lo}, {hi}]")

    def test_slightly_worse_price_trims_flats(self):
        """A 3bb button open (the chart's is 2.5bb) used to cut SB's flats from 85 combos to 24: the chart
        keeps small pairs and suited connectors below the raw-equity bar, and any worse price cut them all."""
        from gto.advisor import preflop_ranges
        T = pf.Table(pf.SEATS[6])
        pre = lambda to: [{"pos": p, "action": "fold"} for p in ("UTG", "HJ", "CO")] + [
            {"pos": "BTN", "action": "raise", "to": to}, {"pos": "SB", "action": "call"}]
        chart, priced = preflop_ranges(pre(2.5), T, 100)["SB"], preflop_ranges(pre(3), T, 100)["SB"]
        self.assertLess(priced.sum(), chart.sum())
        self.assertGreater(priced.sum(), 0.7 * chart.sum())
        # depth: 40bb less used to cost small pairs and suited hands all their implied odds (SB kept 17 of 85
        # flats vs a CO open, no pair or suited connector), and 100bb more made SB flat 276 combos
        co = [{"pos": p, "action": "fold"} for p in ("UTG", "HJ")] + [
            {"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "call"}]
        chart, short = preflop_ranges(co, T, 100)["SB"], preflop_ranges(co, T, 60)["SB"]
        small = np.array([hand_class(c) in ("22", "33", "44", "54s", "65s") for c in ALL_COMBOS])
        self.assertGreater(short.sum(), 0.5 * chart.sum())
        self.assertGreater((short * small).sum(), 0.3 * (chart * small).sum())
        self.assertLess(preflop_ranges(pre(2.5), T, 200)["SB"].sum(), 2 * preflop_ranges(pre(2.5), T, 100)["SB"].sum())

    def test_called_barrels_keep_a_range(self):
        """SB flats a 3bb open, then check-calls flop and turn: a short turn prior solve that folds most
        of SB's range (two pair included) used to leave it under one combo on the river."""
        adv = Advisor(time_budget=0.5, prior_budget=0.5)
        F = lambda *ps: [{"pos": p, "action": "fold"} for p in ps]
        xbc = lambda amt: [{"pos": "SB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": amt},
                           {"pos": "SB", "action": "call"}]
        out = adv.advise({"players": 6, "hero": "BTN", "hole": "As4d", "board": "3s2cQs9hKc", "stack": 100,
                          "preflop": F("UTG", "HJ", "CO") + [{"pos": "BTN", "action": "raise", "to": 3},
                                                             {"pos": "SB", "action": "call"}] + F("BB"),
                          "flop": xbc(5), "turn": xbc(13), "river": [{"pos": "SB", "action": "check"}]})
        self.assertGreater(out["villain_range_combos"], 1.0, out)

    def test_caller_continues_vs_next_raise(self):
        """A flat facing a squeeze continues with 30% of the flatting range or more (the cold-vs-3bet chart
        alone continued 2%), the playable part first, more the wider the squeeze; once the opener 4bets
        too, the flatter folds nearly all."""
        from gto.advisor import preflop_strategy, preflop_ranges
        from gto.cards import COMBO_INDEX, cards
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "call"), R("SB", "fold"),
               R("BB", "raise", 10)]
        w, s = preflop_ranges(pre, T, 100)["BTN"], preflop_strategy("BTN", pre, T, 100)
        cont = float((w * (1 - s["fold"])).sum() / w.sum())
        # 44% against the BB's published 8% 3bet (39% against its old 5.6% one; 30%, the floor, between)
        self.assertTrue(0.36 <= cont <= 0.52, cont)
        ix = lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        self.assertGreater(1 - s["fold"][ix("8s8d")], 0.9)
        # (JTs used to be here too; against the published BTN vs CO chart the flat is mostly pairs and
        # AJs/ATs, ~3% of hands, and it continues 77-JJ and AJs, some ATs and KJs. With a fixed 30% the
        # wider published squeeze pushed 77 out: 88 ranked above it and filled the share)
        self.assertGreater(1 - s["fold"][ix("7s7d")], 0.9)
        after = preflop_ranges(pre + [R("CO", "fold"), R("BTN", "call")], T, 100)["BTN"]
        self.assertGreater(after.sum(), 15)   # of ~38 flatting combos (was 30 of ~125 before the published BTN vs CO chart)
        s4 = preflop_strategy("BTN", pre + [R("CO", "raise", 23)], T, 100)
        self.assertLess(float((w * (1 - s4["fold"])).sum() / w.sum()), 0.12)   # the flat now holds JJ-TT more (it was 125 combos of mostly weaker hands)

    def test_caller_continue_grows_with_raise_width(self):
        """The share a caller continues against the next raise never shrinks as that raise's range widens:
        the BB's squeeze over CO + BTN, from value only to the chart's range with ever more bluffs added;
        against a raise tighter than the reference the floor holds."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "call"), R("SB", "fold"),
               R("BB", "raise", 10)]
        chart = pf.VS_RFI[("BB", "CO")]
        orig = chart["3bet"]
        w = pf.replay_ranges(pre[:4], T)["BTN"]
        bluffs = pf.R("K9s, Q9s, J9s, T8s, 97s, 86s, 75s, 64s, 53s, K8s, Q8s, A9o, KTo, QJo")
        shares = []
        try:
            for r in [pf.R("QQ+, AKs, AKo"), orig] + [np.minimum(1.0, orig + k * bluffs) for k in (0.25, 0.5, 1.0)]:
                chart["3bet"] = r
                pf._STRAT_CACHE.clear()
                s = pf.strategy_for("BTN", pre, T)
                shares.append(float((w * (1 - s["fold"])).sum() / w.sum()))
        finally:
            chart["3bet"] = orig
            pf._STRAT_CACHE.clear()
        # against value only the target is the floor (40% continue: the chart's own calls plus the
        # top-up, with 88 padded up to 77, which edges it against QQ+/AK)
        self.assertGreaterEqual(shares[0], pf.CALLER_CONTINUE[2] - 1e-6)
        for a, b in zip(shares, shares[1:]):
            self.assertGreaterEqual(b, a - 1e-6, shares)
        self.assertGreater(shares[-1], shares[1] + 0.03, shares)
        # the target itself: a range stronger against the raise than against the reference adds to the
        # floor, a weaker one keeps it
        from gto.pf_classes import equity_vs
        self.assertAlmostEqual(pf.caller_target(2, w, equity_vs(pf.R("QQ+, AKs, AKo"))), pf.CALLER_CONTINUE[2])
        ref_eq = pf._CALLER_REF_EQ[2]
        self.assertAlmostEqual(pf.caller_target(2, w, ref_eq), pf.CALLER_CONTINUE[2])
        self.assertAlmostEqual(pf.caller_target(2, w, ref_eq - 0.05), pf.CALLER_CONTINUE[2])
        self.assertAlmostEqual(pf.caller_target(2, w, ref_eq + 0.04), pf.CALLER_CONTINUE[2] + 0.04 * pf.CALLER_EQ_SLOPE)

    def test_sb_flat_vs_early_open_keeps_traps(self):
        """SB flats UTG, BB squeezes, SB calls: the flat held only 66-99 and suited broadways below KQs,
        so the call was 77-99 alone and the squeezer's QQ won every A-K-x flop. The flat now traps a
        share of JJ, TT, AQs, AJs and KQs, and some of them call the squeeze."""
        from gto.advisor import preflop_ranges
        from gto.pf_classes import to_classes, CLASSES
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        T = pf.Table(pf.SEATS[6])
        flat = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "fold"), R("BTN", "fold"), R("SB", "call")]
        line = flat + [R("BB", "raise", 12.5), R("UTG", "fold"), R("SB", "call")]
        c = dict(zip(CLASSES, to_classes(preflop_ranges(line, T, 100)["SB"])))
        for h in ("JJ", "TT", "AQs", "KQs"):
            self.assertGreater(c[h], 0.05, h)
        w = preflop_ranges(flat, T, 100)["SB"]
        self.assertTrue(25 <= w.sum() <= 50, w.sum())   # still a small flat (~3% of hands)

    def test_jam_after_flat_has_a_range(self):
        """CO flats, BB squeezes, HJ 4bets, CO jams: the jam range was CO's flats times the jam chart,
        empty, so equity was NaN and HJ folded AA. It is now the jam chart's range, rebuilt."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        pre = [R("UTG", "fold"), R("HJ", "raise", 2.5), R("CO", "call"), R("BTN", "fold"), R("SB", "fold"),
               R("BB", "raise", 12.5), R("HJ", "raise", 28.7), R("CO", "all-in"), R("BB", "fold")]
        adv = Advisor()
        st = {"players": 6, "hero": "HJ", "board": "", "stack": 100, "preflop": pre}
        self.assertEqual(adv.advise(dict(st, hole="AsAd"))["recommended"], "call")
        self.assertEqual(adv.advise(dict(st, hole="JsJd"))["recommended"], "fold")

    def test_vs_4bet_follows_the_4bet_range(self):
        """The 3bettor's vs-4bet chart is written against the opener's 4bet; against a cold 4bet (no
        bluffs) the marginal continues and bluff jams shrink, mixed value jams turn into calls, and
        against the opener nothing changes."""
        from gto.cards import COMBO_INDEX, cards
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        T = pf.Table(pf.SEATS[6])
        ix = lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        three = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 7.5)] + F("BTN", "SB")
        std = pf.strategy_for("CO", three + [R("BB", "fold"), R("UTG", "raise", 18)], T)
        cold = pf.strategy_for("CO", three + [R("BB", "raise", 18.8), R("UTG", "fold")], T)
        chart = pf.VS_4BET
        self.assertAlmostEqual(float(std["call"][ix("AhQh")]), float(chart["call"][ix("AhQh")]), places=6)
        self.assertLess(cold["call"][ix("AhQh")], 0.5 * std["call"][ix("AhQh")])
        self.assertLess(cold["raise"][ix("Ah5h")], 0.5 * std["raise"][ix("Ah5h")])
        for h in ("KsKd", "QsQd", "AhKd"):
            self.assertAlmostEqual(1 - cold["fold"][ix(h)], 1 - std["fold"][ix(h)], places=6)
        # QQ and AK jam less where the 4bettor folds to a jam much less often (a flat, then a 4bet over
        # an early open), calling instead; KK+ and AKs keep jamming
        flat4 = pf.strategy_for("BB", [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "fold"), R("BTN", "call"),
                                       R("SB", "fold"), R("BB", "raise", 12.5), R("UTG", "fold"), R("BTN", "raise", 28)], T)
        for h in ("QsQd", "AhKd"):
            self.assertLess(flat4["raise"][ix(h)], 0.5 * std["raise"][ix(h)])
            self.assertGreater(flat4["call"][ix(h)], std["call"][ix(h)])
        for h in ("AsAd", "KsKd", "AhKh"):
            self.assertEqual(float(cold["raise"][ix(h)]), 1.0)
            self.assertEqual(float(flat4["raise"][ix(h)]), 1.0)
        # the value bar follows the price, not how wide the 3bet range is: JJ, with more equity against
        # a cold 4bet over the button than over UTG, continues at least as often there
        btn = pf.strategy_for("SB", [R("UTG", "fold"), R("HJ", "fold"), R("CO", "fold"), R("BTN", "raise", 2.5),
                                     R("SB", "raise", 11), R("BB", "raise", 25), R("BTN", "fold")], T)
        self.assertGreaterEqual(1 - btn["fold"][ix("JsJd")], 1 - cold["fold"][ix("JsJd")])

    def test_4bettor_vs_jam_matches_advice(self):
        """The replayed 4bettor facing a 5bet jam calls by equity against the jam range, as hero's advice
        does (the fixed chart called QQ and AK 70% where they have 36% and need 41%)."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        from gto.cards import COMBO_INDEX, cards
        ix = lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 7.5), R("BTN", "fold"), R("SB", "fold"),
               R("BB", "fold"), R("UTG", "raise", 18), R("CO", "all-in")]
        s = pf.strategy_for("UTG", pre, T)
        adv = Advisor()
        for h in ("AsAd", "KsKd", "QsQd", "AhKd", "AhKh"):
            r = adv.advise({"players": 6, "hero": "UTG", "hole": h, "board": "", "stack": 100, "preflop": pre})
            self.assertAlmostEqual(float(s["call"][ix(h)]), r["actions"][0]["freq"], delta=0.02, msg=h)
        self.assertEqual(float(s["call"][ix("QsQd")]), 0.0)

    def test_thin_line_tops_up_from_prior_range(self):
        """An opener calling a cold 4bet: the chart calls only 6.8 combos even from scratch, so filling with
        the action's own hands couldn't reach MIN_LINE_COMBOS; the strongest hands of the range before
        the call top it up (the sanity sweep found 85 such lines under 8 combos)."""
        from gto.advisor import preflop_ranges
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 7.5), R("BTN", "fold"), R("SB", "fold"),
               R("BB", "raise", 18.8), R("UTG", "call")]
        for rng in (preflop_ranges(pre, T, 100)["UTG"], pf.replay_ranges(pre, T)["UTG"]):
            self.assertGreaterEqual(float(rng.sum()), pf.MIN_LINE_COMBOS - 1e-6)
            self.assertLessEqual(float(rng.max()), 1.0)

    def test_postflop_floor_for_thin_ranges(self):
        """A 4bet pot's ~12-combo range checking back the flop left HJ 1.8 combos for the turn, so the turn
        solve faced a near face-up hand; ranges entering a later street are filled out below
        preflop.MIN_LINE_COMBOS. A wide single-raised pot is untouched."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        adv = Advisor(time_budget=1.0, prior_budget=0.5)
        pre = [R("UTG", "raise", 2.5), R("HJ", "raise", 7.5)] + [R(p, "fold") for p in ("CO", "BTN", "SB", "BB")] + \
              [R("UTG", "raise", 18.8), R("HJ", "call")]
        out = adv.advise({"players": 6, "hero": "UTG", "hole": "KhAs", "board": "JsQs2hAh", "stack": 100, "preflop": pre,
                          "flop": [R("UTG", "check"), R("HJ", "check")]})
        self.assertGreater(out["villain_range_combos"], 4.0, out)
        self.assertTrue(any("filled out with the rest of their flop range" in n for n in out["notes"]), out["notes"])
        srp = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "fold"), R("BTN", "raise", 2.5), R("SB", "fold"), R("BB", "call")]
        out = adv.advise({"players": 6, "hero": "BB", "hole": "9h8h", "board": "Kd7c2s5h", "stack": 100, "preflop": srp,
                          "flop": [R("BB", "check"), R("BTN", "check")]})
        self.assertFalse(any("filled out" in n for n in out["notes"]), out["notes"])

    def test_off_chart_line_not_face_up(self):
        """BTN flats CO (the chart's flats never 4bet), then 4bets BB's squeeze: hero's range would be
        empty and KK solved face-up, so BB folds everything to a jam (all-in EV = the pot, +49).
        The range is rebuilt from the 4bet, and BB pays off."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "call"), R("SB", "fold"),
               R("BB", "raise", 10), R("CO", "fold"), R("BTN", "raise", 23), R("BB", "call")]
        out = Advisor(time_budget=1.0, prior_budget=0.5).advise(
            {"players": 6, "hero": "BTN", "hole": "KsKd", "board": "7d2s5s", "stack": 100,
             "preflop": pre, "flop": [R("BB", "check")]})
        self.assertGreater(out["hero_range_combos"], 20, out)
        self.assertGreater(max(a["ev_bb"] for a in out["actions"]), 52, out["actions"])
        self.assertTrue(any("isn't in the chart" in n for n in out["notes"]), out["notes"])



class TestTableSizes(unittest.TestCase):
    def test_replay_hand_dead_money_and_order(self):
        from gto.advisor import replay_hand
        pre = [{"pos": "UTG", "action": "raise", "to": 2.5}, {"pos": "HJ", "action": "call"}, {"pos": "CO", "action": "fold"},
               {"pos": "BTN", "action": "raise", "to": 9}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "fold"},
               {"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "call"}]
        st = replay_hand(pf.SEATS[6], 100, {"preflop": pre, "flop": []}, 2)
        self.assertTrue(st[0].closed)
        self.assertAlmostEqual(st[1].pot_start, 22.0)
        self.assertEqual(st[1].players_start, ["HJ", "BTN"])
        self.assertEqual(st[1].to_act, "HJ")

    def test_heads_up_order(self):
        from gto.advisor import replay_hand
        st = replay_hand(pf.SEATS[2], 100, {"preflop": []}, 1)[0]
        self.assertEqual(st.to_act, "SB")          # button/SB acts first preflop
        st = replay_hand(pf.SEATS[2], 100, {"preflop": [{"pos": "SB", "action": "raise", "to": 2.5}, {"pos": "BB", "action": "call"}], "flop": []}, 2)
        self.assertEqual(st[1].to_act, "BB")        # and last postflop
        self.assertAlmostEqual(st[1].pot_start, 5.0)

    def test_charts_by_table_size(self):
        self.assertEqual(pf.chart_key("LJ", pf.SEATS[9]), "UTG")
        self.assertEqual(pf.chart_key("UTG", pf.SEATS[9]), "EP")
        self.assertEqual(pf.chart_key("UTG", pf.SEATS[6]), "UTG")
        self.assertGreater(pf.RFI["HU_SB"].sum(), pf.RFI["BTN"].sum())
        self.assertLess(pf.RFI["EP"].sum(), pf.RFI["UTG"].sum())
        s = pf.hand_strategy("BB", tuple(cards("Jh9c")), [{"pos": "SB", "action": "raise"}], pf.SEATS[2])
        self.assertGreater(s["call"], 0.5)

    def test_same_high_card_range_syntax(self):
        self.assertEqual(parse_range("ATo-A5o").sum(), 72)

    def test_unsettled_solve_recommends_clear_ev_winner(self):
        from gto.advisor import UNSETTLED, EV_GAP
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": p, "action": "fold"} for p in ["UTG", "HJ", "CO", "BTN"]] + \
              [{"pos": "SB", "action": "raise", "to": 3}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "SB", "hole": "AsJs", "board": "8h5c9s", "preflop": pre,
                        "flop": [{"pos": "SB", "action": "check"}, {"pos": "BB", "action": "bet", "amount": 4.5}]})
        acts = r["actions"]
        top_freq, top_ev = max(acts, key=lambda a: a["freq"]), max(acts, key=lambda a: a["ev_bb"])
        if r["solver"]["exploitability_pct"] >= UNSETTLED and top_ev["ev_bb"] - top_freq["ev_bb"] > EV_GAP / 100 * r["pot_bb"]:
            self.assertEqual(r["recommended"], top_ev["action"])
            self.assertTrue(any(n.startswith("recommending") for n in r["notes"]))
        else:
            self.assertEqual(r["recommended"], top_freq["action"])

    def test_multiway_and_mid_street_hu(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        adv.mw_trials = MW_TRIALS   # fixed trials: the same result however loaded the machine is
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "CO", "hole": "AhAd", "board": "Kh7c2d", "preflop": pre, "flop": [{"pos": "BB", "action": "check"}]})
        self.assertTrue(r.get("multiway"))
        self.assertAlmostEqual(r["pot_bb"], 8.0)
        self.assertEqual(r["recommended"], max(r["actions"], key=lambda a: a["ev_bb"])["action"])
        self.assertTrue(any(n.startswith("if you check, someone bets") for n in r["notes"]))  # BTN still to act
        r = adv.advise({"players": 6, "hero": "CO", "hole": "AhAd", "board": "Kh7c2d", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "bet", "amount": 3.3}]})
        self.assertEqual([a["action"] for a in r["actions"]], ["fold", "call", "raise to 14"])
        self.assertEqual(r["actions"][0]["ev_bb"], 0.0)
        self.assertGreater(min(r["actions"][1]["ev_bb"], r["actions"][2]["ev_bb"]), 5.0)   # AA: call and raise both win
        r = adv.advise({"players": 6, "hero": "BTN", "hole": "KsQs", "board": "Kh7c2d", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "bet", "amount": 4}, {"pos": "CO", "action": "fold"}]})
        self.assertFalse(r.get("multiway"))
        self.assertEqual(r["villain"], "BB")
        self.assertAlmostEqual(r["pot_bb"], 12.0)
        self.assertEqual(r["actions"][0]["action"], "fold")


    def test_multiway_players_behind_and_bluffs(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        adv.mw_trials = MW_TRIALS   # fixed trials: the same result however loaded the machine is
        pre = [{"pos": "UTG", "action": "raise", "to": 2.5}, {"pos": "HJ", "action": "call"}, {"pos": "CO", "action": "call"},
               {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        st = {"players": 6, "hero": "HJ", "hole": "2c3d", "board": "Kc8h4h", "preflop": pre,
              "flop": [{"pos": "BB", "action": "check"}, {"pos": "UTG", "action": "bet", "amount": 5}]}
        r = adv.advise(st)
        self.assertTrue(any(n.startswith("after a call, CO, BB still to act") for n in r["notes"]))
        self.assertEqual(r["recommended"], "fold")
        self.assertLess(r["actions"][2]["ev_bb"], -1.0)        # opponents call too much for a pure bluff raise
        self.assertGreater(r["actions"][1]["ev_bb"], -5.5)     # later streets never cost more than the call
        self.assertIn("error", adv.advise(dict(st, hole="8h2c")))   # 8h is on the board

    def test_multiway_folds_follow_range_strength(self):
        # BB checks flop and turn, then leads the river; CO calls. BB's range is mostly air, CO's
        # call is mostly better than Q9. Each used to fold 64% of their range to a raise, so CO
        # folded QK and AA and a raise with the bluff-catcher Q9 showed +8bb; the folds now come
        # from the weak range.
        from gto.advisor import _joint_top, _top
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        adv.mw_trials = MW_TRIALS   # fixed trials: the same result however loaded the machine is
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "BTN", "hole": "Qc9c", "board": "Qd8s3h2c7d", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 2.5},
                                 {"pos": "BTN", "action": "call"}, {"pos": "BB", "action": "call"}],
                        "turn": [{"pos": p, "action": "check"} for p in ("BB", "CO", "BTN")],
                        "river": [{"pos": "BB", "action": "bet", "amount": 8}, {"pos": "CO", "action": "call"}]})
        self.assertTrue(r.get("multiway"))
        self.assertNotEqual(r["recommended"], "raise to 37.5")
        # (CO's small flop bet is now a wide, merged range and its turn check the weaker part, so Q9 is
        # often best and a raise isn't the disaster it was; it still makes less than a call)
        self.assertLess(r["actions"][2]["ev_bb"], r["actions"][1]["ev_bb"])
        note = next(n for n in r["notes"] if n.startswith("vs raise"))
        bb, co = (int(note.split(f"{p} continues with ")[1].split("%")[0]) for p in ("BB", "CO"))
        # (BB's flop call keeps the hands that would have raised at SLOWPLAY, so its river lead is no longer
        # mostly air: the two continue about equally, 45% and 43%, against 64% each before the joint bar)
        self.assertLess(max(bb, co), 55)
        self.assertLessEqual(abs(bb - co), 8)   # (47 / 41 with the dealt-out multiway preflop equity, 48 / 40 with
                                                # ties kept together in the check ranges)
        # One range: the same as keeping its top share.
        rng = np.random.default_rng(1)
        w, sc = rng.random(1326), rng.random(1326)
        self.assertTrue((_joint_top([w], 0.4, np.argsort(-sc))[0] == _top(w, 0.6, np.argsort(-sc))).all())
        # A strong and a weak range: the weak one folds more, and they fold together 40% of the time.
        strong, weak = np.where(sc > 0.5, 1.0, 0.1), np.where(sc < 0.5, 1.0, 0.1)
        ms, mw = _joint_top([strong, weak], 0.4, np.argsort(-sc))
        share = lambda r, m: (r * m).sum() / r.sum()
        self.assertGreater(share(strong, ms), share(weak, mw))
        self.assertAlmostEqual((1 - share(strong, ms)) * (1 - share(weak, mw)), 0.4, delta=0.01)

    def test_multiway_streets_narrow_ranges(self):
        from gto.advisor import replay_hand, _narrow_street
        import gto.preflop as pf
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "raise", "to": 2.5}, {"pos": "CO", "action": "fold"},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        flop = [{"pos": "BB", "action": "check"}, {"pos": "HJ", "action": "bet", "amount": 5}, {"pos": "BTN", "action": "call"}]
        pos = pf.SEATS[6]
        ranges = pf.replay_ranges(pre, pos)
        before = {p: ranges[p].copy() for p in ("HJ", "BTN", "BB")}
        st = replay_hand(pos, 100.0, {"preflop": pre, "flop": flop}, 2)[1]
        _narrow_street(ranges, cards("Jc7h2s"), st, 100.0)
        from gto.cards import COMBO_INDEX
        sets = COMBO_INDEX[tuple(sorted(cards("JdJh")))]
        for p in ("HJ", "BTN"):
            self.assertLess(ranges[p].sum(), 0.8 * before[p].sum())
        self.assertGreater(ranges["HJ"][sets] / before["HJ"][sets], 0.9)   # a set bets
        from gto.advisor import NARROW_SOFT
        self.assertLessEqual(ranges["BB"][sets] / before["BB"][sets], NARROW_SOFT + 1e-9)   # and doesn't check

    def test_multiway_players_behind_a_call_fold_like_the_field(self):
        # CO pots it 3-way on K72, BTN (hero) calls with BB behind. BB faced the bet with us, so it
        # folds as often as when both of us face it together; it used to price the bet into the pot
        # our call had grown and keep 81% of its range (junk included), paying off our call with dead
        # money: K5s showed +19 bb for an 8 bb call it wins 52% of the time against CO.
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        adv.mw_trials = MW_TRIALS   # fixed trials: the same result however loaded the machine is
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "BTN", "hole": "Kc5c", "board": "Kh7c2d", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 8}]})
        note = next(n for n in r["notes"] if n.startswith("after a call, BB still to act"))
        self.assertLess(int(note.split("BB continues with ")[1].split("%")[0]), 45)   # _keep(8, 8, 2): 39%
        # (about 16 bb; the later streets now let CO and BB bet into us first, and us raise, so more than
        # the 7 bb of a model where we always acted first after the flop; and BB's check now rules out the
        # wider range that would have bet small, so BB is weaker; the dead-money bug gave +19 bb)
        self.assertLess(r["actions"][1]["ev_bb"], 18.0)

    def test_multiway_overcall_narrows_like_the_first_call(self):
        # HJ bets 3-way, BTN calls, BB overcalls. Both faced the same bet; the overcall used to be priced
        # against the pot the first call had grown, so with equal ranges BB kept more than BTN.
        from gto.advisor import replay_hand, _narrow_street
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "raise", "to": 2.5}, {"pos": "CO", "action": "fold"},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        flop = [{"pos": "BB", "action": "check"}, {"pos": "HJ", "action": "bet", "amount": 4},
                {"pos": "BTN", "action": "call"}, {"pos": "BB", "action": "call"}]
        pos = pf.SEATS[6]
        base = pf.replay_ranges(pre, pos)["BTN"]
        st = replay_hand(pos, 100.0, {"preflop": pre, "flop": flop}, 2)[1]
        self.assertEqual((st.bet_pot, st.bet_risk, st.bet_faced), (8.0, 4.0, 2))
        ranges = {"HJ": base.copy(), "BTN": base.copy(), "BB": base.copy()}
        # BB's check narrows its range first; compare the call step alone by checking from equal ranges
        st.actions = flop[1:]
        _narrow_street(ranges, cards("Jc7h2s"), st, 100.0)
        self.assertAlmostEqual(ranges["BB"].sum(), ranges["BTN"].sum(), delta=0.01 * ranges["BTN"].sum())

    def test_multiway_players_behind_a_bettor_act_after_we_check(self):
        # BB checks 3-way: CO may bet with BTN between it and us, or BTN may bet with CO behind us. The
        # first bettor used to end the action, so a check-call faced the bettor alone; now the others
        # call or raise it, and we fold, call or check-raise knowing what the players between did: a set
        # continues against everything, middle pair only some of it (not raises), a gutshot nothing.
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        adv.mw_trials = MW_TRIALS   # fixed trials: the same result however loaded the machine is
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        share = {}
        for hole in ("7s7d", "7s6s", "8s6s"):
            r = adv.advise({"players": 6, "hero": "BB", "hole": hole, "board": "Kh7c2d", "preflop": pre, "flop": []})
            note = next(n for n in r["notes"] if n.startswith("if you check, someone bets"))
            called = int(note.split("between the bettor and you calls it ")[1].split("%")[0])
            raised = int(note.split(" and raises ")[1].split("%")[0])
            self.assertGreater(called, 5)
            self.assertGreater(raised, 0)
            share[hole] = (int(note.split("you'd call ")[1].split("%")[0])
                           + int(note.split("check-raise ")[1].split("%")[0]))
        self.assertGreaterEqual(share["7s7d"], 99)
        self.assertLess(share["7s6s"], share["7s7d"])
        self.assertGreater(share["7s6s"], 0)
        self.assertLessEqual(share["8s6s"], 3)   # (2%: the preflop ranges' small shifts move it a point or two)

    def test_multiway_small_bet(self):
        # Checked to CO 3-way: it chooses between checking, a third of the pot and two thirds by EV, and
        # the field continues wider against the small bet (each folds less, the joint fold rule).
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        adv.mw_trials = MW_TRIALS   # fixed trials: the same result however loaded the machine is
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "BB", "hole": "Ts9s", "board": "Kh7c2d", "preflop": pre, "flop": []})
        self.assertEqual([(a["action"], a.get("amount_bb")) for a in r["actions"]],
                         [("check", None), ("bet 33%", 2.5), ("bet 66%", 5.5)])
        self.assertEqual(r["recommended"], max(r["actions"], key=lambda a: a["ev_bb"])["action"])
        cont = {}
        for size in ("33%", "66%"):
            note = next(n for n in r["notes"] if n.startswith(f"vs bet {size}"))
            cont[size] = [int(note.split(f"{p} continues with ")[1].split("%")[0]) for p in ("CO", "BTN")]
        for small, big in zip(cont["33%"], cont["66%"]):
            self.assertGreater(small, big + 4)
        # Too small a pot for two sizes a 0.5bb step apart: only the 66% bet.
        import time
        from gto.advisor import replay_hand
        st = replay_hand(pf.SEATS[6], 100.0, {"preflop": pre, "flop": []}, 2)[1]
        st.pot_start = 1.0
        out = adv._multiway("BB", tuple(cards("Ts9s")), cards("Kh7c2d"), pf.replay_ranges(pre, pf.SEATS[6]),
                            ["BB", "CO", "BTN"], st, 100.0, time.time(), [])
        self.assertEqual([a["action"] for a in out["actions"]], ["check", "bet 66%"])

    def test_multiway_check_raise(self):
        # Checked to 3-way: after a bet we can check-raise (to 3x the bet plus the calls between), and
        # everyone still in faces it by the joint fold rule. A set check-raises some of the bets (its
        # check counts that line), air never does.
        from gto.advisor import _raise_to
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        adv.mw_trials = MW_TRIALS   # fixed trials: the same result however loaded the machine is
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        line = {}
        for hole in ("7s7d", "4s3d"):
            r = adv.advise({"players": 6, "hero": "BB", "hole": hole, "board": "Kh7c2d", "preflop": pre, "flop": []})
            line[hole] = r["check_line"]
            self.assertTrue(any("check-raise" in n for n in r["notes"]))
            self.assertAlmostEqual(sum(line[hole][k] for k in ("call", "raise", "fold")), 1.0, places=2)
        self.assertGreater(line["7s7d"]["raise"], 0.2)
        self.assertEqual(line["4s3d"]["raise"], 0.0)
        self.assertGreater(line["4s3d"]["fold"], 0.9)
        self.assertEqual(_raise_to(5.5, 5.5), 22.0)     # 3x a 5.5 bet plus one 5.5 call

    def test_multiway_check_raise_frequencies(self):
        # A check-raise has to beat calling (or folding) by two standard errors over the trials of its
        # situation. On the bare means the choice followed the noise: middle pair 7s6s on K72 check-raised
        # about 37% of the bets and the gutshot T9 on 863 anywhere from 15% to 64% depending on when the
        # trials stopped. Now they rarely do, while a set and two pair still check-raise often.
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        adv.mw_trials = MW_TRIALS
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        xr = {}
        for hole, board in (("7s6s", "Kh7c2d"), ("Ts9s", "8d6h3c"), ("7s7d", "Kh7c2d"), ("Kd7d", "Kh7c2d")):
            r = adv.advise({"players": 6, "hero": "BB", "hole": hole, "board": board, "preflop": pre, "flop": []})
            xr[hole] = r["check_line"]["raise"]
        self.assertLessEqual(xr["7s6s"], 0.15)
        self.assertLessEqual(xr["Ts9s"], 0.15)
        self.assertGreater(xr["7s7d"], 0.6)
        self.assertGreater(xr["Kd7d"], 0.25)

    def test_multiway_speed(self):
        # A warm 3-way flop spot (ranges and board cached, a new hand) plays 1200 trials in well under a
        # few CPU-seconds (about 0.5 s on an idle laptop; 1.5 s before the evaluator and equity rework).
        # CPU time, not wall time, and a generous bound, so a loaded machine doesn't fail it.
        import time
        adv = Advisor(Config(depth=1), time_budget=4.0, prior_budget=2.0)
        adv.mw_trials = MW_TRIALS
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        st = {"players": 6, "hero": "BB", "hole": "AsKd", "board": "Kh7c2d", "preflop": pre, "flop": []}
        adv.advise(st)
        c = time.process_time()
        adv.advise(dict(st, hole="7s6s"))
        self.assertLess(time.process_time() - c, 6.0)

    def test_multiway_top_pair_continues_like_draws(self):
        # 8-6-3 checked to the BB 3-way. The bettor's bluffs used to come from the bottom of its range
        # (air) and its value only from what beat what would call as if everyone held its range, so an
        # 8x top pair continued 45% against a bet while 7c5c (an open-ender) continued 100%. Bluffs are now
        # the hands that most often get there (draws first), the value is set against the ranges it is
        # bet into, and a bet our equity alone pays for is called: against the small bets top pair continues
        # at least as often as the open-enders and the gutshot. (The big bet is now the nut-advantage range,
        # sets and overpairs plus draws: top pair folds to it and the draws' price keeps them in, as it
        # should; against the wide small bet a gutshot with two overcards has the price too.)
        from gto.advisor import _street_info
        from gto.cards import COMBO_INDEX
        board = cards("8d6h3c")
        sc, rank, bluff = _street_info(tuple(board))
        pos = {int(c): i for i, c in enumerate(bluff)}
        combo = lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        self.assertLess(pos[combo("7s5s")], pos[combo("Qs2d")])     # an open-ender bluffs before queen-high
        self.assertLess(pos[combo("Qs2d")], pos[combo("9s6s")])     # middle pair has showdown value: last
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        adv.mw_trials = MW_TRIALS   # fixed trials: the same result however loaded the machine is
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        cont = {}
        for hole in ("Ad8s", "7c5c", "5c4c", "Ts9s"):
            r = adv.advise({"players": 6, "hero": "BB", "hole": hole, "board": "8d6h3c", "preflop": pre, "flop": []})
            cont[hole] = (r["check_line"]["continue_small"], r["check_line"]["continue_unraised"])
        self.assertGreaterEqual(cont["Ad8s"][0], max(cont["7c5c"][0], cont["5c4c"][0], cont["Ts9s"][0]))
        self.assertGreater(cont["Ad8s"][0], 0.95)

    def test_multiway_later_streets(self):
        # Later streets inside the trials: bets at either size (33% and 66%), the players behind a bet
        # still act (bets get more than one caller), and raises are sized over the calls in front.
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        adv.mw_trials = MW_TRIALS   # fixed trials: the same result however loaded the machine is
        pre = [{"pos": "UTG", "action": "raise", "to": 2.5}, {"pos": "HJ", "action": "call"}, {"pos": "CO", "action": "call"},
               {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "BB", "hole": "Qh9h", "board": "Kc8h4h", "preflop": pre, "flop": []})
        ls = r["later_streets"]
        self.assertGreater(ls["bets"], 100)
        self.assertGreater(ls["small"], 0.05)
        self.assertLess(ls["small"], 0.95)
        self.assertGreater(ls["multi_called"], 0.02)
        self.assertGreater(ls["raises_over_calls"], 0)
        # from the turn, every river: the same rules on the last street
        r = adv.advise({"players": 6, "hero": "BB", "hole": "Qh9h", "board": "Kc8h4h2s", "preflop": pre,
                        "flop": [{"pos": p, "action": "check"} for p in ("BB", "UTG", "HJ", "CO")], "turn": []})
        self.assertTrue(r.get("multiway"))
        self.assertGreater(r["later_streets"]["small"], 0.0)

    def test_big_bet_range_is_mostly_by_the_rules(self):
        # BB checks a three-diamond flop 3-way, then overbets the fourth diamond: the rule's range is
        # small, and hands off it mustn't swamp it (they used to be ~2/3 of the range).
        from gto.advisor import replay_hand, _narrow_street
        pre = [{"pos": "UTG", "action": "call"}, {"pos": "HJ", "action": "raise", "to": 2.5}, {"pos": "CO", "action": "fold"},
               {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"},
               {"pos": "UTG", "action": "call"}]
        streets = {"preflop": pre, "flop": [{"pos": p, "action": "check"} for p in ("BB", "UTG", "HJ")],
                   "turn": [{"pos": "BB", "action": "bet", "amount": 15}, {"pos": "UTG", "action": "fold"}]}
        pos, board, hole = pf.SEATS[6], cards("6d4d3dKd"), cards("AcJs")
        ranges = pf.replay_ranges(pre, pos)
        rep = replay_hand(pos, 100.0, streets, 3)
        _narrow_street(ranges, board[:3], rep[1], 100.0, "HJ", hole)
        _narrow_street(ranges, board, rep[2], 100.0, "HJ", hole)
        bb = ranges["BB"] * np.prod([1 - MEMBER_FULL[c] for c in board + hole], axis=0)
        flush = np.array([a % 4 == board[0] % 4 or b % 4 == board[0] % 4 for a, b in ALL_COMBOS])
        self.assertGreater(bb[flush].sum() / bb.sum(), 0.45)
        r = Advisor(time_budget=1.0, prior_budget=0.5).advise(dict(
            players=6, hero="HJ", hole="AcJs", board="6d4d3dKd", stack=100, **streets))
        self.assertEqual(r["recommended"], "fold")

    def test_hand_outside_range_still_gets_a_strategy(self):
        # 3s rather than 0.5s: under load a short budget left K2o at 0.87 fold (fewer iterations)
        adv = Advisor(Config(depth=1), time_budget=3.0, prior_budget=0.3)
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "fold"}]
        r = adv.advise({"players": 6, "hero": "BTN", "hole": "Ks2c", "board": "6dQh4h", "preflop": pre,
                        "flop": [{"pos": "CO", "action": "bet", "amount": 9}]})
        self.assertGreater(r["actions"][0]["freq"], 0.9)   # K2o folds, not a uniform mix
        self.assertLess(r["actions"][1]["ev_bb"], 0.0)

    def test_deep_jam_is_in_the_tree(self):
        # a 95bb turn jam into 9.5bb: hero faces the jam, not the tree's biggest bet
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
               {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "BTN", "hole": "7h6h", "board": "Th2hKd2s", "stack": 100, "preflop": pre,
                        "flop": [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 2},
                                 {"pos": "BB", "action": "call"}],
                        "turn": [{"pos": "BB", "action": "all-in"}]})
        self.assertEqual([a["action"] for a in r["actions"]], ["fold", "call"])
        self.assertEqual(r["actions"][1]["amount_bb"], 95.5)
        self.assertFalse(any("mapped 'all-in'" in n for n in r["notes"]))

    def test_real_sizes_reach_the_tree(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
               {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        flop = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 2}, {"pos": "BB", "action": "call"}]
        base = {"players": 6, "hero": "BTN", "hole": "7h6h", "board": "Th2hKd2s", "stack": 100, "preflop": pre, "flop": flop}
        bet = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 5}]
        cases = [
            # a jam entered as a bet of the whole stack
            (dict(base, turn=[{"pos": "BB", "action": "bet", "amount": 95.5}]), 95.5),
            # a raise to the whole stack
            (dict(base, turn=bet + [{"pos": "BB", "action": "raise", "to": 95.5}]), 90.5),
            # an off-size raise and then a jam: the jam's size doesn't replace the raise's
            (dict(base, hero="BB", hole="KhQs", turn=bet + [{"pos": "BB", "action": "raise", "to": 30},
                                                           {"pos": "BTN", "action": "all-in"}]), 65.5),
        ]
        for state, to_call in cases:
            r = adv.advise(state)
            self.assertEqual(dict((a["action"], a["amount_bb"]) for a in r["actions"])["call"], to_call)
            self.assertFalse(any(n.startswith("turn: mapped") and "bet 53%" not in n for n in r["notes"]), r["notes"])
        # a flop 3bet past the tree's raise cap is a 3bet, not an all-in
        r = adv.advise({"players": 6, "hero": "BB", "hole": "KhQs", "board": "Th2hKd", "stack": 100, "preflop": pre,
                        "flop": [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 2},
                                 {"pos": "BB", "action": "raise", "to": 7}, {"pos": "BTN", "action": "raise", "to": 18}]})
        self.assertLess(dict((a["action"], a["amount_bb"]) for a in r["actions"])["call"], 15)
        self.assertFalse(any("to tree action 'all-in'" in n for n in r["notes"]))

    def test_swap_nearest_keeps_sizes_taken(self):
        from gto.advisor import _swap_nearest
        sizes, fixed = [0.5, 1.0], set()
        _swap_nearest(sizes, 1.3, fixed=fixed)
        _swap_nearest(sizes, 5.0, fixed=fixed)
        self.assertEqual(sizes, [5.0, 1.3])
        _swap_nearest(sizes, 0.4, fixed=fixed)
        self.assertEqual(sizes, [5.0, 1.3, 0.4])


def _exact_equity(board, hero, villain_range):
    """Hero's equity against a range, every runout enumerated."""
    import itertools
    from gto.cards import COMBO_INDEX
    hi = COMBO_INDEX[tuple(sorted(hero))]
    vis = [int(j) for j in np.nonzero(villain_range)[0]]
    rest = [c for c in range(52) if c not in board and c not in hero]
    tot = wsum = 0.0
    for run in itertools.combinations(rest, 5 - len(board)):
        b = tuple(board) + run
        st = dict(zip([hi] + vis, eval_board(b, [hi] + vis)))
        for j in vis:
            if set(ALL_COMBOS[j]) & (set(b) | set(hero)):
                continue
            tot += villain_range[j] * ((st[hi] > st[j]) + 0.5 * (st[hi] == st[j]))
            wsum += villain_range[j]
    return tot / wsum


class TestAccuracy(unittest.TestCase):
    """Checks against exact enumeration and against known equilibria."""

    NO_BETS = {s: [] for s in ("flop", "turn", "river")}

    def test_leaf_matrix_averages_over_cards_both_hands_miss(self):
        # AA vs KK on 2c7d9hTs: 44 live rivers, KK wins on 2 of them
        from gto.cards import COMBO_INDEX
        board = tuple(cards("2c7d9hTs"))
        aa, kk = COMBO_INDEX[tuple(sorted(cards("AhAs")))], COMBO_INDEX[tuple(sorted(cards("KhKs")))]
        ctx = Ctx(np.array(sorted([aa, kk])), Config())
        ctx.root_len = 4
        L = ctx.leaf_matrix(board, set(board))
        i, j = list(ctx.active).index(aa), list(ctx.active).index(kk)
        self.assertAlmostEqual(float(L[i, j]), 40 / 44, places=5)

    def test_checked_down_ev_equals_pot_times_equity(self):
        # a tree with no bets: checking is worth exactly pot * equity, through chance nodes
        # (turn -> every river) and through equity leaves (flop -> every turn and river)
        hero, vr = cards("AhAs"), parse_range("KK, 88, QQ, JTs, 76s")
        for board, depth in (("Kh8d5c3s", 2), ("Kh8d5c", 1)):
            cfg = Config(bet_sizes=self.NO_BETS, raise_sizes=self.NO_BETS, allin_spr=0, depth=depth,
                         leaf_turns=49, leaf_rivers_flop=48)
            b = cards(board)
            sol = solve(b, parse_range("AA"), vr, pot=10, stack=100, time_budget=0.1, max_iters=3, cfg=cfg)
            ev = sol.for_combo(tuple(hero))["check"][1]
            self.assertAlmostEqual(ev, 10 * _exact_equity(b, hero, vr), places=3)

    def test_fast_showdown_matches_matrix(self):
        board = tuple(cards("Ks7d2cTh"))
        bm = np.ones(N_FULL)
        for c in board:
            bm *= 1 - MEMBER_FULL[c]
        active = np.nonzero(parse_range("22+, A2s+, K9s+, QTs+, JTs, T9s, 98s, A9o+, KJo+") * bm > 0)[0]
        ctx = Ctx(active, Config())
        boards = [board + (c,) for c in range(52) if c not in board]
        ro = np.random.default_rng(0).random((len(boards), ctx.n))
        for bi, b in enumerate(boards):
            ro[bi] *= 1 - ctx.member[b[-1]]
        fast = ctx.ranking(boards).utility(ro)
        dense = np.matmul(ctx.stacked("showdown", boards, set()), ro[:, :, None].astype(np.float32))[:, :, 0]
        live = np.array([1 - ctx.member[b[-1]] for b in boards])
        self.assertLess(np.abs((fast - dense) * live).max(), 1e-3)
        sel = [3, 17, 40]
        self.assertTrue(np.allclose(ctx.ranking(boards).utility(ro[sel], sel), fast[sel]))

    def test_polarized_river_game(self):
        # nuts-or-air vs a bluff-catcher: bluffs are b/(1+2b) of bets, the catcher calls 1/(1+b)
        board = cards("Kh8d5c3s2h")                       # IP: 88 (3 sets) + T9o (12 air), OOP: QQ
        oop, ip = parse_range("QQ"), parse_range("88, T9o")
        for b in (0.5, 1.0, 2.0):
            cfg = Config(bet_sizes={s: [b] for s in ("flop", "turn", "river")}, raise_sizes=self.NO_BETS, allin_spr=0)
            sol = solve(board, oop, ip, pot=10, stack=100, time_budget=2, max_iters=2000, cfg=cfg)
            ipn = sol.root.children[0]
            k = 1
            st = ipn.avg_strategy0()
            air = parse_range("T9o")[sol.ctx.active] > 0
            self.assertAlmostEqual(float(st[k][air].mean()), 3 * b / (1 + b) / 12, places=2)
            vs_bet = ipn.children[k]
            qq = parse_range("QQ")[sol.ctx.active] > 0
            self.assertAlmostEqual(float(vs_bet.avg_strategy0()[1][qq].mean()), 1 / (1 + b), places=2)
            # includes lines off the equilibrium path (OOP leading), which must be trained too
            self.assertLess(sol.exploitability(), 0.1)

    def test_turn_solve_converges_on_every_river(self):
        r_oop = parse_range("22+, A2s+, KTs+, QTs+, JTs, T9s, 98s, ATo+, KQo")
        r_ip = parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, A8o+, KTo+, QTo+, JTo")
        cfg = Config(bet_sizes={s: [0.75] for s in ("flop", "turn", "river")}, raise_sizes=self.NO_BETS)
        sol = solve(cards("Ks7d2cTh"), r_oop, r_ip, pot=10, stack=50, time_budget=60, max_iters=80, cfg=cfg)
        self.assertLess(sol.exploitability(), 1.0)
        cfg.river_sample = 8                              # public chance sampling runs and stays sane
        sol = solve(cards("Ks7d2cTh"), r_oop, r_ip, pot=10, stack=50, time_budget=60, max_iters=40, cfg=cfg)
        self.assertTrue(np.allclose(sol.strategy[:, sol.ctx.active].sum(axis=0), 1.0, atol=1e-5))

    def test_rfi_widths_match_published_charts(self):
        # 6-max 100bb solver charts: UTG 15-18%, HJ 19-22%, CO 26-29%, BTN 40-48%
        for pos, lo, hi in (("UTG", 15, 18), ("HJ", 19, 22), ("CO", 26, 29), ("BTN", 40, 48)):
            pct = pf.RFI[pos].sum() / 1326 * 100
            self.assertTrue(lo <= pct <= hi, f"{pos} opens {pct:.1f}%")


# Published flop strategies, 6-max cash 100bb, BTN opens 2.5x and BB calls (SRP) or CO opens, BTN 3bets and CO calls
# (3bet). (spot, board, statistic, low, high, source). Statistics: "cbet" (the in-position player's bets when checked
# to, % of its range), "small" (its smallest size), "check" (the out-of-position player's checks), "vs small: fold /
# call / raise" (the checker's response to the smallest bet). Bounds are wide where the sources differ in bet sizes
# (a 25% minimum bet raises frequencies: GTO Wizard's all-flop BTN c-bet is 53% with a 33% minimum and 64% with 27%)
# or come from undisclosed engines. "*" marks the less certain ones.
FLOP_REFERENCES = [
    # K-high dry: among the most c-bet flops, mostly small. BeyondGTO's spot table K72r 98% (*: undisclosed tree);
    # Upswing (Ace-high flops podcast): K83 from MP vs BB ~100% at a third pot; GTO Wizard's C-betting heuristics: 8- to
    # K-high flops are bet more than A-high
    ("srp", "Kd7c2h", "cbet", 65, 100, "BeyondGTO 98% *, Upswing K83 ~100%"),
    ("srp", "Kd7c2h", "small", 50, 100, "RangeConverter 6max 100bb aggregate: 1/3 pot the most common size"),
    ("srp", "Kd7c2h", "vs small: raise", 6, 22, "RangeConverter aggregate x/r 14%; ThinkGTO A74 15.4%"),
    # high paired: range bet small. GTO Wizard (The mechanics of c-bet sizing) QQ6: high frequency, mostly 33%, BB checks
    # 100% and against the 33% bet folds 37 / calls 39 / raises 24; ThinkGTO 25/80 tree QQ6 98.8%; RangeConverter "high
    # paired boards are bet nearly 100%"
    ("srp", "QsQd6c", "cbet", 80, 100, "GTO Wizard QQ6, ThinkGTO 98.8%, RangeConverter ~100%"),
    ("srp", "QsQd6c", "small", 75, 100, "GTO Wizard QQ6 mostly 33%"),
    ("srp", "QsQd6c", "check", 95, 100, "GTO Wizard QQ6 BB checks 100%"),
    # (*: the big blind here flats more hands preflop, so it folds more and calls less: 47 / 30 / 23)
    ("srp", "QsQd6c", "vs small: fold", 27, 52, "GTO Wizard QQ6 37% *"),
    ("srp", "QsQd6c", "vs small: call", 25, 49, "GTO Wizard QQ6 39% *"),
    ("srp", "QsQd6c", "vs small: raise", 14, 34, "GTO Wizard QQ6 24%"),
    # A-high: bet less than K-high. ThinkGTO 25/80 As7h4d 67.3% (62.2 small), BB vs 25%: 34.6 / 50.0 / 15.4;
    # PeakGTO via PokerCoaching As7c2h ~43%, As9h3c 57% (*: trees not given); Upswing A87/A86/A76 55-60%
    ("srp", "As7h4d", "cbet", 40, 80, "ThinkGTO 67%, PeakGTO 43-57% *, Upswing 55-60%"),
    ("srp", "As7h4d", "vs small: raise", 6, 24, "ThinkGTO 15.4% (25% bet)"),
    # low connected: RangeConverter low boards 40-60%; BeyondGTO 543r 49%, 942r 58% *; GTO Wizard 40bb: BB donks ~7% on
    # connected flops (2% over all flops; 100bb cash trees often leave the donk out)
    ("srp", "7h6d5c", "cbet", 30, 60, "RangeConverter low boards 40-60%, BeyondGTO 543 49% *"),
    ("srp", "7h6d5c", "check", 80, 100, "GTO Wizard BB checks 98% of flops, ~93% on connected ones *"),
    # monotone: bet much less than rainbow (GTO Wizard heuristics); QdJdTd ~50/50 (GTO Wizard), T98 monotone 55% *
    ("srp", "Kh7h2h", "cbet", 30, 65, "GTO Wizard QJT monotone ~50%, BeyondGTO T98 monotone 55% *"),
    # 3bet pots: the in-position 3-bettor (BTN vs CO) checks 19.7% of all flops (GTO Wizard, Crush 3-bet pots OOP),
    # c-bets "more than 80%" (GTO Wizard, turn probing); dry K-high near 100%
    ("3bet", "Kd7c2h", "cbet", 85, 100, "GTO Wizard IP 3-bettor c-bets ~80% of all flops; dry K-high higher"),
    ("3bet", "Ah8c3d", "cbet", 65, 100, "GTO Wizard ~80% of all flops"),
]


BUCKET_LOW = {("srp", "QsQd6c", "cbet"): 75}


class TestFlopStrategy(unittest.TestCase):
    """Flop solves against published solver frequencies (FLOP_REFERENCES), and the tree pieces behind them: the river
    beyond `depth` played with its betting (river_play) and one turn per rank class (turn_classes)."""

    ITERS = 150
    _sols = {}
    # Fast's flop model (gto/native.py, static/worker.js): the turn and river by 32 hand buckets
    BUCKETS = dict(river_buckets=32, turn_buckets=True, bucket_river_step=3)

    @classmethod
    def solve_spot(cls, kind, board, buckets=False):
        key = (kind, board, buckets)
        if key not in cls._sols:
            from gto.advisor import preflop_ranges
            pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}]
            if kind == "srp":
                pre += [{"pos": "CO", "action": "fold"}, {"pos": "BTN", "action": "raise", "to": 2.5},
                        {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
                oop, ip, pot, stack = "BB", "BTN", 5.5, 97.5
            else:
                pre += [{"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "raise", "to": 7.5},
                        {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "fold"}, {"pos": "CO", "action": "call"}]
                oop, ip, pot, stack = "CO", "BTN", 16.5, 92.5
            rng = preflop_ranges(pre, pf.Table(pf.SEATS[6]), 100)
            cls._sols[key] = solve(cards(board), rng[oop], rng[ip], pot, stack,
                                   cfg=Config(**(cls.BUCKETS if buckets else {})), time_budget=1e9, max_iters=cls.ITERS)
        return cls._sols[key]

    @staticmethod
    def stats(sol) -> dict:
        root, r0, r1 = sol.root, sol.r0, sol.r1
        st0 = root.avg_strategy0()
        ipn = root.children[0]
        st1 = ipn.avg_strategy0()
        bets = [a for a, n in enumerate(ipn.names) if n != "check"]
        out = {"check": (st0[0] * r0).sum() / r0.sum() * 100,
               "cbet": sum((st1[a] * r1).sum() for a in bets) / r1.sum() * 100,
               "small": (st1[bets[0]] * r1).sum() / r1.sum() * 100}
        node, reach = ipn.children[bets[0]], r0 * st0[0]
        s = node.avg_strategy0()
        for b, m in enumerate(node.names):
            k = "vs small: " + ("raise" if m.startswith("raise") or m == "all-in" else m)
            out[k] = out.get(k, 0.0) + (s[b] * reach).sum() / reach.sum() * 100
        return out

    def test_published_frequencies(self):
        for buckets in (False, True):
            for kind, board, stat, lo, hi, src in FLOP_REFERENCES:
                v = self.stats(self.solve_spot(kind, board, buckets))[stat]
                if buckets:
                    # the buckets' one miss: QQ6's c-bet settles near 80% where a solve over every turn has 88% (the
                    # river checked down had 52-56%)
                    lo = BUCKET_LOW.get((kind, board, stat), lo)
                self.assertTrue(lo <= v <= hi, f"{kind} {board} {stat} {v:.1f} not in [{lo}, {hi}] ({src})"
                                + (" with buckets" if buckets else ""))

    def test_texture_ordering(self):
        # GTO Wizard's c-betting heuristics: A-high flops are bet less than K-high, monotone and low connected ones much less
        for buckets in (False, True):
            c = {b: self.stats(self.solve_spot("srp", b, buckets))["cbet"] for b in ("Kd7c2h", "As7h4d", "7h6d5c", "Kh7h2h")}
            self.assertGreater(c["Kd7c2h"], c["As7h4d"])
            self.assertGreater(c["Kd7c2h"], c["7h6d5c"] + 15)
            self.assertGreater(c["Kd7c2h"], c["Kh7h2h"] + 15)

    def test_buckets_close_to_per_combo(self):
        # the bucketed turn and river against the per-combo model at the same iterations: flop frequencies within a few points
        for kind, board in (("srp", "Kd7c2h"), ("srp", "QsQd6c"), ("srp", "7h6d5c"), ("3bet", "Ah8c3d")):
            a, b = self.stats(self.solve_spot(kind, board)), self.stats(self.solve_spot(kind, board, True))
            for k in ("cbet", "check", "vs small: call"):
                self.assertLess(abs(a[k] - b[k]), 8, f"{kind} {board} {k}: {a[k]:.1f} per combo, {b[k]:.1f} with buckets")

    def test_bucket_tables(self):
        # E[1] = -E[0].T and kappa[1] = kappa[0].T (a zero-sum bucket game), each live combo in one bucket, equal shares
        from gto.advisor import preflop_ranges
        rng = preflop_ranges([{"pos": p, "action": "fold"} for p in ("UTG", "HJ", "CO")]
                             + [{"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"},
                                {"pos": "BB", "action": "call"}], pf.Table(pf.SEATS[6]), 100)
        board = tuple(cards("Kd7c2hTs"))
        live = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
        r0, r1 = rng["BB"] * live, rng["BTN"] * live
        active = np.nonzero((r0 > 0) | (r1 > 0))[0]
        ctx = Ctx(active, Config(river_buckets=16))
        ctx.root_len, ctx.root_reach = 3, (r0[active], r1[active])
        ids, E, kap = ctx.bucket_table(board, set(board), 16)
        self.assertTrue(np.allclose(E[1], -E[0].T) and np.allclose(kap[1], kap[0].T))
        self.assertTrue((kap >= 0).all() and (kap <= 1 + 1e-6).all() and (np.abs(E[0]) <= kap[0] + 1e-6).all())
        for q, r in ((0, r0[active]), (1, r1[active])):
            self.assertTrue(((ids[q] < 16) == (r > 0)).all())
            mass = np.bincount(ids[q][r > 0], weights=r[r > 0], minlength=16)
            self.assertLess(mass.max() / mass.sum(), 2.5 / 16)
        # stronger buckets win more: the top bucket beats the bottom one against every opponent bucket
        self.assertTrue((E[0][-1] >= E[0][0]).all())

    def test_bucket_game_zero_sum(self):
        # both players' values under the average strategies sum to zero, and the best responses gain (exploitability > 0)
        r_oop, r_ip = parse_range("22+, A2s+, K9s+, QTs+, JTs, ATo+, KJo+"), parse_range("44+, A5s+, KTs+, QJs, AJo+, KQo")
        cfg = Config(river_buckets=8, turn_buckets=True, bucket_river_step=3)
        sol = solve(cards("Kd7c2h"), r_oop, r_ip, pot=6, stack=60, cfg=cfg, time_budget=1e9, max_iters=30)
        s = sol.solver
        a, b = sol.r0[None, :].astype(np.float32), sol.r1[None, :].astype(np.float32)
        v0 = float((s._cfr(sol.root, 0, a, b, update=False, use_avg=True)[0] * sol.r0).sum())
        v1 = float((s._cfr(sol.root, 1, a, b, update=False, use_avg=True)[0] * sol.r1).sum())
        pairs = float((s.compat_sum(b)[0] * sol.r0).sum())
        self.assertLess(abs(v0 + v1) / pairs, 1e-4)
        self.assertGreater(sol.exploitability(), 0)
        # exploit mode's pool lock reaches the bucket nodes too (a bucket's percentile is its place in the range)
        from gto.exploit import PoolLock
        lock, seen = PoolLock(sol.ctx, 1), []

        def spy(node, strat, reach, sel=None):
            seen.append(node.bucketed)
            out = lock(node, strat, reach, sel)
            self.assertEqual(out.shape, strat.shape)
            return out
        evs = s.node_action_br_evs(sol.root, sol.r0, sol.r1, lock=spy)
        self.assertTrue(np.isfinite(evs).all() and any(seen) and not all(seen))

    def test_turn_classes_cover_every_turn(self):
        ctx = Ctx(np.arange(N_FULL), Config())
        for flop, n_cls in (("Kd7c2h", 13), ("9h8h6c", 13 + 11), ("Kh7h2h", 13 + 10)):
            f = tuple(cards(flop))
            cls = ctx.turn_classes(f, set(f))
            self.assertEqual(len(cls), n_cls, flop)
            every = sorted(c for _, cs in cls for c in cs)
            self.assertEqual(every, [c for c in range(52) if c not in f])
            for played, cs in cls:
                self.assertIn(played, cs)
                self.assertEqual(len({c // 4 for c in cs}), 1)          # one rank
                suits = [c % 4 for c in cs]
                flushy = {x for x in range(4) if [c % 4 for c in f].count(x) >= 2}
                self.assertTrue(len(cs) == 1 or not flushy & set(suits))  # a flush suit's cards stand alone

    def test_river_play_without_bets_is_the_checked_down_leaf(self):
        # with no river bets the river played on the turn's boards is a check-down: the same EVs as the equity leaf
        hero, vr = cards("AhAs"), parse_range("KK, 88, QQ, JTs, 76s, A5s")
        turn_only = {"flop": [], "turn": [0.75], "river": []}
        for tc in (False, True):
            evs = []
            for rp in (False, True):
                cfg = Config(bet_sizes=turn_only, raise_sizes={s: [] for s in ("flop", "turn", "river")}, allin_spr=0,
                             river_play=rp, turn_classes=tc, later_sizes=None, n_turns=49)
                sol = solve(cards("Kh8d5c"), parse_range("AA, KQs"), vr, pot=10, stack=100, time_budget=1e9,
                            max_iters=30, cfg=cfg)
                evs.append(sol.evs[:, sol.ctx.active])
            self.assertTrue(np.allclose(evs[0], evs[1], atol=1e-3))
        # and checked down from the flop with every turn, a turn class each is close to the exact pot x equity
        cfg = Config(bet_sizes={s: [] for s in ("flop", "turn", "river")}, raise_sizes={s: [] for s in ("flop", "turn", "river")},
                     allin_spr=0, later_sizes=None)
        sol = solve(cards("Kh8d5c"), parse_range("AA"), vr, pot=10, stack=100, time_budget=1e9, max_iters=3, cfg=cfg)
        self.assertAlmostEqual(sol.for_combo(tuple(hero))["check"][1], 10 * _exact_equity(cards("Kh8d5c"), hero, vr), delta=0.15)


class TestExploit(unittest.TestCase):
    """Exploit mode: opponents play like a typical player pool (gto/exploit.py) and hero best-responds."""

    def test_pool_lock_overfolds_weakest_at_every_size(self):
        from gto.exploit import PoolLock
        from gto.solver import Ctx
        board = tuple(cards("Qs7c3d8h2s"))
        live = np.array([i for i, (a, b) in enumerate(ALL_COMBOS) if a not in board and b not in board])
        ctx = Ctx(live, Config())
        lock = PoolLock(ctx, 1)
        n = len(live)
        class Node:   # player 1 faces a bet of `to_call` into `pot`
            def __init__(self, to_call, pot, b=board):
                self.boards, self.names, self.player = [b], ["fold", "call"], 1
                self.bets, self.pot_now = (to_call, 0.0), pot + to_call
        strat = np.full((1, 2, n), 0.5, dtype=np.float32)
        reach = np.ones((1, n))
        big = lock(Node(15.0, 10.0), strat, reach)
        fold = float(big[0, 0].mean())
        self.assertAlmostEqual(fold - 0.5, 0.075, places=2)      # a 1.5x-pot river overbet: 7.5% more of the range folds
        # a third-pot river bet: 20% more folds (the pool over-folds small river bets most)
        self.assertAlmostEqual(float(lock(Node(3.3, 10.0), strat, reach)[0, 0].mean()) - 0.5, 0.20, places=2)
        pct = lock.pct(board)
        self.assertGreater(big[0, 0][pct < 0.2].mean(), big[0, 0][pct > 0.8].mean())   # ...the weakest calls
        flop = tuple(cards("Qs7c3d"))
        small = PoolLock(Ctx(np.array([i for i, (a, b) in enumerate(ALL_COMBOS) if a not in flop and b not in flop]), Config()), 1)
        n2 = len(small.ctx.active)
        out = small(Node(2.0, 6.0, flop), np.full((1, 2, n2), 0.5, dtype=np.float32), np.ones((1, n2)))
        # a third-pot flop bet: 8.5% more folds (population data: the pool over-folds small c-bets, +7-10 points)
        self.assertAlmostEqual(float(out[0, 0].mean()) - 0.5, 0.085, places=2)
        turn = tuple(cards("Qs7c3d8h"))
        tl = PoolLock(Ctx(np.array([i for i, (a, b) in enumerate(ALL_COMBOS) if a not in turn and b not in turn]), Config()), 1)
        n3 = len(tl.ctx.active)
        out = tl(Node(2.0, 6.0, turn), np.full((1, 2, n3), 0.5, dtype=np.float32), np.ones((1, n3)))
        self.assertAlmostEqual(float(out[0, 0].mean()) - 0.5, 0.105, places=2)   # the turn: regulars over-fold a third-pot bet ~10.5 points
        raised = Node(10.0, 10.0); raised.bets = (20.0, 10.0)    # facing a raise of its own bet: no shift
        self.assertAlmostEqual(float(lock(raised, strat, reach)[0, 0].mean()), 0.5, places=5)

    def test_preflop_lines_read_as_real_ranges(self):
        """Lines whose chart or pool range was face-up: every open-limp (the small blind too, in both modes;
        heads-up the pool's was T3o and T4o) reads as LIMP; an over-limp keeps at least the limp range less
        what would iso-raise (the button's chart over-limp was 15 combos); an open-jam is the push/fold range,
        not the whole open; a call of an all-in is by equity against it (the big blind called a 100bb jam
        with its flats, Q2s and 43s, and a 3bet jam was called without AA or KK)."""
        from gto.advisor import preflop_ranges
        from gto.cards import COMBO_INDEX
        idx = lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        C = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        T6 = pf.Table(pf.SEATS[6])
        for line, T in (([C("SB", "call")], pf.Table(pf.SEATS[2])),
                        ([C("UTG", "fold"), C("HJ", "fold"), C("CO", "fold"), C("BTN", "fold"), C("SB", "call")], T6)):
            for pe in (None, "BB"):
                np.testing.assert_allclose(preflop_ranges(line, T, 100, pe)["SB"], pf.LIMP)
        over = preflop_ranges([C("UTG", "call"), C("HJ", "fold"), C("CO", "fold"), C("BTN", "call")], T6, 100, "BB")["BTN"]
        self.assertGreater(over.sum(), 40)
        self.assertEqual(over[idx("AhAd")], 0.0)
        jam = [C("UTG", "fold"), C("HJ", "fold"), C("CO", "all-in"), C("BTN", "fold"), C("SB", "fold"), C("BB", "call")]
        r = preflop_ranges(jam, T6, 100)
        self.assertLess(r["CO"].sum(), 0.25 * preflop_ranges(jam[:2] + [C("CO", "raise", 2.5)], T6, 100)["CO"].sum())
        self.assertEqual(r["BB"][idx("AhAd")], 1.0)
        self.assertEqual(r["BB"][idx("Qh2h")], 0.0)
        self.assertLess(r["BB"].sum(), 60)
        # an iso-raiser facing the limper's re-raise defends as the chart does, not as an opener facing a 3bet
        lrr = [C("UTG", "fold"), C("HJ", "fold"), C("CO", "call"), C("BTN", "raise", 4), C("SB", "fold"),
               C("BB", "fold"), C("CO", "raise", 14), C("BTN", "call")]
        np.testing.assert_allclose(preflop_ranges(lrr, T6, 100, "SB")["BTN"], preflop_ranges(lrr, T6, 100)["BTN"])
        call3 = preflop_ranges([C("UTG", "fold"), C("HJ", "fold"), C("CO", "raise", 2.5), C("BTN", "all-in"),
                                C("SB", "fold"), C("BB", "fold"), C("CO", "call")], T6, 100)["CO"]
        self.assertEqual(call3[idx("AhAd")], 1.0)
        self.assertEqual(call3[idx("9h8h")], 0.0)

    def test_multiway_narrowing_rules(self):
        """Street ranking by percentile over every turn card (24 random runouts of raw strength put K5s
        and K9o above AA on K72); a call of an all-in keeps the hands that would raise (top set was 0.1);
        a check-raise is read from the street's starting range (the check had cut 88 and 44 to 0.3); a
        call keeps its raising hands at SLOWPLAY; a villain who raised postflop or bet the pot is
        'committed' for the pool's over-folds."""
        from gto.advisor import _street_info, _narrow_street, replay_hand, preflop_ranges, _committed, SLOWPLAY
        from gto.cards import COMBO_INDEX
        idx = lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        _, rank, _ = _street_info(tuple(cards("Kd7c2h")))
        pos = {int(j): i for i, j in enumerate(rank)}
        self.assertLess(pos[idx("AhAs")], pos[idx("Ks5s")])
        self.assertLess(pos[idx("AhAs")], pos[idx("Kh9c")])
        R = lambda p, a, x=None: dict({"pos": p, "action": a}, **({"to": x} if a == "raise" else {"amount": x} if x else {}))
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "call"), R("SB", "fold"), R("BB", "call")]
        def narrowed(board, flop):
            ranges = dict(preflop_ranges(pre, T, 100))
            st = replay_hand(T, 100, {"preflop": pre, "flop": flop}, 2)[1]
            _narrow_street(ranges, cards(board), st, 100)
            return ranges
        r = narrowed("Ks9d4c", [R("BB", "all-in"), R("CO", "call")])
        self.assertGreater(r["CO"][idx("KhKd")], 0.9 * preflop_ranges(pre, T, 100)["CO"][idx("KhKd")])
        r = narrowed("Kd8c4h", [R("BB", "check"), R("CO", "bet", 2.8), R("BTN", "call"), R("BB", "raise", 13)])
        before = preflop_ranges(pre, T, 100)["BB"]
        self.assertGreater(r["BB"][idx("8h8s")], 0.9 * before[idx("8h8s")])
        r = narrowed("Kd7c2h", [R("BB", "check"), R("CO", "bet", 2.8), R("BTN", "call")])
        btn = preflop_ranges(pre, T, 100)["BTN"]
        self.assertGreaterEqual(r["BTN"][idx("7h7s")], SLOWPLAY * btn[idx("7h7s")] - 1e-9)
        sts = replay_hand(T, 100, {"preflop": pre, "flop": [R("BB", "check"), R("CO", "bet", 2.8), R("BTN", "raise", 10)]}, 2)[1:]
        self.assertTrue(_committed("BTN", sts))
        self.assertFalse(_committed("CO", sts))
        sts = replay_hand(T, 100, {"preflop": pre, "flop": [R("BB", "check"), R("CO", "bet", 9)]}, 2)[1:]
        self.assertTrue(_committed("CO", sts))

    def test_sweep_regressions(self):
        """From a sweep of obvious spots: (1) multiway, a raise's folds priced on what each opponent must
        add: CO's 85bb turn bet folded to a 12.5bb jam as if it cost 97.5, so 98s air jammed for +2.9;
        (2) heads-up, a river jam the solve never makes is its strongest hands (kept as solved it was 0
        combos; as the rules' range, half bluffs, and a hand playing the board called 99 into 2)."""
        R = lambda p, a, x=None: dict({"pos": p, "action": a}, **({"to": x} if a == "raise" else {"amount": x} if x else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        adv = Advisor(time_budget=1, prior_budget=1)
        adv.mw_trials = MW_TRIALS
        srp = F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "call"), R("SB", "fold"), R("BB", "call")]
        out = adv.advise({"players": 6, "hero": "BTN", "hole": "8s9s", "board": "Tc2h7hAc", "stack": 100, "preflop": srp,
                          "flop": [R("BB", "check"), R("CO", "check"), R("BTN", "check")],
                          "turn": [R("BB", "check"), R("CO", "bet", 85)]})
        self.assertEqual(out["recommended"], "fold", out["actions"])
        out = Advisor(time_budget=2, prior_budget=1).advise(
            {"players": 6, "hero": "BB", "hole": "TdQh", "board": "7sJd4s4h4d", "stack": 100,
             "preflop": F("UTG", "HJ", "CO", "BTN") + [R("SB", "call"), R("BB", "check")],
             "flop": [R("SB", "check"), R("BB", "check")], "turn": [R("SB", "check"), R("BB", "check")],
             "river": [R("SB", "all-in")]})
        self.assertEqual(out["recommended"], "fold", out["actions"])
        self.assertGreater(out["villain_range_combos"], 0.5)

    def test_pool_preflop_shows_chart_share(self):
        """Exploit mode preflop plays a chart mix one way (55 flats a CO open 100%, the chart 40%): each
        action carries the chart's share as gto_freq, as postflop shows the equilibrium's."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        st = {"players": 6, "hero": "BTN", "hole": "5h5d", "board": "", "stack": 100, "opponents": "pool",
              "preflop": [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5)]}
        acts = {a["action"]: a for a in Advisor().advise(st)["actions"]}
        gto = {a["action"]: a["freq"] for a in Advisor().advise(dict(st, opponents="gto"))["actions"]}
        for k, a in acts.items():
            self.assertAlmostEqual(a["gto_freq"], gto[k])
        self.assertGreater(acts["call"]["freq"], acts["call"]["gto_freq"])

    def test_slowplay_carries_and_passive_top_up(self):
        """A hand a call slowplayed keeps SLOWPLAY_AGAIN on the next street's call, not another SLOWPLAY
        (three calls left an overcaller TT-55, KJs at 0.07); a thin passive line is topped up with hands
        like its own, not the strongest (an over-limp got AA)."""
        from gto.advisor import _narrow_street, replay_hand, preflop_ranges, SLOWPLAY, SLOWPLAY_AGAIN
        from gto.cards import COMBO_INDEX
        idx = lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        R = lambda p, a, x=None: dict({"pos": p, "action": a}, **({"to": x} if a == "raise" else {"amount": x} if x else {}))
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "call"), R("SB", "fold"), R("BB", "call")]
        streets = {"preflop": pre, "flop": [R("BB", "check"), R("CO", "bet", 2.8), R("BTN", "call"), R("BB", "fold")],
                   "turn": [R("CO", "bet", 6), R("BTN", "call")]}
        rep = replay_hand(T, 100, streets, 3)
        ranges, slow = dict(preflop_ranges(pre, T, 100)), {}
        _narrow_street(ranges, cards("Kd7c2h"), rep[1], 100, slow=slow)
        flop = ranges["BTN"].copy()
        rep[2].players_start = ["CO", "BTN", "BB"]   # (narrowed as if multiway, to reach the rule)
        _narrow_street(ranges, cards("Kd7c2h5s"), rep[2], 100, slow=slow)
        j = idx("7h7s")
        self.assertTrue(slow["BTN"][j])          # (set of sevens: in the raise band, slowplayed on the flop)
        self.assertGreaterEqual(ranges["BTN"][j], SLOWPLAY_AGAIN * flop[j] - 1e-9)
        lean = np.zeros(1326); lean[idx("7h6h")] = 1.0
        out, _ = pf.thin_line(lean, lean, np.ones(1326), passive=True)
        self.assertEqual(out[idx("AhAd")], 0.0)
        self.assertGreater(pf.thin_line(lean, lean, np.ones(1326))[0][idx("AhAd")], 0.0)

    def test_sweep_regressions_2(self):
        """Second sweep: AA facing a 3bet and a cold 4bet folded half the time (half the cold-vs-3bet chart),
        now COLD_VS_4BET; behind two checks the button's small-bet value took nearly its whole range (bet
        93%) until SMALL_VALUE_MAX; a 96%-pot bet marks its maker committed (COMMIT_BET)."""
        from gto.advisor import _bet_ranges, _street_info, preflop_ranges, _share, _committed, replay_hand
        R = lambda p, a, x=None: dict({"pos": p, "action": a}, **({"to": x} if a == "raise" else {"amount": x} if x else {}))
        pre = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 8), R("BTN", "raise", 20), R("SB", "fold"), R("BB", "fold")]
        for hole in ("AhAs", "KhKs"):
            out = Advisor().advise({"players": 6, "hero": "UTG", "hole": hole, "board": "", "stack": 100, "preflop": pre})
            self.assertEqual(next(a["freq"] for a in out["actions"] if a["action"] == "fold"), 0.0, out["actions"])
        # at 20bb the same line (to 11) jams the continuing hands: a push/fold equilibrium had KK jamming 25%,
        # and a call of 9 with 9 behind is committed anyway
        short = [R("UTG", "raise", 2.2), R("HJ", "fold"), R("CO", "raise", 5), R("BTN", "raise", 11), R("SB", "fold"), R("BB", "fold")]
        for hole in ("KhKs", "QhQs", "AhKd"):
            out = Advisor().advise({"players": 6, "hero": "UTG", "hole": hole, "board": "", "stack": 20, "preflop": short})
            self.assertEqual(out["recommended"], "all-in", out["actions"])
        T = pf.Table(pf.SEATS[6])
        srp = [R("UTG", "fold"), R("HJ", "raise", 2.5), R("CO", "fold"), R("BTN", "call"), R("SB", "fold"), R("BB", "call")]
        rng = preflop_ranges(srp, T, 100)
        sc, rank, bluff = _street_info(tuple(cards("9s5d2h")))
        live = lambda r: r * (sc >= 0)
        bg, sm = _bet_ranges(live(rng["BTN"]), 2, 8.0, 5.3, 2.6, rank, bluff, [live(rng["BB"]), live(rng["HJ"])])
        self.assertLess(_share(live(rng["BTN"]), bg | sm), 0.8)
        sts = replay_hand(T, 100, {"preflop": srp, "flop": [R("BB", "bet", 7.7)]}, 2)[1:]
        self.assertTrue(_committed("BB", sts))
        # a raise into two players carries half the heads-up bluffs
        from gto.advisor import _aggro
        r = live(rng["BTN"])
        one, two = _aggro(r, 0.1, 20, 10, rank, bluff, 1), _aggro(r, 0.1, 20, 10, rank, bluff, 2)
        self.assertAlmostEqual(_share(r, two) - 0.1, (_share(r, one) - 0.1) / 2, delta=0.01)

    def test_check_range_not_split_ties(self):
        """A range that is mostly one nut hand (57% KQ on AhJdTc5s4h, 99 and 88 the rest) checks with little of
        it: the value a check rules out was cut by rank order, which splits hands of equal strength, and with
        the small bet's value capped (SMALL_VALUE_MAX) 53% of the check range was KQ; the cut now takes every
        hand as strong as its weakest, so KQ is only the off-rule share (NARROW_SOFT, at most OFF_RULE_MAX)."""
        from gto.advisor import _bet_ranges, _street_info, _soften, OFF_RULE_MAX
        from gto.ranges import remove_blockers
        board = tuple(cards("AhJdTc5s4h"))
        sc, rank, bluff = _street_info(board)
        opp = remove_blockers(parse_range("KQ,99,88"), board)
        hero = remove_blockers(parse_range("AA,JJ,TT,55,44,AJ,AK,AQ,KK,QQ,KQ"), board)
        kq = parse_range("KQ")
        _, _, drop = _bet_ranges(opp, 2, 20.0, 13.2, 6.6, rank, bluff, [opp, hero], drop=True, sc=sc)
        chk = opp * _soften(opp, ~drop)
        self.assertLessEqual((chk * kq).sum() / chk.sum(), OFF_RULE_MAX + 1e-9)
        self.assertTrue(drop[kq > 0].all())   # every KQ alike
        # without strengths (sc) the old cut
        _, _, drop0 = _bet_ranges(opp, 2, 20.0, 13.2, 6.6, rank, bluff, [opp, hero], drop=True)
        chk0 = opp * _soften(opp, ~drop0)
        self.assertGreater((chk0 * kq).sum() / chk0.sum(), 0.5)

    def test_rare_jam_is_stable(self):
        """A flop jam the solve rarely makes keeps the solve's hands with the rules' value added, every run:
        kept alone when its combos sat just above a threshold, AK called it at +48 bb one run in four."""
        R = lambda p, a, x=None: dict({"pos": p, "action": a}, **({"to": x} if a == "raise" else {"amount": x} if x else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        st = {"players": 6, "hero": "BB", "hole": "AsKd", "board": "Kc7d2s", "stack": 100,
              "preflop": F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)] + F("SB") + [R("BB", "call")],
              "flop": [R("BB", "check"), R("BTN", "all-in")]}
        # (and at every budget: the solve's own jamming hands are the ones it still jams now, not an average
        # that keeps its uniform start: K8o-KQo jammed 18-45% on average at 1-2.5 s, and AK called at +6 to +19)
        for tb in (0.5, 1, 1, 2.5):
            res = Advisor(time_budget=tb, prior_budget=0.5).advise(st)
            self.assertEqual(res["recommended"], "fold")
            self.assertLess(next(a["ev_bb"] for a in res["actions"] if a["action"] == "call"), -5)

    def test_rare_check_by_value_range_keeps_range(self):
        """BB 3bets, overbets the flop 150% (rare: the rules' value, JJ+ and sets) and checks the turn, which the
        rules have none of that range doing: the range stays as it was, not 0 combos (where every hand of ours
        was worth exactly half the pot, check and jam tied)."""
        from gto.advisor import _soften
        r = np.zeros(1326); r[:10] = 1.0
        self.assertTrue(np.all(_soften(r, np.zeros(1326, dtype=bool)) == 1.0))
        R = lambda p, a, x=None: dict({"pos": p, "action": a}, **({"to": x} if a == "raise" else {"amount": x} if x else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        st = {"players": 6, "hero": "BTN", "hole": "Qd5c", "board": "9sJd3d6c", "stack": 100,
              "preflop": F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)] + F("SB") + [R("BB", "raise", 11), R("BTN", "call")],
              "flop": [R("BB", "bet", 33.8), R("BTN", "call")], "turn": [R("BB", "check")]}
        # (the page's Fast tree, one turn size: there the overbet is rare)
        fast = Config(n_turns=3, n_rivers=8, leaf_turns=4, leaf_rivers_flop=4, n_rivers_from_flop=4,
                      bet_sizes={"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]},
                      raise_sizes={"flop": [0.5], "turn": [0.75], "river": [0.75]}, river_play=False, turn_classes=False)
        res = Advisor(fast, time_budget=1, prior_budget=0.5).advise(st)
        self.assertGreater(res["villain_range_combos"], 10)
        ev = {a["action"]: a["ev_bb"] for a in res["actions"]}
        self.assertGreater(ev["check"], ev["all-in"] + 5)

    def test_multiway_line_filled_out(self):
        """A line the multiway rules leave nearly face-up is filled out (as heads-up streets are): HJ's pot-sized
        turn bet into two players in a 60bb 3bet pot kept 1.9 combos, and its river check 0.5."""
        R = lambda p, a, x=None: dict({"pos": p, "action": a}, **({"to": x} if a == "raise" else {"amount": x} if x else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        st = {"players": 6, "hero": "BTN", "hole": "3h8c", "board": "Qc2dTc9c2c", "stack": 60,
              "preflop": F("UTG") + [R("HJ", "raise", 2.5), R("CO", "call"), R("BTN", "raise", 10)] + F("SB", "BB")
              + [R("HJ", "call"), R("CO", "call")],
              "flop": [R("HJ", "check"), R("CO", "check"), R("BTN", "check")],
              "turn": [R("HJ", "bet", 31.5), R("CO", "fold"), R("BTN", "call")], "river": [R("HJ", "check")]}
        res = Advisor(time_budget=1, prior_budget=0.5).advise(st)
        self.assertGreater(res["villain_range_combos"], 3)
        self.assertTrue(any("filled out" in n for n in res["notes"]))

    def test_rare_turn_reraise_keeps_solved_range(self):
        """BTN 3bets AQ, HJ calls; Qs9dKc checks through; on the Td turn HJ bets 1, BTN raises to 12 and HJ
        min-reraises to 23. The solve rarely has HJ raise there: its range stays the solve's (straights,
        sets), not the rules' (99, 66 and air), and the spot is solved again from there, so AQ folds. Read
        by the rules off the barely-converged tree it jammed for +31.8 bb on some runs and folded on others."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        pre = [R("UTG", "fold"), R("HJ", "raise", 2), R("CO", "fold"), R("BTN", "raise", 6), R("SB", "fold"),
               R("BB", "fold"), R("HJ", "call")]
        st = {"players": 6, "hero": "BTN", "hole": "QhAd", "board": "Qs9dKcTd", "stack": 100, "preflop": pre,
              "flop": [R("HJ", "check"), R("BTN", "check")],
              "turn": [R("HJ", "bet", 1), R("BTN", "raise", 12), R("HJ", "raise", 23)]}
        out = Advisor(time_budget=3, prior_budget=2).advise(st)
        self.assertEqual(out["recommended"], "fold", out["actions"])
        self.assertTrue(all(a["ev_bb"] < 0.5 for a in out["actions"] if a["action"] != "fold"), out["actions"])

    def test_pool_limper_overbet_and_call_folds_bottom_pair(self):
        """Exploit mode, CO limps, SB completes, and on Q43 CO bets 23.5 into 3bb and SB calls: the big
        blind's K3o folds. The pool's limps alone (K7o, K8o, Q9o, K5s, Q8s) made CO's bet Q8/Q9 or
        K-high, so bottom pair overcalled with "33%" equity; the limp is read as the LIMP range now."""
        R = lambda p, a, amt=None: dict({"pos": p, "action": a}, **({"amount": amt} if amt else {}))
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "call"), R("BTN", "fold"), R("SB", "call"), R("BB", "check")]
        st = {"players": 6, "hero": "BB", "hole": "3dKh", "board": "3s4cQh", "stack": 100, "preflop": pre,
              "flop": [R("SB", "check"), R("BB", "check"), R("CO", "bet", 23.5), R("SB", "call")], "opponents": "pool"}
        adv = Advisor(time_budget=1, prior_budget=1)
        adv.mw_trials = 1000
        out = adv.advise(st)
        self.assertEqual(out["recommended"], "fold", out["actions"])
        self.assertLess(out["equity_vs_field"], 0.25)

    def test_pool_best_response_ties_within_noise(self):
        """Exploit mode: a short solve's best-response EVs that sit within its noise of the best are tied and
        follow the equilibrium mix (AQo checked 100% at 6.45 over bets at 6.40 in a solve 3.1% of the pot
        from settled); a settled solve still picks the clear best."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "fold"), R("BTN", "raise", 2.5), R("SB", "fold"), R("BB", "call")]
        st = {"players": 6, "hero": "BTN", "hole": "AhQd", "board": "Qs8d3c", "stack": 100, "preflop": pre,
              "flop": [R("BB", "check")], "opponents": "pool"}
        rough = Advisor(time_budget=0.2, prior_budget=0.2).advise(st)
        from gto.advisor import BR_TIE_MAX   # (ties are capped at BR_TIE_MAX of the pot, as the advisor does)
        tol = min(max(0.005, rough["solver"]["exploitability_pct"] / 100), BR_TIE_MAX) * rough["pot_bb"]
        best = max(a["ev_bb"] for a in rough["actions"])
        for a in rough["actions"]:
            if a["ev_bb"] >= best - tol and a["gto_freq"] > 0.05:
                self.assertGreater(a["freq"], 0.0, rough["actions"])       # a tied action keeps its GTO share

    def test_pool_preflop_flats_and_3bets_more(self):
        """Population HUD stats (GGPoker 6-max NL50-500: 3bet 8.9% against the charts' 8.7%, VPIP 28.75
        against 24.7; scripts/calibrate_pool.py): facing an open the pool 3bets POOL_3BET_SCALE (1.13) x the chart's share
        (from its strongest calls) and flats POOL_EXTRA_ENTER of its folds more (its strongest)."""
        from gto.advisor import preflop_ranges
        from gto import exploit
        T = pf.Table(pf.SEATS[6])
        F = lambda *ps: [{"pos": p, "action": "fold"} for p in ps]
        line = lambda act: F("UTG", "HJ") + [{"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": act}]
        gto_call, pool_call = preflop_ranges(line("call"), T, 100)["BTN"], preflop_ranges(line("call"), T, 100, "BB")["BTN"]
        gto_3b, pool_3b = preflop_ranges(line("raise"), T, 100)["BTN"], preflop_ranges(line("raise"), T, 100, "BB")["BTN"]
        # POOL_3BET_SCALE: the extra comes from calls (the published BTN vs CO chart calls 2.9% of hands
        # against 12.2% 3bets, enough for the extra 3bets)
        self.assertAlmostEqual(pool_3b.sum() / gto_3b.sum(), exploit.POOL_3BET_SCALE, delta=0.01)
        self.assertGreater(pool_call.sum(), 2 * gto_call.sum())
        # the extra 3bets are the strongest calls: the pool's 3bet range is the chart's plus better hands
        self.assertTrue(np.all(pool_3b >= gto_3b - 1e-9))
        # hero's own range is the chart's
        self.assertAlmostEqual(preflop_ranges(line("call"), T, 100, "BTN")["BTN"].sum(), gto_call.sum())

    def test_pool_preflop_extra_vpip_is_spread_over_limps_and_flats(self):
        """The pool's extra VPIP (28.75 against the charts' 24.7) isn't split by line in any data found, so
        it is spread evenly: first in it enters (POOL_EXTRA_OPEN of them open-raising, the strongest, the
        rest open-limping), and facing an open it flats, with the same share (POOL_EXTRA_ENTER) of what the
        chart folds, strongest first; the big blind's defence is the chart's (the small blind's extra limps:
        the next test)."""
        from gto.advisor import preflop_ranges, preflop_strategy, pool_strategy
        from gto.cards import COMBO_INDEX
        from gto.ranges import N_COMBOS
        from gto import exploit
        T = pf.Table(pf.SEATS[6])
        q, o = exploit.POOL_EXTRA_ENTER, exploit.POOL_EXTRA_OPEN
        F = lambda *ps: [{"pos": p, "action": "fold"} for p in ps]
        ones = {p: np.ones(N_COMBOS) for p in T}
        chart = preflop_strategy("HJ", F("UTG"), T, 100)
        pool = pool_strategy(chart, "HJ", F("UTG"), ones)
        self.assertAlmostEqual(pool["call"].mean(), q * (1 - o) * chart["fold"].mean(), places=4)
        self.assertAlmostEqual((pool["raise"] - chart["raise"]).mean(), q * o * chart["fold"].mean(), places=4)
        self.assertTrue(np.all(pool["raise"] >= chart["raise"] - 1e-9))
        # the extra hands are the strongest folds: A9o (by equity against any hand) enters before 72o,
        # and the extra opens are stronger than the limps (A9o raises before K8o limps)
        idx = lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        extra = pool["call"] + pool["raise"] - chart["raise"]
        self.assertGreater(extra[idx("Ah9d")], extra[idx("7h2d")])
        self.assertGreaterEqual(pool["raise"][idx("Ah9d")] - chart["raise"][idx("Ah9d")], pool["raise"][idx("Kh8d")] - chart["raise"][idx("Kh8d")])
        # but an observed limp reads as the chart's LIMP range: the pool's limps alone are a face-up band
        limp = preflop_ranges(F("UTG") + [{"pos": "HJ", "action": "call"}], T, 100, "BB")["HJ"]
        np.testing.assert_allclose(limp, pf.LIMP)
        # behind a limper the chart's over-limps stand
        behind = preflop_strategy("CO", F("UTG") + [{"pos": "HJ", "action": "call"}], T, 100)
        self.assertIs(pool_strategy(behind, "CO", F("UTG") + [{"pos": "HJ", "action": "call"}], ones), behind)
        # facing an open: the extra flats are the same share of the folds; the big blind is the chart's
        line = F("UTG", "HJ") + [{"pos": "CO", "action": "raise", "to": 2.5}]
        ranges = dict(ones, CO=preflop_ranges(line, T, 100)["CO"])
        btn = preflop_strategy("BTN", line, T, 100)
        pb = pool_strategy(btn, "BTN", line, ranges)
        self.assertAlmostEqual(pb["fold"].mean(), (1 - q) * btn["fold"].mean(), places=4)
        bb_line = line + F("BTN", "SB")
        bb = preflop_strategy("BB", bb_line, T, 100)
        self.assertAlmostEqual(pool_strategy(bb, "BB", bb_line, ranges)["fold"].mean(), bb["fold"].mean(), places=6)

    def test_pool_preflop_small_blind_limps_its_weakest_opens(self):
        """First in from the small blind the pool open-limps POOL_SB_LIMP of its range out of the chart's
        weakest opens (whole-sample CoinPoker NL50-NL100 2025: SB open-limps 8.5-9.7%, other seats 1.5-1.6%),
        on top of its extra limps; heads-up (the small blind is the button) it keeps the chart's opens."""
        from gto.advisor import preflop_strategy, pool_strategy
        from gto.cards import COMBO_INDEX
        from gto.ranges import N_COMBOS
        from gto import exploit
        F = lambda *ps: [{"pos": p, "action": "fold"} for p in ps]
        T = pf.Table(pf.SEATS[6])
        ones = {p: np.ones(N_COMBOS) for p in T}
        chart = preflop_strategy("SB", F("UTG", "HJ", "CO", "BTN"), T, 100)
        pool = pool_strategy(chart, "SB", F("UTG", "HJ", "CO", "BTN"), ones)
        q, o = exploit.POOL_EXTRA_ENTER, exploit.POOL_EXTRA_OPEN
        extra = q * chart["fold"].mean()
        self.assertAlmostEqual(pool["call"].mean(), exploit.POOL_SB_LIMP + (1 - o) * extra, places=4)
        self.assertAlmostEqual(pool["raise"].mean(), chart["raise"].mean() + o * extra - exploit.POOL_SB_LIMP, places=4)
        self.assertGreater(pool["call"].mean(), 0.08)
        # the limps that were opens are the chart's weakest opens: aces still raise
        idx = lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        self.assertAlmostEqual(pool["raise"][idx("AhAd")], chart["raise"][idx("AhAd")], places=6)
        # other seats limp only their extra hands
        hj = pool_strategy(preflop_strategy("HJ", F("UTG"), T, 100), "HJ", F("UTG"), ones)
        self.assertLess(hj["call"].mean(), 0.04)
        # heads-up the small blind keeps the chart's opens
        T2 = pf.Table(pf.SEATS[2])
        hu = preflop_strategy("SB", [], T2, 100)
        if "call" not in hu and "fold" in hu and "raise" in hu:
            phu = pool_strategy(hu, "SB", [], {p: np.ones(N_COMBOS) for p in T2})
            self.assertGreaterEqual(phu["raise"].mean(), hu["raise"].mean() - 1e-9)

    def test_pool_preflop_opener_defends_3bets_more(self):
        """Population fold to 3bet 48.25% against the charts' 54.2%: the opener facing a 3bet calls with
        POOL_3BET_DEFEND more of its range (its strongest folds); its 4bets stay the chart's (the pool's
        tree then 4bets 12.5% against the population's 12.75%)."""
        from gto.advisor import preflop_ranges, preflop_strategy, pool_strategy
        from gto import exploit
        T = pf.Table(pf.SEATS[6])
        F = lambda *ps: [{"pos": p, "action": "fold"} for p in ps]
        three = F("UTG", "HJ") + [{"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "raise", "to": 7.5}] + F("SB", "BB")
        ranges = preflop_ranges(three, T, 100)
        co = ranges["CO"]
        chart = preflop_strategy("CO", three, T, 100)
        pool = pool_strategy(chart, "CO", three, ranges)
        share = lambda v: float((co * v).sum() / co.sum())
        np.testing.assert_allclose(pool["raise"], chart["raise"])
        self.assertAlmostEqual(share(pool["call"]) - share(chart["call"]), exploit.POOL_3BET_DEFEND, places=3)

    def test_pool_advice_best_responds(self):
        adv = Advisor(time_budget=1.0, prior_budget=0.5)
        F = lambda *ps: [{"pos": p, "action": "fold"} for p in ps]
        st = {"players": 6, "hero": "BB", "hole": "QdJd", "board": "Qs7c3d8h2s", "stack": 100,
              "preflop": F("UTG", "HJ", "CO") + [{"pos": "BTN", "action": "raise", "to": 2.5}] + F("SB") + [{"pos": "BB", "action": "call"}],
              "flop": [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 2}, {"pos": "BB", "action": "call"}],
              "turn": [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 6}, {"pos": "BB", "action": "call"}],
              "river": [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 15}]}
        gto, pool = adv.advise(st), adv.advise(dict(st, opponents="pool"))
        self.assertNotIn("gto_freq", gto["actions"][0])
        self.assertTrue(all("gto_freq" in a for a in pool["actions"]))
        # the recommendation is the best EV, or an action within the solve's noise of it (tied actions
        # follow the equilibrium mix)
        best = max(a["ev_bb"] for a in pool["actions"])
        tol = max(0.005, pool["solver"]["exploitability_pct"] / 100) * pool["pot_bb"]
        rec = next(a for a in pool["actions"] if a["action"] == pool["recommended"])
        self.assertGreaterEqual(rec["ev_bb"], best - tol - 1e-6, pool["actions"])
        self.assertAlmostEqual(sum(a["freq"] for a in pool["actions"]), 1.0, places=2)
        # the pool under-bluffs river bets: the best response folds more of the range than GTO
        self.assertGreater(pool["range_action_freq"]["fold"], gto["range_action_freq"]["fold"] + 0.1)

    def test_pool_preflop_advice(self):
        """Facing a pool 4bet (as wide as the chart's: the population 4bets as often) KK still continues;
        opens stay the chart's."""
        adv = Advisor()
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        four = F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "raise", 7.5), R("SB", "fold"), R("BB", "fold"), R("CO", "raise", 22)]
        st = {"players": 6, "hero": "BTN", "hole": "KsKh", "board": "", "stack": 100, "preflop": four}
        fold = lambda r: next(a["freq"] for a in r["actions"] if a["action"] == "fold")
        self.assertLess(fold(adv.advise(dict(st, opponents="pool"))), 0.01)
        opens = dict(st, hero="CO", hole="As9h", preflop=F("UTG", "HJ"))
        strip = lambda acts: [{k: v for k, v in a.items() if k != "gto_freq"} for a in acts]   # (pool shows the chart's share too)
        self.assertEqual(strip(adv.advise(dict(opens, opponents="pool"))["actions"]), adv.advise(opens)["actions"])

    def test_multiway_narrowing_leaves_cached_ranges_alone(self):
        """Multiway narrowing used to write into the cached preflop ranges, so the same request twice gave
        different answers (and the next street started from ranges narrowed twice)."""
        from gto.advisor import preflop_ranges
        pre = [{"pos": "UTG", "action": "raise", "to": 2.5}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "call"},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "fold"}]
        T = pf.Table(pf.SEATS[6])
        before = {p: v.sum() for p, v in preflop_ranges(pre, T, 100).items()}
        st = {"players": 6, "hero": "BTN", "hole": "AhKh", "board": "Ks7d2c9h", "stack": 100, "preflop": pre,
              "flop": [{"pos": "UTG", "action": "bet", "amount": 4}, {"pos": "CO", "action": "call"}, {"pos": "BTN", "action": "call"}],
              "turn": [{"pos": "UTG", "action": "check"}, {"pos": "CO", "action": "check"}]}
        adv = Advisor(time_budget=0.5, prior_budget=0.3)
        r1 = adv.advise(st)
        self.assertEqual({p: v.sum() for p, v in preflop_ranges(pre, T, 100).items()}, before)
        self.assertEqual([a["ev_bb"] for a in r1["actions"]], [a["ev_bb"] for a in adv.advise(st)["actions"]])


if __name__ == "__main__":
    unittest.main()


class TestPoolOverfold(unittest.TestCase):
    def test_overfold_keeps_strong_calls(self):
        """The pool's extra folds to a big bet were a flat share of the range: against a turn jam that
        equilibrium calls with a few percent of the range, every call went, queens full included, and a jam
        was worth exactly the pot (found by scripts/sanity_sweep.py). They now take at most half the calls
        and never the strongest hands. Checked on the lock itself: the spot as found (SB jamming the turn
        of Qd6dQcAc against CO's flat of a UTG open) stopped showing it once the pool's flats changed, since
        how many calls CO has there moves with the preflop pool model."""
        from gto.exploit import PoolLock
        from gto import exploit
        board = tuple(cards("Qd6dQcAc"))
        lock = PoolLock(Ctx(np.array([i for i, (a, b) in enumerate(ALL_COMBOS) if a not in board and b not in board]),
                            Config()), 1)
        n = len(lock.ctx.active)
        pct = lock.pct(board)
        class Node:   # player 1 faces a 2x-pot jam: 7.5% more of the range folds than at equilibrium
            boards, names, player, bets, pot_now = [board], ["fold", "call"], 1, (60.0, 0.0), 90.0
        call = (pct >= 0.96).astype(np.float32)        # equilibrium calls with its top 4% (full houses and up)
        strat = np.stack([1.0 - call, call])[None]
        reach = np.ones((1, n))
        cap, protect = exploit.OVERFOLD_CAP, exploit.OVERFOLD_PROTECT
        try:
            capped = lock(Node(), strat, reach)[0, 1]
            exploit.OVERFOLD_CAP, exploit.OVERFOLD_PROTECT = 1e9, 2.0   # the old rule: a flat share of the range
            uncapped = lock(Node(), strat, reach)[0, 1]
        finally:
            exploit.OVERFOLD_CAP, exploit.OVERFOLD_PROTECT = cap, protect
        self.assertLess(uncapped.sum(), 1e-6)                        # the old rule: every call folds
        self.assertAlmostEqual(capped.sum(), call.sum(), places=3)   # now: the top 10% never folds, so all stay
        # equilibrium calling its top 20%: at most half of the calls fold, never the top 10%
        call = (pct >= 0.8).astype(np.float32)
        out = lock(Node(), np.stack([1.0 - call, call])[None], reach)[0, 1]
        self.assertGreaterEqual(out.sum(), 0.5 * call.sum() - 1e-3)
        self.assertTrue(np.all(out[pct >= exploit.OVERFOLD_PROTECT] > 0.999))


class TestQuickSweep(unittest.TestCase):
    """scripts/sanity_sweep.py --quick runs before each deploy: its spot set (not the sweep itself)."""

    @classmethod
    def setUpClass(cls):
        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        import sanity_sweep
        cls.sw = sanity_sweep
        cls.pre, cls.post = sanity_sweep.quick_jobs(7)

    def test_covers_every_setup_and_kind(self):
        sw = self.sw
        tags = {s[0] for s in sw.SETUPS}
        self.assertEqual({j[0][0] for j in self.pre}, tags)
        self.assertEqual({j[0][0] for j in self.post}, tags)
        self.assertTrue(any(j[4] for j in self.pre), "no exploit-mode preflop nodes")
        hu = {j[2] for j in self.post if not self._multiway(j)}
        self.assertEqual(hu, set(sw.POST_KINDS))
        self.assertEqual({j[2] for j in self.post if self._multiway(j)}, set(sw.MULTIWAY_KINDS))
        self.assertGreaterEqual({j[2] for j in self.post if j[3]}, {"vs_bet", "turn", "river_vs_bet"})
        streets = {len(j[1]["board"]) // 2 for j in self.post}
        self.assertEqual(streets, {3, 4, 5})
        self.assertLess(len(self.pre) + len(self.post), 340)

    def test_rare_lines_covered(self):
        """The hand-picked rare lines (outside the quick preflop trees' reach cut) are all in the quick set:
        every kind, every setup, each line on the charts and built in full, closed pots with flop and
        turn spots (heads-up) or 3-way spots, and the lines ending at a decision checked there."""
        import json
        from collections import defaultdict
        sw = self.sw
        self.assertEqual({k for _, k, _ in sw.RARE_LINES}, set(sw.RARE_KINDS))
        self.assertEqual({t for t, _, _ in sw.RARE_LINES}, {s[0] for s in sw.SETUPS})
        by_tag = {s[0]: s for s in sw.SETUPS}
        rare_pre, rare_post = sw.rare_jobs(7)
        self.assertEqual((rare_pre, rare_post), sw.rare_jobs(7))
        self.assertEqual(self.pre[-len(rare_pre):], rare_pre)
        self.assertEqual(self.post[-len(rare_post):], rare_post)
        pre_lines = {(j[0][0], json.dumps(j[1])) for j in rare_pre}
        post_lines = defaultdict(set)
        for j in rare_post:
            post_lines[(j[0][0], json.dumps(j[1]["preflop"]))].add(j[2])
        closed_kinds, threeway = set(), 0
        for tag, kind, steps in sw.RARE_LINES:
            acts, prefixes, reach, closed = sw.build_line(by_tag[tag], steps)
            self.assertGreater(reach, 0, (tag, kind))
            self.assertEqual(len([a for a in acts if a["action"] != "fold"]),
                             len([s for s in steps if s[1] != "f"]), (tag, kind))
            self.assertIn((tag, json.dumps(acts)), pre_lines, (tag, kind))
            for node in prefixes[-sw.RARE_PRE_NODES:]:
                self.assertIn((tag, json.dumps(node)), pre_lines, (tag, kind))
            if closed:
                closed_kinds.add(kind)
                kinds = post_lines[(tag, json.dumps(acts))]
                self.assertTrue(kinds == set(sw.RARE_POST_KINDS) or kinds == set(sw.MULTIWAY_KINDS), (tag, kind, kinds))
                threeway += kinds == set(sw.MULTIWAY_KINDS)
        self.assertGreaterEqual(closed_kinds, {"3bet pot", "4bet pot", "cold 4bet", "squeeze called", "straddled 3bet pot"})
        self.assertGreaterEqual(threeway, 2)

    def test_rare_line_range_check(self):
        """A closed rare line's range job (no hands) runs the preflop range checks: the flat-then-4bet
        range is filled out, not left thin."""
        sw = self.sw
        setup = sw.SETUPS[0]
        steps = next(s for t, k, s in sw.RARE_LINES if t == setup[0] and k == "flat then 4bet")
        acts = sw.build_line(setup, steps)[0]
        names = {n for n, _ in sw.check_preflop((setup, acts, None, [], False))}
        self.assertNotIn("error", names)
        self.assertNotIn("thin_range", names)
        self.assertIn("filled_range", names)

    def _multiway(self, job):
        from gto.advisor import replay_hand
        setup, state = job[0], job[1]
        return len(replay_hand(self.sw._table(setup), setup[2], {"preflop": state["preflop"]}, 1)[0].active) == 3

    def test_deterministic(self):
        pre, post = self.sw.quick_jobs(7)
        self.assertEqual(pre, self.pre)
        self.assertEqual(post, self.post)

    def test_blocking_flags(self):
        self.assertEqual(self.sw.blocking({"unsettled": [0, 3], "filled_range": [2, 0]}), [])
        self.assertEqual(self.sw.blocking({"unsettled": [0, 3], "ev_is_pot": [0, 1], "error": [0, 0]}),
                         ["ev_is_pot"])


class TestPreflopSubgame(unittest.TestCase):
    def test_converges_and_is_sane(self):
        """The 3bet-pot subgame (pf_solve) converges, premiums never fold (AA and KK call the jam),
        trash never continues, and there are 4bets beyond the value core and non-premium jams."""
        from gto.pf_solve import solve, Spot
        from gto.pf_classes import to_classes, CLASS_INDEX as C
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "raise", 7.5), R("SB", "fold"), R("BB", "fold")]
        rng = pf.replay_ranges(pre, T)
        S = solve(to_classes(rng["CO"]), to_classes(rng["BTN"]), Spot(11.5, 2.5, 7.5, 18.0, 100.0, False), iters=1000)
        self.assertLess(S["exploitability"], 0.05)
        for h in ("AA", "KK"):
            self.assertLess(S["o_fold"][C[h]], 0.01)
            self.assertLess(S["t_fold"][C[h]], 0.01)
            self.assertGreater(S["o_vs_jam"][C[h]], 0.99)
        # the weakest hands the opener opened with fold to the 3bet
        from gto.pf_classes import equity_matrix, PAIR_COMPAT
        wo, wt = to_classes(rng["CO"]), to_classes(rng["BTN"])
        eq = (equity_matrix() * PAIR_COMPAT * wt).sum(1) / (PAIR_COMPAT * wt).sum(1)
        weak = _weakest_offsuit(eq, wo)
        self.assertGreater(min(S["o_fold"][i] for i in weak), 0.99)
        # 4bets beyond the value core (KQs/KJs blocker bluffs against the old, narrower BTN 3bet; against
        # the published 12.5% one, AQo, KQo, TT-99 and small pairs)
        core = [C[h] for h in ("AA", "KK", "QQ", "JJ", "AKs", "AKo")]
        w4 = wo * S["o_4bet"]
        self.assertGreater(float(w4.sum() - w4[core].sum()), 0.5)
        # some jams aren't premiums (which hands bluff depends on the realization model)
        prem = [C[h] for h in ("AA", "KK", "QQ", "AKs", "AKo")]
        jam = wt_ = to_classes(rng["BTN"]) * S["t_jam"]
        self.assertGreater(float(jam.sum() - jam[prem].sum()), 0.5)



def _weakest_offsuit(e, wo):
    """The opener's five lowest-equity unpaired offsuit hands against the 3bet range (classes it opens
    more than half the time). Suited connectors and small pairs are left out: against the published,
    wider 3bet ranges (BTN 3betting 65s/54s against a CO open, CO 3betting 7% against UTG) 54s-76s and
    22-33 rank among the lowest-equity opens, and their playability makes calling them sound, so they
    aren't trash that must fold."""
    from gto.pf_classes import CLASSES
    return [i for i in np.argsort(e) if wo[i] > 0.5 and CLASSES[i].endswith("o")][:5]


def _subgame_cases():
    """The six standard 3bet pots of scripts/calibrate_pf_solve.py: (name, opener classes, 3bettor
    classes, Spot, chart shares [opener call, opener 4bet, 3bettor call, 3bettor jam])."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import calibrate_pf_solve
    return calibrate_pf_solve.prepare()


def _continue_evs(S, wo, wt, spot):
    """Per opener class, the EV (bb, relative to folding) of its best continue against the 3bet, under
    the solved strategies."""
    from gto import pf_solve as P
    from gto.pf_classes import PAIR_COMPAT
    st = lambda *k: np.stack([S[x] for x in k])
    v0, _, _ = P._values(P._payoffs(spot), wo, wt, spot.pot, st("o_fold", "o_call", "o_4bet"),
                         st("t_fold", "t_call", "t_jam"), np.stack([1 - S["o_vs_jam"], S["o_vs_jam"]]))
    return np.max(v0[1:], axis=0) / np.maximum(PAIR_COMPAT @ wt, 1e-12)


def _half_table(name):
    import os
    ns = {}
    with open(os.path.join(os.path.dirname(__file__), "data", f"pf_realize_{name}.py")) as f:
        exec(f.read(), ns)
    return ns["TABLE"]


class TestRealizationTable(unittest.TestCase):
    """gto/pf_realize.py (measured with the postflop solver on the charts' ranges) and the preflop
    subgame solved with it."""

    @classmethod
    def setUpClass(cls):
        from gto import pf_solve as P
        cls.P = P
        cls._mode = P.MODE
        P.MODE = "table"
        cls.cases = _subgame_cases()
        cls.sols = {name: P.solve(wo, wt, spot, iters=1500) for name, wo, wt, spot, _ in cls.cases}

    @classmethod
    def tearDownClass(cls):
        cls.P.MODE = cls._mode

    def test_table_shape_and_bounds(self):
        from gto.pf_realize import TABLE
        from gto.pf_solve import bucket_of, K
        nb = int(bucket_of().max()) + 1
        for pt in ("3bet", "4bet"):
            self.assertEqual(sorted(TABLE[pt]), ["aggr_ip", "aggr_oop", "caller_ip", "caller_oop"])
            for pos in TABLE[pt]:
                rows = TABLE[pt][pos]
                self.assertEqual(len(rows), K)
                for row in rows:
                    self.assertEqual(len(row), nb)
                    for v in row:
                        if v is not None:
                            self.assertTrue(-3.0 < v[0] < 3.0 and v[1] >= 0.0, (pt, pos, v))

    def test_table_makes_poker_sense(self):
        """Where well measured: KK loses chips after the flop against AA and AA wins them against KK;
        AA runs at or above its equity against everything weaker; AKo and AJo run below it against the
        top buckets (they pay off)."""
        from gto.pf_realize import TABLE
        from gto.pf_classes import CLASS_INDEX as C
        for pt in ("3bet", "4bet"):
            for pos in TABLE[pt]:
                t = TABLE[pt][pos]
                kk_aa, aa_kk = t[C["KK"]][0], t[C["AA"]][1]
                if kk_aa and kk_aa[1] > 0.2:
                    self.assertLess(kk_aa[0], 0.0, (pt, pos))
                if aa_kk and aa_kk[1] > 0.2:
                    self.assertGreater(aa_kk[0], 0.0, (pt, pos))
                for b in range(1, 9):
                    v = t[C["AA"]][b]
                    if v and v[1] > 0.5:
                        self.assertGreater(v[0], -0.05, (pt, pos, b))
                for h in ("AKo", "AJo"):
                    for b in (0, 1):
                        v = t[C[h]][b]
                        if v and v[1] > 0.2:
                            self.assertLess(v[0], 0.0, (pt, pos, h, b))

    def test_shares_bounded_and_follow_equity(self):
        from gto.pf_classes import equity_matrix
        eq = equity_matrix()
        for name, wo, wt, spot, _ in self.cases:
            for pt in ("3bet", "4bet"):
                sh = self.P._shares(eq, 1.0, 1.0, pt, spot.opener_ip)
                self.assertTrue(np.isfinite(sh).all(), name)
                self.assertTrue((sh > -2.0).all() and (sh < 3.0).all(), name)
                m = (wo[:, None] * wt[None, :]) > 0
                self.assertGreater(np.corrcoef(sh[m], eq[m])[0, 1], 0.8, (name, pt))

    def test_every_spot_converges_and_plays_sanely(self):
        from gto.pf_classes import CLASS_INDEX as C, equity_matrix, PAIR_COMPAT
        for name, wo, wt, spot, _ in self.cases:
            S = self.sols[name]
            self.assertLess(S["exploitability"], 0.05, name)
            for h in ("AA", "KK"):
                self.assertLess(S["o_fold"][C[h]], 0.01, (name, h))
                self.assertLess(S["t_fold"][C[h]], 0.01, (name, h))
            self.assertGreater(S["o_vs_jam"][C["AA"]], 0.99, name)
            # (KK facing the jam: see TestRealizationModel.test_kk_calls_the_jam)
            # the weakest hands the opener opened don't call the 3bet
            e = (equity_matrix() * PAIR_COMPAT * wt).sum(1) / (PAIR_COMPAT * wt).sum(1)
            weak = _weakest_offsuit(e, wo)
            # never call (a blocker 4bet bluff, KTo against BTN's wider published 3bet, is fine here)
            self.assertLess(max(S["o_call"][i] for i in weak), 0.05, name)


    # Known failure: the raw table is noisy per class, so pair EVs can come out of order (CO vs BTN: 99
    # continues for 1.76bb, 88 for 2.31); the smooth model (TestRealizationModel) passes this.
    @unittest.expectedFailure
    def test_pairs_in_order(self):
        """A pair's EV of continuing against the 3bet doesn't trail a smaller pair's by more than 0.5bb
        (frequencies can flip between hands near break-even, so the check is on EVs)."""
        from gto.pf_classes import CLASS_INDEX as C
        for name, wo, wt, spot, _ in self.cases:
            ev = _continue_evs(self.sols[name], wo, wt, spot)
            pairs = [ev[C[p]] for p in ("AA", "KK", "QQ", "JJ", "TT", "99", "88", "77") if wo[C[p]] > 0.5]
            for hi, lo in zip(pairs, pairs[1:]):
                self.assertGreater(hi, lo - 0.5, (name, [round(x, 2) for x in pairs]))

    # Known failure: per class the raw table is still a little noisy: on the two (stratified, weighted)
    # halves of the flops the opener's call share in BTN vs SB is 0.55/0.45 (every other share agrees
    # within 7 points; with random flops it moved up to 22). The smooth model passes this.
    @unittest.expectedFailure
    def test_stable_across_flop_halves(self):
        """Solved with the table from each half of the flops, the range-level shares and the key hands
        come out nearly the same: the measurement isn't noise-driven."""
        from gto.pf_classes import CLASS_INDEX as C
        P = self.P
        res = {}
        try:
            for half in ("a", "b"):
                P.TABLE_OVERRIDE = _half_table(half)
                res[half] = {name: P.solve(wo, wt, spot, iters=1500) for name, wo, wt, spot, _ in self.cases}
        finally:
            P.TABLE_OVERRIDE = None
        for name, wo, wt, spot, _ in self.cases:
            a, b = res["a"][name], res["b"][name]
            sh = lambda S, k, w: float((w * S[k]).sum() / w.sum())
            for k, w in (("o_call", wo), ("o_4bet", wo), ("t_call", wt), ("t_jam", wt)):
                self.assertLess(abs(sh(a, k, w) - sh(b, k, w)), 0.08, (name, k, sh(a, k, w), sh(b, k, w)))
            for h in ("AA", "KK"):
                self.assertLess(abs(a["o_vs_jam"][C[h]] - b["o_vs_jam"][C[h]]), 0.15, (name, h))

    # Known failure: UTG vs CO calls 14% (chart 26%), BTN vs SB 50% (chart 36%); see the model's
    # test_close_to_charts.
    @unittest.expectedFailure
    def test_close_to_charts(self):
        """The opener's response to the 3bet stays near the charts (hand-written solver approximations)."""
        for name, wo, wt, spot, target in self.cases:
            S = self.sols[name]
            sh = lambda k, w: float((w * S[k]).sum() / w.sum())
            self.assertLess(abs(sh("o_call", wo) - target[0]), 0.12, (name, sh("o_call", wo), target[0]))
            self.assertLess(abs(sh("o_4bet", wo) - target[1]), 0.08, (name, sh("o_4bet", wo), target[1]))


def _half_model(name):
    import os
    ns = {}
    with open(os.path.join(os.path.dirname(__file__), "data", f"pf_realize_model_{name}.py")) as f:
        exec(f.read(), ns)
    return ns["COEF"]


class TestRealizationModel(unittest.TestCase):
    """gto/pf_realize_model.py: the smooth model of the measured excess pot shares
    (scripts/fit_realization.py), and the preflop subgame solved with it."""

    @classmethod
    def setUpClass(cls):
        from gto import pf_solve as P
        cls.P = P
        cls._mode = P.MODE
        P.MODE = "model"
        cls.cases = _subgame_cases()
        cls.sols = {name: P.solve(wo, wt, spot, iters=1500) for name, wo, wt, spot, _ in cls.cases}

    @classmethod
    def tearDownClass(cls):
        cls.P.MODE = cls._mode

    def test_model_shape_and_sense(self):
        """Every class gets a finite excess against every bucket; big favourites gain (AA against KK)
        and big underdogs lose (KK against AA), in every pot type and position."""
        from gto.pf_classes import CLASS_INDEX as C
        for pt in ("3bet", "4bet"):
            for pos in ("caller_oop", "caller_ip", "aggr_oop", "aggr_ip"):
                m = self.P._model(pt, pos)
                self.assertEqual(m.shape, (len(C), int(self.P.bucket_of().max()) + 1))
                self.assertTrue(np.isfinite(m).all())
                self.assertGreater(m[C["AA"], 1], 0.0, (pt, pos))
                self.assertLess(m[C["KK"], 0], 0.0, (pt, pos))

    def test_every_spot_converges_and_plays_sanely(self):
        from gto.pf_classes import CLASS_INDEX as C, equity_matrix, PAIR_COMPAT
        for name, wo, wt, spot, _ in self.cases:
            S = self.sols[name]
            self.assertLess(S["exploitability"], 0.05, name)
            for h in ("AA", "KK"):
                self.assertLess(S["o_fold"][C[h]], 0.01, (name, h))
                self.assertLess(S["t_fold"][C[h]], 0.01, (name, h))
            self.assertGreater(S["o_vs_jam"][C["AA"]], 0.99, name)
            e = (equity_matrix() * PAIR_COMPAT * wt).sum(1) / (PAIR_COMPAT * wt).sum(1)
            weak = _weakest_offsuit(e, wo)
            # never call (a blocker 4bet bluff, KTo against BTN's wider published 3bet, is fine here)
            self.assertLess(max(S["o_call"][i] for i in weak), 0.05, name)
            ev = _continue_evs(S, wo, wt, spot)
            pairs = [ev[C[p]] for p in ("AA", "KK", "QQ", "JJ", "TT", "99", "88", "77") if wo[C[p]] > 0.5]
            for hi, lo in zip(pairs, pairs[1:]):
                self.assertGreater(hi, lo - 0.5, (name, [round(x, 2) for x in pairs]))

    # (With realization keyed by position only, the 3bettor trapped KK against the 4bet in UTG vs CO and
    # HJ vs BTN and the opener folded KK to the AA-heavy jams; keyed by role it calls them everywhere.)
    def test_kk_calls_the_jam(self):
        from gto.pf_classes import CLASS_INDEX as C
        for name, *_ in self.cases:
            self.assertGreater(self.sols[name]["o_vs_jam"][C["KK"]], 0.9, name)

    # Known failure: keyed by role and position each fit has half the data it had keyed by position, and
    # the 3bettor's call share against the 4bet moves 10-11 points between halves in CO vs BTN
    # (0.41/0.31) and BTN vs SB (0.28/0.18); every other share agrees within 5.
    @unittest.expectedFailure
    def test_stable_across_flop_halves(self):
        """Fitted on each half of the (texture-stratified, weighted) flops, the model gives nearly the same
        strategies."""
        P = self.P
        res = {}
        try:
            for half in ("a", "b"):
                P.MODEL_OVERRIDE = _half_model(half)
                res[half] = {name: P.solve(wo, wt, spot, iters=1500) for name, wo, wt, spot, _ in self.cases}
        finally:
            P.MODEL_OVERRIDE = None
        for name, wo, wt, spot, _ in self.cases:
            a, b = res["a"][name], res["b"][name]
            sh = lambda S, k, w: float((w * S[k]).sum() / w.sum())
            for k, w in (("o_call", wo), ("o_4bet", wo), ("t_call", wt), ("t_jam", wt)):
                self.assertLess(abs(sh(a, k, w) - sh(b, k, w)), 0.08, (name, k, sh(a, k, w), sh(b, k, w)))

    # Known failure: with the in-position caller shifted to the public rules of thumb (pf_solve.MODEL_SHIFT)
    # in-position openers call about as often as the charts (BTN vs BB 41% against 34%, CO vs BB 31%
    # against 41%), but out-of-position openers call 2-2.5 times as often (CO vs BTN 47% against 18%,
    # UTG vs CO 39% against 18%, HJ vs BTN 30% against 17%), though the measured out-of-position caller
    # (0.80) matches public EQR for that role (0.78-0.87); fitting the charts would take it to ~0.64.
    @unittest.expectedFailure
    def test_close_to_charts(self):
        for name, wo, wt, spot, target in self.cases:
            S = self.sols[name]
            sh = lambda k, w: float((w * S[k]).sum() / w.sum())
            self.assertLess(abs(sh("o_call", wo) - target[0]), 0.12, (name, sh("o_call", wo), target[0]))
            self.assertLess(abs(sh("o_4bet", wo) - target[1]), 0.08, (name, sh("o_4bet", wo), target[1]))


class TestOpenerVs3bet(unittest.TestCase):
    def test_oop_openers_match_published_solver_shares(self):
        """Openers out of position facing a 3bet: fold/call/4bet shares of the opening range near published
        solver output (CO vs BTN 3bet to 7.5bb, 100bb cash: 63.7/17.2/19.1; the SB vs a BB 3bet folds about
        half and 4bets a third of its continues), and the in-position charts against a blind's 3bet keep
        calling widely."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        T = pf.Table(pf.SEATS[6])
        def shares(hero, pre):
            w = pf.replay_ranges(pre, T)[hero]
            s = pf.strategy_for(hero, pre, T)
            return [float((w * s.get(k, 0)).sum() / w.sum()) for k in ("fold", "call", "raise")]
        co = shares("CO", F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "raise", 7.5)] + F("SB", "BB"))
        for got, want in zip(co, (0.637, 0.172, 0.191)):
            self.assertAlmostEqual(got, want, delta=0.03)
        for hero, pre in (("UTG", [R("UTG", "raise", 2.5)] + F("HJ", "CO") + [R("BTN", "raise", 7.5)] + F("SB", "BB")),
                          ("HJ", F("UTG") + [R("HJ", "raise", 2.5), R("CO", "raise", 7.5)] + F("BTN", "SB", "BB"))):
            f, c, r = shares(hero, pre)
            self.assertTrue(0.58 <= f <= 0.68 and 0.14 <= c <= 0.21 and 0.16 <= r <= 0.23, (hero, f, c, r))
            self.assertTrue(0.4 <= r / (1 - f) <= 0.65, (hero, r / (1 - f)))   # about half the continues 4bet
        f, c, r = shares("SB", F("UTG", "HJ", "CO", "BTN") + [R("SB", "raise", 3), R("BB", "raise", 9)])
        self.assertTrue(0.45 <= f <= 0.58 and 0.28 <= r / (1 - f) <= 0.40, (f, c, r))
        f, c, r = shares("BTN", F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)] + F("SB") + [R("BB", "raise", 10)])
        self.assertGreater(c, 0.3)

    def test_thin_lines_filled_out(self):
        """A line that leaves fewer than MIN_LINE_COMBOS gets the hands that take its last action mixed in,
        weighing 1 - combos / threshold: all of them for an empty line, none at the threshold; the hands
        that got there are kept."""
        n = pf.N_COMBOS
        scratch = np.zeros(n); scratch[:40] = 1.0
        empty, mostly = pf.thin_line(np.zeros(n), scratch)
        self.assertAlmostEqual(empty.sum(), 40.0); self.assertTrue(mostly)
        r = np.zeros(n); r[100:104] = 1.0                       # 4 combos, half the threshold
        out, mostly = pf.thin_line(r, scratch)
        self.assertTrue((out[100:104] == 1.0).all())
        self.assertAlmostEqual(out[:40].sum(), 40.0 * (1 - 4.0 / pf.MIN_LINE_COMBOS)); self.assertTrue(mostly)
        wide = np.zeros(n); wide[:10] = 1.0
        self.assertIs(pf.thin_line(wide, scratch)[0], wide)    # at or above the threshold: unchanged
        # BB cold-calls a 3bet, then calls HJ's 4bet: 6.7 combos, filled out to ~13 with a few trapped
        # premiums; now that the cold caller defends more against a 4bet wider than UTG's (caller_target)
        # it is ~10 combos of JJ-TT, AK, AQs and QQ without filling
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        from gto.advisor import preflop_ranges
        from gto.cards import COMBO_INDEX, cards
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "fold"), R("HJ", "raise", 2.5), R("CO", "raise", 7.5), R("BTN", "fold"), R("SB", "fold"),
               R("BB", "call"), R("HJ", "raise", 18.8), R("CO", "fold"), R("BB", "call")]
        bb = preflop_ranges(pre, T, 100)["BB"]
        self.assertGreater(bb.sum(), pf.MIN_LINE_COMBOS)
        self.assertGreater(bb[COMBO_INDEX[tuple(sorted(cards("QsQd")))]], 0.0)


class TestPublishedBlindFrequencies(unittest.TestCase):
    """Charts matched to published solver frequencies (see the comments in gto/preflop.py): the button
    facing either blind's 3bet, the big blind's 3bets against UTG/HJ/CO opens, and squeezes."""

    R = staticmethod(lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {})))
    F = staticmethod(lambda *ps: [{"pos": p, "action": "fold"} for p in ps])

    def _shares(self, hero, pre, of_range=True):
        from gto.advisor import preflop_strategy, preflop_ranges
        T = pf.Table(pf.SEATS[6])
        s = preflop_strategy(hero, pre, T, 100)
        w = preflop_ranges(pre, T, 100)[hero] if of_range else np.ones(pf.N_COMBOS)
        return [float((w * s.get(k, 0)).sum() / w.sum()) for k in ("fold", "call", "raise")]

    def test_button_vs_blind_3bets(self):
        # gtopreflop.com: the button 4bets 10.4% of its opens against the SB and 10.3% against the BB, and
        # calls 6 points less against the SB; 888poker has BTN vs BB at 54.6/33.9/11.5
        R, F = self.R, self.F
        btn = F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)]
        f_sb, c_sb, r_sb = self._shares("BTN", btn + [R("SB", "raise", 11), R("BB", "fold")])
        f_bb, c_bb, r_bb = self._shares("BTN", btn + F("SB") + [R("BB", "raise", 11)])
        self.assertTrue(0.10 <= r_sb <= 0.14 and 0.25 <= c_sb <= 0.32 and 0.56 <= f_sb <= 0.64, (f_sb, c_sb, r_sb))
        self.assertTrue(0.10 <= r_bb <= 0.14 and 0.31 <= c_bb <= 0.37, (f_bb, c_bb, r_bb))
        self.assertGreater(r_sb, r_bb - 0.01)          # at least as many 4bets against the SB
        self.assertLess(c_sb, c_bb - 0.03)             # fewer calls
        # the SB chart's hand mix: QQ and AKo mostly 4bet, AQs and KQs call, offsuit non-4bets fold
        from gto.cards import COMBO_INDEX
        sb = pf.vs_3bet("BTN", "SB")
        ix =lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        self.assertGreater(sb["4bet"][ix("QsQh")], 0.8)
        self.assertGreater(sb["4bet"][ix("AsKd")], 0.7)
        self.assertEqual(sb["4bet"][ix("AsQs")], 0.0)
        self.assertGreater(sb["call"][ix("AsQs")], 0.9)
        self.assertEqual(sb["call"][ix("KsQd")], 0.0)

    def test_big_blind_3bets_vs_early_and_middle_opens(self):
        # published (gtopreflop.com raked 8-max/6-max, Poker Academy, pokalab, BeyondGTO): vs LJ/UTG 4.5-6%,
        # vs HJ 5.9-8%, vs CO 7.7-10%; total defence kept
        R, F = self.R, self.F
        lines = {"UTG": [R("UTG", "raise", 2.5)] + F("HJ", "CO", "BTN", "SB"),
                 "HJ": F("UTG") + [R("HJ", "raise", 2.5)] + F("CO", "BTN", "SB"),
                 "CO": F("UTG", "HJ") + [R("CO", "raise", 2.5)] + F("BTN", "SB")}
        want = {"UTG": (0.045, 0.055, 0.30, 0.36), "HJ": (0.058, 0.07, 0.36, 0.42), "CO": (0.075, 0.09, 0.42, 0.48)}
        got = {}
        for o, pre in lines.items():
            f, c, r = self._shares("BB", pre, of_range=False)
            lo, hi, dlo, dhi = want[o]
            self.assertTrue(lo <= r <= hi and dlo <= c + r <= dhi, (o, f, c, r))
            got[o] = r
        self.assertTrue(got["UTG"] < got["HJ"] < got["CO"], got)
        self.assertIsNot(pf.vs_rfi_chart("BB", "HJ"), pf.vs_rfi_chart("BB", "UTG"))

    def test_squeezes_near_heads_up_3bets(self):
        # cash solves squeeze about as often as they 3bet heads-up (GTO Wizard NL500 40bb: 9.8% vs 10.3%
        # over CO + BTN; Poker Academy NL200: 12.7% vs 11.5% over BTN + SB)
        R, F = self.R, self.F
        co = F("UTG", "HJ") + [R("CO", "raise", 2.5)]
        hu = self._shares("BB", co + F("BTN", "SB"), of_range=False)[2]
        sq = self._shares("BB", co + [R("BTN", "call"), R("SB", "fold")], of_range=False)[2]
        self.assertTrue(0.065 <= sq <= 0.09 and sq >= 0.85 * hu, (sq, hu))
        utg = [R("UTG", "raise", 2.5)]
        hu = self._shares("BB", utg + F("HJ", "CO", "BTN", "SB"), of_range=False)[2]
        sq = self._shares("BB", utg + F("HJ", "CO") + [R("BTN", "call"), R("SB", "fold")], of_range=False)[2]
        self.assertTrue(0.04 <= sq <= 0.055 and sq >= 0.85 * hu, (sq, hu))
        btn = F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)]
        sq = self._shares("BB", btn + [R("SB", "call")], of_range=False)[2]
        self.assertTrue(0.09 <= sq <= 0.14, sq)

    def test_sb_short_stack_flats(self):
        # at 40bb the small blind's flat of a CO open holds suited broadways, 77-99 and AQo (Poker Academy's
        # 40bb solves: JTs, KJs, KTs, 88, AQo call or jam), not only small pairs and A6s-A8s (~45 combos,
        # the 100bb chart re-priced), at about the 100bb chart's totals
        from gto.advisor import preflop_strategy, preflop_ranges
        from gto.cards import COMBO_INDEX
        R, F = self.R, self.F
        T = pf.Table(pf.SEATS[6])
        pre = F("UTG", "HJ") + [R("CO", "raise", 2.5)] + F("BTN")
        s = preflop_strategy("SB", pre, T, 40)
        call = s["call"]
        cls = lambda h: float(np.mean([call[i] for c, i in COMBO_INDEX.items() if hand_class(c) == h]))
        for h in ("JTs", "QJs", "KJs", "KTs", "88", "AQo"):
            self.assertGreater(cls(h), 0.4, h)
        self.assertTrue(70 <= call.sum() <= 110, call.sum())
        f, c, r = (float(s[k].sum()) / len(call) for k in ("fold", "call", "raise"))
        self.assertTrue(0.77 <= f <= 0.84 and 0.11 <= r <= 0.15, (f, c, r))
        # the flatting range that reaches the flop (BB folds) is the same at a table built at 40bb
        rng = preflop_ranges(pre + [R("SB", "call"), R("BB", "fold")], T, 40)["SB"]
        self.assertTrue(70 <= rng.sum() <= 110, rng.sum())
        # 100bb keeps the 100bb chart
        self.assertIs(pf.vs_rfi_chart("SB", "CO"), pf.VS_RFI[("SB", "CO")])
        self.assertIs(pf.vs_rfi_chart("SB", "CO", 40.0), pf.SB_VS_LATE_40)


class TestRake(unittest.TestCase):
    """Optional rake: the winner of a postflop pot pays a share of it, to a cap; pots that end preflop
    aren't raked ("no flop, no drop"). Off by default."""

    NO_BETS = {s: [] for s in ("flop", "turn", "river")}
    R = staticmethod(lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {})))

    def test_zero_rake_is_identical(self):
        r0 = parse_range("22+, A2s+, K9s+, QTs+, JTs, ATo+, KJo+")
        r1 = parse_range("33+, A5s+, KTs+, QJs, AJo+, KQo")
        small = dict(n_turns=2, leaf_turns=2, leaf_rivers_flop=2, n_rivers_from_flop=2)
        for board, kw in (("Ks7d2c", small), ("Ks7d2cTh", dict(n_rivers=4)), ("Ks7d2cTh3s", {})):
            a = solve(cards(board), r0, r1, pot=10, stack=90, cfg=Config(**kw), max_iters=8, time_budget=100)
            b = solve(cards(board), r0, r1, pot=10, stack=90, cfg=Config(rake_cap=3.0, **kw), max_iters=8, time_budget=100)
            self.assertTrue(np.array_equal(a.evs, b.evs) and np.array_equal(a.strategy, b.strategy), board)
            self.assertEqual(a.exploitability(), b.exploitability())
        self.assertEqual(Config(rake_cap=3.0).sizes_key(), Config().sizes_key())
        self.assertNotEqual(Config(rake_rate=0.05).sizes_key(), Config().sizes_key())

    def test_showdown_pays_the_pot_less_rake(self):
        # checked down, a hand wins (pot - rake) * equity: through every river (a merged chance node),
        # an equity leaf, and turn decisions over every turn with equity leaves below (tiled leaves)
        hero, vr = cards("AhAs"), parse_range("KK, 88, QQ, JTs, 76s")
        for board, depth, rate, cap in (("Kh8d5c3s", 2, 0.05, None), ("Kh8d5c", 1, 0.05, 0.3), ("Kh8d5c", 2, 0.1, 3.0),
                                        ("Kh8d5c3s2h", 1, 0.05, 3.0)):
            cfg = Config(bet_sizes=self.NO_BETS, raise_sizes=self.NO_BETS, allin_spr=0, depth=depth,
                         n_turns=49, leaf_turns=49, leaf_rivers_flop=48, rake_rate=rate, rake_cap=cap,
                         turn_classes=False, later_sizes=None)
            b = cards(board)
            sol = solve(b, parse_range("AA"), vr, pot=10, stack=100, time_budget=100, max_iters=2, cfg=cfg)
            ev = sol.for_combo(tuple(hero))["check"][1]
            rake = min(rate * 10, cap or 1e9)
            self.assertAlmostEqual(ev, (10 - rake) * _exact_equity(b, hero, vr), places=3, msg=board)

    def test_fold_pays_rake_on_the_called_pot(self):
        # the winner of a fold pays rake on the pot without its own uncalled bet
        cfg = Config(bet_sizes={s: [1.0] for s in ("flop", "turn", "river")}, raise_sizes=self.NO_BETS, allin_spr=0,
                     rake_rate=0.05, rake_cap=3.0)
        sol = solve(cards("Kh8d5c3s2h"), parse_range("QQ"), parse_range("88, T9o"), pot=10, stack=100,
                    time_budget=100, max_iters=2, cfg=cfg)
        vs_bet = sol.root.children[0].children[1]           # OOP checks, IP bets 10, OOP to act
        fold = vs_bet.children[vs_bet.names.index("fold")]
        self.assertAlmostEqual(fold.rake, 0.5)                # 5% of the 10bb pot, not of 20
        self.assertAlmostEqual(fold.fold_pay(1), 5 - 0.5)
        self.assertAlmostEqual(fold.fold_pay(0), -5)
        call = vs_bet.children[vs_bet.names.index("call")]
        self.assertAlmostEqual(call.rake, 1.5)                # 5% of 30
        big = solve(cards("Kh8d5c3s2h"), parse_range("QQ"), parse_range("88, T9o"), pot=50, stack=100,
                    time_budget=100, max_iters=2, cfg=cfg)
        vs_big = big.root.children[0].children[1]
        self.assertAlmostEqual(vs_big.children[vs_big.names.index("fold")].rake, 2.5)
        self.assertAlmostEqual(vs_big.children[vs_big.names.index("call")].rake, 3.0)   # capped

    def test_raked_polarized_river_game(self):
        # nuts-or-air vs a bluff-catcher, raked 10% (no cap), pot P = 10, bet b = 10: the bluff wins P less
        # rake on P when called off, so the catcher calls (P - R_P) / (P - R_P + b) = 9/19 (unraked 1/2); the
        # catcher wins P + b less rake on P + 2b, so bets are b / (P + 2b - R_c) = 10/27 bluffs (unraked 1/3)
        board = cards("Kh8d5c3s2h")
        cfg = Config(bet_sizes={s: [1.0] for s in ("flop", "turn", "river")}, raise_sizes=self.NO_BETS, allin_spr=0,
                     rake_rate=0.1)
        sol = solve(board, parse_range("QQ"), parse_range("88, T9o"), pot=10, stack=100, time_budget=5,
                    max_iters=2000, cfg=cfg)
        ipn = sol.root.children[0]
        air = parse_range("T9o")[sol.ctx.active] > 0
        x = 10 / 27
        self.assertAlmostEqual(float(ipn.avg_strategy0()[1][air].mean()), 3 * x / (1 - x) / 12, places=2)
        qq = parse_range("QQ")[sol.ctx.active] > 0
        self.assertAlmostEqual(float(ipn.children[1].avg_strategy0()[1][qq].mean()), 9 / 19, places=2)
        # the exploitability counts what each best response gains over its player's raked payoff
        self.assertLess(sol.exploitability(), 0.1)

    def test_no_blas_path_matches(self):
        # the browser's einsum path, with leaf rows trimmed to the live combos, gives the same raked utilities
        import gto.solver as S
        r0 = parse_range("22+, A2s+, K9s+, QTs+, JTs, ATo+, KJo+")
        r1 = parse_range("33+, A5s+, KTs+, QJs, AJo+, KQo")
        # (the browser's Fast flop tree: random turns, the river checked down; and the default: turn classes, the
        # river played on the turn's boards)
        for cfg in (Config(n_turns=2, leaf_turns=2, leaf_rivers_flop=2, n_rivers_from_flop=2, rake_rate=0.05, rake_cap=3.0,
                           river_play=False, turn_classes=False), Config(rake_rate=0.05, rake_cap=3.0)):
            sol = solve(cards("Ks7d2c"), r0, r1, pot=10, stack=90, cfg=cfg, max_iters=6, time_budget=100)
            s = sol.solver
            a, b = sol.r0[None, :].astype(np.float32), sol.r1[None, :].astype(np.float32)
            live = (np.flatnonzero(sol.r0 > 0), np.flatnonzero(sol.r1 > 0))
            # the root and the merged subtrees (tiled equity leaves), each player, current strategies
            spots = [(s.root, a, b)] + [(m, np.repeat(a, len(m.boards), axis=0), np.repeat(b, len(m.boards), axis=0))
                                        for m, _ in s._groups]
            leaf = lambda n: n if not hasattr(n, "children") else leaf(n.children[0])
            self.assertTrue(any(leaf(m.child).tile for m, _ in s._groups if hasattr(m, "child")))
            saved = S.NO_BLAS
            try:
                for node, x0, x1 in spots:
                    for p in (0, 1):
                        S.NO_BLAS, s._live, s._trimmed = False, None, {}
                        ref = s._cfr(node, p, x0, x1, update=False, use_avg=False)
                        S.NO_BLAS, s._live = True, live
                        got = s._cfr(node, p, x0, x1, update=False, use_avg=False, strats=({}, {}))   # a training pass
                        self.assertLess(np.abs(ref - got)[:, live[p]].max(), 1e-5 * np.abs(ref).max())
            finally:
                S.NO_BLAS, s._live, s._trimmed = saved, None, {}

    def test_preflop_fold_has_no_rake(self):
        # pots that end preflop aren't raked: opens, and the steal's folds, are the chart's with rake on
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        R = self.R
        for hero, hole, pre in (("BTN", "Kd9c", [R("UTG", "fold"), R("HJ", "fold"), R("CO", "fold")]),
                                ("SB", "Qc5d", [R("UTG", "fold"), R("HJ", "fold"), R("CO", "fold"), R("BTN", "raise", 2.5)])):
            st = {"players": 6, "hero": hero, "hole": hole, "board": "", "stack": 100, "preflop": pre}
            a, b = adv.advise(st), adv.advise(dict(st, rake=0.05, rake_cap=3))
            self.assertEqual(a["actions"], b["actions"])
            self.assertEqual(a["grid"], b["grid"])
        # a zero rate is no rake
        st = {"players": 6, "hero": "BB", "hole": "Kd9c", "board": "", "stack": 100,
              "preflop": [R("UTG", "fold"), R("HJ", "fold"), R("CO", "fold"), R("BTN", "raise", 2.5), R("SB", "fold")]}
        self.assertEqual(adv.advise(st)["grid"], adv.advise(dict(st, rake=0, rake_cap=3))["grid"])

    def _rake_shares(self, hero, pre, rake):
        # fold/call/raise shares of hero's range at a preflop decision, by the adjusted strategy
        from gto.advisor import preflop_strategy, preflop_ranges
        T = pf.Table(pf.SEATS[6], rake=rake)
        w = preflop_ranges(pre, T, 100)[hero]
        s = preflop_strategy(hero, pre, T, 100)
        return {k: float((w * v).sum() / w.sum()) for k, v in s.items()}

    def _rake_spots(self):
        R = self.R
        F = lambda *ps: [R(p, "fold") for p in ps]
        return {
            "BB vs BTN": ("BB", F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5), R("SB", "fold")]),
            "BB vs CO": ("BB", F("UTG", "HJ") + [R("CO", "raise", 2.5)] + F("BTN", "SB")),
            "UTG vs SB": ("UTG", [R("UTG", "raise", 2.5)] + F("HJ", "CO", "BTN") + [R("SB", "raise", 11), R("BB", "fold")]),
            "CO vs BB": ("CO", F("UTG", "HJ") + [R("CO", "raise", 2.5)] + F("BTN", "SB") + [R("BB", "raise", 11)]),
            "BTN vs BB": ("BTN", F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5), R("SB", "fold"), R("BB", "raise", 11)]),
            "CO vs BTN": ("CO", F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "raise", 7.5)] + F("SB", "BB")),
            "BTN cold vs CO": ("BTN", F("UTG", "HJ") + [R("CO", "raise", 2.5)]),
        }

    def test_big_blind_defends_less_with_rake(self):
        # the big blind's calls are priced by the pot it wins less the rake it goes on to pay; its 3bets stay
        pre = self._rake_spots()["BB vs BTN"][1]
        off, on = self._rake_shares("BB", pre, None), self._rake_shares("BB", pre, (0.05, 3.0))
        self.assertLess(on["call"], off["call"] - 0.10)
        self.assertAlmostEqual(on["raise"], off["raise"], places=3)
        # the advisor reads the state's rake (cap in chips, scaled by the big blind) and says why
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        st = {"players": 6, "hero": "BB", "hole": "Kd4c", "board": "", "stack": 200, "bb": 2, "preflop": [
            dict(a, to=5.0) if a.get("to") else a for a in pre], "rake": 0.05, "rake_cap": 6}
        self.assertIn("after rake", " ".join(adv.advise(st)["notes"]))

    def test_preflop_calls_with_rake_match_raked_solves(self):
        # gtopreflop.com's raked solve (8-max 100bb, 5% capped at 4bb, 12bb blind 3bets): UTG vs SB
        # 71.4/14.1/14.5, CO vs BB 68.0/22.4/9.6, BTN vs BB 69.7/20.0/10.3 fold/call/4bet; the big blind
        # calls 23.8% against a CO open, 29.8% against the button. The unraked charts call 36/41/34 and
        # 37/53: with rake they drop toward those, the 4bets and 3bets unchanged
        spots = self._rake_spots()
        target = {"UTG vs SB": 0.141, "CO vs BB": 0.224, "BTN vs BB": 0.200, "BB vs CO": 0.238, "BB vs BTN": 0.298}
        for name, pub in target.items():
            hero, pre = spots[name]
            off, on = self._rake_shares(hero, pre, None), self._rake_shares(hero, pre, (0.05, 4.0))
            self.assertLess(abs(on["call"] - pub), abs(off["call"] - pub) - 0.08, (name, off, on))
            self.assertLess(abs(on["call"] - pub), 0.11, (name, on))
            self.assertAlmostEqual(on["raise"], off["raise"], places=3)
        for name in ("CO vs BB", "BTN vs BB"):   # within 3 points
            hero, pre = spots[name]
            self.assertLess(abs(self._rake_shares(hero, pre, (0.05, 4.0))["call"] - target[name]), 0.03, name)

    def test_preflop_rake_is_monotonic(self):
        # more rake (a higher rate, a higher cap) never adds calls; charts fit to raked solves (the opener
        # out of position facing a 3bet, cold calls) change only with more rake than theirs (5%, 4bb cap)
        for name, (hero, pre) in self._rake_spots().items():
            call = lambda rk: self._rake_shares(hero, pre, rk)["call"]
            off = call(None)
            by_rate = [off] + [call((r, None)) for r in (0.02, 0.05, 0.07, 0.10)]
            by_cap = [off] + [call((0.05, c)) for c in (0.5, 1.0, 3.0, 4.0, None)]
            for calls in (by_rate, by_cap):
                for x, y in zip(calls, calls[1:]):
                    self.assertLessEqual(y, x + 1e-9, (name, calls))
            if name in ("CO vs BTN", "BTN cold vs CO"):
                self.assertAlmostEqual(by_cap[4], off, places=9)
                self.assertLess(by_rate[4], off)
            else:
                self.assertLess(by_cap[4], off - 0.05, (name, by_cap))

    def test_preflop_zero_rake_is_identical(self):
        from gto.advisor import _call_rake
        self.assertEqual(_call_rake(pf.Table(pf.SEATS[6], rake=(0.0, 3.0)), 20.0), 0.0)
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        for hero, pre in self._rake_spots().values():
            for hole in ("9h8h", "AdJc", "5s5c"):
                st = {"players": 6, "hero": hero, "hole": hole, "board": "", "stack": 100, "preflop": pre}
                a, z = adv.advise(st), adv.advise(dict(st, rake=0, rake_cap=4))
                self.assertEqual((a["actions"], a["grid"], a["notes"]), (z["actions"], z["grid"], z["notes"]))

    def test_push_fold_with_rake(self):
        # a called jam runs out a board, so its pot is raked; an uncalled one ends preflop and isn't
        from gto.pf_classes import push_fold
        one = np.ones(1326)
        base = push_fold(one, one, 10, 0.5, 1.0, 1.5)
        self.assertTrue(np.array_equal(base, push_fold(one, one, 10, 0.5, 1.0, 1.5, rake=None)))
        self.assertTrue(np.array_equal(base, push_fold(one, one, 10, 0.5, 1.0, 1.5, rake=(0.0, 3.0))))
        # the cap holds: the called pot is 20bb, so 10% capped at 1bb is the same rake as 5% uncapped
        self.assertTrue(np.array_equal(push_fold(one, one, 10, 0.5, 1.0, 1.5, rake=(0.10, 1.0)),
                                       push_fold(one, one, 10, 0.5, 1.0, 1.5, rake=(0.05, None))))
        self.assertFalse(np.array_equal(push_fold(one, one, 10, 0.5, 1.0, 1.5, rake=(0.10, 1.0)),
                                        push_fold(one, one, 10, 0.5, 1.0, 1.5, rake=(0.10, None))))
        # the big blind facing a 10bb small-blind jam: rake costs the caller (the pot it wins is short the
        # rake), so the jam range widens on the extra folds and K5s, a marginal call, calls less
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [self.R(p, "fold") for p in ("UTG", "HJ", "CO", "BTN")] + [self.R("SB", "all-in")]
        st = {"players": 6, "hero": "BB", "hole": "Kh5h", "board": "", "stack": 10, "preflop": pre}
        a, b = adv.advise(st), adv.advise(dict(st, rake=0.05, rake_cap=3))
        z = adv.advise(dict(st, rake=0, rake_cap=3))
        self.assertEqual((a["actions"], a["grid"], a["notes"]), (z["actions"], z["grid"], z["notes"]))
        share = lambda r: int(next(n for n in r["notes"] if "jam range" in n).split("(")[1].split("%")[0])
        call = lambda r: next(x["freq"] for x in r["actions"] if x["action"] == "call")
        self.assertGreater(share(b), share(a))
        self.assertLess(call(b), call(a) - 0.1)

    def test_multiway_with_rake(self):
        # the multiway heuristic pays every pot hero wins less the rake on it (to the cap). A fixed number
        # of trials (no time caps), so each run plays the same hands and runouts however loaded the machine.
        import time
        from gto.advisor import replay_hand
        R = self.R
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "call"), R("SB", "fold"), R("BB", "call")]
        ranges = pf.replay_ranges(pre, pf.SEATS[6])

        def run(cfg, hole, flop, rules=False, stats=False):
            adv = Advisor(cfg, time_budget=0.5, prior_budget=0.3)
            adv.mw_trials = 376   # two passes over the runouts
            adv.mw_rake_rules = rules
            st = replay_hand(pf.SEATS[6], 100.0, {"preflop": pre, "flop": flop}, 2)[1]
            out = adv._multiway("BB" if not flop else "CO", tuple(cards(hole)), cards("Kh7c2d"), ranges,
                                ["BB", "CO", "BTN"], st, 100.0, time.time(), [])
            return out if stats else {a["action"]: a["ev_bb"] for a in out["actions"]}

        # The payouts alone (our in-trial rules unraked, as they were): raked EVs are lower, by at most the cap.
        for hole, flop in (("AhKd", []), ("Qs7s", [R("BB", "bet", 3)])):
            off = run(Config(), hole, flop)
            self.assertEqual(off, run(Config(rake_rate=0.0, rake_cap=3.0), hole, flop))     # zero rake: identical
            on, capped = run(Config(rake_rate=0.05), hole, flop), run(Config(rake_rate=0.05, rake_cap=0.25), hole, flop)
            for k in off:
                if k == "fold":
                    self.assertEqual(on[k], 0.0)
                    continue
                # raked EVs are lower; the cap limits it to 0.25bb per pot hero wins (EVs are rounded)
                self.assertLess(on[k], off[k], (hole, k))
                self.assertLessEqual(on[k], capped[k] + 0.011, (hole, k))
                self.assertLessEqual(off[k] - capped[k], 0.25 + 0.011, (hole, k))
                self.assertGreater(off[k] - capped[k], 0.0, (hole, k))
            if flop:   # calling a bet: the pots the call goes on to win are raked
                self.assertLess(on["call"], off["call"] - 0.05)

        # Our in-trial calls, leads and raises count the rake too: zero rake is bit-identical to no rake at
        # all, and with rake we lead later streets less often and fold to bets more (the same trials, so the
        # counts compare directly), where the payout-only rules decide exactly as without rake.
        for hole, flop in (("AhKd", []), ("Qs7s", [R("BB", "bet", 3)])):
            base = run(Config(), hole, flop, rules=True, stats=True)
            zero = run(Config(rake_rate=0.0, rake_cap=3.0), hole, flop, rules=True, stats=True)
            self.assertEqual((base["actions"], base["later_streets"], base.get("check_line")),
                             (zero["actions"], zero["later_streets"], zero.get("check_line")))
            raked = run(Config(rake_rate=0.10), hole, flop, rules=True, stats=True)["later_streets"]
            payout_only = run(Config(rake_rate=0.10), hole, flop, stats=True)["later_streets"]
            ls = base["later_streets"]
            self.assertLess(raked["ours"], ls["ours"], hole)
            self.assertGreater(raked["our_folds"], ls["our_folds"], hole)
            self.assertEqual(payout_only["ours"], ls["ours"], hole)

    def test_postflop_advice_with_rake(self):
        # the advisor's postflop solve is raked: a checked-down river pays the pot less 5%
        adv = Advisor(Config(depth=1, bet_sizes=self.NO_BETS, raise_sizes=self.NO_BETS, allin_spr=0),
                      time_budget=0.3, prior_budget=0.2)
        R = self.R
        st = {"players": 6, "hero": "BB", "hole": "AhAs", "board": "Kh8d5c3s2h", "stack": 100,
              "preflop": [R("UTG", "fold"), R("HJ", "fold"), R("CO", "fold"), R("BTN", "raise", 2.5), R("SB", "fold"),
                          R("BB", "call")], "flop": [R("BB", "check"), R("BTN", "check")],
              "turn": [R("BB", "check"), R("BTN", "check")]}
        a = adv.advise(st)
        b = adv.advise(dict(st, rake=0.05, rake_cap=3))
        ev = lambda r: next(x["ev_bb"] for x in r["actions"] if x["action"] == "check")
        # the ranges differ a little (the big blind defends less preflop), the pot the same: 5.5bb less 5%
        self.assertAlmostEqual(ev(b) / ev(a), 0.95, places=2)


class TestPractice(unittest.TestCase):
    """Advisor.play: practice opponents' moves, drawn from the GTO mix for their hand."""
    PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
           {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
    FLOP = {"players": 6, "hero": "BTN", "hole": "AhKh", "board": "Kd7s2c", "stack": 100, "preflop": PRE, "flop": []}
    LEGAL = {"fold", "check", "call", "bet", "raise", "all-in"}

    @staticmethod
    def counts(adv, st, seat, hole, n):
        c = {}
        for s in range(n):
            m = adv.play(st, seat, hole, seed=s)
            c[m["name"]] = c.get(m["name"], 0) + 1
        return c

    def test_preflop_follows_the_chart(self):
        adv = Advisor()
        st = {"players": 6, "hero": "UTG", "hole": "2c2d", "stack": 100, "preflop": self.PRE[:4]}
        for h in ["As5s", "KsJd", "Ts8s", "Qs9s", "As9d", "9s9d", "Ks9s", "Js9s", "6s5s", "Kc8c"]:
            mix = {a["action"]: a["freq"] for a in adv.advise(dict(st, hero="SB", hole=h))["actions"]}
            if sum(0.15 < f < 0.85 for f in mix.values()) >= 2:
                break
        else:
            self.skipTest("no mixed hand found for the small blind")
        n = 2000
        c = self.counts(adv, st, "SB", h, n)
        for a, f in mix.items():
            self.assertAlmostEqual(c.get(a, 0) / n, f, delta=0.04, msg=(h, mix, c))
        m = adv.play(st, "SB", h, seed=3)
        self.assertEqual(m, adv.play(st, "SB", h, seed=3))   # repeatable with a seed
        self.assertIn(m["action"], self.LEGAL)
        if m["action"] == "raise":
            self.assertGreater(m["to"], 2.5)

    def test_postflop_reads_the_heroes_solve(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        st = dict(self.FLOP, flop=[{"pos": "BB", "action": "check"}])
        r = adv.advise(st)                                      # the hero's advice: BTN's solve of the flop
        # the bet BTN's range makes most (a line the solve rarely takes is answered from BB's own seat)
        name = max((a for a in r["range_action_freq"] if a.startswith("bet")), key=r["range_action_freq"].get)
        size = next(a["amount_bb"] for a in r["actions"] if a["action"] == name)
        bet = dict(self.FLOP, flop=st["flop"] + [{"pos": "BTN", "action": "bet", "amount": size}])
        n_cached = len(adv._cache)
        m = adv.play(bet, "BB", "8c8d", seed=1)
        self.assertEqual(len(adv._cache), n_cached)             # no new solve: read off the hero's
        self.assertNotIn("ev_bb", m["actions"][0])
        self.assertIn(m["action"], {"fold", "call", "raise", "all-in"})
        mix = {a["action"]: a["freq"] for a in m["actions"]}
        n = 1500
        c = self.counts(adv, bet, "BB", "8c8d", n)
        for a, f in mix.items():
            self.assertAlmostEqual(c.get(a, 0) / n, f, delta=0.045, msg=(mix, c))
        self.assertEqual(adv.play(bet, "BB", "8c8d", seed=9), adv.play(bet, "BB", "8c8d", seed=9))
        # each move is one the page records and the advisor replays for the hero's next decision
        for s in range(6):
            m = adv.play(bet, "BB", "8c8d", seed=s)
            if m["action"] == "fold":
                continue
            nxt = dict(bet, flop=bet["flop"] + [{"pos": "BB", "action": m["action"], **({"to": m["to"]} if "to" in m else {})}])
            if m["action"] == "call":   # the turn: the big blind acts first
                self.assertIn(adv.play(dict(nxt, board="Kd7s2c9h", turn=[]), "BB", "8c8d", seed=s)["action"],
                              {"check", "bet", "all-in"})
            else:
                out = adv.advise(nxt)
                self.assertNotIn("error", out)

    def test_villain_to_act_first_and_fallbacks(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        m = adv.play(self.FLOP, "BB", "8c8d", seed=2)            # first to act: one solve serves both
        self.assertIn(m["action"], {"check", "bet", "all-in"})
        n_cached = len(adv._cache)
        self.assertNotIn("error", adv.advise(dict(self.FLOP, flop=[{"pos": "BB", "action": "check"}])))
        self.assertEqual(len(adv._cache), n_cached)
        # a hand the big blind 3bets (not in its calling range): answered from its own seat
        m = adv.play(self.FLOP, "BB", "QcQd", seed=2)
        self.assertIn(m["action"], {"check", "bet", "all-in"})
        self.assertIn("ev_bb", m["actions"][0])
        # no hero hand in the state (an API caller): from the seat's own view
        m = adv.play(dict(self.FLOP, hero="BB", hole=""), "BB", "8c8d", seed=2)
        self.assertIn(m["action"], self.LEGAL)
        # the wrong seat to act is an error, not a move
        self.assertIn("error", adv.play(self.FLOP, "BTN", "QsJs"))
        # multiway: the seat's own Monte Carlo
        adv.mw_trials = 300
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        st = {"players": 6, "hero": "BTN", "hole": "AhKh", "board": "Kd7s2c", "stack": 100, "preflop": pre, "flop": []}
        self.assertIn(adv.play(st, "BB", "8c8d", seed=1)["action"], {"check", "bet", "all-in"})


class TestAdvanced(unittest.TestCase):
    """Optional state keys: custom postflop sizes ("sizes", % of the pot), custom ranges entering the flop
    ("ranges"), and bet-size optimization ("optimize_sizes"); plus the villain's range by hand class."""

    R = staticmethod(lambda p, a, to=None, amount=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}),
                                                             **({"amount": amount} if amount else {})))
    PRE = [("UTG", "fold"), ("HJ", "fold"), ("CO", "fold"), ("BTN", "raise", 2.5), ("SB", "fold"), ("BB", "call")]

    def river(self, **kw):
        R = self.R
        return dict({"players": 6, "hero": "BB", "hole": "AhQd", "board": "Kh8d5c3s2h", "stack": 100,
                     "preflop": [R(*a) for a in self.PRE], "flop": [R("BB", "check"), R("BTN", "check")],
                     "turn": [R("BB", "check"), R("BTN", "check")], "river": []}, **kw)

    def adv(self, budget=0.5):
        return Advisor(Config(depth=1), time_budget=budget, prior_budget=0.3)

    def test_custom_sizes(self):
        adv = self.adv()
        out = adv.advise(self.river(sizes={"river": [40, 150], "raise": [50]}))
        self.assertEqual([a["action"] for a in out["actions"]], ["check", "bet 40%", "bet 150%"])
        self.assertEqual(out["actions"][1]["amount_bb"], 2.0)   # 40% of 5.5, to the half bb
        # facing a bet the custom raise menu is one size (and all-in); the villain's off-menu bet keeps its real size
        out = adv.advise(self.river(river=[self.R("BB", "check"), self.R("BTN", "bet", amount=4)],
                                    sizes={"river": [33], "raise": [50]}))
        names = [a["action"] for a in out["actions"]]
        self.assertEqual(names[:2], ["fold", "call"])
        self.assertEqual(len([n for n in names if n.startswith("raise")]), 1, names)
        self.assertEqual(out["to_call_bb"], 4.0)
        # the default menu is untouched by a request that set sizes
        self.assertEqual([a["action"] for a in adv.advise(self.river())["actions"]], ["check", "bet 50%", "bet 100%"])

    def test_custom_range(self):
        adv = self.adv()
        base = adv.advise(self.river())
        out = adv.advise(self.river(ranges={"BTN": "AA, QQ, 76s"}))
        self.assertLess(out["villain_range_combos"], 13)   # AA and QQ less our blockers, 76s
        self.assertGreater(base["villain_range_combos"], 50)
        cells = {c for c, v in out["villain_grid"]["cells"].items() if v["w"] > 0}
        self.assertLessEqual(cells, {"AA", "QQ", "76s"})
        self.assertIn("AA", cells)
        self.assertEqual(out["villain_grid"]["actions"], ["in range"])
        self.assertTrue(any("custom" in n for n in out["notes"]))
        # multiway: the opener's custom range (aces only) is what the field's equity is computed against
        R = self.R
        mw = {"players": 6, "hero": "BB", "hole": "KsQs", "board": "Kd7c2h", "stack": 100,
              "preflop": [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "call"), R("SB", "fold"),
                          R("BB", "call")], "flop": []}
        adv.mw_trials = MW_TRIALS
        a = adv.advise(mw)
        b = adv.advise(dict(mw, ranges={"CO": "AA"}))
        self.assertTrue(a.get("multiway") and b.get("multiway"))
        self.assertLess(b["equity_vs_field"], a["equity_vs_field"] - 0.1)

    def test_optimize_sizes(self):
        adv = self.adv(1.0)
        out = adv.advise(self.river(optimize_sizes=True))
        use, kept = out["size_usage"], out["sizes_kept"]
        self.assertAlmostEqual(sum(use.values()), 1.0, delta=0.01)   # hero's whole mix at the node
        self.assertIn("check", use)
        self.assertEqual(set(use) - {"check", "all-in"}, {"25%", "33%", "50%", "75%", "100%", "150%", "200%"})
        self.assertTrue(1 <= len(kept) <= 3 and set(kept) <= set(use) - {"check"})
        # the advice is solved with the kept sizes only
        bets = [a["action"] for a in out["actions"] if a["action"] != "check"]
        self.assertEqual(sorted(b.replace("bet ", "") for b in bets), sorted(kept))
        # facing a bet there's nothing to choose: noted, no usage
        out = adv.advise(self.river(river=[self.R("BB", "check"), self.R("BTN", "bet", amount=4)], optimize_sizes=True))
        self.assertNotIn("size_usage", out)
        self.assertTrue(any("optimized only" in n for n in out["notes"]))

    def test_bad_input(self):
        adv = self.adv()
        for kw, msg in ((dict(sizes={"flop": ["x"]}), "positive numbers"), (dict(sizes={"flop": [-10]}), "positive numbers"),
                        (dict(sizes={"river": [10, 20, 30, 40]}), "at most 3"), (dict(sizes={"preflop": [50]}), "unknown street"),
                        (dict(sizes=[33]), "must be an object"), (dict(ranges={"BTN": "ZZ+"}), "can't read BTN's range"),
                        (dict(ranges={"LJ": "AA"}), "not a seat"), (dict(ranges={"BTN": "KhKd"}), "no combos left on this board"),
                        (dict(ranges="AA"), "must be an object")):
            out = adv.advise(self.river(**kw))
            self.assertIn("error", out, kw)
            self.assertIn(msg, out["error"], kw)
        # sizes are held to 5-500% of the pot; an empty range box is the chart's range
        out = adv.advise(self.river(sizes={"river": [1, 900]}, ranges={"BTN": " "}))
        self.assertEqual([a["action"] for a in out["actions"]][:2], ["check", "bet 5%"])

