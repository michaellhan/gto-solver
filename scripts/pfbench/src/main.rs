// Benchmark harness: reads a spot (board, ranges, pot, stack, the exact betting tree as lines) as JSON on stdin,
// builds that tree in b-inary's postflop-solver and either solves it ("solve") or measures a given strategy profile
// ("lock": one player's strategy locked at every node, the other's best response computed).
use postflop_solver::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::Read;

#[derive(Deserialize)]
struct Act { t: String, #[serde(default)] a: i32 }

#[derive(Deserialize)]
struct Lock { path: Vec<String>, acts: Vec<Act>, hands: Vec<String>, strat: Vec<Vec<f32>> }

#[derive(Deserialize)]
struct Spec {
    board: Vec<String>,
    oop: (Vec<String>, Vec<f32>),
    ip: (Vec<String>, Vec<f32>),
    pot: i32,
    stack: i32,
    lines: Vec<Vec<Act>>,
    mode: String,
    #[serde(default)] iters: u32,
    #[serde(default)] target_pct: f32,
    #[serde(default)] lock_player: usize,
    #[serde(default)] locks: Vec<Lock>,
    #[serde(default)] root_lock: Option<Lock>,
    #[serde(default)] compress: bool,
    #[serde(default)] trace: bool,
    #[serde(default)] dump: Vec<Vec<String>>,     // solve: paths of nodes whose equilibrium strategy to output too
}

fn to_action(a: &Act) -> Action {
    match a.t.as_str() {
        "fold" => Action::Fold,
        "check" => Action::Check,
        "call" => Action::Call,
        "bet" => Action::Bet(a.a),
        "raise" => Action::Raise(a.a),
        "allin" => Action::AllIn(a.a),
        _ => panic!("bad action {}", a.t),
    }
}

fn act_str(a: &Action) -> String {
    match a {
        Action::Fold => "fold".into(),
        Action::Check => "check".into(),
        Action::Call => "call".into(),
        Action::Bet(x) => format!("bet {x}"),
        Action::Raise(x) => format!("raise {x}"),
        Action::AllIn(x) => format!("allin {x}"),
        Action::Chance(c) => format!("card {}", card_to_string(*c).unwrap()),
        Action::None => "none".into(),
    }
}

fn hole(s: &str) -> (Card, Card) {
    let a = card_from_str(&s[0..2]).unwrap();
    let b = card_from_str(&s[2..4]).unwrap();
    (a.min(b), a.max(b))
}

fn range(r: &(Vec<String>, Vec<f32>)) -> Range {
    let hs: Vec<(Card, Card)> = r.0.iter().map(|s| hole(s)).collect();
    Range::from_hands_weights(&hs, &r.1).unwrap()
}

// Navigate to a path: action strings ("check", "bet 650", ...) and "card Xx" at chance nodes.
fn goto(game: &mut PostFlopGame, path: &[String]) {
    game.back_to_root();
    for p in path {
        if let Some(c) = p.strip_prefix("card ") {
            game.play(card_from_str(c).unwrap() as usize);
            continue;
        }
        let acts = game.available_actions();
        let i = acts.iter().position(|a| &act_str(a) == p).unwrap_or_else(|| panic!("no action {p} in {:?}", acts));
        game.play(i);
    }
}

fn apply_lock(game: &mut PostFlopGame, l: &Lock) {
    goto(game, &l.path);
    let player = game.current_player();
    let theirs = game.available_actions();
    let ours: Vec<String> = l.acts.iter().map(|a| act_str(&to_action(a))).collect();
    let perm: Vec<usize> = theirs
        .iter()
        .map(|a| ours.iter().position(|o| *o == act_str(a))
            .unwrap_or_else(|| panic!("action {:?} not ours {:?} at {:?}", a, ours, l.path)))
        .collect();
    let hands = holes_to_strings(game.private_cards(player)).unwrap();
    let idx: std::collections::HashMap<String, usize> =
        l.hands.iter().enumerate().map(|(i, h)| (hole_to_string(hole(h)).unwrap(), i)).collect();
    let n = hands.len();
    let k_n = theirs.len();
    let mut s = vec![0.0f32; k_n * n];
    for (j, h) in hands.iter().enumerate() {
        let mut tot = 0.0;
        if let Some(&i) = idx.get(h) {
            for (k, &o) in perm.iter().enumerate() {
                s[k * n + j] = l.strat[o][i].max(0.0);
                tot += s[k * n + j];
            }
        }
        if tot <= 0.0 {
            for k in 0..k_n {
                s[k * n + j] = 1.0 / k_n as f32;
            }
        }
    }
    game.lock_current_strategy(&s);
}

fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap();
    let spec: Spec = serde_json::from_str(&buf).unwrap();
    let cards: Vec<Card> = spec.board.iter().map(|c| card_from_str(c).unwrap()).collect();
    let state = match cards.len() {
        3 => BoardState::Flop,
        4 => BoardState::Turn,
        _ => BoardState::River,
    };
    let card_config = CardConfig {
        range: [range(&spec.oop), range(&spec.ip)],
        flop: [cards[0], cards[1], cards[2]],
        turn: if cards.len() > 3 { cards[3] } else { NOT_DEALT },
        river: if cards.len() > 4 { cards[4] } else { NOT_DEALT },
    };
    // no sizes of its own: the tree is exactly the lines given (bets, raises and all-ins added one by one)
    let none = BetSizeOptions::try_from(("", "")).unwrap();
    let tree_config = TreeConfig {
        initial_state: state,
        starting_pot: spec.pot,
        effective_stack: spec.stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [none.clone(), none.clone()],
        turn_bet_sizes: [none.clone(), none.clone()],
        river_bet_sizes: [none.clone(), none],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        merging_threshold: 0.0,
    };
    let mut tree = ActionTree::new(tree_config).unwrap();
    for line in &spec.lines {
        let l: Vec<Action> = line.iter().map(to_action).collect();
        if let Some(Action::Bet(_)) | Some(Action::Raise(_)) | Some(Action::AllIn(_)) = l.last() {
            tree.add_line(&l).unwrap_or_else(|e| panic!("add_line {:?}: {e}", l));
        }
    }
    let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
    let (mem, _) = game.memory_usage();
    game.allocate_memory(spec.compress);
    let t0 = std::time::Instant::now();
    let pot = spec.pot as f32;
    let (mem_c, _) = (game.memory_usage().1, ());
    let mut out = json!({"mem_gb": (if spec.compress { mem_c } else { mem }) as f64 / 1e9});
    if spec.mode == "solve" {
        // node locks (any node; root_lock is the root's): the rest is solved around them
        for l in spec.locks.iter().chain(spec.root_lock.iter()) {
            apply_lock(&mut game, l);
        }
        game.back_to_root();
        let target = pot * spec.target_pct / 100.0;
        let mut expl = compute_exploitability(&game);
        let mut it = 0;
        let mut trace = Vec::new();
        while it < spec.iters && expl > target {
            solve_step(&game, it);
            it += 1;
            if it % 10 == 0 {
                expl = compute_exploitability(&game);
                if spec.trace {
                    trace.push((it, expl / pot * 100.0));
                }
            }
        }
        if spec.trace {
            out["trace"] = json!(trace);
        }
        expl = compute_exploitability(&game);
        let ev_cur = compute_current_ev(&game);
        finalize(&mut game);
        game.back_to_root();
        game.cache_normalized_weights();
        let p = game.current_player();
        let mut root = serde_json::Map::new();
        for pl in 0..2 {
            root.insert(format!("hands{pl}"), json!(holes_to_strings(game.private_cards(pl)).unwrap()));
            root.insert(format!("ev{pl}"), json!(game.expected_values(pl)));
            root.insert(format!("eq{pl}"), json!(game.equity(pl)));
        }
        let acts: Vec<String> = game.available_actions().iter().map(act_str).collect();
        let n = game.private_cards(p).len();
        let strat = game.strategy();
        let rows: Vec<Vec<f32>> = (0..acts.len()).map(|k| strat[k * n..(k + 1) * n].to_vec()).collect();
        let det = game.expected_values_detail(p);
        let evs: Vec<Vec<f32>> = (0..acts.len()).map(|k| det[k * n..(k + 1) * n].to_vec()).collect();
        out["iters"] = json!(it);
        out["expl_pct"] = json!(expl / pot * 100.0);
        out["ev_cur"] = json!(ev_cur);
        out["player"] = json!(p);
        out["acts"] = json!(acts);
        out["strat"] = json!(rows);
        out["action_evs"] = json!(evs);
        out["root"] = Value::Object(root);
        // the equilibrium's strategy at other nodes: the acting player's hands, their reach there, the strategy
        let mut nodes = Vec::new();
        for path in &spec.dump {
            goto(&mut game, path);
            game.cache_normalized_weights();
            let p = game.current_player();
            let acts: Vec<String> = game.available_actions().iter().map(act_str).collect();
            let n = game.private_cards(p).len();
            let strat = game.strategy();
            let rows: Vec<Vec<f32>> = (0..acts.len()).map(|k| strat[k * n..(k + 1) * n].to_vec()).collect();
            nodes.push(json!({"path": path, "player": p, "acts": acts, "strat": rows,
                              "hands": holes_to_strings(game.private_cards(p)).unwrap(),
                              "reach": game.normalized_weights(p)}));
        }
        out["nodes"] = json!(nodes);
    } else {
        // lock: every node of lock_player fixed at the given strategy; the other plays a best response
        for l in &spec.locks {
            apply_lock(&mut game, l);
        }
        game.back_to_root();
        let mes = compute_mes_ev(&game);
        out["mes_ev"] = json!(mes);
        out["br_gain_pct"] = json!(mes[1 - spec.lock_player] / pot * 100.0);
    }
    out["secs"] = json!(t0.elapsed().as_secs_f64());
    println!("{}", out);
}
