use wasm_bindgen::prelude::*;
use web_sys::{window, Document, Element};
use std::cell::RefCell;
use serde::{Serialize, Deserialize};

// ---------------- Neural network ----------------

const HID: usize = 128;

#[derive(Serialize, Deserialize)]
struct Weights {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct Net {
    w1: Vec<f32>, // HID x 9
    b1: Vec<f32>, // HID
    w2: Vec<f32>, // 9 x HID
    b2: Vec<f32>, // 9
    #[serde(skip)]
    z1: Vec<f32>, // HID pre-activations
    #[serde(skip)]
    h: Vec<f32>,  // HID activations
}

impl Net {
    fn weights(&self) -> Weights {
        Weights {
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            w2: self.w2.clone(),
            b2: self.b2.clone(),
        }
    }

    fn from_weights(w: Weights) -> Self {
        Net {
            w1: w.w1,
            b1: w.b1,
            w2: w.w2,
            b2: w.b2,
            z1: vec![0.0; HID],
            h: vec![0.0; HID],
        }
    }
}

fn save_net_local(net: &Net) {
    if let Some(storage) = window().and_then(|w| w.local_storage().ok()).flatten() {
        if let Ok(s) = serde_json::to_string(&net.weights()) {
            let _ = storage.set_item("ttt_net_v1", &s);
        }
    }
}

fn load_net_local() -> Option<Net> {
    let storage = window()?.local_storage().ok()??;
    let s = storage.get_item("ttt_net_v1").ok()??;
    serde_json::from_str::<Weights>(&s).ok().map(Net::from_weights)
}

// --- Shared brain over HTTP (falls back to localStorage when offline) ---

async fn fetch_brain() -> Option<Net> {
    let window = window()?;
    let resp_val = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str("/api/brain"))
        .await
        .ok()?;
    let resp: web_sys::Response = resp_val.dyn_into().ok()?;
    if !resp.ok() {
        return None;
    }
    let text = wasm_bindgen_futures::JsFuture::from(resp.text().ok()?)
        .await
        .ok()?;
    let s = text.as_string()?;
    serde_json::from_str::<Weights>(&s).ok().map(Net::from_weights)
}

async fn put_brain_json(s: String) {
    let headers = web_sys::Headers::new().unwrap();
    let _ = headers.set("content-type", "application/json");
    let init = web_sys::RequestInit::new();
    init.set_method("PUT");
    init.set_headers(&headers);
    init.set_body(&JsValue::from_str(&s));
    if let (Some(window), Ok(req)) = (
        window(),
        web_sys::Request::new_with_str_and_init("/api/brain", &init),
    ) {
        let _ = wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&req)).await;
    }
}

fn upload_brain(net: &Net) {
    save_net_local(net);
    if let Ok(s) = serde_json::to_string(&net.weights()) {
        wasm_bindgen_futures::spawn_local(put_brain_json(s));
    }
}

// Deterministic small PRNG so the crate stays dependency-light.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 = (self.0 ^ (self.0 >> 33)).wrapping_mul(0xff51afd7ed558ccd);
        self.0
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

impl Net {
    fn new(rng: &mut Rng) -> Self {
        let mut w1 = vec![0.0; HID * 9];
        for v in w1.iter_mut() {
            *v = (rng.f32() - 0.5) * 0.5;
        }
        let mut w2 = vec![0.0; 9 * HID];
        for v in w2.iter_mut() {
            *v = (rng.f32() - 0.5) * 0.2;
        }
        Net {
            w1,
            b1: vec![0.0; HID],
            w2,
            b2: vec![0.0; 9],
            z1: vec![0.0; HID],
            h: vec![0.0; HID],
        }
    }

    fn forward(&mut self, x: &[f32; 9]) -> [f32; 9] {
        for j in 0..HID {
            let mut s = self.b1[j];
            for i in 0..9 {
                s += self.w1[j * 9 + i] * x[i];
            }
            self.z1[j] = s;
            self.h[j] = if s > 0.0 { s } else { 0.0 };
        }
        let mut out = [0.0f32; 9];
        for k in 0..9 {
            let mut s = self.b2[k];
            for j in 0..HID {
                s += self.w2[k * HID + j] * self.h[j];
            }
            out[k] = s;
        }
        out
    }

    // REINFORCE-style update: push probability of taken move toward advantage.
    fn train_sample(&mut self, x: &[f32; 9], legal: &[bool; 9], taken: usize, adv: f32, lr: f32) {
        let logits = self.forward(x);
        let mut max = f32::NEG_INFINITY;
        for k in 0..9 {
            if legal[k] && logits[k] > max {
                max = logits[k];
            }
        }
        let mut p = [0.0f32; 9];
        let mut sum = 0.0f32;
        for k in 0..9 {
            if legal[k] {
                p[k] = (logits[k] - max).exp();
                sum += p[k];
            }
        }
        for k in 0..9 {
            p[k] = if legal[k] { p[k] / sum } else { 0.0 };
        }
        // dz = adv * (p - onehot(taken))
        let mut dz = [0.0f32; 9];
        for k in 0..9 {
            dz[k] = adv * p[k];
        }
        dz[taken] = adv * (p[taken] - 1.0);

        let mut dh = [0.0f32; HID];
        for j in 0..HID {
            let mut s = 0.0f32;
            for k in 0..9 {
                s += dz[k] * self.w2[k * HID + j];
            }
            dh[j] = if self.z1[j] > 0.0 { s } else { 0.0 };
        }
        for k in 0..9 {
            self.b2[k] -= lr * dz[k];
            for j in 0..HID {
                self.w2[k * HID + j] -= lr * (dz[k] * self.h[j] + 0.0001 * self.w2[k * HID + j]);
            }
        }
        for j in 0..HID {
            self.b1[j] -= lr * dh[j];
            for i in 0..9 {
                self.w1[j * 9 + i] -= lr * (dh[j] * x[i] + 0.0001 * self.w1[j * 9 + i]);
            }
        }
    }

    fn policy(&mut self, x: &[f32; 9], legal: &[bool; 9]) -> [f32; 9] {
        let logits = self.forward(x);
        let mut max = f32::NEG_INFINITY;
        for k in 0..9 {
            if legal[k] && logits[k] > max {
                max = logits[k];
            }
        }
        let mut p = [0.0f32; 9];
        let mut sum = 0.0f32;
        for k in 0..9 {
            if legal[k] {
                p[k] = (logits[k] - max).exp();
                sum += p[k];
            }
        }
        for k in 0..9 {
            p[k] = if sum > 0.0 { p[k] / sum } else { 0.0 };
        }
        p
    }
}

// ---------------- Game logic ----------------

const LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

fn winner(board: &[i8; 9]) -> i8 {
    for line in LINES.iter() {
        let a = board[line[0]];
        if a != 0 && a == board[line[1]] && a == board[line[2]] {
            return a;
        }
    }
    0
}

fn board_full(board: &[i8; 9]) -> bool {
    board.iter().all(|&c| c != 0)
}

fn features(board: &[i8; 9], player: i8) -> [f32; 9] {
    let mut x = [0.0f32; 9];
    for i in 0..9 {
        if board[i] == player {
            x[i] = 1.0;
        } else if board[i] != 0 {
            x[i] = -1.0;
        }
    }
    x
}

fn legal_moves(board: &[i8; 9]) -> [bool; 9] {
    let mut m = [false; 9];
    for i in 0..9 {
        m[i] = board[i] == 0;
    }
    m
}

// ---------------- Self-play training ----------------

fn train(net: &mut Net, games: usize, rng: &mut Rng) {
    for _ in 0..games {
        let mut board = [0i8; 9];
        let mut player: i8 = 1;
        // (x, legal, taken, player)
        let mut traj: Vec<([f32; 9], [bool; 9], usize, i8)> = Vec::new();
        let mut result: i8 = 0; // winner, 0 => draw
        loop {
            let legal = legal_moves(&board);
            let x = features(&board, player);
            let p = net.policy(&x, &legal);
            // epsilon-greedy: mostly sample from policy, sometimes random
            let mv = if rng.f32() < 0.15 {
                let options: Vec<usize> = (0..9).filter(|&i| legal[i]).collect();
                options[rng.range(options.len())]
            } else {
                let mut r = rng.f32();
                let mut chosen = 0;
                for i in 0..9 {
                    if legal[i] {
                        r -= p[i];
                        if r <= 0.0 {
                            chosen = i;
                            break;
                        }
                        chosen = i;
                    }
                }
                chosen
            };
            traj.push((x, legal, mv, player));
            board[mv] = player;
            let w = winner(&board);
            if w != 0 {
                result = w;
                break;
            }
            if board_full(&board) {
                break;
            }
            player = if player == 1 { 2 } else { 1 };
        }
        for (x, legal, mv, p) in traj {
            let adv = if result == 0 {
                0.0
            } else if result == p {
                1.0
            } else {
                -1.0
            };
            // update each side's moves; advantage sign flips for the loser
            net.train_sample(&x, &legal, mv, adv * 0.5, 0.03);
        }
    }
}

impl Game {
    // Learn from the finished human game, then sharpen with a burst of
    // self-play, and persist the updated weights to localStorage.
    fn learn_from_game(&mut self, ai_outcome: f32) {
        for (x, legal, mv) in self.traj.drain(..) {
            self.ai.train_sample(&x, &legal, mv, ai_outcome * 0.5, 0.05);
        }
        train(&mut self.ai, 25, &mut self.rng);
        upload_brain(&self.ai);
    }
}

// ---------------- UI ----------------

struct Game {
    board: [i8; 9],
    turn: i8, // 1 = human (X), 2 = AI (O)
    over: bool,
    ai: Net,
    rng: Rng,
    // (x, legal, taken) for the AI's moves in the current human game
    traj: Vec<([f32; 9], [bool; 9], usize)>,
}

thread_local! {
    static GAME: RefCell<Game> = RefCell::new(Game {
        board: [0; 9],
        turn: 1,
        over: false,
        ai: Net::new(&mut Rng(0xC0FFEE)),
        rng: Rng(0x12345678),
        traj: Vec::new(),
    });
    static HANDLES: RefCell<Vec<wasm_bindgen::closure::Closure<dyn FnMut()>>> = RefCell::new(Vec::new());
}

fn document() -> Document {
    window().unwrap().document().unwrap()
}

fn el(tag: &str) -> Element {
    document().create_element(tag).unwrap()
}

fn set_status(msg: &str) {
    if let Some(node) = document().get_element_by_id("status") {
        node.set_inner_html(msg);
    }
}

fn render() {
    GAME.with(|g| {
        let g = g.borrow();
        for i in 0..9 {
            if let Some(cell) = document().get_element_by_id(&format!("cell-{}", i)) {
                let txt = match g.board[i] {
                    1 => "X",
                    2 => "O",
                    _ => "",
                };
                cell.set_inner_html(txt);
                let mut class = "cell".to_string();
                if g.board[i] == 1 {
                    class.push_str(" x");
                } else if g.board[i] == 2 {
                    class.push_str(" o");
                }
                cell.set_attribute("class", &class).unwrap();
            }
        }
        if g.over {
        } else if g.turn == 1 {
            set_status("Your turn — you are <b>X</b>");
        } else {
            set_status("AI thinking…");
        }
    });
}

fn ai_move() {
    GAME.with(|g| {
        let mut g = g.borrow_mut();
        let legal = legal_moves(&g.board);
        let x = features(&g.board, 2);
        let p = g.ai.policy(&x, &legal);
        let mut best = 0;
        let mut bestv = f32::NEG_INFINITY;
        for i in 0..9 {
            if legal[i] && p[i] > bestv {
                bestv = p[i];
                best = i;
            }
        }
        g.traj.push((x, legal, best));
        g.board[best] = 2;
        let w = winner(&g.board);
        if w == 2 {
            g.over = true;
            set_status("<b>AI wins!</b>");
            g.learn_from_game(1.0);
        } else if board_full(&g.board) {
            g.over = true;
            set_status("<b>Draw.</b>");
            g.learn_from_game(0.0);
        } else {
            g.turn = 1;
        }
    });
    render();
}

fn human_move(i: usize) {
    let mut play = false;
    GAME.with(|g| {
        let mut g = g.borrow_mut();
        if g.over || g.turn != 1 || g.board[i] != 0 {
            return;
        }
        g.board[i] = 1;
        let w = winner(&g.board);
        if w == 1 {
            g.over = true;
            set_status("<b>You win!</b> Nice.");
            g.learn_from_game(-1.0);
        } else if board_full(&g.board) {
            g.over = true;
            set_status("<b>Draw.</b>");
            g.learn_from_game(0.0);
        } else {
            g.turn = 2;
            play = true;
        }
    });
    render();
    if play {
        ai_move();
    }
}

#[wasm_bindgen(start)]
pub fn main() {
    // Try the shared server brain, then this browser's saved brain,
    // otherwise train a fresh network and publish it.
    wasm_bindgen_futures::spawn_local(async {
        let loaded = match fetch_brain().await {
            Some(net) => Some(net),
            None => load_net_local(),
        };
        GAME.with(|g| {
            let mut g = g.borrow_mut();
            match loaded {
                Some(saved) => g.ai = saved,
                None => {
                    let mut rng_seed = Rng(0xDEADBEEF);
                    train(&mut g.ai, 6000, &mut rng_seed);
                    upload_brain(&g.ai);
                }
            }
        });
        build_ui();
    });
}

fn build_ui() {

    let doc = document();
    let body = doc.body().unwrap();
    body.set_inner_html(
        r#"
        <h1>Neural Tic&#8211;Tac&#8211;Toe</h1>
        <p class="sub">A neural network (Rust + WebAssembly) that learns from every game and keeps improving. Its weights are shared by all players via the server.</p>
        <div id="status"></div>
        <div id="board"></div>
        <button id="restart">New game</button>
        "#,
    );

    let board_el = doc.get_element_by_id("board").unwrap();
    for i in 0..9 {
        let cell = el("div");
        cell.set_attribute("id", &format!("cell-{}", i)).unwrap();
        cell.set_attribute("class", "cell").unwrap();
        let cb = wasm_bindgen::closure::Closure::<dyn FnMut()>::new(move || human_move(i));
        cell.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())
            .unwrap();
        board_el.append_child(&cell).unwrap();
        HANDLES.with(|h| h.borrow_mut().push(cb));
    }
    let restart = doc.get_element_by_id("restart").unwrap();
    let cb = wasm_bindgen::closure::Closure::<dyn FnMut()>::new(|| {
        GAME.with(|g| {
            let mut g = g.borrow_mut();
            g.board = [0; 9];
            g.turn = 1;
            g.over = false;
            g.traj.clear();
        });
        render();
        set_status("Your turn — you are <b>X</b>");
    });
    restart
        .add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())
        .unwrap();
    HANDLES.with(|h| h.borrow_mut().push(cb));

    render();
    set_status("Your turn — you are <b>X</b>");
}
