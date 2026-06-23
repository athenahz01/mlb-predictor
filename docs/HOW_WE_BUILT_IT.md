# How We Built the MLB Predictor — Full Reference

Four versions of the same story: English Technical, English ELI5, 中文技术版,
中文白话版. Read whichever fits. Everything here reflects the code as it actually
runs, not a plan.

---
---

# PART A — ENGLISH (TECHNICAL)

## A0. The goal and the philosophy
Build a full MLB prediction pipeline that produces both game-level outcomes
(moneyline, run total, run line) and player/in-game props (pitcher strikeouts,
batter hits/total bases/home runs, NRFI, first-to-score) for the 2026 season,
and benchmark every number against the Kalshi market. The brand rule is **"model
vs. market, calibration in public, not betting advice."** We never claim to "beat
the market," only to measure our *edge* against it and own our misses with logged
receipts.

## A1. How we started: research before code
We ran a deep research pass on how the best public and private MLB models work.
The decisive finding: in baseball, the prediction targets we care about (game
lines AND props) all fall out of **one generative simulation**, so we should not
train a separate ML model per market. The architecture bet:

- **Simulation is the spine.** A base-out Monte Carlo engine simulates each game
  thousands of times and reads every outcome — win/loss, total, props — off the
  same joint distribution. One coherent model, internally consistent.
- **The RF+XGBoost ensemble is a future blending layer** on the moneyline only,
  not the core. (Stage 4, not yet built.)
- Calibration + significance-gating + ledger discipline carry over unchanged from
  the WC/NHL/NBA pipelines.

Why simulation over pure ML for baseball specifically: baseball run distributions
are **overdispersed** (variance ≈ 2× mean), so naive Poisson models misfit; an
event-level sim captures this naturally and also lets us attribute every plate
appearance to a batter and every strikeout to a pitcher, which is what props need.

## A2. Data sources (what, why, how)
- **Statcast via `pybaseball`** — pitch-level data (~770k pitches/season). We pull
  a season and derive each player's **per-PA outcome rates** (BB, HBP, 1B, 2B, 3B,
  HR, K, in-play out). This is the simulator's fuel. `ingest/pull_statcast.py`.
- **MLB Stats API** (`statsapi.mlb.com`, free, no key) — schedule, **probable
  pitchers**, **confirmed lineups**, venue. The operational backbone.
  `ingest/pull_mlb_statsapi.py`.
- **FanGraphs / Baseball-Reference via `pybaseball`** — season stats + Steamer/ZiPS
  projections as priors for early-season shrinkage. FanGraphs' legacy endpoint
  returns HTTP 403 (a known upstream break), so we fall back to Baseball-Reference
  automatically. `ingest/pull_fangraphs.py`. Not needed mid-season once Statcast
  samples are large.
- **Kalshi REST API** (public, no auth for market data) — `KXMLBGAME` (winner) and
  `KXMLBTOTAL` (run total) markets, the benchmark. `ingest/pull_kalshi.py`.
- **Retrosheet** (planned) — historical play-by-play to replace the hand-set
  base-running advancement constants with empirical rates.

## A3. The model: from a matchup to a probability
**Step 1 — per-PA probabilities (`features/pa_probabilities.py`).** For a given
batter vs pitcher, we combine their rates with the league baseline using the
multinomial generalization of Bill James' **log5** (Tango's odds-ratio method):

    OR_o = odds(batter_o) · odds(pitcher_o) / odds(league_o),   rate_o = OR_o/(1+OR_o)

then renormalize across the 8 outcomes. We then adjust for:
- **Park** (handedness-controlled HR and hit factors; Coors is the extreme),
- **Platoon** (L/R handedness advantage),
- **Umpire** (strike-zone / K tendency),
- **Times-through-order** (a small offense bump the 3rd time a lineup sees a
  starter),
- **Morey-Cohen HR shrink** — log5 over-estimates HR% at the extremes, so we
  regress the HR term toward league for outlier matchups.

**Step 2 — the simulation (`sim/markov_game.py`).** A 24-state base-out engine
(3 out-counts × 8 base configurations). Each plate appearance samples an outcome
from the matchup probabilities; runners advance by documented (refinable) rules;
the starter is pulled for the bullpen at a pitch/IP cap; extras use the Manfred
ghost runner; home-field advantage is a small home-offense bump. We run 20,000
games and aggregate the joint distribution.

**Step 3 — outputs.** From that one run: P(home win), expected total + over grid,
run line, NRFI/YRFI, F5, first-to-score, each starter's strikeout distribution,
and each batter's hits / total bases / HR / P(≥1 HR) / P(≥2 hits).

## A4. Market benchmarking
- **De-vig (`market/devig.py`).** Market prices include vig (overround > 100%).
  We strip it with the **power method** (solve for k so Σ pᵢᵏ = 1), which removes
  more juice from favorites than naive normalization — important for lopsided MLB
  moneylines.
- **Matching (`market/kalshi_match.py`).** We parse the Kalshi ticker
  (`KXMLBGAME-<date><time><AWAY><HOME>-<TEAM>`) to find a game's two winner markets
  and its total ladder, de-vig, and compute **edge = model − market**.
- **Guards.** A **thin-market flag** when the bid/ask spread is wide (the de-vig is
  noise), and a **settled-market filter** so finished games (pinned 100/0) are
  excluded.
- **Price parsing.** Kalshi's 2026 API moved prices to `*_dollars` string fields;
  we read those and normalize to cents.

## A5. The ledger (`ledger/ledger.py`)
Append-only receipts. We **log the model number before checking the market**
(out-of-sample integrity), attach the de-vigged market price, and on resolution
score log-loss, Brier, accuracy, **model-vs-market delta**, **CLV** (did our number
beat the close), and a reliability curve for calibration.

## A6. Discipline / house rules
- **Snapshot discipline** — pin one frozen pull per matchday; never auto-download
  mid-run (the data-hygiene lesson).
- **Listed-pitcher / lineup confirmation** — re-run on any scratch; one starter is
  30–40% of run-scoring. The card flags CONFIRMED vs PROJECTED.
- **Significance gate** — no model version ships without clearing paired-bootstrap
  **p < 0.05** on held-out log-loss. (Harness in `backtest/walk_forward.py`.)
- **Honest framing** — "edge," never "beating the market."

## A7. Lessons caught while building
- **Base-running bug** suppressed run totals to ~6.9; fixing batter placement on
  doubles/singles brought it to a realistic ~8.6.
- **Stale player-team knowledge** (mine, not the code's): players traded after the
  training cutoff looked "wrong" but the API was correct — log the API's truth, not
  memory. This is exactly why snapshot discipline exists.
- **Kalshi API schema change** — prices moved to `*_dollars`; the old fields read
  as empty until we updated the parser.

## A8. Status and what's next
- **Done (the operational core):** scaffold, data plumbing, log5+simulation engine,
  loader joining real lineups, de-vig + Kalshi edge + guards, ledger.
- **Done — baselines + the gate:** Elo + Pythagenpat baselines, plus a calibrated
  RF+XGBoost "second model" and a paired-bootstrap significance test that only
  ships a model when it beats the baseline at p<0.05 (`models/ensemble.py`).
- **Later — Stage 4:** RF+XGBoost ensemble stacked with the sim + isotonic/beta
  calibration. **Stage 5 polish:** reliability-diagram reports, auto-CLV.

## A9. Daily workflow
```
python -m ingest.pull_mlb_statsapi --date today     # probables + lineups
python -m ingest.pull_kalshi --series winner         # market (winner)
python -m ingest.pull_kalshi --series total          # market (totals)
python -m market.kalshi_match                         # which games are live now
python run_predict.py --game "NYM@PHI" --sims 20000  # predict (add --log to record)
```

---
---

# PART B — ENGLISH (ELI5)

Think of it like building a **weather forecast for baseball**, then checking the
forecast against what the betting market thinks.

**The big idea.** Instead of guessing each thing separately (who wins, how many
runs, will Schwarber homer), we built a video-game version of the game and let the
computer **play it 20,000 times**. Count up all those pretend games and you get
the odds of everything at once, and they all agree with each other.

**How the pretend game knows what happens.** For every at-bat, we mix three things:
how good the hitter is, how good the pitcher is, and what's normal for the league.
Then we nudge it for the ballpark (Coors makes homers), lefty-vs-righty, the
umpire, and so on. That gives the chance of a walk, single, homer, strikeout, etc.,
and the computer rolls the dice.

**Where the numbers come from.** Free baseball data: every pitch thrown last
season (to learn each player), the official MLB schedule (to know who's pitching
and batting today), and Kalshi (the market we compare against).

**The honesty part.** We always write our prediction down **before** we look at the
market price, so we can't fool ourselves. Then we compare. If our number and the
market's number are close, good. If they're far apart, that's a flag to look
closer — not an automatic bet. Later we check who was right and keep score.

**Rules we never break.** Use the official lineup, not a guess (one pitcher is a
third of the game). Don't trust a market with no real prices in it. Never say
"we beat the market" — only "here's our edge."

**Where we are.** The machine works end to end today: it pulls today's games,
simulates them, prints win odds + run totals + player props, and shows how that
compares to Kalshi. Next we're adding two simpler "sanity-check" models so we can
prove the big simulation is actually better before we ever trust a new version.

---
---

# PART C — 中文（技术版）

## C0. 目标与理念
为 2026 赛季搭建一条完整的 MLB 预测流水线，既能产出**整场比赛层级**的结果（胜负盘、
总分、让分盘），也能产出**球员/局内 prop**（投手三振、打者安打/垒打数/全垒打、首局
无失分 NRFI、谁先得分），并把每一个数字拿去与 **Kalshi 市场**对标。品牌原则是
**「模型对市场、公开校准、非投注建议」**。我们从不宣称「击败市场」，只衡量自己相对
市场的 **edge（优势）**，并用记录在案的 receipts 公开认错。

## C1. 起步：先研究再写码
我们先做了一轮深度研究。关键结论：在棒球里，我们关心的预测目标（整场盘口 + props）
都可以从**同一个生成式模拟**里导出，所以不该为每个市场单独训练一个 ML 模型。架构选择：

- **模拟是主干。** 一个 base-out 蒙特卡洛引擎把每场比赛模拟上千次，所有结果——胜负、
  总分、props——都从同一个联合分布里读出，内部一致。
- **RF+XGBoost 集成模型**是未来仅用于胜负盘的**融合层**，不是核心（Stage 4，尚未建）。
- 校准、显著性门槛、账本纪律全部沿用 WC/NHL/NBA 的做法。

为何棒球用模拟而非纯 ML：棒球得分分布**过度离散**（方差约为均值的 2 倍），朴素泊松
会拟合不良；事件级模拟天然能捕捉这一点，还能把每个打席归到某打者、每次三振归到某投手，
这正是 props 所需。

## C2. 数据来源（用什么、为什么、怎么用）
- **Statcast（经 `pybaseball`）**——逐球数据（每季约 77 万球）。我们拉一个赛季，导出
  每名球员的**每打席结果费率**（保送、触身、一/二/三垒安打、全垒打、三振、场内出局）。
  这是模拟器的燃料。`ingest/pull_statcast.py`。
- **MLB 官方 API**（`statsapi.mlb.com`，免费免钥）——赛程、**先发投手**、**确认打线**、
  场地。运营骨干。`ingest/pull_mlb_statsapi.py`。
- **FanGraphs / Baseball-Reference**——赛季数据 + Steamer/ZiPS 预测作为先验。FanGraphs
  旧接口返回 403（已知上游故障），我们自动回退到 Baseball-Reference。
  `ingest/pull_fangraphs.py`。赛季中期样本够大后可不依赖它。
- **Kalshi REST API**（行情数据公开免钥）——`KXMLBGAME`（胜负）与 `KXMLBTOTAL`（总分）
  市场，作为基准。`ingest/pull_kalshi.py`。
- **Retrosheet**（计划中）——历史逐打席数据，用以把手设的跑垒推进常数替换为经验值。

## C3. 模型：从对位到概率
**第一步——每打席概率（`features/pa_probabilities.py`）。** 用 Bill James 的 **log5**
多项式推广（Tango 的赔率比法）把打者、投手与联盟基线结合：

    OR_o = odds(打者_o) · odds(投手_o) / odds(联盟_o)，  rate_o = OR_o/(1+OR_o)

再在 8 种结果上归一化。随后调整：**球场**（按左右手分的全垒打/安打因子，Coors 极端）、
**左右投打**、**主审**（好球带/三振倾向）、**第几次面对打线**（先发第三轮起的小幅进攻
加成）、以及 **Morey-Cohen 全垒打收缩**（log5 在极端处高估 HR%，故把 HR 项往联盟回归）。

**第二步——模拟（`sim/markov_game.py`）。** 24 状态 base-out 引擎（3 种出局数 × 8 种
垒包组合）。每个打席按对位概率抽样；跑者按可细化的规则推进；先发达到投球数/局数上限后
换牛棚；延长赛用 Manfred 幽灵跑者；主场优势是对主队进攻的小幅加成。跑 20,000 场后聚合
联合分布。

**第三步——输出。** 同一次运行即得：主队胜率、期望总分 + over 网格、让分盘、NRFI/YRFI、
前五局、谁先得分、两位先发的三振分布，以及每位打者的安打/垒打数/全垒打/至少一轰/至少两安。

## C4. 市场对标
- **去水 De-vig（`market/devig.py`）。** 盘口含水（总和 > 100%）。我们用**幂方法**
  （求 k 使 Σ pᵢᵏ = 1），它从热门一侧去掉更多水，对 MLB 这种一边倒的胜负盘很重要。
- **匹配（`market/kalshi_match.py`）。** 解析 Kalshi 代码
  （`KXMLBGAME-<日期><时间><客><主>-<队>`），找出该场的两个胜负市场与总分梯队，去水后算
  **edge = 模型 − 市场**。
- **护栏。** 买卖价差过大时打**薄市场标记**（去水是噪声）；并有**已结算过滤**，剔除
  已结束的比赛（钉在 100/0）。
- **价格解析。** Kalshi 2026 接口把价格改到 `*_dollars` 字符串字段，我们读取并换算成美分。

## C5. 账本（`ledger/ledger.py`）
只增不改的 receipts。**先记录模型数字，再看市场**（样本外诚实度），附上去水后的市场价；
结算后计算 log-loss、Brier、准确率、**模型对市场差值**、**CLV**（我们的数字是否优于收盘），
以及用于校准的可靠性曲线。

## C6. 纪律 / 内部规矩
- **快照纪律**——每个比赛日固定一次冻结拉取，绝不在运行中自动下载。
- **先发/打线确认**——任何先发被划掉都要重跑；单一先发占 30–40% 得分环境。卡片标注
  CONFIRMED / PROJECTED。
- **显著性门槛**——新模型版本必须在留出集 log-loss 上通过配对自助法 **p < 0.05** 才能上线。
- **诚实措辞**——只说「edge」，不说「击败市场」。

## C7. 搭建中抓到的教训
- **跑垒 bug** 把总分压到约 6.9；修正二垒/一垒安打后的打者落点后回到现实的约 8.6。
- **我对球员所属队的过时认知**（是我的问题，不是代码）：训练截止后被交易的球员看起来「不对」，
  但 API 是对的——记录 API 的事实，而非记忆。这正是快照纪律存在的理由。
- **Kalshi 接口改版**——价格移到 `*_dollars`；旧字段读出来是空，直到我们更新解析器。

## C8. 现状与下一步
- **已完成（运营核心）：** 脚手架、数据管道、log5+模拟引擎、接真实打线的加载器、去水 +
  Kalshi edge + 护栏、账本。
- **下一步 Stage 2：** Elo + Pythagenpat 基准（模拟必须先在 p<0.05 上击败的底线）。
- **之后 Stage 4：** RF+XGBoost 集成与模拟堆叠 + isotonic/beta 校准。**Stage 5 收尾：**
  可靠性图报告、自动 CLV。

## C9. 每日流程
```
python -m ingest.pull_mlb_statsapi --date today     # 先发 + 打线
python -m ingest.pull_kalshi --series winner         # 市场（胜负）
python -m ingest.pull_kalshi --series total          # 市场（总分）
python -m market.kalshi_match                         # 现在哪些比赛有活跃盘口
python run_predict.py --game "NYM@PHI" --sims 20000  # 预测（加 --log 记录）
```

---
---

# PART D — 中文（白话版）

把它想成做一个**棒球版的天气预报**，再拿这个预报去对一对赌盘怎么想。

**核心点子。** 我们不去一样一样猜（谁赢、几分、Schwarber 会不会轰一发），而是做了一个
电子游戏版的比赛，让电脑**自己打 20,000 遍**。把这两万场假比赛统计一下，所有结果的概率
一次就都出来了，而且彼此自洽。

**假比赛怎么知道会发生什么。** 每个打席，我们把三样东西混在一起：打者多强、投手多强、
联盟平均什么样。然后按球场（Coors 容易出全垒打）、左右手、主审等等微调，得出保送、安打、
全垒打、三振的概率，电脑掷骰子。

**数字哪来的。** 都是免费棒球数据：上赛季每一个投球（学每位球员）、官方赛程（知道今天谁
先发谁上场）、还有 Kalshi（拿来对比的市场）。

**诚实那部分。** 我们永远在**看市场价之前**先把自己的预测写下来，免得自欺。然后再比。
我们的数字和市场接近，很好；差很远，那是一个「值得再看一眼」的信号，不是自动下注。事后
再核对谁对了，记分。

**绝不破的规矩。** 用官方打线，不要猜（一个投手就占整场三分之一）。市场里没有真实报价
就别信。永远不说「我们赢了市场」，只说「这是我们的 edge」。

**现在到哪了。** 这台机器今天已经能从头跑到尾：拉今天的比赛、模拟、印出胜率 + 总分 +
球员 prop，并显示和 Kalshi 的对比。下一步我们再加两个更简单的「检查用」模型，好在真正
信任新版本之前，证明那个大模拟确实更准。
