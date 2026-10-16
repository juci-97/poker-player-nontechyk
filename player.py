import random
import sys
import time
from collections import Counter, deque

RANKS = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
         '10': 10, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
SUITS = {'spades': 0, 'hearts': 1, 'clubs': 2, 'diamonds': 3}
FULL_DECK = [(r, s) for r in range(2, 15) for s in range(4)]

MC_TIME_BUDGET = 0.1  # seconds per decision; total limit is ~30s per sit'n'go
MC_MAX_SIMS = 1500


def log(*args):
    print(*args, file=sys.stderr)


def parse_card(card):
    return RANKS[str(card['rank'])], SUITS.get(card['suit'], 0)


def straight_high(unique_desc):
    """Highest straight top card from a descending list of unique ranks, or 0."""
    ranks = list(unique_desc)
    if 14 in ranks:
        ranks.append(1)
    run = 1
    for i in range(1, len(ranks)):
        if ranks[i] == ranks[i - 1] - 1:
            run += 1
            if run >= 5:
                return ranks[i] + 4
        else:
            run = 1
    return 0


def hand_value(cards):
    """Comparable tuple for the best 5-card hand in 5..7 cards."""
    ranks = sorted((c[0] for c in cards), reverse=True)
    suit_counts = Counter(c[1] for c in cards)
    flush_suit, flush_count = suit_counts.most_common(1)[0]
    flush_ranks = None
    if flush_count >= 5:
        flush_ranks = sorted((c[0] for c in cards if c[1] == flush_suit), reverse=True)
        sf = straight_high(flush_ranks)
        if sf:
            return (8, sf)

    groups = sorted(Counter(ranks).items(), key=lambda x: (x[1], x[0]), reverse=True)
    top_rank, top_count = groups[0]
    second_count = groups[1][1] if len(groups) > 1 else 0

    if top_count == 4:
        return (7, top_rank, max(r for r in ranks if r != top_rank))
    if top_count == 3 and second_count >= 2:
        return (6, top_rank, groups[1][0])
    if flush_ranks:
        return (5,) + tuple(flush_ranks[:5])
    st = straight_high(sorted(set(ranks), reverse=True))
    if st:
        return (4, st)
    if top_count == 3:
        kickers = [r for r in ranks if r != top_rank][:2]
        return (3, top_rank) + tuple(kickers)
    if top_count == 2 and second_count == 2:
        p1, p2 = groups[0][0], groups[1][0]
        kicker = max(r for r in ranks if r != p1 and r != p2)
        return (2, p1, p2, kicker)
    if top_count == 2:
        kickers = [r for r in ranks if r != top_rank][:3]
        return (1, top_rank) + tuple(kickers)
    return (0,) + tuple(ranks[:5])


def equity(hole, board, num_opponents):
    """Monte Carlo win probability vs random hands (ties count fractionally)."""
    num_opponents = max(1, min(num_opponents, 6))
    known = set(hole) | set(board)
    deck = [c for c in FULL_DECK if c not in known]
    need_board = 5 - len(board)
    draw = need_board + 2 * num_opponents
    deadline = time.time() + MC_TIME_BUDGET
    score = 0.0
    sims = 0
    while sims < MC_MAX_SIMS and (sims < 100 or time.time() < deadline):
        sample = random.sample(deck, draw)
        full_board = board + sample[:need_board]
        mine = hand_value(hole + full_board)
        ties = 0
        lost = False
        for i in range(num_opponents):
            start = need_board + 2 * i
            opp = hand_value(sample[start:start + 2] + full_board)
            if opp > mine:
                lost = True
                break
            if opp == mine:
                ties += 1
        if not lost:
            score += 1.0 / (ties + 1)
        sims += 1
    return score / sims


# Opponent tracking across hands; the server process stays up between requests.
SHOVES = Counter()
HANDS_SEEN = Counter()
RECENT = {}
RAISES = {}
AGG = {}
_counted = set()


def observe(gs, players, my_index, bb):
    if len(_counted) > 20000:
        _counted.clear()
    hand = (gs.get('game_id'), gs.get('round'))
    for i, p in enumerate(players):
        if i == my_index or p.get('status') == 'out':
            continue
        name = p.get('name')
        if (hand, name, 'seen') not in _counted:
            _counted.add((hand, name, 'seen'))
            HANDS_SEEN[name] += 1
            RECENT.setdefault(name, deque(maxlen=20)).append(0)
            RAISES.setdefault(name, deque(maxlen=20)).append(0)
            AGG.setdefault(name, deque(maxlen=20)).append(0)
        if not gs.get('community_cards') and p.get('bet', 0) >= 8 * bb:
            RAISES[name][-1] = 1
        if not gs.get('community_cards') and p.get('bet', 0) > bb:
            AGG[name][-1] = 1
        total = p.get('bet', 0) + p.get('stack', 0)
        big = p.get('bet', 0) >= max(10 * bb, 0.8 * total)
        if not gs.get('community_cards') and big and (hand, name, 'shove') not in _counted:
            _counted.add((hand, name, 'shove'))
            SHOVES[name] += 1
            RECENT[name][-1] = 1


def is_maniac(name):
    # Judge on recent hands only, so a bot that tightens up loses the label quickly.
    recent = RECENT.get(name, ())
    shoves = sum(recent)
    return shoves >= 2 and shoves >= 0.3 * len(recent)


def agg_rate(name):
    # Share of recent hands where this player raised preflop; None until we have seen enough.
    recent = AGG.get(name, ())
    return sum(recent) / float(len(recent)) if len(recent) >= 8 else None


def is_loose(name):
    rate = agg_rate(name)
    return is_maniac(name) or is_aggressor(name) or (rate is not None and rate >= 0.4)


def is_hyper(name):
    # Shoves most hands (e.g. every hand heads-up): its range really is random.
    recent = RECENT.get(name, ())
    return len(recent) >= 8 and sum(recent) >= 0.6 * len(recent)


def is_aggressor(name):
    # Opens big preflop most hands (e.g. a fixed 100-chip raise to steal blinds).
    recent = RAISES.get(name, ())
    return sum(recent) >= 3 and sum(recent) >= 0.4 * len(recent)


def pot_odds_of(to_call, pot):
    return to_call / float(pot + to_call) if to_call else 0.0


def chen_score(hole):
    (r1, s1), (r2, s2) = hole
    hi, lo = max(r1, r2), min(r1, r2)
    points = {14: 10, 13: 8, 12: 7, 11: 6}.get(hi, hi / 2.0)
    if r1 == r2:
        return max(5, points * 2)
    if s1 == s2:
        points += 2
    gap = hi - lo - 1
    points -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 12:
        points += 1
    return points


_COMBOS = []


def range_combos(f):
    # All 1326 starting hands, strongest first by Chen score; the top f fraction is the range.
    if not _COMBOS:
        combos = [(a, b) for i, a in enumerate(FULL_DECK) for b in FULL_DECK[i + 1:]]
        combos.sort(key=lambda h: -chen_score(list(h)))
        _COMBOS.extend(combos)
    return _COMBOS[:max(20, int(len(_COMBOS) * min(1.0, max(0.05, f))))]


def equity_vs_range(hole, f):
    """Preflop heads-up equity against a player who only plays its top f of hands."""
    rng = [h for h in range_combos(f) if h[0] not in hole and h[1] not in hole]
    deck = [c for c in FULL_DECK if c not in hole]
    deadline = time.time() + MC_TIME_BUDGET
    score = 0.0
    sims = 0
    while sims < MC_MAX_SIMS and (sims < 100 or time.time() < deadline):
        opp = random.choice(rng)
        rest = [c for c in deck if c != opp[0] and c != opp[1]]
        board = random.sample(rest, 5)
        mine = hand_value(list(hole) + board)
        theirs = hand_value(list(opp) + board)
        score += 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
        sims += 1
    return score / sims


class Player:
    VERSION = "nontechyk v25"

    def betRequest(self, game_state):
        try:
            return int(self.decide(game_state))
        except Exception as e:
            log("betRequest error:", repr(e))
            try:
                me = game_state['players'][game_state['in_action']]
                to_call = game_state['current_buy_in'] - me['bet']
                return to_call if to_call <= 2 * game_state.get('small_blind', 0) * 2 else 0
            except Exception:
                return 0

    def showdown(self, game_state):
        pass

    def decide(self, gs):
        players = gs['players']
        me = players[gs['in_action']]
        stack = me['stack']
        to_call = max(0, gs['current_buy_in'] - me['bet'])
        min_raise = gs.get('minimum_raise', 0)
        pot = gs.get('pot', 0)
        bb = max(1, 2 * gs.get('small_blind', 1))

        hole = [parse_card(c) for c in me['hole_cards']]
        board = [parse_card(c) for c in gs.get('community_cards', [])]

        opponents = [p for i, p in enumerate(players)
                     if i != gs['in_action'] and p['status'] == 'active']
        alive = [p for i, p in enumerate(players)
                 if i != gs['in_action'] and p['status'] != 'out']
        num_opp = max(1, len(opponents))
        max_opp_stack = max([p['stack'] + p['bet'] for p in opponents] or [stack])
        effective = min(stack + me['bet'], max_opp_stack)

        def call():
            return min(to_call, stack)

        def check_fold():
            return 0

        def raise_by(amount):
            amount = max(min_raise, int(amount))
            total = to_call + amount
            # If this commits most of our stack, just shove.
            if total >= stack * 0.6:
                return stack
            return min(total, stack)

        def all_in():
            return stack

        observe(gs, players, gs['in_action'], bb)
        maniacs = [p for p in opponents if is_maniac(p['name'])]

        # Preflop shove from a bot that shoves nearly every hand: its range is random,
        # so call whenever we beat a random hand by more than the pot odds.
        if not board and to_call > 2 * bb:
            bettors = [p for p in opponents if p['bet'] > 2 * bb]
            # A single shover: estimate its range from how often it shoves, and call
            # whenever our equity against that range beats the price.
            if len(bettors) == 1 and bettors[0]['bet'] >= 0.8 * (bettors[0]['bet'] + bettors[0]['stack']):
                recent = RECENT.get(bettors[0]['name'], ())
                width = sum(recent) / float(len(recent)) if len(recent) >= 8 else 0.15
                # Shove rates jump when the table shrinks: trust the last 5 hands if they are wider.
                last = list(recent)[-5:]
                if len(last) == 5 and sum(last) >= 3:
                    width = max(width, sum(last) / 5.0)
                eq = equity_vs_range(hole, max(width, 0.08))
                pot_odds = min(to_call, stack) / float(pot + min(to_call, stack))
                log("vs shove range=%.2f eq=%.2f pot_odds=%.2f" % (width, eq, pot_odds))
                return call() if eq >= pot_odds + 0.03 else check_fold()
            if bettors and all(is_maniac(p['name']) for p in bettors):
                eq = equity(hole, [], len(bettors))
                pot_odds = min(to_call, stack) / float(pot + min(to_call, stack))
                log("vs maniac eq=%.2f pot_odds=%.2f" % (eq, pot_odds))
                if all(is_hyper(p['name']) for p in bettors):
                    return call() if eq >= pot_odds + 0.02 else check_fold()
                return call() if eq >= max(0.55, pot_odds + 0.12) else check_fold()
            # Habitual big raiser: its range is wide, so re-shove good hands and call fair prices.
            if bettors and all(is_aggressor(p['name']) for p in bettors):
                eq = equity(hole, [], len(bettors))
                pot_odds = min(to_call, stack) / float(pot + min(to_call, stack))
                log("vs aggressor eq=%.2f pot_odds=%.2f" % (eq, pot_odds))
                if eq >= 0.68:
                    return all_in()
                return call() if eq >= max(0.55, pot_odds + 0.12) else check_fold()

        # We opened and a single player re-raised (not all-in): resteal bots 3-bet wide
        # because we fold too often, so fight back with a 4-bet shove or a call by range equity.
        if not board and me['bet'] > bb and to_call > 0:
            raisers = [p for p in opponents if p['bet'] > me['bet']]
            if len(raisers) == 1 and raisers[0]['stack'] > 0:
                rate = agg_rate(raisers[0]['name'])
                eq = equity_vs_range(hole, max(0.2, rate or 0.2))
                pot_odds = min(to_call, stack) / float(pot + min(to_call, stack))
                log("vs 3-bet range=%.2f eq=%.2f pot_odds=%.2f" % (max(0.2, rate or 0.2), eq, pot_odds))
                if eq >= 0.5:
                    return all_in()
                return call() if eq >= pot_odds + 0.05 else check_fold()

        # A loose raiser attacks our blinds (bots now target "known folders"): defend by
        # equity against its actual raising range instead of folding almost everything.
        if not board and to_call > 0 and me['bet'] <= bb:
            raisers = [p for p in opponents if p['bet'] > bb]
            if len(raisers) == 1 and raisers[0]['stack'] > 0:
                rate = agg_rate(raisers[0]['name'])
                if rate is not None and rate >= 0.3:
                    eq = equity_vs_range(hole, rate)
                    pot_odds = min(to_call, stack) / float(pot + min(to_call, stack))
                    log("defend vs steal range=%.2f eq=%.2f pot_odds=%.2f" % (rate, eq, pot_odds))
                    if eq >= 0.55:
                        return all_in() if stack <= 25 * bb else raise_by(3 * to_call)
                    return call() if eq >= max(0.4, pot_odds + 0.02) else check_fold()

        # Judge by behaviour only (version strings can lie): if every opponent has stayed
        # passive over its recent hands and nobody has bet into us, a min-raise takes the pot.
        passive = opponents and not maniacs and all(
            agg_rate(p['name']) is not None and agg_rate(p['name']) <= 0.1
            and not sum(RECENT.get(p['name'], ())) for p in opponents)
        if passive and to_call <= (bb if not board else 0):
            log("passive table: min-raise steal")
            return raise_by(min_raise)

        if len(board) == 0:
            return self.preflop(hole, to_call, stack, bb, pot, effective, num_opp,
                                len(alive), call, check_fold, raise_by, all_in)
        # Two or more players already committing big money: someone has a real hand.
        big_in = [p for p in opponents if p['bet'] >= max(4 * bb, 0.25 * (p['bet'] + p['stack']))]
        if board and to_call > 0 and len(big_in) >= 2:
            eq = equity(hole, board, len(big_in)) - 0.3
            price = to_call / float(pot + to_call)
            log("multiway commitment eq=%.2f price=%.2f" % (eq, price))
            return call() if eq >= max(0.5, price + 0.1) else check_fold()

        top_bet = max([p['bet'] for p in opponents] or [0])
        loose_bettor = any(is_loose(p['name']) for p in opponents if p['bet'] == top_bet and top_bet > 0)
        tight_bettor = any(agg_rate(p['name']) is not None and agg_rate(p['name']) <= 0.2 and not is_loose(p['name'])
                           for p in opponents if p['bet'] == top_bet and top_bet > 0)
        reraised = me['bet'] > 0 and to_call > 0
        return self.postflop(hole, board, to_call, stack, pot, num_opp, loose_bettor, reraised, tight_bettor,
                             call, check_fold, raise_by, all_in)

    def preflop(self, hole, to_call, stack, bb, pot, effective, num_opp, table_size,
                call, check_fold, raise_by, all_in):
        chen = chen_score(hole)
        m = effective / float(bb)
        heads_up = table_size <= 1
        log("preflop chen=%.1f m=%.1f to_call=%d opp=%d" % (chen, m, to_call, num_opp))

        # Heads-up, unopened: the small blind must play most hands or the blinds bleed us dry.
        if heads_up and to_call <= bb:
            eq = equity(hole, [], 1)
            log("heads-up open eq=%.2f m=%.1f" % (eq, m))
            if m <= 12:
                return all_in() if eq >= 0.45 else (call() if to_call == 0 else check_fold())
            if eq >= 0.45:
                return raise_by(2 * bb)
            if to_call == 0 or eq >= 0.38:
                return call()
            return check_fold()

        # Short stacked: push/fold.
        if m <= 12:
            eq = equity(hole, [], num_opp)
            share = eq * (num_opp + 1)
            log("push/fold eq=%.2f share=%.2f" % (eq, share))
            if to_call >= stack * 0.4:
                # Shovers' ranges beat random hands, so demand extra equity to call.
                pot_odds = min(to_call, stack) / float(pot + min(to_call, stack))
                # Multiway shoves make pot odds look cheap: also need more than a fair share.
                return all_in() if eq >= max(pot_odds + 0.15, 1.1 / (num_opp + 1)) else check_fold()
            # The shorter we are, the wider we shove (blinds eat us otherwise).
            push_share = 1.25 if m > 8 else (1.1 if m > 5 else 0.95)
            if heads_up:
                push_share -= 0.1
            # Someone already raised: shoving over a raiser needs a much stronger hand.
            if to_call > bb:
                push_share += 0.4
            if share >= push_share:
                return all_in()
            return call() if to_call == 0 else check_fold()

        facing_raise = to_call > bb

        if chen >= 12:  # AA, KK, QQ, AKs
            if to_call >= stack * 0.25:
                return all_in()
            return raise_by(max(3 * bb, 3 * to_call, pot))
        if chen >= 9:  # JJ-99, AK, AQs, KQs...
            if not facing_raise:
                return raise_by(3 * bb)
            if to_call <= stack * 0.15:
                return call()
            return all_in() if chen >= 10 and to_call >= stack * 0.5 else check_fold()
        if chen >= 7 or (heads_up and chen >= 5):
            if not facing_raise:
                return raise_by(2.5 * bb) if num_opp <= 2 or heads_up else call()
            if to_call <= max(3 * bb, stack * 0.05):
                return call()
            return check_fold()
        if chen >= 5 and to_call <= bb:
            return call()
        return call() if to_call == 0 else check_fold()

    def postflop(self, hole, board, to_call, stack, pot, num_opp, loose_bettor, reraised, tight_bettor,
                 call, check_fold, raise_by, all_in):
        eq = equity(hole, board, num_opp)
        # A big bet narrows the bettor's range well above random hands: discount our equity.
        if to_call > 0:
            pressure = min(1.0, to_call / float(max(pot, 1)))
            discount = 0.08 + 0.25 * pressure
            # Loose bettors bet big with weak hands too: discount less so they can't bully us.
            if loose_bettor:
                discount *= 0.4
            # Tight bettors bet for value: a bet from them means a real hand.
            elif tight_bettor:
                discount *= 1.5
            eq -= discount
        # They raised over our own bet on this street: that range is strong, and we stop
        # re-raising unless we hold a near-lock (avoids raise wars into sets and flushes).
        if reraised:
            eq -= 0.1
            if eq < 0.75:
                return call() if eq >= pot_odds_of(to_call, pot) + 0.05 else check_fold()
        pot_odds = to_call / float(pot + to_call) if to_call else 0.0
        # Strength relative to a fair share of the pot, so multiway hands aren't underplayed.
        share = eq * (num_opp + 1)
        log("postflop eq=%.2f share=%.2f pot=%d to_call=%d opp=%d" % (eq, share, pot, to_call, num_opp))

        # Calling off a big chunk of the stack needs a clearly winning hand, not one pair.
        if to_call >= stack * 0.3:
            return call() if eq >= max(0.6, pot_odds + 0.1) else check_fold()

        if eq >= 0.85 or (share >= 1.7 and eq >= 0.4):
            if to_call >= stack * 0.3:
                return all_in()
            return raise_by(pot)
        if eq >= 0.68 or (share >= 1.4 and eq >= 0.33):
            if to_call == 0:
                return raise_by(pot * 0.66)
            if to_call <= pot * 0.5:
                return raise_by(pot * 0.75)
            return call()
        if eq >= 0.5:
            if to_call == 0:
                # Callers now hold real hands: only bet when clearly ahead, else check.
                return raise_by(pot * 0.5) if eq >= 0.6 else check_fold()
            # Big bets usually mean a strong hand; demand extra equity.
            margin = 0.1 if to_call > pot * 0.5 else 0.0
            return call() if eq >= pot_odds + margin else check_fold()
        if to_call == 0:
            return check_fold()
        margin = 0.05 if to_call <= pot * 0.3 else 0.15
        return call() if eq >= pot_odds + margin else check_fold()
