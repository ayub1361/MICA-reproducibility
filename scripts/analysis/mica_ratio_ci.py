import argparse
import csv
import math
import random
import sys
from collections import defaultdict


def load(path, numerator, denominator):
    by_city = defaultdict(lambda: defaultdict(dict))
    seen_policies = set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"city", "group_id", "policy", "n_decisions", "n_overrides"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit("ERROR: CSV is missing required columns: %s" % sorted(missing))
        for row in reader:
            pol = row["policy"].strip()
            seen_policies.add(pol)
            if pol not in (numerator, denominator):
                continue
            rec = {
                "n_decisions": int(row["n_decisions"]),
                "n_overrides": int(row["n_overrides"]),
                "members": [m for m in row.get("members", "").split("|") if m],
            }
            for opt, cast in (("irrecoverable", int), ("base_score_loss", float)):
                if row.get(opt) not in (None, ""):
                    rec[opt] = cast(row[opt])
            by_city[row["city"].strip()][row["group_id"].strip()][pol] = rec

    if numerator not in seen_policies or denominator not in seen_policies:
        sys.exit(
            "ERROR: policy labels not found in CSV.\n"
            "  --numerator   : %r %s\n"
            "  --denominator : %r %s\n"
            "  labels present: %s"
            % (numerator, "OK" if numerator in seen_policies else "MISSING",
               denominator, "OK" if denominator in seen_policies else "MISSING",
               sorted(seen_policies))
        )

    paired = {}
    for city, groups in by_city.items():
        kept = {g: d for g, d in groups.items()
                if numerator in d and denominator in d}
        dropped = len(groups) - len(kept)
        if dropped:
            print("  note: %s -- %d group(s) dropped, not present under both policies"
                  % (city, dropped), file=sys.stderr)
        paired[city] = kept
    return paired


def rates(groups, gids, numerator, denominator):
    num_ov = num_de = den_ov = den_de = 0
    for g in gids:
        d = groups[g]
        num_ov += d[numerator]["n_overrides"]
        num_de += d[numerator]["n_decisions"]
        den_ov += d[denominator]["n_overrides"]
        den_de += d[denominator]["n_decisions"]
    r_num = num_ov / num_de if num_de else float("nan")
    r_den = den_ov / den_de if den_de else float("nan")
    return r_num, r_den


def ratio_of(groups, gids, numerator, denominator):
    r_num, r_den = rates(groups, gids, numerator, denominator)
    if r_den == 0:
        return float("inf")
    return r_num / r_den


def diff_of(groups, gids, numerator, denominator):
    r_num, r_den = rates(groups, gids, numerator, denominator)
    return r_den - r_num


def components(groups, gids):
    user_to_groups = defaultdict(list)
    for g in gids:
        rec = next(iter(groups[g].values()))
        for m in rec["members"]:
            user_to_groups[m].append(g)

    parent = {g: g for g in gids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for gs in user_to_groups.values():
        for other in gs[1:]:
            union(gs[0], other)

    blocks = defaultdict(list)
    for g in gids:
        blocks[find(g)].append(g)
    return list(blocks.values())


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def bootstrap(units, draw_fn, stat_fn, reps, rng):
    out = []
    n_degenerate = 0
    n = len(units)
    for _ in range(reps):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        gids = draw_fn(sample)
        v = stat_fn(gids)
        if v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))):
            n_degenerate += 1
            continue
        out.append(v)
    out.sort()
    return out, n_degenerate


def run_city(city, groups, numerator, denominator, reps, rng, alpha, rows):
    gids = sorted(groups)
    n = len(gids)
    r_num, r_den = rates(groups, gids, numerator, denominator)
    point_ratio = ratio_of(groups, gids, numerator, denominator)
    point_diff = diff_of(groups, gids, numerator, denominator)

    print("\n" + "=" * 74)
    print("CITY: %s   (n = %d paired evaluation groups)" % (city, n))
    print("=" * 74)
    print("  %-34s %.4f" % (numerator + " rate", r_num))
    print("  %-34s %.4f" % (denominator + " rate", r_den))
    print("  %-34s %.2fx" % ("point ratio (num/den)", point_ratio))
    print("  %-34s %+.4f" % ("paired difference (den - num)", point_diff))

    lo_p, hi_p = alpha / 2, 1 - alpha / 2

    ratios, deg = bootstrap(
        gids, lambda s: s,
        lambda ids: ratio_of(groups, ids, numerator, denominator),
        reps, rng)
    r_lo, r_hi = percentile(ratios, lo_p), percentile(ratios, hi_p)

    diffs, _ = bootstrap(
        gids, lambda s: s,
        lambda ids: diff_of(groups, ids, numerator, denominator),
        reps, rng)
    d_lo, d_hi = percentile(diffs, lo_p), percentile(diffs, hi_p)

    print("\n  -- group bootstrap (%d reps) --" % reps)
    print("     ratio  95%% CI : [%.2fx, %.2fx]" % (r_lo, r_hi))
    print("     diff   95%% CI : [%+.4f, %+.4f]" % (d_lo, d_hi))
    if deg:
        print("     note: %d/%d resamples had a zero denominator rate and were "
              "dropped (%.2f%%)" % (deg, reps, 100.0 * deg / reps))

    blocks = components(groups, gids)
    have_members = any(next(iter(groups[g].values()))["members"] for g in gids)
    if have_members:
        cratios, cdeg = bootstrap(
            blocks, lambda s: [g for b in s for g in b],
            lambda ids: ratio_of(groups, ids, numerator, denominator),
            reps, rng)
        c_lo, c_hi = percentile(cratios, lo_p), percentile(cratios, hi_p)
        sizes = sorted((len(b) for b in blocks), reverse=True)
        print("\n  -- cluster bootstrap over membership-overlap blocks --")
        print("     blocks: %d   largest: %s" % (len(blocks), sizes[:5]))
        print("     ratio  95%% CI : [%.2fx, %.2fx]" % (c_lo, c_hi))
        if cdeg:
            print("     note: %d/%d resamples dropped (zero denominator)"
                  % (cdeg, reps))
    else:
        c_lo = c_hi = float("nan")
        print("\n  -- cluster bootstrap skipped: 'members' column empty --")

    rows.append({
        "city": city, "n_groups": n,
        "rate_numerator": round(r_num, 6),
        "rate_denominator": round(r_den, 6),
        "ratio_point": round(point_ratio, 4),
        "ratio_lo_group": round(r_lo, 4), "ratio_hi_group": round(r_hi, 4),
        "ratio_lo_cluster": round(c_lo, 4) if c_lo == c_lo else "",
        "ratio_hi_cluster": round(c_hi, 4) if c_hi == c_hi else "",
        "diff_point": round(point_diff, 6),
        "diff_lo": round(d_lo, 6), "diff_hi": round(d_hi, 6),
        "n_blocks": len(blocks) if have_members else "",
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--numerator", default="horizon max-min control",
                    help="policy in the numerator of the ratio (the control)")
    ap.add_argument("--denominator", default="MICA",
                    help="policy in the denominator of the ratio")
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="optional CSV of results")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = load(args.csv, args.numerator, args.denominator)

    print("mica_ratio_ci.py")
    print("  input      : %s" % args.csv)
    print("  numerator  : %s" % args.numerator)
    print("  denominator: %s" % args.denominator)
    print("  reps       : %d    seed: %d" % (args.reps, args.seed))

    rows = []
    for city in sorted(data):
        run_city(city, data[city], args.numerator, args.denominator,
                 args.reps, rng, args.alpha, rows)

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
