import json
import sys
from tasks.game24 import verify

in_path, out_path = sys.argv[1], sys.argv[2]

rows = [json.loads(l) for l in open(in_path)]
n = len(rows)
n_stored = sum(r["verified"] for r in rows)
mismatches = []
new_rows = []
for r in rows:
    actual = verify(r["equation"], tuple(r["inputs"]))
    # if not actual:
    #     print(r["equation"])
    if actual != r["verified"]:
        mismatches.append((r["idx"], r["inputs"], r["equation"],
                           r["verified"], actual))
    new_rows.append({**r, "verified": actual})

n_now = sum(r["verified"] for r in new_rows)
print(f"  stored:    {n_stored}/{n} = {100*n_stored/n:.1f}%")
print(f"  re-verify: {n_now}/{n} = {100*n_now/n:.1f}%   delta = {n_now - n_stored:+d}")
if mismatches:
    print(f"  mismatches ({len(mismatches)}):")
    for idx, inputs, eq, was, now in mismatches:
        print(f"    [{idx:3d}] {tuple(inputs)} {eq!r}")
        print(f"          stored={was}  current={now}")

with open(out_path, "w") as fp:
    for r in new_rows:
        fp.write(json.dumps(r) + "\n")
print(f"  wrote: {out_path}")
