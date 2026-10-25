# Preregistration

`PREREG.md` is the analysis plan for the confirmatory replication (fresh seeds 30-39) reported in the paper. It is
byte-for-byte the file that was committed before those runs started.

## Check that the file is unchanged

The paper gives the file's git blob hash, `c01901dc`. Check it:

```bash
git hash-object preregistration/PREREG.md
# c01901dc3dbba5e9a05d78f21978d93378af6796
```

## When it was written

The plan was committed to the authors' then-private working repository on 6 October 2026 at 22:33 PDT. Each run of the
replication then recorded the plan's hash and its own start time; those records are in `run_provenance/`, one file per
run.

Fields in each record:
- `prereg_blob`: the hash of the plan.
- `prereg_commit`: the registration commit and its date.
- `started`: when the job began, the earliest on 7 October.

Git commit dates are self-reported, and the original repository was private. So this shows that the file has not
changed since the runs recorded it. It does not provide an independent third-party timestamp.

The commit hash in the records (`d006e016`) is the commit as it existed before the working repository's history was
rewritten. The plan's content hash is the same.
