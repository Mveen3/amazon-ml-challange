"""Exact-key blocking channels (A2 address keys, N2 name keys).

Oversized blocks are not dropped wholesale: every name key also exists in a
locality-conditioned form (``skeleton|admin token``), so when a generic name
("Primary Care Group") exceeds the block cap, the per-locality sub-blocks survive.
"""
from __future__ import annotations

import polars as pl

KEY_TYPES = ["kn_skel", "kn_sorted", "kn_concat", "kn_skel_loc", "ka_num_tok", "ka_pin_num", "ka_exact"]
KT = {k: i for i, k in enumerate(KEY_TYPES)}


def make_keys(part: pl.DataFrame, cfg) -> pl.DataFrame:
    """part: one country partition with norm columns. Returns (uid, src, kt, key)."""
    kc = cfg.blocking.keyblock
    base = part.select(["uid", "src"])
    frames = []

    def add(df: pl.DataFrame, kt: str):
        frames.append(df.select(["uid", pl.col("key").cast(pl.Utf8)]).filter(pl.col("key").str.len_chars() > 0)
                      .join(base, on="uid").with_columns(pl.lit(KT[kt], dtype=pl.Int8).alias("kt")))

    # ---- name keys
    add(part.select(["uid", pl.col("name_skel").alias("key")]).filter(pl.col("key").str.len_chars() >= 3), "kn_skel")
    add(part.select(["uid", pl.col("name_core_sorted").alias("key")]), "kn_sorted")
    concat = pl.when((pl.col("src") != 1) & pl.col("is_domain")).then(pl.col("domain_body")) \
        .otherwise(pl.col("name_core").str.replace_all(" ", ""))
    add(part.select(["uid", concat.alias("key")]).filter(pl.col("key").str.len_chars() >= 6), "kn_concat")

    # ---- address token statistics within the partition
    tok = (part.select(["uid", "addr_tokens"]).explode("addr_tokens").rename({"addr_tokens": "tok"})
           .drop_nulls().unique())
    tok = tok.join(tok.group_by("tok").agg(pl.len().alias("df")), on="tok")
    words = tok.filter(~pl.col("tok").str.contains(r"^\d+$") & (pl.col("tok").str.len_chars() >= 3))
    admin = (words.filter(pl.col("df").is_between(int(kc.admin_df_min), int(kc.admin_df_max)))
             .sort(["uid", "df", "tok"], descending=[False, True, False])
             .group_by("uid", maintain_order=True).head(int(kc.loc_tokens)))
    skel = part.select(["uid", "name_skel"]).filter(pl.col("name_skel").str.len_chars() >= 3)
    add(skel.join(admin, on="uid").select(["uid", (pl.col("name_skel") + "|" + pl.col("tok")).alias("key")]),
        "kn_skel_loc")

    # ---- address keys
    rare = words.sort(["uid", "df", "tok"]).group_by("uid", maintain_order=True).head(int(kc.rare_tokens))
    nums = (part.select(["uid", "nums"]).explode("nums").drop_nulls().rename({"nums": "num"})
            .group_by("uid", maintain_order=True).head(int(kc.num_tokens)))
    add(nums.join(rare, on="uid").select(["uid", (pl.col("num") + "|" + pl.col("tok")).alias("key")]), "ka_num_tok")
    first_num = nums.group_by("uid", maintain_order=True).first()
    pin = part.select(["uid", "pin"]).filter(pl.col("pin").str.len_chars() > 0)
    add(pin.join(first_num, on="uid").select(["uid", (pl.col("pin") + "|" + pl.col("num")).alias("key")]),
        "ka_pin_num")
    add(part.filter(~pl.col("addr_missing")).select(["uid", pl.col("addr_key").alias("key")]), "ka_exact")
    return pl.concat(frames)


def key_pairs(keys: pl.DataFrame, cfg) -> pl.DataFrame:
    """Join S1 keys with record keys under block-size caps -> (s1_uid, rec_uid, kt, block)."""
    kc = cfg.blocking.keyblock
    s = keys.filter(pl.col("src") == 1).select(["uid", "kt", "key"]).unique()
    r = keys.filter(pl.col("src") != 1).select(["uid", "kt", "key"]).unique()
    n1 = s.group_by(["kt", "key"]).agg(pl.len().alias("n1"))
    n2 = r.group_by(["kt", "key"]).agg(pl.len().alias("n2"))
    blk = (n1.join(n2, on=["kt", "key"])
           .filter((pl.col("n1") <= int(kc.max_block_s1)) & (pl.col("n1") * pl.col("n2") <= int(kc.max_block_pairs)))
           .with_columns((pl.col("n1") * pl.col("n2")).cast(pl.Int32).alias("block")))
    s = s.join(blk.select(["kt", "key", "block"]), on=["kt", "key"])
    pairs = s.join(r, on=["kt", "key"], suffix="_r")
    return pairs.select([pl.col("uid").alias("s1_uid"), pl.col("uid_r").alias("rec_uid"), "kt", "block"])
