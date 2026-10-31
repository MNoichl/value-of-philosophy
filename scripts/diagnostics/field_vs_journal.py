import numpy as np, pandas as pd, pyfixest as pf, warnings, glob
warnings.filterwarnings("ignore"); pd.set_option("display.width",230); pd.set_option("display.max_rows",200); pd.set_option("display.max_colwidth",46)
from vop.config import load_config, connect
from vop import s07_hurdle as h
cfg=load_config(); con=connect(cfg)
slim=cfg.derived/"s02"/"works_slim"; print(slim, len(glob.glob(str(slim/"**/*.parquet"),recursive=True)),flush=True)
print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{slim}/**/*.parquet')").fetchdf().column_name.tolist())
j=con.execute(f"""SELECT source_id, coalesce(field,'No field') field, count(*) n FROM read_parquet('{slim}/**/*.parquet') GROUP BY 1,2""").fetchdf()
print("source-field rows",len(j),flush=True)
tot=j.groupby("source_id").n.sum().rename("tot"); j=j.join(tot,on="source_id"); j["sh"]=j.n/j.tot
top=j.sort_values("n",ascending=False).groupby("source_id").head(1).set_index("source_id")[["field","sh","tot"]].rename(columns={"field":"j_field","sh":"j_share"})
ms=cfg.derived/"s05"/"matched_sets.parquet"
d=con.execute(f"SELECT set_id, work_id, is_focal, source_id, field FROM read_parquet('{ms}') WHERE k=3").fetchdf()
f=d[d.is_focal].rename(columns={"field":"set_field"})[["set_id","set_field","source_id"]].join(top,on="source_id")
c=d[~d.is_focal].merge(f[["set_id","set_field"]],on="set_id")
c["same"]=c.field==c.set_field
print("sets",len(f),"journals",f.source_id.nunique())
print("== how pure are the journals these sets sit in (top field's share of ALL articles in the journal), weighted by sets")
print(f.j_share.describe(percentiles=[.1,.25,.5,.75,.9]).round(3).to_string())
print("share of sets in journals where top field holds >=90%:",round((f.j_share>=.9).mean(),3),">=75%:",round((f.j_share>=.75).mean(),3),"<50%:",round((f.j_share<.5).mean(),3))
print("== does the philosopher paper's field equal its journal's top field; do controls share the philosopher paper's field")
s=f.assign(match=f.set_field==f.j_field).groupby(f.set_field.fillna("No field")).agg(n_sets=("set_id","size"),paper_eq_journal=("match","mean"),journal_purity=("j_share","mean"))
s["control_same_field"]=c.assign(sf=c.set_field.fillna("No field")).groupby("sf").same.mean()
s["control_no_field"]=c.assign(sf=c.set_field.fillna("No field")).groupby("sf").field.apply(lambda x:x.isna().mean())
print(s.sort_values("n_sets",ascending=False).round(2).to_string())
print("overall: paper field == journal top field",round((f.set_field==f.j_field).mean(),3),"; control field == philosopher paper field",round(c.same.mean(),3),flush=True)
print("== cross-tab, rows = philosopher paper's field, columns = journal's top field, row shares >=8%")
x=pd.crosstab(f.set_field.fillna("No field"),f.j_field,normalize="index")
for r in s.sort_values("n_sets",ascending=False).head(14).index:
    v=x.loc[r].sort_values(ascending=False); v=v[v>=.08]; print(f"{r[:40]:42}",", ".join(f"{k[:28]} {a:.0%}" for k,a in v.items()))
print("== reverse: of sets in journals whose top field is X, where do philosopher papers get classified")
y=pd.crosstab(f.j_field,f.set_field.fillna("No field"),normalize="index"); nj=f.j_field.value_counts()
for r in nj.head(12).index:
    v=y.loc[r].sort_values(ascending=False); v=v[v>=.08]; print(f"{r[:34]:36} n={nj[r]:>6}",", ".join(f"{k[:28]} {a:.0%}" for k,a in v.items()))
# Fixed effects under three definitions
a=h.load(cfg,cfg.derived/"s05",3); a=a.merge(f[["set_id","j_field"]],on="set_id",how="left").merge(d[["set_id","work_id","field"]],on=["set_id","work_id"],how="left")
cov=["log_refs","year_dev","nauth_dev","has_abstract","prestige_cnp","no_prior_record","log_prior_works_max"]
def fe(df,col,y,sub=None):
    df=df[df[col].notna()].copy(); lv=sorted(df[col].unique()); P=[]
    for i,l in enumerate(lv): df[f"p{i}"]=df.phil*(df[col]==l).astype(float); P.append(f"p{i}")
    if y=="log_cites": df=df[df.cited==1]
    m=pf.feols(f"{y} ~ "+"+".join(P+cov)+" | set_id",data=df,vcov={"CRV1":"set_id"})
    return pd.DataFrame({"coef":[m.coef()[p] for p in P],"se":[m.se()[p] for p in P]},index=lv),df.groupby(col).set_id.nunique()
pa,n1=fe(a,"set_field","log_cites"); ja,n2=fe(a,"j_field","log_cites")
same=a[(a.field==a.set_field)|(a.phil==1)]; sa,n3=fe(same,"set_field","log_cites")
pc,_=fe(a,"set_field","cited"); jc,_=fe(a,"j_field","cited"); sc,_=fe(same,"set_field","cited")
t=pd.DataFrame({"n_paper":n1,"cites_paperfield":pa.coef,"cites_samefieldctrl":sa.coef,"n_journal":n2,"cites_journalfield":ja.coef,"se_j":ja.se,"cited_paperfield":pc.coef*100,"cited_samefieldctrl":sc.coef*100,"cited_journalfield":jc.coef*100})
print("== philosopher effect, fixed effects: log citations if cited, and cited points, by field definition");print(t.sort_values("n_paper",ascending=False).round(3).to_string())
print("mean sets per same-field subset control count:",same.groupby("set_id").phil.size().describe().round(2).to_dict())
print("DONE",flush=True)
