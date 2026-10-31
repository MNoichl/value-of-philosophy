"""Compare journal-and-field matching with journal-only matching: coverage and fixed-effects estimates."""
import os, warnings
import numpy as np, pandas as pd, pyfixest as pf
warnings.filterwarnings("ignore"); pd.set_option("display.width",230); pd.set_option("display.max_rows",200)
from vop.config import load_config, connect
from vop import s07_hurdle as h
cfg=load_config(); con=connect(cfg)
new=h.attach_journal_field(cfg,h.load(cfg,cfg.derived/"s05",3))
os.environ["VOP_VARIANT"]="journal"; old_cfg=load_config()
old=h.attach_journal_field(old_cfg,h.load(old_cfg,old_cfg.derived/"s05",3))
os.environ["VOP_VARIANT"]="journal_field"
cov=[c for c in ["log_refs","year_dev","nauth_dev","has_abstract","prestige_cnp","no_prior_record","log_prior_works_max"]]
s=new.groupby("set_id").phil.size()-1
print("focal in journal-only sets",old.set_id.nunique(),"; journal+field sets",new.set_id.nunique(),"; share kept",round(new.set_id.nunique()/old.set_id.nunique(),3))
print("controls per set"); print(s.value_counts().sort_index().to_string())
print("check: controls in same field as philosopher paper",round((new.loc[new.phil==0].set_field==new.loc[new.phil==0].merge(con.execute(f"SELECT set_id,work_id,field FROM read_parquet('{cfg.derived/'s05'/'matched_sets.parquet'}') WHERE k=3").fetchdf(),on=['set_id','work_id']).field.values).mean(),4))
print("== kept share of old sets, by journal field and by paper field")
ids=set(new.set_id); o=old.drop_duplicates("set_id").assign(kept=lambda x:x.set_id.isin(ids))
print(o.groupby("j_field").kept.agg(["size","mean"]).sort_values("size",ascending=False).head(16).round(3).to_string())
print(o.groupby(o.set_field.fillna("No field")).kept.agg(["size","mean"]).sort_values("size",ascending=False).head(16).round(3).to_string())
g=new.groupby("phil"); print("== new sets, controls vs philosophers"); print(pd.DataFrame({"cited":g.cited.mean(),"abstract":g.has_abstract.mean(),"zero_refs":g.n_refs.apply(lambda x:(x==0).mean()),"mean_refs":g.n_refs.mean(),"track":g.prestige_cnp.mean(),"no_record":g.no_prior_record.mean(),"mean_cites":g.cited_by_count.mean()}).round(3).to_string())
def overall(d,y,x): return pf.feols(f"{y} ~ {x} | set_id",data=d[d.cited==1] if y=="log_cites" else d,vcov={"CRV1":"set_id"})
print("== overall philosopher effect: old sets, new sets, old sets limited to kept")
keep=old[old.set_id.isin(ids)]
for name,dd in [("journal only",old),("journal only, sets kept below",keep),("journal and field",new)]:
    for lbl,x in [("no covariates","phil"),("covariates","phil+"+"+".join(cov))]:
        a=overall(dd,"cited",x); b=overall(dd,"log_cites",x)
        print(f"{name:32} {lbl:14} cited {100*a.coef()['phil']:+.2f} (se {100*a.se()['phil']:.2f}) cites {100*(np.exp(b.coef()['phil'])-1):+.1f}% (se {100*b.se()['phil']:.1f}) sets {dd.set_id.nunique()}")
def fe(df,col,y):
    df=df[df[col].notna()].copy(); lv=sorted(df[col].unique()); P=[]
    for i,l in enumerate(lv): df[f"p{i}"]=df.phil*(df[col]==l).astype(float); P.append(f"p{i}")
    if y=="log_cites": df=df[df.cited==1]
    m=pf.feols(f"{y} ~ "+"+".join(P+cov)+" | set_id",data=df,vcov={"CRV1":"set_id"})
    return pd.DataFrame({"coef":[m.coef()[p] for p in P],"se":[m.se()[p] for p in P]},index=lv)
n=new.drop_duplicates("set_id"); no=old.drop_duplicates("set_id")
for y,sc,l in [("log_cites",1,"LOG CITATIONS IF CITED"),("cited",100,"CITED POINTS")]:
    print("==",l)
    t=pd.DataFrame({"sets_paperfield":n.set_field.value_counts(),"new_field":fe(new,"set_field",y).coef*sc,"se":fe(new,"set_field",y).se*sc,"sets_journalfield":n.j_field.value_counts(),"new_journal":fe(new,"j_field",y).coef*sc,"se_j":fe(new,"j_field",y).se*sc,"old_paper":fe(old,"set_field",y).coef*sc,"old_journal":fe(old,"j_field",y).coef*sc})
    print(t.sort_values("sets_paperfield",ascending=False).round(3).to_string())
print("DONE")
