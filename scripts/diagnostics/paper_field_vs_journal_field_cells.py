import numpy as np, pandas as pd, pyfixest as pf, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width",230); pd.set_option("display.max_rows",200)
from vop.config import load_config
from vop import s07_hurdle as h
cfg=load_config(); a=h.attach_journal_field(cfg,h.load(cfg,cfg.derived/"s05",3))
cov=["log_refs","year_dev","nauth_dev","has_abstract","prestige_cnp","no_prior_record","log_prior_works_max"]
a["home"]=(a.set_field==a.j_field).astype(int)
def fe(df,cols,y):
    df=df.dropna(subset=cols).copy(); key=df[cols].astype(str).agg("|".join,axis=1); lv=sorted(key.unique()); P=[]
    for i,l in enumerate(lv): df[f"p{i}"]=df.phil*(key==l).astype(float); P.append(f"p{i}")
    if y=="log_cites": df=df[df.cited==1]
    m=pf.feols(f"{y} ~ "+"+".join(P+cov)+" | set_id",data=df,vcov={"CRV1":"set_id"})
    return pd.DataFrame({"coef":[m.coef()[p] for p in P],"se":[m.se()[p] for p in P]},index=lv)
for y,l,s in [("log_cites","LOG CITATIONS IF CITED",1),("cited","CITED POINTS",100)]:
    print("==",l,flush=True)
    j=fe(a,["j_field"],y)*s; pf_=fe(a,["set_field"],y)*s
    jh=fe(a,["j_field","home"],y)*s; ph=fe(a,["set_field","home"],y)*s
    n=a.drop_duplicates("set_id")
    big=n.j_field.value_counts(); big=big[big>=900].index
    rows=[]
    for f in big:
        g=lambda t,k: t.coef.get(k,np.nan)
        rows.append(dict(field=f[:30],by_paper=g(pf_,f),by_journal=g(j,f),
          paper_in_home=g(ph,f"{f}|1"),paper_elsewhere=g(ph,f"{f}|0"),se_else=ph.se.get(f"{f}|0",np.nan),
          journal_offtopic=g(jh,f"{f}|0"),se_off=jh.se.get(f"{f}|0",np.nan),
          n_home=int(((n.set_field==f)&(n.j_field==f)).sum()),n_paper_else=int(((n.set_field==f)&(n.j_field!=f)).sum()),n_journal_off=int(((n.j_field==f)&(n.set_field!=f)).sum())))
    print(pd.DataFrame(rows).round(3).to_string(index=False),flush=True)
print("DONE")
