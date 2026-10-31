import numpy as np, pandas as pd, pyfixest as pf, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width",220)
from vop.config import load_config
from vop import s07_hurdle as h
cfg=load_config(); d=h.load(cfg,cfg.derived/"s05",3); d["set_field"]=d.set_field.fillna("No field assigned")
fields=sorted(d.set_field.unique()); cov=["log_refs","year_dev","nauth_dev","has_abstract","prestige_cnp","no_prior_record","log_prior_works_max"]
for i,f in enumerate(fields):
    d[f"p{i}"]=d.phil*(d.set_field==f).astype(float); d[f"f{i}"]=(d.set_field==f).astype(float)
P=[f"p{i}" for i in range(len(fields))]; F=[f"f{i}" for i in range(len(fields))]
bay=pd.read_csv(cfg.derived/"s07"/"field_model_effects_k3.csv").set_index("field")
c=d[d.cited==1].copy(); c["const"]=1.0
b=pf.feols("log_cites ~ "+"+".join(P+cov)+" | set_id",data=c).coef()
se2,su2=1.367,0.7409**2
def gls(df,cols):
    n=df.groupby("set_id").log_cites.transform("size").values; th=1-np.sqrt(se2/(se2+n*su2))
    mean=df[["log_cites"]+cols].groupby(df.set_id).transform("mean")
    Y=df.log_cites.values-th*mean.log_cites.values; X=df[cols].values-th[:,None]*mean[cols].values
    return pd.Series(np.linalg.lstsq(X,Y,rcond=None)[0],index=cols)
M=[]
for v in ["log_refs","has_abstract","prestige_cnp","no_prior_record","log_prior_works_max"]:
    c["m_"+v]=c.groupby("set_id")[v].transform("mean"); M.append("m_"+v)
A=gls(c,["const"]+P+cov); B=gls(c,F+P+cov); C=gls(c,F+P+cov+M)
for v in ["log_refs","has_abstract","prestige_cnp","no_prior_record","log_prior_works_max"]+["phil"]:
    pass
# centred philosopher indicator, no field intercepts
c2=c.copy(); 
for p in P: c2[p]=c2[p]-c2.groupby("set_id")[p].transform("mean")
D=gls(c2,["const"]+P+cov+M)
o=pd.DataFrame({"field":fields,"FE":[b[p] for p in P],"RE_no_field":[A[p] for p in P],"RE_field":[B[p] for p in P],"RE_field_means":[C[p] for p in P],"RE_centred_means":[D[p] for p in P]}).set_index("field")
o["bayes"]=bay.m_mean; o["n_sets"]=bay.n_sets
ctrl=c[c.phil==0].groupby("set_field").log_cites.mean(); o["ctrl_logcites"]=ctrl
print(o.sort_values("n_sets",ascending=False).round(3).to_string())
big=o[(o.n_sets>=300)&(o.index!="No field assigned")]
print("fields with >=300 sets:",len(big))
for col in ["RE_no_field","RE_field","RE_field_means","RE_centred_means","bayes"]:
    print(col,"corr(gap from FE, control mean log cites)=",round(np.corrcoef(big[col]-big.FE,big.ctrl_logcites)[0,1],2),"mean abs gap",round((big[col]-big.FE).abs().mean(),3))
print("covariates has_abstract / prestige with field and means:",round(C["has_abstract"],3),round(C["prestige_cnp"],3))
# hurdle: baseline vs gap
hm=pf.feglm("cited ~ "+"+".join(P+cov)+" | set_id",data=d,family="logit").coef()
hh=pd.DataFrame({"field":fields,"FE":[hm[p] for p in P]}).set_index("field"); hh["bayes"]=bay.h_mean; hh["n_sets"]=bay.n_sets; p0=d[d.phil==0].groupby("set_field").cited.mean(); hh["ctrl_logit"]=np.log(p0/(1-p0))
hb=hh[(hh.n_sets>=300)&(hh.index!="No field assigned")]
print("hurdle corr(bayes, ctrl baseline)=",round(np.corrcoef(hb.bayes,hb.ctrl_logit)[0,1],2),"corr(gap, ctrl baseline)=",round(np.corrcoef(hb.bayes-hb.FE,hb.ctrl_logit)[0,1],2),"corr(FE, baseline)=",round(np.corrcoef(hb.FE,hb.ctrl_logit)[0,1],2))
print("corr(bayes_m gap...) done; weighted FE mean log cites",round((big.FE*big.n_sets).sum()/big.n_sets.sum(),4))
print("DONE",flush=True)
