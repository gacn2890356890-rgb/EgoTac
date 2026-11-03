"""NumPy contact and response metrics on exact validity masks."""
import numpy as np

def counts(pred,gt,mask):
 p=pred[mask]; y=gt[mask]
 return int((p&y).sum()),int((p&~y).sum()),int((~p&y).sum())
def prf(tp,fp,fn):
 return {'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'F1':2*tp/max(2*tp+fp+fn,1)}
def ap(score,gt):
 if not gt.any(): return 0.
 order=np.argsort(-score,kind='stable');y=gt[order];s=score[order]
 ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
 tp=np.cumsum(y)[ends];precision=tp/(ends+1);recall=tp/y.sum()
 return float(np.sum(np.diff(np.r_[0.,recall])*precision))
def onset_counts(pred,gt,valid,times):
 tp=fp=fn=0
 for k in range(12):
  good=valid[1:,k]&valid[:-1,k]
  a=np.flatnonzero(pred[1:,k]&~pred[:-1,k]&good)+1
  b=np.flatnonzero(gt[1:,k]&~gt[:-1,k]&good)+1
  used=set()
  for i in a:
   options=[j for j in b if j not in used and abs(times[i]-times[j])<=.200001]
   if options:used.add(min(options,key=lambda j:abs(times[i]-times[j])));tp+=1
   else:fp+=1
  fn+=len(b)-len(used)
 return tp,fp,fn

def evaluate(records,threshold):
 episodes={};patches=[[] for _ in range(12)]
 for row in records:
  y,v,prob,amp,times=[row[k] for k in ['target','valid','probability','amplitude','timestamps']]
  gt=y>=1; pred=prob>threshold;response=np.where(pred,amp,0.);error=response-y
  c=counts(pred,gt,v);ev=onset_counts(pred,gt,v,times);pairs=v[1:]&v[:-1]
  false_changes=((pred[1:]!=pred[:-1])&(gt[1:]==gt[:-1])&pairs).sum()
  d={**prf(*c),'AUPRC':ap(prob[v],gt[v]),'Brier':float(((prob-gt)**2)[v].mean()),
     'MAE':float(np.abs(error[v]).mean()),'RMSE':float(np.sqrt((error[v]**2).mean())),
     'positive_MAE':float(np.abs(error[v&gt]).mean()) if (v&gt).any() else None,
     'onset_event_F1_0.2s':prf(*ev)['F1'],'false_transitions_per_minute':float(false_changes/max(float(times[-1]-times[0])/60,1e-6)),
     'counts':{'TP':c[0],'FP':c[1],'FN':c[2]},'valid_values':int(v.sum())}
  episodes[row['name']]=d
  for k in range(12): patches[k].append((pred[:,k],gt[:,k],v[:,k],prob[:,k]))
 summary={key:float(np.mean([d[key] for d in episodes.values() if d[key] is not None])) if any(d[key] is not None for d in episodes.values()) else None for key in next(iter(episodes.values())) if key not in ['counts','valid_values']}
 perpatch=[]
 for parts in patches:
  pp,gg,vv,ss=[np.concatenate([x[k] for x in parts]) for k in range(4)]
  perpatch.append({**prf(*counts(pp,gg,vv)),'AUPRC':ap(ss[vv],gg[vv]),'valid_values':int(vv.sum())})
 return {'summary':summary,'episodes':episodes,'per_patch':perpatch,'threshold':float(threshold)}

def select_threshold(records):
 choices=[]
 for t in np.arange(.05,.951,.05):
  m=evaluate(records,float(t));choices.append((m['summary']['F1'],-m['summary']['MAE'],float(t)))
 _,_,threshold=max(choices)
 return threshold,evaluate(records,threshold)
