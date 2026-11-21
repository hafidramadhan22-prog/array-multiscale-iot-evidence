"""Relative retained-host timing; unchanged multiscale vectors are strict controls."""
import numpy as np
import pandas as pd
from multiscale_context import MultiscaleRing,WINDOWS,CONTEXT_NAMES
from e4 import DomainFeatures,columns

TIME_NAMES=('log_mean_iat_ms','log_std_iat_ms','last_packet_age_fraction','observed_span_fraction')
TIME_COLUMNS=[f'hosttime_{slot}_{int(w*1000)}ms_{name}' for slot in ('higher','lower') for w in WINDOWS for name in TIME_NAMES]
AGE_INDEX=np.arange(24).reshape(2,3,4)[:,:,2].reshape(-1)
VIEWS=('time_context84','legacy25_time_context84','legacy25_age_context66')

class HostTimeRing(MultiscaleRing):
    def _time(self,records,clock,host):
        if not host:return np.zeros(12)
        involved=records[(records[:,1]==host)|(records[:,2]==host)];out=[]
        for window in self.windows:
            r=involved[involved[:,0]>=clock-int(window*1e9)]
            if not len(r):out.extend([0.,0.,1.,0.]);continue
            stamps=r[:,0];diff=np.diff(stamps)
            if (diff<0).any():raise ValueError('Noncausal record order')
            mean=np.log1p(diff.mean()/1e6) if len(diff) else 0.
            std=np.log1p(diff.std()/1e6) if len(diff) else 0.
            out.extend([mean,std,(clock-int(stamps[-1]))/(window*1e9),
                        (int(stamps[-1])-int(stamps[0]))/(window*1e9)])
        return np.asarray(out,dtype=float)
    def query_time(self,timestamp,src,dst):
        records,clock=self._records_before(timestamp);ends=[]
        for host in [src,dst]:
            vector=self._host_vector(records,clock,host);direction=self._direction(records,clock,host);timing=self._time(records,clock,host)
            ends.append((vector,direction,timing))
        ends.sort(key=lambda z:self._activity_key(z[0])+tuple(z[1])+tuple(z[2]),reverse=True)
        return tuple(np.concatenate([z[j] for z in ends]) for j in range(3))

def repair_frame(raw,data,view):
    if view not in VIEWS:raise ValueError('Unreviewed time view')
    if not np.array_equal(raw.index.to_numpy(),data['row_ids']):raise ValueError('Feature identity mismatch')
    host=np.asarray(data['prior_multiscale']);time=np.asarray(data['prior_hosttime'])
    if host.shape!=(len(raw),60) or time.shape!=(len(raw),24) or not np.isfinite(time).all():raise ValueError('Invalid host timing')
    if view=='legacy25_age_context66':time=time[:,AGE_INDEX];tn=[TIME_COLUMNS[j] for j in AGE_INDEX]
    else:tn=list(TIME_COLUMNS)
    names=list(CONTEXT_NAMES)+tn;x=pd.DataFrame(np.c_[host,time],index=raw.index,columns=names);cat=[]
    if view.startswith('legacy25'):
        num,cat=columns('noport25');x=pd.concat([DomainFeatures('noport25').transform(raw),x],axis=1);names=list(num)+names
    return x,names,list(cat)
