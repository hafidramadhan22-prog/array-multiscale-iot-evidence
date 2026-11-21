"""Prior-only bounded host history with direction-aware probe/response summaries."""
import numpy as np
import pandas as pd
from wb_behavior import PriorEndpointRing,HOST_NAMES
from e4 import DomainFeatures,columns

WINDOWS=(.1,1.,10.)
DIRECTION_NAMES=('log_out_syn','log_out_syn_peerports','out_syn_port_diversity',
                 'incoming_rst_per_out_syn_plus1','incoming_synack_per_out_syn_plus1','outbound_fraction')
CONTEXT_NAMES=[f'multiscale_{slot}_{int(w*1000)}ms_{name}' for slot in ('higher','lower') for w in WINDOWS for name in HOST_NAMES]
DIRECTION_COLUMNS=[f'directional_{slot}_{int(w*1000)}ms_{name}' for slot in ('higher','lower') for w in WINDOWS for name in DIRECTION_NAMES]

class MultiscaleRing(PriorEndpointRing):
    def __init__(self,capacity=16384):super().__init__(capacity,WINDOWS)
    def _direction(self,records,clock,host):
        if not host:return np.zeros(18)
        records=records[(records[:,1]==host)|(records[:,2]==host)]
        out=[]
        for w in self.windows:
            r=records[records[:,0]>=clock-int(w*1e9)]
            if not len(r):out.extend([0.]*6);continue
            tcp=r[:,7]==6;sent=r[:,1]==host;received=r[:,2]==host
            syn=(r[:,6]&2)!=0;ack=(r[:,6]&16)!=0;rst=(r[:,6]&4)!=0
            probes=tcp&sent&syn&~ack;n=int(probes.sum());ports=len(np.unique(r[probes,4]))
            out.extend([np.log1p(n),np.log1p(ports),ports/n if n else 0.,
                        np.sum(tcp&received&rst)/(n+1),np.sum(tcp&received&syn&ack)/(n+1),sent.mean()])
        return np.asarray(out,dtype=float)
    def query_multiscale(self,timestamp,src,dst):
        records,clock=self._records_before(timestamp)
        source=self._host_vector(records,clock,src);dest=self._host_vector(records,clock,dst)
        endpoints=[(source,self._direction(records,clock,src)),(dest,self._direction(records,clock,dst))]
        endpoints.sort(key=lambda z:self._activity_key(z[0])+tuple(z[1]),reverse=True)
        return source,np.concatenate([z[0] for z in endpoints]),np.concatenate([z[1] for z in endpoints])

def repair_frame(raw,data,view):
    if view not in ('multiscale60','legacy25_multiscale60','legacy25_multiscale_directional96'):
        raise ValueError('Unreviewed multiscale view')
    if not np.array_equal(raw.index.to_numpy(),data['row_ids']):raise ValueError('Feature identity mismatch')
    context=np.asarray(data['prior_multiscale'],dtype=float)
    if context.shape!=(len(raw),60) or not np.isfinite(context).all():raise ValueError('Invalid multiscale data')
    names=list(CONTEXT_NAMES);x=pd.DataFrame(context,index=raw.index,columns=names);cat=[]
    if view.endswith('directional96'):
        direct=np.asarray(data['prior_directional'],dtype=float)
        if direct.shape!=(len(raw),36) or not np.isfinite(direct).all():raise ValueError('Invalid directional data')
        x=pd.concat([x,pd.DataFrame(direct,index=raw.index,columns=DIRECTION_COLUMNS)],axis=1);names+=DIRECTION_COLUMNS
    if view.startswith('legacy25'):
        num,cat=columns('noport25');x=pd.concat([DomainFeatures('noport25').transform(raw),x],axis=1);names=list(num)+names
    return x,names,list(cat)
