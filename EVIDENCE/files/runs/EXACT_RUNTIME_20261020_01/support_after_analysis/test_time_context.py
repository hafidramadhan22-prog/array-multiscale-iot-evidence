import numpy as np
from sbfe import Packet
from time_context import HostTimeRing,AGE_INDEX
from multiscale_context import MultiscaleRing

def test_timing_uses_only_relative_observed_history_and_queries_are_read_only():
    ring=HostTimeRing(32)
    for stamp in [0,10_000_000,30_000_000]:ring.push(stamp,Packet(src=1,dst=2,wire=60,protocol=6,flags=2))
    host,direction,time=ring.query_time(50_000_000,1,2)
    np.testing.assert_allclose(time[:4],[np.log1p(15),np.log1p(5),.2,.3])
    assert ring.count==3 and len(host)==60 and len(direction)==36 and len(time)==24
    assert len(AGE_INDEX)==6

def test_old_host_and_direction_controls_are_exact_and_swap_invariant():
    a,b=HostTimeRing(16),MultiscaleRing(16)
    for i in range(40):
        p=Packet(src=1+i%3,dst=4,wire=60+i,sp=i+1,dp=80,protocol=6,flags=2 if i%2 else 20)
        a.push(i*5_000_000,p);b.push(i*5_000_000,p)
    host,direction,time=a.query_time(205_000_000,1,4);_,old,od=b.query_multiscale(205_000_000,1,4)
    np.testing.assert_array_equal(host,old);np.testing.assert_array_equal(direction,od)
    reverse=a.query_time(205_000_000,4,1)
    for left,right in zip((host,direction,time),reverse):np.testing.assert_array_equal(left,right)

def test_absolute_clock_translation_port_labels_missing_hosts_and_reset():
    a,b=HostTimeRing(8),HostTimeRing(8)
    for i in range(6):
        a.push(i*100,Packet(src=1,dst=2,sp=500+i,dp=20+i,wire=60,protocol=6,flags=2))
        b.push(10**15+i*100,Packet(src=1,dst=2,sp=1500+i,dp=1020+i,wire=60,protocol=6,flags=2))
    for x,y in zip(a.query_time(700,1,2),b.query_time(10**15+700,1,2)):np.testing.assert_array_equal(x,y)
    assert not a._time(a.data[:0],800,0).any()
    np.testing.assert_array_equal(a._time(a.data[:0],800,9)[2::4],np.ones(3))
    a.reset();assert a.count==0
