"""Own-child Windows Job accounting; no global settings or registry changes."""
import ctypes as C
from ctypes import wintypes as W
import os
assert os.name=='nt','Windows-only footprint counter'
K=C.WinDLL('kernel32',use_last_error=True)
class BASIC(C.Structure):
    _fields_=[('PerProcessUserTimeLimit',C.c_longlong),('PerJobUserTimeLimit',C.c_longlong),('LimitFlags',W.DWORD),('MinimumWorkingSetSize',C.c_size_t),('MaximumWorkingSetSize',C.c_size_t),('ActiveProcessLimit',W.DWORD),('Affinity',C.c_size_t),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]
class IO(C.Structure):
    _fields_=[(n,C.c_ulonglong) for n in ['ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount']]
class EXT(C.Structure):
    _fields_=[('BasicLimitInformation',BASIC),('IoInfo',IO),('ProcessMemoryLimit',C.c_size_t),('JobMemoryLimit',C.c_size_t),('PeakProcessMemoryUsed',C.c_size_t),('PeakJobMemoryUsed',C.c_size_t)]
K.CreateJobObjectW.argtypes=[C.c_void_p,W.LPCWSTR];K.CreateJobObjectW.restype=W.HANDLE
K.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];K.OpenProcess.restype=W.HANDLE
K.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE];K.AssignProcessToJobObject.restype=W.BOOL
K.QueryInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD,C.c_void_p];K.QueryInformationJobObject.restype=W.BOOL
K.SetInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD];K.SetInformationJobObject.restype=W.BOOL
K.CloseHandle.argtypes=[W.HANDLE];K.CloseHandle.restype=W.BOOL
def check(ok):
    if not ok:raise C.WinError(C.get_last_error())
class Job:
    def __init__(self,pid):
        self.h=K.CreateJobObjectW(None,None);check(self.h)
        # Enable commit accounting on this own-child job only. The high cap is
        # far above these measured predictors; no global setting is changed.
        limits=EXT();limits.BasicLimitInformation.LimitFlags=0x100|0x200
        limits.ProcessMemoryLimit=4*2**30;limits.JobMemoryLimit=4*2**30
        check(K.SetInformationJobObject(self.h,9,C.byref(limits),C.sizeof(limits)))
        p=K.OpenProcess(0x0100|0x0001|0x0400,False,pid)
        try:check(p);check(K.AssignProcessToJobObject(self.h,p))
        finally:
            if p:K.CloseHandle(p)
    def peak(self):
        v=EXT();check(K.QueryInformationJobObject(self.h,9,C.byref(v),C.sizeof(v),None))
        return dict(PeakProcessMemoryUsed=int(v.PeakProcessMemoryUsed),PeakJobMemoryUsed=int(v.PeakJobMemoryUsed))
    def close(self):
        if self.h:check(K.CloseHandle(self.h));self.h=None
def pulse(size):
    K.VirtualAlloc.argtypes=[C.c_void_p,C.c_size_t,W.DWORD,W.DWORD];K.VirtualAlloc.restype=C.c_void_p
    K.VirtualFree.argtypes=[C.c_void_p,C.c_size_t,W.DWORD];K.VirtualFree.restype=W.BOOL
    p=K.VirtualAlloc(None,size,0x1000|0x2000,0x04);check(p)
    check(K.VirtualFree(p,0,0x8000))
