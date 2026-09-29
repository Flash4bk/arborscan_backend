import ctypes
from ctypes import wintypes
from pathlib import Path
import os
import subprocess
import sys
import unittest


@unittest.skipUnless(os.name=='nt','Windows Job Object integration')
class WindowsJobTest(unittest.TestCase):
    def test_parent_termination_stops_background_child(self):
        root=str(Path(__file__).resolve().parents[1]/'deploy-vps')
        code=(f'import sys;sys.path.insert(0,{root!r});'
              'from ops_windows_job import contain_children;contain_children();'
              'import subprocess,time;'
              'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],creationflags=subprocess.CREATE_NO_WINDOW);'
              'print(p.pid,flush=True);time.sleep(60)')
        parent=subprocess.Popen([sys.executable,'-B','-c',code],stdout=subprocess.PIPE,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            child=int(parent.stdout.readline().strip())
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
            kernel.CloseHandle.argtypes=[wintypes.HANDLE]
            handle=kernel.OpenProcess(0x100000,False,child);self.assertTrue(handle)
            try:
                parent.terminate();parent.wait(timeout=10)
                self.assertEqual(kernel.WaitForSingleObject(handle,5000),0)
            finally:kernel.CloseHandle(handle)
        finally:
            if parent.poll() is None:parent.terminate();parent.wait(timeout=10)
            parent.stdout.close()


if __name__=='__main__':unittest.main()
