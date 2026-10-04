"""Offline smoke test for the included Go adaptations."""
import numpy as np
import torch
from common import ROOT
from nicad_go import extract
from go_code2vec import GoCode2VecEncoder
from statement_vectors import encode_statements

def main():
    source='package demo\nfunc Sum(xs []int) int {\n total := 0\n for _, x := range xs { total += x }\n return total\n}\n'
    target=source.replace('total','result').replace('Sum','Add')
    a=extract(source,'source.go');b=extract(target,'target.go')
    assert len(a)==len(b)==1 and a[0].rows==b[0].rows
    assert all(a[0].line_map)
    print('NiCad-Go preprocessing passed; official comparison engine not run.')
    torch.set_num_threads(2)
    encoder=GoCode2VecEncoder(ROOT/'models/go_code2vec.pt')
    v=encoder.file_vector(source)
    assert v is not None and np.isfinite(v).all() and np.isclose(v@v,1.)
    units=encode_statements(encoder,source)
    assert units and all(np.isfinite(u['vector']).all() for u in units)
    print('Go code2vec checkpoint, file and statement encoding passed.')

if __name__=='__main__':main()
