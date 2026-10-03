"""Fetch public repository objects and read the exact frozen revisions."""
import subprocess
from pathlib import Path
from common import ROOT

REPOSITORIES={'ethereum/go-ethereum','bnb-chain/bsc','0xPolygon/bor','celo-org/celo-blockchain'}

def git(repo,*args):
    if repo not in REPOSITORIES:raise ValueError('Unsupported repository')
    directory=ROOT/'.cache/git_repos'/ (repo.replace('/','_')+'.git')
    if not directory.exists():
        directory.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--mirror','https://github.com/'+repo+'.git',str(directory)],check=True)
    return subprocess.check_output(['git','--git-dir='+str(directory),*args]).decode('utf-8')

def code(repo,revision,path):
    return git(repo,'show',revision+':'+path)
