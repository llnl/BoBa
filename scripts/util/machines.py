# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import socket
import os
import platform

def get_machine():
    machine = str(socket.gethostname())
    system  = str(platform.system())

    #
    # MSU h200s
    #
    if 'dev-amd24-h200' in machine:
        return ('msu_h200', 'MSU', 'cuda_h100')

    #
    # LLNL machines
    # 
    if 'matrix' in machine:
        return ('matrix', 'TOSS4', 'cuda_h100')

    if 'dane' in machine:
        return ('dane', 'TOSS4', 'cpu')
    if 'rzhound' in machine:
        return ('rzhound', 'TOSS4', 'cpu')
    if 'rzwhippet' in machine:
        return ('rzwhippet', 'TOSS4', 'cpu')
    if 'toss_4_x86_64_ib' == os.environ.get("SYS_TYPE"):
        return ('dane', 'TOSS4', 'hip')

    if 'tuolumne' in machine:
        return ('tuolumne', 'TOSS4_CRAY', 'hip')
    if 'rzadams' in machine:
        return ('rzadams', 'TOSS4_CRAY', 'hip')
    if 'toss_4_x86_64_ib_cray' == os.environ.get("SYS_TYPE"):
        return ('tuolumne', 'TOSS4_CRAY', 'hip')
    #
    # MSU machines
    #
    if 'dev-amd20-v100' in machine:
        return ('msu_v100', 'MSU', 'cuda_h100')
    if 'dev-amd20' in machine:
        return ('msu_amd_cpu', 'MSU', 'cpu')

    #
    # Apple
    #
    if 'MacBook' in machine:
        return ('macbook', 'MACOS', 'cpu')
    if 'Darwin' in system:
        return ('macbook', 'MACOS', 'cpu')
    if 'MBP' in machine:
        return ('macbook', 'MACOS', 'cpu')
    
    #
    # Linux
    #
    if 'Linux' in system:
        return ('linux','LINUX','cpu')

    print('unsupported machine! ' + machine)
    quit()
