#!/usr/bin/env python3
import gvar as gv
import sys

if len(sys.argv) != 2:
    sys.exit('USAGE: [python] print_params.py <result_file>')

fit_result = gv.load(sys.argv[1])

state = ('0', 'T1g', 1)
if state not in [('0', 'T1g', 0), ('0', 'T1g', 1)]:
    sys.exit("only supports T1g irrpes currently")

n_params  = []
nn_params = []

for k in fit_result:
    if k[0][0] == state:
        if len(k[0]) > 1 and k[0][1] == 'N':
            n_params.append(k[1])
        if len(k[0]) > 1 and k[0][1] == 'R':
            nn_params.append(k[1])

print(state)
n_params.sort()
print('single nucleon params')
print('---------------------------------------------------------------')
#print(n_params)
n = state[2]
for p in n_params:
    post  = fit_result[((state, 'N', '%d' %n), p)]
    prior = fit_result[((state,), 'prior')][((state, 'N', '%d' %n), p)]
    print('%5s    %15s    [ %s ]' %(p, post, prior))

print()
print(state, 'params')
print('---------------------------------------------------------------')
nn_params.sort()
print(nn_params)
for p in nn_params:
    post  = fit_result[((state, 'R', ('%d' %n,'%d' %n)), p)]
    prior = fit_result[((state,), 'prior')][((state, 'R', ('%d' %n,'%d' %n)), p)]
    rel_diff = (post-prior).mean / (post-prior).sdev
    print('%5s    %15s    [ %s ]' %(p, post, prior))

#import IPython; IPython.embed()
