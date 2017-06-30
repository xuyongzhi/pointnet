from __future__ import print_function


def print_n(list,n):
    for i,data in enumerate(list):
        if i >= n:
            break
        print('{}: {}'.format(i,data))
