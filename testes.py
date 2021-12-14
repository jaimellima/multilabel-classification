# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 00:31:17 2021

@author: jaimel
"""

import argparse
parser = argparse.ArgumentParser(description = 'Um programa de exemplo.')

parser.add_argument('--frase', 
                    action='store', 
                    dest='frase', 
                    default='hello world', 
                    required=True, 
                    help='informe a frase!')

arguments = parser.parse_args()

print(arguments.frase)