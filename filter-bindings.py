#!/usr/bin/env python3

"""
    Filters the EMSCRIPTEN_BINDINGS section of the bindings.cpp file, keeping only what is used in every file given as parameters.
    Outputs the result to bindings2.cpp
"""

import sys
import os
import re

import argparse

SRC_BINDINGS = "bindings.cpp"
DST_BINDINGS = "bindings2.cpp"


parser = argparse.ArgumentParser()
parser.add_argument( "files", nargs = "+", default = None, help="The .js files to analyze" )

args = parser.parse_args()

props = set()
rePropFinder = re.compile( r"\bcv\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\b" )
#rePropFinder = re.compile( "\bcv\.([a-zA-Z_][a-zA-Z0-9_]*)" )

for f in args.files:
    jssource = open( f, "r" ).read()
    for match in rePropFinder.finditer( jssource ):
        props.add( match.group( 1 ) )

#print( props )

def replace( match ):
    lines = []
    inSection = False
    keepSection = False
    #   each "section" starts with a first line naming the prop, and ends with a line ending with a semicolon (it may be the same)
    for line in match.group(2).split( "\n" ):
        if not inSection:
            inSection = bool( line.strip() )
            if inSection:   #   start of a section
                identifierMatched = re.search( r'"([^"]+)"', line )
                keepSection = identifierMatched and identifierMatched.group(1) in props
                if keepSection:
                    lines.append( "" )  #   new section: insert a blank line before it

        if inSection and keepSection:
            lines.append( line )

        if inSection and re.search( r";$", line ):
            inSection = keepSection = False

    return match.group(1) + "\n".join(lines) + match.group(3)

text = open( SRC_BINDINGS, 'r' ).read()
text = re.sub( r"(\bEMSCRIPTEN_BINDINGS\b\s*\(\s*\btestBinding\b\s*\)\s*\{)([^{}]*)(\})", replace, text )

open( DST_BINDINGS, 'w' ).write( text )
