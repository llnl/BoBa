#!/bin/bash
htlatex boba_bibliography.tex
bibtex boba_bibliography
htlatex boba_bibliography.tex
htlatex boba_bibliography.tex