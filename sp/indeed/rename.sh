#!/bin/bash

i=1
for f in *.html; do
    mv "$f" "$i.html"
    i=$((i+1))
done
