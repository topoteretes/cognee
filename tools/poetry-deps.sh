#!/usr/bin/env bash

set -e
set -u

shopt -s dotglob
shopt -s nullglob
# verifies if python packages have proper structure
# find all directories not containing __init__
error=
while IFS= read -r d; do
  myarray=(`find $d -maxdepth 1 -name "*.py"`)
  if [ ${#myarray[@]} -gt 0 ]; then
    if [[ $@ == *--fix* ]]; then
      echo Will create "$d/__init__.py"
      touch "$d/__init__.py"
    else
      echo Folder "$d" lacks __init__.py file
      error="yes"
    fi
  fi
done < <(find . -mindepth 1 -not -path "./docs/website/node_modules*" -type d -regex "^./[^.^_].*" '!' -exec test -e "{}/__init__.py" ';' -print)

if [ -z $error ]; then
  exit 0
fi

# error in package
exit 1