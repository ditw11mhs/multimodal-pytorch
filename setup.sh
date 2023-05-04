#!/bin/bash

# helper function to create a conda environment for the current directory
function mkcenv() {
	if [[ -f ".cenv" ]]; then
		printf ".cenv file already exists. If this is a mistake use the rmcenv command\n"
	else
		cenv_name="$(basename $PWD)"
		conda create --name "$cenv_name" "$@"
		conda run -n "$cenv_name" poetry install
		printf "$cenv_name\n" >".cenv"
		chmod 600 .cenv
	fi
}

mkcenv python=3.10 -y
