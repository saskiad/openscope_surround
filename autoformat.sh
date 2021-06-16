#!/bin/sh

# Autoformat one or more Python files using Black with consistent settings.
# -S disables string normalization (leave single quotes as single quotes)
# -l 80 sets line length to 80 characters
black -S -l 80 "$@"
